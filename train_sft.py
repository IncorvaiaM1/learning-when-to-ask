"""
Part A -- LoRA SFT training entrypoint.

SFT warm-starts the target behavior: it fine-tunes the instruct base model on the
PREFERRED response of each preference pair you built (a clarifying question for an
ambiguous prompt, a direct answer for a clear one), so that DPO in Part B has a sensible
policy to sharpen rather than starting cold.

CLI args, tokenizer setup, base model + LoRA construction, and checkpointing are wired
up already. You implement utils/masking_utils.tokenize_with_response_mask() and
train_loop() -- see the TODO blocks and README.md's Part A for two gotchas: freezing
base weights, masking prompt tokens.
"""

import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup

from configs.paths_config import SFT_CHECKPOINT_DIR
from configs.train_config import (
    MODEL_ID,
    RANK,
    ALPHA,
    DROPOUT,
    TARGET_MODULES,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    SFT_MAX_SEQ_LEN,
    SFT_GRAD_ACCUM,
    SFT_WARMUP_RATIO,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    LOG_EVERY,
)
from data_loader import load_sft_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask


def build_model_and_tokenizer():
    """Loads the base model + tokenizer, and wraps the model with a LoRA adapter."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model()

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Sanity check: should be a small fraction of total params.

    return model, tokenizer


def build_sft_example(messages: list[dict], tokenizer, max_seq_len: int = SFT_MAX_SEQ_LEN) -> dict:
    """
    Wired up: converts utils.masking_utils.tokenize_with_response_mask()'s response_mask into
    the -100-masked labels tensor the causal-LM loss needs. Once you implement
    tokenize_with_response_mask(), this works for free -- nothing to implement here.

    Returns a dict with "input_ids", "attention_mask", "labels": labels equal
    input_ids where response_mask == 1 (the final assistant turn), -100 elsewhere.
    """
    example = tokenize_with_response_mask(messages, tokenizer, max_seq_len)

    labels = example["input_ids"].clone()
    labels[example["response_mask"] == 0] = -100

    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": labels,
    }


def train_loop(model, tokenizer, dataset) -> None:
    """
    Implements the Part A SFT training loop.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_sft_example() per example
         (e.g. via `dataset.map(...)`) to get input_ids/attention_mask/labels -- do not
         call build_sft_example() inside the per-step loop below, or you'll re-tokenize
         the same examples on every epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate into a batch, forward pass,
         compute loss, backward, optimizer step. Only LoRA parameters should have
         gradients -- true by default from build_model_and_tokenizer() above, but verify
         it if you change the LoRA setup (README.md Part A, gotcha #1).
      3. Log the loss periodically so you can sanity-check training is converging.

    Hyperparameters (SFT_BATCH_SIZE, SFT_EPOCHS, SFT_LEARNING_RATE, SFT_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune, not fixed by this skeleton.
    """
    # --- Gotcha #1: only LoRA params may be trainable. Fail loudly if not. ---
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    leaked = [n for n, _ in trainable if "lora_" not in n]
    if leaked:
        raise RuntimeError(f"Non-LoRA params are trainable (first: {leaked[0]}); freeze them.")
    print(f"[sft] {len(trainable)} trainable tensors, all LoRA.")

    # --- 1. Tokenize + mask once, up front. ---
    def _prep(ex):
        out = build_sft_example(ex["messages"], tokenizer, SFT_MAX_SEQ_LEN)
        return {k: v.tolist() for k, v in out.items()}

    data = dataset.map(_prep, remove_columns=dataset.column_names, desc="tokenize SFT")
    # Drop any example whose response got fully truncated away (no supervised tokens).
    data = data.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    examples = [
        {k: torch.tensor(ex[k], dtype=torch.long) for k in ("input_ids", "attention_mask", "labels")}
        for ex in data
    ]
    n_sup = sum(int((e["labels"] != -100).sum()) for e in examples)
    n_tok = sum(len(e["input_ids"]) for e in examples)
    print(f"[sft] {len(examples)} examples, {n_tok} tokens, {n_sup} supervised "
          f"({n_sup / max(n_tok, 1):.1%}) -- the rest is masked prompt.")

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        return {
            "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "labels": pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
        }

    # --- 2. Optimizer / schedule over LoRA params only. ---
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable], lr=SFT_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = math.ceil(len(examples) / (SFT_BATCH_SIZE * SFT_GRAD_ACCUM))
    total_steps = steps_per_epoch * SFT_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(SFT_WARMUP_RATIO * total_steps), total_steps
    )

    device = next(model.parameters()).device
    model.config.use_cache = False
    model.train()
    rng = random.Random(torch.initial_seed())
    history = []
    step, t0 = 0, time.time()

    for epoch in range(SFT_EPOCHS):
        order = list(range(len(examples)))
        rng.shuffle(order)
        batches = [order[i : i + SFT_BATCH_SIZE] for i in range(0, len(order), SFT_BATCH_SIZE)]
        running, n_running = 0.0, 0

        for bi, idxs in enumerate(batches):
            batch = {k: v.to(device) for k, v in collate([examples[i] for i in idxs]).items()}
            # HF shifts labels internally and ignores -100 => loss is over response tokens only.
            loss = model(**batch).loss
            (loss / SFT_GRAD_ACCUM).backward()
            running += loss.item()
            n_running += 1

            if (bi + 1) % SFT_GRAD_ACCUM == 0 or bi == len(batches) - 1:
                gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % LOG_EVERY == 0 or step == total_steps:
                    rec = {
                        "epoch": epoch,
                        "step": step,
                        "loss": running / n_running,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(gnorm),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    history.append(rec)
                    print(f"[sft] ep {epoch} step {step}/{total_steps} loss {rec['loss']:.4f} "
                          f"lr {rec['lr']:.2e} gnorm {rec['grad_norm']:.2f}", flush=True)
                    running, n_running = 0.0, 0

    model.train_history = history  # picked up by main() for the run log


def _write_run_log(args, history) -> None:
    """Config + loss curve next to the adapter, for the report and reproducibility."""
    import configs.train_config as tc

    cfg = {k: getattr(tc, k) for k in dir(tc) if k.isupper()}
    log = {"stage": "sft", "seed": args.seed, "config": cfg, "history": history}
    with open(os.path.join(args.checkpoint_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading base model {MODEL_ID} and building LoRA adapter...")
    model, tokenizer = build_model_and_tokenizer()

    print("Loading SFT examples (preferred responses from your preference data)...")
    dataset = load_sft_examples()

    print("Training starts...")
    train_loop(model, tokenizer, dataset)

    print(f"Saving LoRA adapter to {args.checkpoint_dir}...")
    model.save_pretrained(args.checkpoint_dir)
    # evaluate.py batch-generates with the tokenizer saved here. Decoder-only batched
    # generation needs LEFT padding; Llama's default (right) would put pad/eot tokens between
    # the prompt and the generation for every shorter prompt in a batch.
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"  # so it's written to tokenizer_config.json
    tokenizer.save_pretrained(args.checkpoint_dir)
    _write_run_log(args, getattr(model, "train_history", []))
    print("Part A SFT complete.")


if __name__ == "__main__":
    main()
