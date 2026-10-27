"""
Part B -- DPO training entrypoint.

DPO sharpens the Part A policy on your preference pairs: push up the log-prob of the
preferred response (ask when ambiguous / answer when clear) relative to the dispreferred
one, under a KL leash to a frozen reference (your Part A checkpoint). You must sweep beta
over at least three values -- run this script once per beta (`--beta`), each writing its
own adapter.

CLI args, loading your Part A LoRA checkpoint as the policy's starting point, and
per-beta checkpointing are wired up already. You implement sequence_log_prob(),
dpo_loss(), and the training loop around them -- see the TODO blocks and README.md's
Part B for the three things to know going in (the disable_adapter() trick, sequence
log-prob masking, the loss formula). build_dpo_example() itself is wired, built on top of
utils/masking_utils.tokenize_with_response_mask() -- see Part A for where that TODO lives.
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup

from configs.paths_config import SFT_CHECKPOINT_DIR, dpo_checkpoint_dir
from configs.train_config import (
    DPO_BATCH_SIZE,
    DPO_EPOCHS,
    DPO_LEARNING_RATE,
    DPO_BETA,
    DPO_MAX_SEQ_LEN,
    DPO_GRAD_ACCUM,
    DPO_EPOCHS as _DPO_EPOCHS,
    DPO_WARMUP_RATIO,
    DPO_REFERENCE,
    DPO_SFT_NLL_WEIGHT,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    LOG_EVERY,
)
from data_loader import load_dpo_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask


def build_policy_and_tokenizer(sft_checkpoint_dir: str):
    """
    Loads the base model and attaches your Part A LoRA adapter as the DPO policy's
    starting point. The *same* model, with its adapter disabled via
    `policy.disable_adapter()`, doubles as the frozen reference model -- see
    README.md Part B, gotcha #1. You do not need to load a second copy of the base model.
    """
    tokenizer = load_tokenizer(sft_checkpoint_dir)
    base_model = load_base_model()
    policy = PeftModel.from_pretrained(base_model, sft_checkpoint_dir, is_trainable=True)

    return policy, tokenizer


def build_dpo_example(
    prompt: str,
    chosen_messages: list[dict],
    rejected_messages: list[dict],
    tokenizer,
    max_seq_len: int = DPO_MAX_SEQ_LEN,
) -> dict:
    """
    Wired up: calls utils.masking_utils.tokenize_with_response_mask() once for
    chosen_messages and once for rejected_messages. Once you implement that one
    function (see Part A), this works for free -- nothing to implement here.

    `chosen_messages`/`rejected_messages` are each a list of {"role", "content"} dicts:
    the shared user prompt followed by the differing final assistant completion -- see
    data_loader.py. `prompt` itself isn't needed here (it's already the shared prefix
    inside both messages lists) but is accepted for a uniform call signature with
    data_loader's (prompt, chosen, rejected) triples.

    Returns a dict with, for both "chosen" and "rejected":
      - "{chosen,rejected}_input_ids", "{chosen,rejected}_attention_mask",
        "{chosen,rejected}_response_mask" -- see
        utils.masking_utils.tokenize_with_response_mask()'s return value.
    """
    chosen = tokenize_with_response_mask(chosen_messages, tokenizer, max_seq_len)
    rejected = tokenize_with_response_mask(rejected_messages, tokenizer, max_seq_len)

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_response_mask": chosen["response_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_response_mask": rejected["response_mask"],
    }


def sequence_log_prob(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """
    TODO (student): computes, for each sequence in the batch, the sum of log-probs of
    the response tokens only (response_mask == 1), under `model`'s current weights.
    Returns a 1D tensor of shape (batch_size,).

    Call this once with the policy active, and once more inside a
    `with model.disable_adapter():` block to get the reference log-probs -- see
    README.md Part B, gotcha #1.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    # Position t predicts token t+1: drop the last logit and the first label.
    logits = logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:].to(logits.dtype)

    # log_softmax + gather, done in fp32 for a numerically stable sum over ~100s of tokens.
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = DPO_BETA,
) -> torch.Tensor:
    """
    TODO (student): implements the DPO loss --

        L_DPO = -log sigma( beta * [ (policy_chosen_logps - reference_chosen_logps)
                                      - (policy_rejected_logps - reference_rejected_logps) ] )

    averaged over the batch.

    This function's signature must not change. The GTA reference tests exercise it
    directly (tests/test_dpo_loss.py).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    # -log sigma(x) == softplus(-x); logsigmoid is the numerically stable form.
    return -F.logsigmoid(logits).mean()


def train_loop(policy, tokenizer, dataset, beta: float = DPO_BETA) -> None:
    """
    Implements the Part B DPO training loop for a single beta.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_dpo_example() per
         (prompt, chosen, rejected) triple (e.g. via `dataset.map(...)`) -- do not call
         it inside the per-step loop below, or you'll re-tokenize every example on every
         epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate, then compute
         policy_chosen_logps / policy_rejected_logps via sequence_log_prob() with the
         adapter enabled.
      3. Compute reference_chosen_logps / reference_rejected_logps via the same
         sequence_log_prob() call, inside `with policy.disable_adapter():`.
      4. Compute dpo_loss(..., beta=beta), backward, optimizer step.
      5. Log the loss periodically.

    Hyperparameters (DPO_BATCH_SIZE, DPO_EPOCHS, DPO_LEARNING_RATE, DPO_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune. `beta` is passed in per run (--beta) so
    the sweep trains a separate adapter for each value.
    """
    trainable = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    leaked = [n for n, _ in trainable if "lora_" not in n]
    if not trainable or leaked:
        raise RuntimeError(f"Expected only LoRA params trainable; got leaked={leaked[:1]}, n={len(trainable)}")

    # --- 1. Tokenize once. ---
    def _prep(ex):
        out = build_dpo_example(ex["prompt"], ex["chosen"], ex["rejected"], tokenizer, DPO_MAX_SEQ_LEN)
        return {k: v.tolist() for k, v in out.items()}

    data = dataset.map(_prep, remove_columns=dataset.column_names, desc="tokenize DPO")
    data = data.filter(lambda ex: any(ex["chosen_response_mask"]) and any(ex["rejected_response_mask"]))
    keys = [f"{side}_{k}" for side in ("chosen", "rejected") for k in ("input_ids", "attention_mask", "response_mask")]
    examples = [{k: torch.tensor(ex[k], dtype=torch.long) for k in keys} for ex in data]
    print(f"[dpo] {len(examples)} preference pairs after tokenization.")

    pad_id = tokenizer.pad_token_id
    device = next(policy.parameters()).device

    def collate(batch):
        """Stack chosen then rejected into ONE padded batch of size 2B -> one forward pass."""
        ids = [b["chosen_input_ids"] for b in batch] + [b["rejected_input_ids"] for b in batch]
        att = [b["chosen_attention_mask"] for b in batch] + [b["rejected_attention_mask"] for b in batch]
        rsp = [b["chosen_response_mask"] for b in batch] + [b["rejected_response_mask"] for b in batch]
        return (
            pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device),
            pad_sequence(att, batch_first=True, padding_value=0).to(device),
            pad_sequence(rsp, batch_first=True, padding_value=0).to(device),
        )

    # --- Reference log-probs. ---
    # "base": the README's recipe -- the same model with the LoRA adapter switched off
    #         (policy.disable_adapter()), i.e. the untouched instruct checkpoint.
    # "sft":  textbook DPO -- reference = the SFT policy itself. Since the reference is
    #         frozen, its log-probs are constants: we compute them ONCE here (adapter on,
    #         no grad) before any update, still with a single model copy.
    # Either way the reference never changes during training, so we cache it per example.
    policy.eval()
    with torch.no_grad():
        for i in range(0, len(examples), DPO_BATCH_SIZE):
            chunk = examples[i : i + DPO_BATCH_SIZE]
            ids, att, rsp = collate(chunk)
            if DPO_REFERENCE == "base":
                with policy.disable_adapter():
                    ref = sequence_log_prob(policy, ids, att, rsp)
            else:
                ref = sequence_log_prob(policy, ids, att, rsp)
            b = len(chunk)
            for j, ex in enumerate(chunk):
                ex["ref_chosen"] = ref[j].detach().cpu()
                ex["ref_rejected"] = ref[b + j].detach().cpu()
    print(f"[dpo] reference = {DPO_REFERENCE!r}; cached reference log-probs.")

    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=DPO_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(examples) / (DPO_BATCH_SIZE * DPO_GRAD_ACCUM))
    total_steps = steps_per_epoch * _DPO_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(DPO_WARMUP_RATIO * total_steps), total_steps)

    policy.config.use_cache = False
    policy.train()
    rng = random.Random(torch.initial_seed())
    history, step, t0 = [], 0, time.time()
    agg = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "r_chosen": 0.0, "r_rejected": 0.0,
           "logp_chosen": 0.0, "logp_rejected": 0.0, "n": 0}

    for epoch in range(_DPO_EPOCHS):
        order = list(range(len(examples)))
        rng.shuffle(order)
        batches = [order[i : i + DPO_BATCH_SIZE] for i in range(0, len(order), DPO_BATCH_SIZE)]

        for bi, idxs in enumerate(batches):
            chunk = [examples[i] for i in idxs]
            ids, att, rsp = collate(chunk)
            b = len(chunk)

            # --- 2. Policy log-probs (adapter ON, with grad). ---
            logps = sequence_log_prob(policy, ids, att, rsp)
            pc, pr = logps[:b], logps[b:]
            # --- 3. Reference log-probs (cached above; adapter OFF or frozen SFT). ---
            rc = torch.stack([ex["ref_chosen"] for ex in chunk]).to(device)
            rr = torch.stack([ex["ref_rejected"] for ex in chunk]).to(device)

            # --- 4. Loss. ---
            loss = dpo_loss(pc, pr, rc, rr, beta=beta)
            if DPO_SFT_NLL_WEIGHT > 0:
                # Optional anchor (RPO-style): length-normalized NLL on the chosen response.
                n_tok = rsp[:b, 1:].sum(-1).clamp(min=1)
                loss = loss + DPO_SFT_NLL_WEIGHT * (-(pc / n_tok).mean())
            (loss / DPO_GRAD_ACCUM).backward()

            with torch.no_grad():
                r_c = beta * (pc - rc)
                r_r = beta * (pr - rr)
                agg["loss"] += loss.item()
                agg["acc"] += (r_c > r_r).float().mean().item()
                agg["margin"] += (r_c - r_r).mean().item()
                agg["r_chosen"] += r_c.mean().item()
                agg["r_rejected"] += r_r.mean().item()
                agg["logp_chosen"] += pc.mean().item()
                agg["logp_rejected"] += pr.mean().item()
                agg["n"] += 1

            if (bi + 1) % DPO_GRAD_ACCUM == 0 or bi == len(batches) - 1:
                gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # --- 5. Log. ---
                if step % LOG_EVERY == 0 or step == total_steps:
                    n = max(agg.pop("n"), 1)
                    rec = {k: v / n for k, v in agg.items()}
                    rec.update(epoch=epoch, step=step, lr=scheduler.get_last_lr()[0],
                               grad_norm=float(gnorm), elapsed_s=round(time.time() - t0, 1))
                    history.append(rec)
                    print(f"[dpo b={beta}] ep {epoch} step {step}/{total_steps} loss {rec['loss']:.4f} "
                          f"acc {rec['acc']:.2f} margin {rec['margin']:.3f} "
                          f"logp(c/r) {rec['logp_chosen']:.1f}/{rec['logp_rejected']:.1f}", flush=True)
                    agg = {k: 0.0 for k in agg}
                    agg["n"] = 0

    policy.train_history = history


def _write_run_log(args, checkpoint_dir, history) -> None:
    import configs.train_config as tc

    cfg = {k: getattr(tc, k) for k in dir(tc) if k.isupper()}
    log = {"stage": "dpo", "beta": args.beta, "seed": args.seed,
           "sft_checkpoint_dir": args.sft_checkpoint_dir, "config": cfg, "history": history}
    with open(os.path.join(checkpoint_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)

    parser.add_argument(
        "--beta",
        type=float,
        default=DPO_BETA,
        help="DPO beta for this run (sweep by running once per value).",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override; defaults to a per-beta directory.",
    )

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or dpo_checkpoint_dir(args.beta)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading Part A checkpoint from {args.sft_checkpoint_dir} as DPO policy init...")
    policy, tokenizer = build_policy_and_tokenizer(args.sft_checkpoint_dir)

    print("Loading DPO preference pairs...")
    dataset = load_dpo_examples()

    print(f"Training starts (beta={args.beta})...")
    train_loop(policy, tokenizer, dataset, beta=args.beta)

    print(f"Saving DPO-aligned LoRA adapter to {checkpoint_dir}...")
    policy.save_pretrained(checkpoint_dir)
    # evaluate.py batch-generates with the tokenizer saved here. Decoder-only batched
    # generation needs LEFT padding; Llama's default (right) would put pad/eot tokens between
    # the prompt and the generation for every shorter prompt in a batch.
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"  # so it's written to tokenizer_config.json
    tokenizer.save_pretrained(checkpoint_dir)
    _write_run_log(args, checkpoint_dir, getattr(policy, "train_history", []))
    print(f"Part B DPO complete (beta={args.beta}).")


if __name__ == "__main__":
    main()
