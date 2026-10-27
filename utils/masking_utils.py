"""
Shared tokenization + response-masking logic used by both Part A (train_sft.py) and
Part B (train_dpo.py). "Mask everything except the final assistant turn" is the same
operation in both places, so it lives here once instead of being re-derived
independently in two files.
"""

import torch


def tokenize_with_response_mask(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """
    TODO (student): tokenizes one chat-formatted conversation and marks which tokens
    belong to the final assistant turn. `messages` is a list of {"role", "content"}
    dicts (user/assistant, alternating, ending on an assistant turn) -- see data_loader.py.

    Must return a dict with:
      - "input_ids": 1D LongTensor -- the full tokenized conversation
        (`tokenizer.apply_chat_template(messages, tokenize=True, ...)`), truncated to
        at most max_seq_len tokens.

      - "attention_mask": 1D LongTensor, same length, all 1s (no padding needed here --
        batching/collating happens in the caller's train_loop()).

      - "response_mask": 1D LongTensor, same length: 1 for the final assistant-turn
        tokens, 0 everywhere else (prompt tokens and any earlier turns).

    This is the function the autograder unit-tests directly (see README.md's
    Autograder section) -- its signature and the three return keys above must not
    change. Both train_sft.build_sft_example() and train_dpo.build_dpo_example() call
    this rather than re-deriving the masking logic independently -- implement it once,
    here, and both Part A and Part B get it for free.

    Hint: `tokenizer.apply_chat_template(messages[:-1], tokenize=True,
    add_generation_prompt=True)` tokenizes everything up through the prompt that
    invites the final assistant turn. Its length tells you exactly where the response
    starts in the full conversation's tokenization -- everything before that index is
    prompt, everything from it onward (up to the end-of-turn token) is response.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("messages must end on an assistant turn")

    full_ids = _as_id_list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    prompt_ids = _as_id_list(
        tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    )

    # The prompt (everything through "<|start_header_id|>assistant<|end_header_id|>\n\n")
    # must be an exact prefix of the full conversation; otherwise the boundary is wrong
    # and we'd silently train on prompt tokens. Fall back to the longest common prefix.
    start = len(prompt_ids)
    if full_ids[:start] != prompt_ids:
        start = 0
        for a, b in zip(full_ids, prompt_ids):
            if a != b:
                break
            start += 1

    # Response = everything from `start` to the end, which for the Llama-3 template is the
    # assistant content followed by <|eot_id|>. Keeping <|eot_id|> in the loss is what
    # teaches the model to stop after a short clarifying question.
    response_mask = [0] * start + [1] * (len(full_ids) - start)

    # Truncate from the right (keep the prompt intact; very long answers just get clipped).
    full_ids = full_ids[:max_seq_len]
    response_mask = response_mask[:max_seq_len]

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.long),
    }


def _as_id_list(out) -> list[int]:
    """apply_chat_template(tokenize=True) returns a list[int] in transformers 4.57, but
    some versions/flags return a BatchEncoding/dict; normalize to a flat list of ints."""
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    if isinstance(out, torch.Tensor):
        out = out.tolist()
    if out and isinstance(out[0], list):  # batched shape [[...]]
        out = out[0]
    return list(out)
