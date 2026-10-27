#!/usr/bin/env python3
"""Checks tokenize_with_response_mask() against the REAL base-model tokenizer/chat template.
Runs on the login node (tokenizer only, no weights). Exits non-zero if the mask is wrong."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.train_config import MODEL_ID  # noqa: E402
from utils.masking_utils import tokenize_with_response_mask  # noqa: E402
from utils.model_utils import load_tokenizer  # noqa: E402

tok = load_tokenizer(MODEL_ID)
reply = "Which file is the bug in, and what error do you see?"
msgs = [{"role": "user", "content": "Fix the bug."}, {"role": "assistant", "content": reply}]
o = tokenize_with_response_mask(msgs, tok, 1024)
ids, m = o["input_ids"], o["response_mask"]

resp = tok.decode(ids[m == 1])
prompt = tok.decode(ids[m == 0])
print(f"model      : {MODEL_ID}")
print(f"tokens     : {len(ids)} total, {int(m.sum())} in response")
print(f"response   : {resp!r}")
print(f"prompt tail: {prompt[-70:]!r}")
print(f"pad side   : {tok.padding_side}, pad token: {tok.pad_token!r}")

ok = resp.strip() == reply + "<|eot_id|>" and prompt.endswith("assistant<|end_header_id|>\n\n")
ok &= len(tokenize_with_response_mask(msgs, tok, 8)["input_ids"]) == 8
print("MASKING OK" if ok else "MASKING MISMATCH -- tell Claude what printed above")
sys.exit(0 if ok else 1)
