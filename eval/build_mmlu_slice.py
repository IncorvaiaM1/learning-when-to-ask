#!/usr/bin/env python3
"""Builds a DEV MMLU slice for the capability-retention self-check.

The graders use their own fixed 200-question slice; this one is ours, for tracking the
trend (base vs. SFT vs. DPO) during development. Deterministic given --seed.

    python3 eval/build_mmlu_slice.py                 # 200 questions -> eval/mmlu_slice.jsonl
    python3 eval/build_mmlu_slice.py --n 400 --seed 1
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.evaluation_config import MMLU_SLICE_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=6604)
    ap.add_argument("--out", default=MMLU_SLICE_PATH)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(args.seed).sample(range(len(ds)), args.n)
    with open(args.out, "w", encoding="utf-8") as f:
        for k, i in enumerate(idx):
            r = ds[i]
            f.write(json.dumps({"id": f"m_{k:03d}", "subject": r["subject"], "question": r["question"],
                                "choices": list(r["choices"]), "answer": int(r["answer"])}) + "\n")
    print(f"Wrote {args.n} MMLU questions -> {args.out}")


if __name__ == "__main__":
    main()
