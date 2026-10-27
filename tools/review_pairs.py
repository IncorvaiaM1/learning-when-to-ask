#!/usr/bin/env python3
"""Interactive helper for the 100-pair hand verification (Part 1).

Walks data/hand_verified.jsonl one pair at a time and records YOUR verdict. It never
decides anything itself -- it only saves typing. Resumable: already-filled rows are skipped.

    python3 tools/review_pairs.py          # a = accept, r = reject (then a short reason), q = quit

Reject if: the prompt's class label is wrong (the "ambiguous" one is actually answerable,
or the "clear" one is missing info), the chosen response isn't better than the rejected one,
the clarifying question isn't specific, or the direct answer is wrong / low quality.
"""
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths_config import HAND_VERIFIED_PATH  # noqa: E402


def show(label, text):
    print(f"\n\033[1m{label}\033[0m")
    print(textwrap.indent(textwrap.fill(text, 100, replace_whitespace=False), "  "))


def main():
    path = Path(HAND_VERIFIED_PATH)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [i for i, r in enumerate(rows) if r.get("verdict") not in ("accept", "reject")]
    print(f"{len(rows) - len(todo)}/{len(rows)} already done.")

    for n, i in enumerate(todo, 1):
        r = rows[i]
        print("\n" + "=" * 100)
        print(f"[{len(rows) - len(todo) + n}/{len(rows)}] {r['id']}   class = {r['class'].upper()}")
        show("PROMPT", r["prompt"])
        show("CHOSEN  (should be " + ("a specific clarifying question" if r["class"] == "ambiguous" else "a direct, correct answer") + ")", r["chosen"])
        show("REJECTED (should be " + ("a confident guess" if r["class"] == "ambiguous" else "a needless question") + ")", r["rejected"])
        while True:
            v = input("\n[a]ccept / [r]eject / [q]uit > ").strip().lower()
            if v in ("a", "r", "q"):
                break
        if v == "q":
            break
        r["verdict"] = "accept" if v == "a" else "reject"
        if v == "r":
            r["reject_reason"] = input("reason (short) > ").strip()
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    done = sum(1 for r in rows if r.get("verdict") in ("accept", "reject"))
    print(f"\nSaved. {done}/{len(rows)} verified. Run: python3 generate_preference_data.py --report")


if __name__ == "__main__":
    main()
