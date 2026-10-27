#!/usr/bin/env python3
"""Build figs/beta_sweep.pdf for the A2 report.

Left panel : ask-rate / answer-rate / MMLU vs beta (from output/results.json), base as dashed lines.
Right panel: DPO reward margin vs optimizer step per beta (from checkpoints/dpo/beta_*/train_log.json).

    python make_figures.py --repo ../CAI_Assignment_2
Also prints a LaTeX-ready table row block you can paste into Table 1.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../CAI_Assignment_2")
    ap.add_argument("--results", default=None, help="defaults to <repo>/output/results.json")
    ap.add_argument("--out", default="figs/beta_sweep.pdf")
    a = ap.parse_args()

    res_path = a.results or os.path.join(a.repo, "output", "results.json")
    with open(res_path, encoding="utf-8") as f:
        res = json.load(f)
    runs = res["runs"]
    dpo = sorted(runs["dpo"], key=lambda r: r["beta"])
    betas = [r["beta"] for r in dpo]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    metrics = [("ask_rate", "Ask (ambig.)", "C0"), ("answer_rate", "Answer (clear)", "C1"),
               ("mmlu_acc", "MMLU", "C2")]
    for key, label, c in metrics:
        ys = [100 * (r.get(key) or 0) for r in dpo]
        ax1.plot(betas, ys, "o-", color=c, label=label)
        if runs["base"].get(key) is not None:
            ax1.axhline(100 * runs["base"][key], color=c, ls="--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks(betas)
    ax1.set_xticklabels([str(b) for b in betas])
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel("%")
    ax1.legend(fontsize=6, frameon=False)

    for log_path in sorted(glob.glob(os.path.join(a.repo, "checkpoints", "dpo", "beta_*", "train_log.json"))):
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        h = log.get("history", [])
        if h:
            ax2.plot([r["step"] for r in h], [r["margin"] for r in h], label=rf"$\beta$={log['beta']}")
    ax2.set_xlabel("step")
    ax2.set_ylabel("reward margin")
    ax2.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)

    def pct(x):
        return "--" if x is None else f"{100 * x:.1f}"
    print("\n% ---- paste into Table 1 ----")
    for name, r in [("Base", runs["base"]), ("SFT", runs["sft"])] + \
                   [(rf"DPO $\beta$={r['beta']}", r) for r in dpo]:
        print(f"{name:18s} & {pct(r.get('ask_rate'))} & {pct(r.get('answer_rate'))} & {pct(r.get('mmlu_acc'))} \\\\")


if __name__ == "__main__":
    main()
