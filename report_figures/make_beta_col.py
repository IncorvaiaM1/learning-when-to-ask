import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/mnt/user-data/uploads/MAIStuff--randLab/convAI/report_data"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 7.5, "axes.titleweight": "bold", "savefig.dpi": 300})
r = json.load(open(f"{D}/output__results.json"))["runs"]
dpo = sorted(r["dpo"], key=lambda d: d["beta"]); betas = [d["beta"] for d in dpo]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.1, 3.7), gridspec_kw={"height_ratios": [1.15, 1]})
for k, lab, c in (("answer_rate", "answer (clear)", ORANGE), ("ask_rate", "ask (ambiguous)", BLUE), ("mmlu_acc", "MMLU", AQUA)):
    ys = [100 * d[k] for d in dpo]
    a1.axhline(100 * r["base"][k], color=c, ls=(0, (3, 2)), lw=0.9, alpha=0.9)
    a1.plot(range(3), ys, color=c, lw=1.8, marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=1)
    for i, y in enumerate(ys):
        below = k in ("ask_rate", "mmlu_acc")
        a1.text(i, y - 2.2 if below else y + 2.2, f"{y:.0f}" if k != "mmlu_acc" else f"{y:.1f}", ha="center", va="top" if below else "bottom",
                fontsize=6, color=INK, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    a1.text(2.18, ys[-1] + (3 if k == "answer_rate" else 0), lab, va="center", fontsize=6.3, color=INK, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
a1.text(-0.12, 100 * r["base"]["ask_rate"] + 1.5, "dashed = base", fontsize=5.8, color=INK2)
a1.set_xticks(range(3)); a1.set_xticklabels([f"β={b}" for b in betas]); a1.set_xlim(-0.2, 2.9); a1.set_ylim(30, 108)
a1.set_ylabel("% of dev prompts"); a1.set_title("Behavior vs. β (final run)", loc="left")
for b, c in ((0.05, VIOLET), (0.1, BLUE), (0.5, ORANGE)):
    h = [x for x in json.load(open(f"{D}/checkpoints__dpo__beta_{b}__train_log.json"))["history"] if x["step"] % 10 == 0]
    a2.plot([x["step"] for x in h], [x["margin"] for x in h], color=c, lw=1.4)
    a2.text(h[-1]["step"] + 5, h[-1]["margin"], f"β={b}", va="center", fontsize=6.3, color=INK)
a2.set_xlim(0, 390); a2.set_xlabel("optimizer step"); a2.set_ylabel("reward margin"); a2.set_title("DPO implicit reward margin", loc="left")
for ax in (a1, a2):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(h_pad=1.2)
fig.savefig("beta_sweep.pdf"); fig.savefig("beta_sweep.png")
print("ok")
