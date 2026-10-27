import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/mnt/user-data/uploads/MAIStuff--randLab/convAI/report_data"
OUT = "/home/claude/figs"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300})

def res(name):
    p = f"{D}/output__results.json" if name == "final" else f"{D}/{name}__output__results.json"
    r = json.load(open(p))["runs"]
    out = {"base": r["base"], "sft": r["sft"]}
    for d in r["dpo"]: out[d["beta"]] = d
    return out
def log(prefix, beta):
    p = f"{D}/{prefix}checkpoints__dpo__beta_{beta}__train_log.json"
    h = json.load(open(p))["history"]
    return [r for r in h if r["step"] % 10 == 0]   # drop the final partial logging window

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

base = res("final")["base"]
METRICS = [("ask_rate", "Ask-rate, ambiguous", BLUE, base["ask_rate"] + 0.25, "target ≥ base+25"),
           ("answer_rate", "Answer-rate, clear", ORANGE, base["answer_rate"] - 0.05, "target ≥ base−5"),
           ("mmlu_acc", "MMLU accuracy", AQUA, base["mmlu_acc"] - 0.03, "target ≥ base−3")]

# ---------------- Figure 1: beta sweep (final run) ----------------
fin = res("final")
cols = ["base", "sft", 0.05, 0.1, 0.5]
labels = ["Base", "SFT", "DPO\nβ=0.05", "DPO\nβ=0.1", "DPO\nβ=0.5"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
for ax, (k, title, color, tgt, tlab) in zip(axes, METRICS):
    vals = [fin[c][k] * 100 for c in cols]
    bars = ax.bar(range(len(cols)), vals, width=0.62, color=[MUTED] + [color] * 4, edgecolor=SURF, linewidth=2, zorder=3)
    ax.axhline(tgt * 100, color=INK2, ls=(0, (4, 3)), lw=1, zorder=4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.6 if k == "mmlu_acc" else 1.5), f"{v:.1f}" if k == "mmlu_acc" else f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text(0, 1.0, f"dashed: {tlab} = {tgt*100:.0f}", transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, fontsize=7.5)
    lo = 0 if k != "mmlu_acc" else 20
    ax.set_ylim(lo, 108 if k != "mmlu_acc" else 47); ax.set_title(title, loc="left", pad=14); style(ax)
axes[0].set_ylabel("% of dev prompts")
fig.text(0.01, 0.005, "Dev sets: 50 ambiguous / 50 clear prompts (hand-written), 200 MMLU questions. Judge: course rubric, Llama-3.1-8B-Instruct.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/fig1_beta_sweep.png"); fig.savefig(f"{OUT}/fig1_beta_sweep.pdf"); plt.close(fig)

# ---------------- Figure 2: what broke and when (data recipe versions) ----------------
versions = [("runs__v1_overasker", "v1"), ("runs__v2", "v2"), ("runs__v3", "v3"), ("final", "v4\nfinal")]
R = [res(v) for v, _ in versions]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, (k, title, _, tgt, tlab) in zip(axes, METRICS):
    x = range(len(versions))
    ax.axhline(base[k] * 100, color=MUTED, lw=1, zorder=2)
    ax.text(-0.25, base[k] * 100 + 1.5, "base", va="bottom", fontsize=7, color=MUTED)
    ax.axhline(tgt * 100, color=INK2, ls=(0, (4, 3)), lw=1, zorder=2)
    for key, color, name, mk in (("sft", BLUE, "SFT", "o"), (0.5, ORANGE, "DPO β=0.5", "s"), (0.05, VIOLET, "DPO β=0.05", "^")):
        ys = [r[key][k] * 100 for r in R]
        ax.plot(list(x), ys, color=color, lw=2, marker=mk, ms=6, markeredgecolor=SURF, markeredgewidth=1.5, zorder=3, label=name)
    ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in versions], fontsize=8)
    ax.set_xlim(-0.3, 3.3); ax.set_ylim(-3, 105); ax.set_title(title, loc="left"); style(ax)
axes[1].annotate("over-asker", xy=(0, R[0][0.05]["answer_rate"] * 100), xytext=(0.35, 22), fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
axes[2].annotate("forgetter", xy=(1, R[1][0.05]["mmlu_acc"] * 100), xytext=(1.25, 8), fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
axes[0].set_ylabel("% of dev prompts")
h, l = axes[0].get_legend_handles_labels()
fig.legend( loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0),
           handles=h + [plt.Line2D([], [], color=INK2, ls=(0, (4, 3)), lw=1)], labels=l + ["target"])
fig.text(0.01, 0.005, "v1 single-call JSON generation · v2 length-matched minimal pairs · v3 + base-1B self-distillation + ARC letter-format · v4 + Alpaca everyday instructions",
         fontsize=6.3, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(f"{OUT}/fig2_failure_modes.png"); fig.savefig(f"{OUT}/fig2_failure_modes.pdf"); plt.close(fig)

# ---------------- Figure 3: DPO training dynamics ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axes[0]
for prefix, name, ls in (("runs__v1_overasker__", "v1: pure DPO, 2 ep, lr 2e-5", "-"), ("", "v4: + NLL anchor 0.5, 1 ep, lr 1e-5", (0, (1.5, 0.8)))):
    h = log(prefix, 0.05)
    st = [r["step"] for r in h]
    ax.plot(st, [r["logp_chosen"] for r in h], color=BLUE, lw=2.2 if ls != "-" else 1.6, ls=ls, label=f"chosen, {name}")
    ax.plot(st, [r["logp_rejected"] for r in h], color=ORANGE, lw=2.2 if ls != "-" else 1.6, ls=ls, label=f"rejected, {name}")
ax.set_title("Response log-prob, β=0.05", loc="left"); ax.set_xlabel("optimizer step"); ax.set_ylabel("mean log π(y|x)")
style(ax); L3 = ax.get_legend_handles_labels()
ax = axes[1]
for b, color in ((0.05, VIOLET), (0.1, BLUE), (0.5, ORANGE)):
    h = log("", b)
    ax.plot([r["step"] for r in h], [r["margin"] for r in h], color=color, lw=2, label=f"β={b}")
    ax.text(h[-1]["step"] + 4, h[-1]["margin"], f"β={b}", va="center", fontsize=7.5, color=INK)
ax.set_title("Reward margin by β, final run", loc="left"); ax.set_xlabel("optimizer step"); ax.set_ylabel("implicit reward margin")
ax.set_xlim(0, 370); style(ax)
fig.set_size_inches(7.2, 3.0)
fig.legend(*L3, loc="lower center", ncol=2, frameon=False, fontsize=6.5, columnspacing=1.5, handlelength=2.2)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(f"{OUT}/fig3_dpo_dynamics.png"); fig.savefig(f"{OUT}/fig3_dpo_dynamics.pdf"); plt.close(fig)
print("ok")
