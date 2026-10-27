#!/usr/bin/env python3
"""v3 data = v2 pairs + two retention fixes (run on a GPU, CAI_A2 env; ~10 min).

1. Self-distillation for CLEAR prompts: chosen := the BASE 1B model's own greedy answer, generated
   exactly as evaluate.py generates (same chat template, no system prompt). Training then barely moves
   the model on clear prompts, so answer-rate and capability are preserved (v2 SFT lost 8 pts answer,
   6.5 pts MMLU by imitating the 8B's style). Falls back to the 8B answer if the 1B response looks like
   a question (course MockJudge heuristic), is empty, or hit the token limit.
2. Letter-format retention: ~300 ARC (allenai/ai2_arc, NOT MMLU) 4-option questions in the same
   "Answer with the single letter" format evaluate.py uses; chosen = the correct letter,
   rejected = a needless clarifying question.

The v2 pairs are backed up to data/v2/preference_pairs_v2.jsonl. hand_verified.jsonl is NOT touched
(its sample measures the 8B synthetic data quality, which v3 inherits).
    python tools/build_v3.py
"""
import json, os, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import generate_preference_data as g
from configs.train_config import DATA_SEED, MODEL_ID
from configs.paths_config import DATA_DIR, PREFERENCE_DATA_PATH
from eval.judge import MockJudge
from evaluate import _mmlu_prompt, LETTERS
from utils.model_utils import load_base_model, load_tokenizer

N_MCQ = int(os.environ.get("A2_N_MCQ", "300"))
N_ALPACA = int(os.environ.get("A2_N_ALPACA", "500"))
GENERIC_CLARIFY = [
    "Before I start, could you tell me a bit more about what you're looking for?",
    "Could you give me more details about what you need?",
    "Can you clarify what exactly you have in mind?",
    "What specifically would you like me to focus on?",
    "Could you tell me more about the context so I can help?",
    "Do you have any particular requirements or preferences for this?",
    "Before I answer, what level of detail would you like?",
    "Can you say a little more about what this is for?",
]
REFERS_TO_MISSING = ("following", "given", "below", "above", "this ", "these ", "the text", "the passage",
                     "the sentence", "the article", "the paragraph", "the code", "the data", "the list", "the table",
                     "the story", "the poem", "the input", "provided", "attached")
MAX_NEW = 384
NEEDLESS_TEMPLATES = [
    "Before I answer, do you want me to choose just one of the four options?",
    "Should I answer with a single letter, or explain each option?",
    "Just to confirm, do you want the letter of the correct option?",
    "Do you want me to pick from options A through D?",
    "Are you asking which of the listed options is correct?",
    "Would you like only the letter, or the full text of the answer?",
]

def self_distill(prompts, bs=32):
    tok = load_tokenizer(MODEL_ID)  # left padding, pad fallback (utils/model_utils.py)
    model = load_base_model().eval()
    outs, hit_limit = [], []
    for i in range(0, len(prompts), bs):
        chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(chat, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        for row in o[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            ended = tok.eos_token_id in ids or any(t in ids for t in (tok.convert_tokens_to_ids("<|eot_id|>"),) if t is not None)
            outs.append(tok.decode(row, skip_special_tokens=True).strip()); hit_limit.append(not ended)
        print(f"[distill] {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    del model; torch.cuda.empty_cache()
    return outs, hit_limit

def alpaca_prompts(n, seen):
    """Everyday self-contained instructions (no `input` field, no reference to missing material):
    the 'loosely specified but answerable with reasonable defaults' middle ground v3 lacked."""
    from datasets import load_dataset
    rows = [r["instruction"].strip() for r in load_dataset("tatsu-lab/alpaca", split="train")
            if not r["input"].strip() and 4 <= len(r["instruction"].split()) <= 25
            and not any(k in r["instruction"].lower() for k in REFERS_TO_MISSING)]
    rng = random.Random(DATA_SEED + 7); rng.shuffle(rows)
    out = []
    for p in rows:
        if g._norm(p) in seen: continue
        seen.add(g._norm(p)); out.append(p)
        if len(out) >= n: break
    return out

def arc_items(n, seen):
    from datasets import load_dataset
    rows = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for r in load_dataset("allenai/ai2_arc", cfg, split="train"):
            if list(r["choices"]["label"]) == LETTERS and r["answerKey"] in LETTERS:
                rows.append({"question": r["question"], "choices": list(r["choices"]["text"]), "answer": LETTERS.index(r["answerKey"]), "src": cfg})
    rng = random.Random(DATA_SEED); rng.shuffle(rows)
    out = []
    for r in rows:
        p = _mmlu_prompt(r)
        if g._norm(p) in seen: continue
        seen.add(g._norm(p))
        out.append({"class": "clear", "prompt": p, "chosen": LETTERS[r["answer"]], "rejected": rng.choice(NEEDLESS_TEMPLATES),
                    "meta": {"source": "arc_mcq", "arc_config": r["src"]}})
        if len(out) >= n: break
    return out

if __name__ == "__main__":
    v2_backup = Path(DATA_DIR) / "v2" / "preference_pairs_v2.jsonl"
    v2_backup.parent.mkdir(parents=True, exist_ok=True)
    if not v2_backup.exists():
        shutil.copy(PREFERENCE_DATA_PATH, v2_backup)
    pairs = [json.loads(l) for l in open(v2_backup, encoding="utf-8") if l.strip()]
    # Per-pair metadata keyed by PROMPT (ids are renumbered every rebuild). Built once from the
    # current pair_meta.jsonl + preference_pairs.jsonl (same ids), then cached.
    mbp_path = Path(DATA_DIR) / "v2" / "meta_by_prompt.json"
    if mbp_path.exists():
        meta_by_prompt = json.load(open(mbp_path, encoding="utf-8"))
    else:
        cur = {json.loads(l)["id"]: json.loads(l) for l in open(PREFERENCE_DATA_PATH, encoding="utf-8") if l.strip()}
        meta_by_prompt = {}
        for l in open(Path(DATA_DIR) / "pair_meta.jsonl", encoding="utf-8"):
            if l.strip():
                m = json.loads(l)
                if m["id"] in cur:
                    meta_by_prompt[cur[m["id"]]["prompt"]] = {k: v for k, v in m.items() if k not in ("id", "chosen_source", "chosen_8b", "distill_fallback")}
        json.dump(meta_by_prompt, open(mbp_path, "w", encoding="utf-8"), ensure_ascii=False)
    meta_rows = {p["id"]: meta_by_prompt.get(p["prompt"], {}) for p in pairs}
    clear_idx = [i for i, p in enumerate(pairs) if p["class"] == "clear"]
    seen = {g._norm(p["prompt"]) for p in pairs} | g._eval_prompt_blocklist()
    alp = alpaca_prompts(N_ALPACA, seen)
    print(f"[alpaca] {len(alp)} self-contained instructions selected")
    all_outs, all_limit = self_distill([pairs[i]["prompt"] for i in clear_idx] + alp)
    outs, limit = all_outs[:len(clear_idx)], all_limit[:len(clear_idx)]
    alp_outs, alp_limit = all_outs[len(clear_idx):], all_limit[len(clear_idx):]
    mock, used, why = MockJudge(), 0, {}
    for i, o, lim in zip(clear_idx, outs, limit):
        m = dict(meta_rows.get(pairs[i]["id"], {}))
        reason = "empty" if not o else "hit token limit" if lim else "looks like a question" if mock.label_raw(pairs[i]["prompt"], o) != "answer" else None
        if reason is None:
            m.update(chosen_source="self_distill_1b", chosen_8b=pairs[i]["chosen"]); pairs[i]["chosen"] = o; used += 1
        else:
            m.update(chosen_source="8b", distill_fallback=reason); why[reason] = why.get(reason, 0) + 1
        pairs[i]["meta"] = m
    for p in pairs:
        p.setdefault("meta", meta_rows.get(p["id"], {}))
    print(f"[distill] clear chosen from 1B: {used}/{len(clear_idx)}; kept 8B for {why}")
    rng_a, alpaca_pairs, alp_drop = random.Random(DATA_SEED + 9), [], 0
    for p, o, lim in zip(alp, alp_outs, alp_limit):
        if not o or lim or mock.label_raw(p, o) != "answer":
            alp_drop += 1; continue
        alpaca_pairs.append({"class": "clear", "prompt": p, "chosen": o, "rejected": rng_a.choice(GENERIC_CLARIFY),
                             "meta": {"source": "alpaca_self_distill", "chosen_source": "self_distill_1b"}})
    print(f"[alpaca] kept {len(alpaca_pairs)} (dropped {alp_drop}: 1B response empty/truncated/question)")
    mcq = arc_items(N_MCQ, seen)
    print(f"[arc] added {len(mcq)} letter-format MCQ pairs")
    allp = [{k: p[k] for k in ("class", "prompt", "chosen", "rejected", "meta")} for p in pairs] + alpaca_pairs + mcq
    random.Random(DATA_SEED + 3).shuffle(allp)
    with open(Path(DATA_DIR) / "pair_meta.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(allp):
            f.write(json.dumps({**{k: v for k, v in p["meta"].items() if k != "id"}, "id": f"pair_{i:06d}", "class": p["class"]}, ensure_ascii=False) + "\n")
    g.write_pairs(allp)
    json.dump({"clear_self_distilled": used, "clear_total": len(clear_idx), "fallbacks": why, "alpaca": len(alpaca_pairs), "arc_mcq": len(mcq)},
              open(Path(DATA_DIR) / "v2" / "v3_stats.json", "w"), indent=2)
    print("v3 data written. hand_verified.jsonl untouched.")
