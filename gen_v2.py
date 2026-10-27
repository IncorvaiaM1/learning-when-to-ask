#!/usr/bin/env python3
"""Preference data v2 -- length-matched minimal pairs, per-response generation, judge-verified.

Stages (each cached under data/v2/, so a rerun resumes after a disconnect):
  A. prompts   : one call per seed -> {ambiguous, missing, clear}, BOTH inside the seed's length
                 bucket (short/medium/long) so prompt length cannot leak the class label (v1 bug).
  B. responses : four focused plain-text calls per minimal pair:
                 clarify (chosen|amb), guess (rejected|amb), direct (chosen|clear), needless (rejected|clear)
  C. verify    : the 8B judge with the COURSE rubric (eval/judge.py) must label chosen/rejected as
                 expected (ask vs answer), and an answerability check must agree with the class.
  D. balance   : equal ambiguous/clear per length bucket -> data/preference_pairs.jsonl (+ meta,
                 stats, and the 100-pair hand-verification sample).

Backend: vLLM if importable (10-20x faster), else HF transformers.
    python gen_v2.py                      # defaults: 2600 seeds, target 3000 pairs
    A2_SEEDS=3000 A2_TARGET=3200 python gen_v2.py
"""
import json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_preference_data as g
from configs.train_config import DATA_SEED
from configs.paths_config import DATA_DIR
from eval.judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, _normalize_label

MODEL = os.environ.get("A2_SYNTH_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
N_SEEDS = int(os.environ.get("A2_SEEDS", "2600"))
TARGET = int(os.environ.get("A2_TARGET", "3000"))
USE_ANSWERABLE = os.environ.get("A2_USE_ANSWERABLE", "0") == "1"
OUT = Path(DATA_DIR) / "v2"; OUT.mkdir(parents=True, exist_ok=True)

BUCKETS = {"short": (3, 12), "medium": (13, 30), "long": (31, 70)}
STYLES = ["casual", "terse", "polite", "rushed, lowercase", "formal", "conversational"]

# ----------------------------------------------------------------------------- backends
class VLLMBackend:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        self.llm = LLM(MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, seed=DATA_SEED)
    def run(self, convs, temperature, max_tokens):
        sp = self.SP(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens, seed=DATA_SEED)
        return [o.outputs[0].text for o in self.llm.chat(convs, sp, use_tqdm=True)]

class HFBackend:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
        torch.manual_seed(DATA_SEED)
    def run(self, convs, temperature, max_tokens, bs=64):
        outs = []
        for i in range(0, len(convs), bs):
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs[i:i + bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False)
            with self.torch.no_grad():
                o = self.model.generate(**enc, max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id, **kw)
            outs += self.tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"    {min(i + bs, len(convs))}/{len(convs)}", flush=True)
        return outs

def backend():
    if os.environ.get("A2_BACKEND", "auto") != "hf":
        try:
            import vllm  # noqa
            print("[gen_v2] backend: vLLM"); return VLLMBackend()
        except Exception as e:
            print(f"[gen_v2] vLLM unavailable ({type(e).__name__}); using HF transformers")
    return HFBackend()

def cached(name, fn):
    p = OUT / f"{name}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        print(f"[gen_v2] {name}: {len(rows)} cached rows"); return rows
    t = time.time(); rows = fn()
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_v2] {name}: {len(rows)} rows in {time.time() - t:.0f}s"); return rows

def words(t): return len(t.split())
def strip_quotes(t): return t.strip().strip('"').strip()

# ----------------------------------------------------------------------------- stage A
A_SYS = "You write realistic user requests for training data. Reply with ONE valid JSON object and nothing else."
A_USR = """Domain: {domain}. Task type: {task}. Writing style: {style}.
Write a MINIMAL PAIR of user requests for the same task:
- "ambiguous": genuinely underspecified because of {ambiguity}. A competent expert could NOT respond well without asking. Do not say it is ambiguous.
- "missing": a few words naming exactly what is missing.
- "clear": the SAME task, fully specified (include the actual text/numbers/options/target needed) so an expert could respond immediately.
LENGTH: each request must be between {lo} and {hi} words. Long requests may add realistic background, but in "ambiguous" the key detail must still be missing.
Reply with ONLY: {{"ambiguous": "...", "missing": "...", "clear": "..."}}"""

def stage_a(be):
    rng = random.Random(DATA_SEED)
    seeds = []
    for i in range(N_SEEDS):
        b = list(BUCKETS)[i % 3]
        seeds.append({"bucket": b, "domain": rng.choice(g.DOMAINS), "task": rng.choice(g.TASK_TYPES),
                      "ambiguity": rng.choice(g.AMBIGUITY_TYPES), "style": rng.choice(STYLES)})
    convs = [[{"role": "system", "content": A_SYS},
              {"role": "user", "content": A_USR.format(lo=BUCKETS[s["bucket"]][0], hi=BUCKETS[s["bucket"]][1], **s)}] for s in seeds]
    outs = be.run(convs, 0.9, 400)
    rows, block, seen, drops = [], g._eval_prompt_blocklist(), set(), Counter()
    for s, raw in zip(seeds, outs):
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])
            a, m, c = (strip_quotes(obj[k]) for k in ("ambiguous", "missing", "clear"))
        except Exception:
            drops["unparseable"] += 1; continue
        lo, hi = BUCKETS[s["bucket"]]
        if not all(lo * 0.6 <= words(x) <= hi * 1.4 for x in (a, c)): drops["length"] += 1; continue
        ka, kc = g._norm(a), g._norm(c)
        if ka == kc or ka in seen or kc in seen: drops["duplicate"] += 1; continue
        if ka in block or kc in block: drops["eval overlap"] += 1; continue
        seen |= {ka, kc}
        rows.append({**s, "ambiguous": a, "missing": m, "clear": c})
    print(f"[A] drops: {dict(drops)}"); return rows

# ----------------------------------------------------------------------------- stage B
CLARIFY = ("You are a helpful assistant. The user's request is missing this information: {missing}. "
           "Reply with ONE short, specific clarifying question about exactly that (you may offer 2-3 concrete options). "
           "Do not attempt the task or give any partial answer. Under 40 words.")
GUESS = ("You are a confident assistant. Respond to the request directly and completely. If anything is unspecified, "
         "silently assume the most likely interpretation and do not mention that you assumed anything. "
         "Do not ask any questions. Under 150 words.")
DIRECT = ("You are a helpful assistant. Respond to the request directly, correctly and concisely. Do not ask questions. "
          "Under 150 words. For multiple-choice questions, begin with the letter of the correct option.")
NEEDLESS = ("Reply to the user ONLY with one unnecessary clarifying question about a detail they ALREADY specified "
            "in their request. Do not answer or attempt the task. One sentence ending with a question mark.")

def stage_b(be, A):
    convs, keys = [], []
    for i, r in enumerate(A):
        for kind, sys_p, prompt in (("clarify", CLARIFY.format(missing=r["missing"]), r["ambiguous"]), ("guess", GUESS, r["ambiguous"]),
                                    ("direct", DIRECT, r["clear"]), ("needless", NEEDLESS, r["clear"])):
            convs.append([{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}]); keys.append((i, kind))
    outs = be.run(convs, 0.7, 300)
    resp = defaultdict(dict)
    for (i, kind), o in zip(keys, outs): resp[i][kind] = o.strip()
    return [{**A[i], **resp[i]} for i in range(len(A))]

def trim_q(t):

    t = t.strip().strip('"').strip()

    return t[: t.rfind("?") + 1].strip() if "?" in t else t



def b_ok(r):

    q = lambda t: t.rstrip().rstrip('"').endswith("?")

    return {"amb": "?" in r["clarify"] and words(trim_q(r["clarify"])) <= 60 and not q(r["guess"]) and words(r["guess"]) >= 10,

            "clr": "?" in r["needless"] and words(trim_q(r["needless"])) <= 60 and not q(r["direct"]) and words(r["direct"]) >= 1}



# ----------------------------------------------------------------------------- stage C
ANSWERABLE = ("Could a careful expert give a genuinely good, complete response to the following request WITHOUT asking "
              "for any more information? Reply with only yes or no.\n\nREQUEST:\n{p}")

def stage_c(be, B):
    convs, keys = [], []
    def judge(i, tag, prompt, response):
        convs.append([{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                      {"role": "user", "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response)}]); keys.append((i, tag))
    for i, r in enumerate(B):
        judge(i, "clarify", r["ambiguous"], r["clarify"]); judge(i, "guess", r["ambiguous"], r["guess"])
        judge(i, "direct", r["clear"], r["direct"]); judge(i, "needless", r["clear"], r["needless"])
        for tag, p in (("ans_amb", r["ambiguous"]), ("ans_clr", r["clear"])):
            convs.append([{"role": "user", "content": ANSWERABLE.format(p=p)}]); keys.append((i, tag))
    outs = be.run(convs, 0.0, 4)
    lab = defaultdict(dict)
    for (i, tag), o in zip(keys, outs):
        lab[i][tag] = ("yes" if "yes" in o.lower() else "no") if tag.startswith("ans_") else _normalize_label(o)
    return [{**B[i], "labels": lab[i]} for i in range(len(B))]

# ----------------------------------------------------------------------------- stage D
def stage_d(C):
    by = {"ambiguous": defaultdict(list), "clear": defaultdict(list)}
    drops = Counter()
    for r in C:
        ok, L = b_ok(r), r["labels"]
        meta = {k: r[k] for k in ("bucket", "domain", "task", "ambiguity", "style", "missing")}
        if not ok["amb"]:
            drops["amb: format"] += 1
            if drops["amb: format"] <= 3: print("[D] amb format example:", repr(r["clarify"][:200]), "| GUESS:", repr(r["guess"][:120]))
        elif not (L["clarify"] == "ask" and L["guess"] == "answer"): drops["amb: judge disagrees"] += 1
        elif USE_ANSWERABLE and L["ans_amb"] != "no": drops["amb: prompt judged answerable"] += 1
        else: by["ambiguous"][r["bucket"]].append({"class": "ambiguous", "prompt": r["ambiguous"], "chosen": trim_q(r["clarify"]), "rejected": r["guess"], "meta": meta})
        if not ok["clr"]: drops["clr: format"] += 1
        elif not (L["direct"] == "answer" and L["needless"] == "ask"): drops["clr: judge disagrees"] += 1
        elif USE_ANSWERABLE and L["ans_clr"] != "yes": drops["clr: prompt judged unanswerable"] += 1
        else: by["clear"][r["bucket"]].append({"class": "clear", "prompt": r["clear"], "chosen": r["direct"], "rejected": trim_q(r["needless"]), "meta": meta})
    rng, pairs, per = random.Random(DATA_SEED), [], TARGET // (2 * len(BUCKETS))
    counts = {}
    for b in BUCKETS:
        n = min(len(by["ambiguous"][b]), len(by["clear"][b]), per)   # balanced per bucket
        counts[b] = {"ambiguous_avail": len(by["ambiguous"][b]), "clear_avail": len(by["clear"][b]), "used_each": n}
        for cls in ("ambiguous", "clear"):
            pool = by[cls][b][:]; rng.shuffle(pool); pairs += pool[:n]
    rng.shuffle(pairs)
    stats = {"answerability_check_used": USE_ANSWERABLE, "amb_judged_unanswerable_frac": sum(r["labels"].get("ans_amb") == "no" for r in C) / max(len(C), 1), "clr_judged_answerable_frac": sum(r["labels"].get("ans_clr") == "yes" for r in C) / max(len(C), 1), "seeds": N_SEEDS, "minimal_pairs_after_A": len(C), "drops": dict(drops), "per_bucket": counts, "total_pairs": len(pairs)}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    with open(Path(DATA_DIR) / "pair_meta.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            f.write(json.dumps({"id": f"pair_{i:06d}", "class": p["class"], **p["meta"]}, ensure_ascii=False) + "\n")
    g.write_pairs(pairs)
    # Never clobber a hand-verified sample that already has verdicts (the graded audit).
    hv = Path(DATA_DIR) / "hand_verified.jsonl"
    filled = hv.exists() and any(json.loads(l).get("verdict") for l in hv.read_text(encoding="utf-8").splitlines() if l.strip())
    if filled:
        print(f"[gen_v2] {hv} already has verdicts; not resampling")
    else:
        g.sample_for_verification()

if __name__ == "__main__":
    be = None
    def need():
        global be
        if be is None: be = backend()
        return be
    A = cached("A_prompts", lambda: stage_a(need()))
    B = cached("B_responses", lambda: stage_b(need(), A))
    C = cached("C_labels", lambda: stage_c(need(), B))
    stage_d(C)
