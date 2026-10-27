"""
Part 1 -- build your preference data.

You construct at least MIN_PREFERENCE_PAIRS (1,000) preference pairs over two input
classes:

  - ambiguous: a request that is genuinely underspecified. The PREFERRED (chosen)
    response asks a specific clarifying question; the DISPREFERRED (rejected) response
    is a confident guess that commits to one interpretation.

  - clear: a fully-specified request. The PREFERRED (chosen) response answers directly;
    the DISPREFERRED (rejected) response is a needless clarifying question.

You MAY synthesize these with an LLM. The course model (Llama-3.1-8B-Instruct) runs on
PACE-ICE at no cost -- load it locally the same way eval/judge.py's LocalHFJudge does.
You MUST hand-verify a sample of HAND_VERIFY_SAMPLE (100) pairs and report the
rejection rate of your own synthetic data (spec, Part 1 + the "preference data quality" rubric line).

What is wired for you: the output schema + writer, the verification sampler, and the
rejection-rate computation. What you implement: `synthesize_pairs()` -- the actual data generation.

INTEGRITY: build your own prompts. Do NOT reuse the held-out eval prompts in
eval/held_out_*.jsonl as training data -- that is training on the test set.

Usage:
    python3 generate_preference_data.py --generate   # Synthesize + write pairs, then
                                                      # sample 100 for hand-verification.

    # ... now open data/hand_verified.jsonl and fill in each "verdict" ...
    python3 generate_preference_data.py --report      # Compute + print the rejection rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from configs.train_config import (
    DATA_SEED,
    HAND_VERIFY_SAMPLE,
    MIN_PREFERENCE_PAIRS,
)
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH

VALID_CLASSES = {"ambiguous", "clear"}


# =====================================================================================
# Student implementation: synthesize_pairs()
# =====================================================================================
#
# Recipe ("minimal pairs"):
#   1. Sample a seed = (domain, task type, ambiguity type) with a fixed RNG.
#   2. One LLM call per seed returns JSON with BOTH an ambiguous request and a
#      fully-specified rewrite of the same task, plus four responses:
#         clarifying_question  -> chosen   for the ambiguous prompt
#         confident_guess      -> rejected for the ambiguous prompt
#         direct_answer        -> chosen   for the clear prompt
#         needless_question    -> rejected for the clear prompt
#      Because the two prompts share a topic and differ only in whether the key detail is
#      present, the model must learn "is information missing?" rather than "topic X => ask".
#   3. Rule-based filters (format, '?' placement, length, dedup, eval-set contamination).
#   4. Keep generating until we have >= n valid pairs, then shuffle with DATA_SEED.
#
# Backends:
#   local_hf           -- Llama-3.1-8B-Instruct in-process (the course model; run on PACE-ICE).
#   openai_compatible  -- any /chat/completions endpoint (e.g. the course endpoint / vLLM).
#   template           -- offline, model-free templates. ONLY for smoke-testing the pipeline.
#
# Every raw generation is saved to data/raw_generations.jsonl for the AI-use disclosure.

import os
import re

SYNTH_BACKEND = os.environ.get("A2_SYNTH_BACKEND", "local_hf")
SYNTH_MODEL = os.environ.get("A2_SYNTH_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SYNTH_BASE_URL = os.environ.get("A2_SYNTH_BASE_URL", "")
SYNTH_BATCH_SIZE = int(os.environ.get("A2_SYNTH_BATCH", "32"))
SYNTH_TEMPERATURE = 0.9
SYNTH_TOP_P = 0.95
SYNTH_MAX_NEW_TOKENS = 900
OVERGEN_FACTOR = 1.35  # filters reject some generations; ask for ~35% more seeds than needed.

RAW_GENERATIONS_PATH = os.path.join(os.path.dirname(str(PREFERENCE_DATA_PATH)), "raw_generations.jsonl")

DOMAINS = [
    "Python programming", "JavaScript/web development", "SQL and databases", "Linux command line",
    "Excel/spreadsheets", "data analysis", "machine learning", "cooking and recipes",
    "baking", "personal fitness", "travel planning", "personal finance and budgeting",
    "email and business writing", "resume and job applications", "academic essay writing",
    "high-school math", "calculus", "physics homework", "chemistry", "biology",
    "world history", "geography", "English grammar", "translation between languages",
    "music theory", "photography", "home repair and DIY", "gardening", "pet care",
    "car maintenance", "event planning", "project management", "marketing copy",
    "social media posts", "unit conversion", "scheduling and time zones", "board games and puzzles",
    "health and nutrition facts", "legal/administrative paperwork", "electronics and circuits",
    "statistics", "economics", "philosophy", "creative writing", "customer support",
    "smartphone/computer troubleshooting", "shopping and product comparison", "sports rules",
]

TASK_TYPES = [
    "write or edit a piece of text", "fix or debug something", "convert or translate something",
    "compute a number", "recommend or choose between options", "explain a concept",
    "summarize something", "plan a schedule or itinerary", "write code", "answer a factual question",
    "give step-by-step instructions", "compare two things", "rewrite in a different tone",
    "create a list", "estimate a quantity or cost",
    "answer a multiple-choice question (clear version lists the A-D options)",
    "answer a short-answer quiz question",
]

AMBIGUITY_TYPES = [
    "a missing referent: the request refers to 'this', 'it', 'the file', 'my code', etc. that was never provided",
    "a missing key parameter: a number, date, quantity, size, or budget the answer depends on",
    "multiple plausible interpretations of a key word or phrase",
    "an unspecified target: which language, format, platform, version, or audience",
    "unclear scope: how long, how detailed, or how much to cover",
    "an unstated goal or constraint that changes the right answer",
    "an ambiguous pronoun or entity that could refer to more than one thing",
    "missing units, location, or time zone that the answer depends on",
]

GEN_SYSTEM = """You generate training data that teaches a small assistant WHEN to ask a clarifying question. \
You always reply with a single valid JSON object and nothing else."""

GEN_USER = """Create one MINIMAL PAIR of user requests and four assistant responses.

Domain: {domain}
Task type: {task}
Kind of ambiguity: {ambiguity}

Fields (all strings):
- "ambiguous_prompt": a realistic user request in this domain that is GENUINELY underspecified because of the ambiguity kind above. A competent human expert could NOT give a good answer without asking. Do not mention that it is ambiguous. 5-40 words.
- "missing_info": a few words naming exactly what is missing.
- "clear_prompt": the SAME task rewritten so it is fully specified: it includes every detail needed (the actual text/code/numbers/options, the target format, etc.), so an expert could answer immediately. 10-120 words.
- "clarifying_question": the ideal reply to ambiguous_prompt. It asks ONE specific question about the missing info (optionally offering 2-3 concrete options). 1-2 sentences, under 50 words, ends with "?". It must NOT attempt the task.
- "confident_guess": a bad reply to ambiguous_prompt that silently assumes one interpretation and fully completes the task as if nothing were missing. Confident tone, no questions, no caveats. 30-150 words.
- "direct_answer": the ideal reply to clear_prompt. It directly and correctly completes the task. No questions. If it is a multiple-choice question, start with the letter. Under 180 words.
- "needless_question": a bad reply to clear_prompt that asks for clarification the user ALREADY gave (an unnecessary question). 1-2 sentences, ends with "?". It must NOT complete the task.

Reply with ONLY the JSON object."""

REQUIRED_FIELDS = [
    "ambiguous_prompt", "missing_info", "clear_prompt", "clarifying_question",
    "confident_guess", "direct_answer", "needless_question",
]


def _sample_seeds(k: int, rng: random.Random) -> list[dict]:
    return [
        {"domain": rng.choice(DOMAINS), "task": rng.choice(TASK_TYPES), "ambiguity": rng.choice(AMBIGUITY_TYPES)}
        for _ in range(k)
    ]


def _seed_messages(seed: dict) -> list[dict]:
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": GEN_USER.format(**seed)},
    ]


# --- Backends ----------------------------------------------------------------------


class _LocalHFGenerator:
    """Llama-3.1-8B-Instruct in-process (same loading recipe as eval/judge.py's LocalHFJudge)."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"  # decoder-only batched generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        torch.manual_seed(DATA_SEED)

    def generate(self, batch_messages: list[list[dict]]) -> list[str]:
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=SYNTH_MAX_NEW_TOKENS, do_sample=True,
                temperature=SYNTH_TEMPERATURE, top_p=SYNTH_TOP_P, pad_token_id=self.tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


class _OpenAICompatibleGenerator:
    def __init__(self, model_name: str, base_url: str):
        if not base_url:
            raise ValueError("Set A2_SYNTH_BASE_URL for the openai_compatible backend.")
        self.model_name, self.base_url = model_name, base_url.rstrip("/")

    def generate(self, batch_messages: list[list[dict]]) -> list[str]:
        import requests

        key = os.environ.get("OPENAI_API_KEY", "")
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        outs = []
        for i, m in enumerate(batch_messages):
            payload = {"model": self.model_name, "messages": m, "temperature": SYNTH_TEMPERATURE,
                       "top_p": SYNTH_TOP_P, "max_tokens": SYNTH_MAX_NEW_TOKENS, "seed": DATA_SEED + i}
            try:
                r = requests.post(self.base_url + "/chat/completions", headers=headers, json=payload, timeout=120)
                r.raise_for_status()
                outs.append(r.json()["choices"][0]["message"]["content"])
            except Exception as exc:  # one failed call shouldn't kill the whole run
                outs.append(f"__ERROR__ {exc}")
        return outs


class _TemplateGenerator:
    """Model-free stand-in so the pipeline runs offline. NOT real training data."""

    def __init__(self):
        self.rng = random.Random(DATA_SEED)
        self.i = 0

    def generate(self, batch_messages: list[list[dict]]) -> list[str]:
        outs = []
        for m in batch_messages:
            self.i += 1
            dom = re.search(r"Domain: (.*)", m[1]["content"]).group(1)
            x, y = self.rng.randint(2, 99), self.rng.randint(2, 99)
            outs.append(json.dumps({
                "ambiguous_prompt": f"Can you help me with the {dom} thing from earlier? (#{self.i})",
                "missing_info": "which thing",
                "clear_prompt": f"In {dom}, what is {x} plus {y}? (#{self.i})",
                "clarifying_question": f"Happy to help -- which {dom} task do you mean, and what have you tried so far?",
                "confident_guess": f"Sure. For your {dom} task, the best approach is to start with the basics, "
                                   f"break it into steps, and apply the standard method carefully until done.",
                "direct_answer": f"{x} plus {y} is {x + y}.",
                "needless_question": f"Do you want me to add {x} and {y}?",
            }))
        return outs


def _make_backend():
    if SYNTH_BACKEND == "local_hf":
        return _LocalHFGenerator(SYNTH_MODEL)
    if SYNTH_BACKEND == "openai_compatible":
        return _OpenAICompatibleGenerator(SYNTH_MODEL, SYNTH_BASE_URL)
    if SYNTH_BACKEND == "template":
        print("[warn] template backend: offline smoke-test data only -- do NOT train for real on this.")
        return _TemplateGenerator()
    raise ValueError(f"unknown A2_SYNTH_BACKEND={SYNTH_BACKEND!r}")


# --- Parsing + filtering -------------------------------------------------------------


def _parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0), strict=False)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not all(isinstance(obj.get(k), str) and obj[k].strip() for k in REQUIRED_FIELDS):
        return None
    return {k: obj[k].strip() for k in REQUIRED_FIELDS}


def _words(t: str) -> int:
    return len(t.split())


def _is_question(t: str) -> bool:
    return "?" in t and t.rstrip().endswith(("?", '?"', "?)"))


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _reject_reason(r: dict) -> str | None:
    """Rule-based quality gate. Returns why a generation is dropped, or None if it passes."""
    if _norm(r["ambiguous_prompt"]) == _norm(r["clear_prompt"]):
        return "prompts identical"
    if not (3 <= _words(r["ambiguous_prompt"]) <= 60):
        return "ambiguous_prompt length"
    if not (5 <= _words(r["clear_prompt"]) <= 200):
        return "clear_prompt length"
    if not _is_question(r["clarifying_question"]) or _words(r["clarifying_question"]) > 70:
        return "clarifying_question not a short question"
    if not _is_question(r["needless_question"]) or _words(r["needless_question"]) > 70:
        return "needless_question not a short question"
    if "?" in r["confident_guess"] or _words(r["confident_guess"]) < 15:
        return "confident_guess asks or is too short"
    if r["direct_answer"].rstrip().endswith("?") or _words(r["direct_answer"]) > 260:
        return "direct_answer ends in a question or too long"
    if _words(r["direct_answer"]) < 1:
        return "direct_answer empty"
    return None


def _eval_prompt_blocklist() -> set[str]:
    """Normalized prompts of any eval/held_out_*.jsonl present -- never train on them."""
    from configs.paths_config import EVAL_DIR

    block = set()
    for name in ("held_out_ambiguous.jsonl", "held_out_clear.jsonl"):
        p = Path(EVAL_DIR) / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    block.add(_norm(json.loads(line)["prompt"]))
    return block


def synthesize_pairs(n: int) -> list[dict]:
    """
    Produce `n` (or more) preference pairs, balanced across the two classes -- each
    accepted generation yields exactly one ambiguous pair and one clear pair.
    """
    rng = random.Random(DATA_SEED)
    backend = _make_backend()
    blocklist = _eval_prompt_blocklist()
    seen: set[str] = set()
    pairs: list[dict] = []
    drop_reasons: dict[str, int] = {}

    raw_path = Path(RAW_GENERATIONS_PATH)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_f = raw_path.open("w", encoding="utf-8")

    target_minimal_pairs = (n + 1) // 2
    rounds = 0
    while len(pairs) < 2 * target_minimal_pairs and rounds < 6:
        rounds += 1
        need = target_minimal_pairs - len(pairs) // 2
        seeds = _sample_seeds(max(int(need * OVERGEN_FACTOR), 8), rng)
        print(f"[gen] round {rounds}: {len(seeds)} seeds (have {len(pairs)} pairs)")

        for i in range(0, len(seeds), SYNTH_BATCH_SIZE):
            batch = seeds[i : i + SYNTH_BATCH_SIZE]
            outs = backend.generate([_seed_messages(sd) for sd in batch])

            for sd, raw in zip(batch, outs):
                rec = _parse_json(raw)
                reason = "unparseable JSON" if rec is None else _reject_reason(rec)
                if reason is None:
                    ka, kc = _norm(rec["ambiguous_prompt"]), _norm(rec["clear_prompt"])
                    if ka in seen or kc in seen:
                        reason = "duplicate prompt"
                    elif ka in blocklist or kc in blocklist:
                        reason = "matches an eval prompt"
                raw_f.write(json.dumps({"seed": sd, "raw": raw, "dropped": reason}, ensure_ascii=False) + "\n")

                if reason is not None:
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue
                seen.update({ka, kc})
                meta = {"domain": sd["domain"], "task": sd["task"], "ambiguity": sd["ambiguity"],
                        "missing_info": rec["missing_info"]}
                pairs.append({"class": "ambiguous", "prompt": rec["ambiguous_prompt"],
                              "chosen": rec["clarifying_question"], "rejected": rec["confident_guess"], "meta": meta})
                pairs.append({"class": "clear", "prompt": rec["clear_prompt"],
                              "chosen": rec["direct_answer"], "rejected": rec["needless_question"], "meta": meta})

            print(f"[gen]   {min(i + SYNTH_BATCH_SIZE, len(seeds))}/{len(seeds)} seeds -> {len(pairs)} pairs", flush=True)

    raw_f.close()
    total = sum(drop_reasons.values()) + len(pairs) // 2
    print(f"[gen] kept {len(pairs) // 2}/{total} generations; automatic drops: {drop_reasons}")

    pairs = pairs[: 2 * target_minimal_pairs]
    random.Random(DATA_SEED).shuffle(pairs)

    # Sidecar with per-pair provenance (write_pairs() assigns ids by position: pair_{i:06d}).
    meta_path = Path(PREFERENCE_DATA_PATH).with_name("pair_meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            f.write(json.dumps({"id": f"pair_{i:06d}", "class": p["class"], **p["meta"]}, ensure_ascii=False) + "\n")
    return pairs


# --- Wired: everything below is provided -------------------------------------------


def write_pairs(pairs: list[dict], path: str | Path = PREFERENCE_DATA_PATH) -> None:
    """Validates, assigns stable ids, and writes pairs as JSONL."""
    required = {"class", "prompt", "chosen", "rejected"}

    for i, p in enumerate(pairs):
        missing = required - p.keys()

        if missing:
            raise ValueError(f"pair #{i} missing keys {sorted(missing)}")

        if p["class"] not in VALID_CLASSES:
            raise ValueError(f"pair #{i} has invalid class {p['class']!r}")

    if len(pairs) < MIN_PREFERENCE_PAIRS:
        raise ValueError(
            f"Only {len(pairs)} pairs; the assignment requires at least " f"{MIN_PREFERENCE_PAIRS}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {
                "id": f"pair_{i:06d}",
                "class": p["class"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_amb = sum(1 for p in pairs if p["class"] == "ambiguous")
    n_clr = sum(1 for p in pairs if p["class"] == "clear")
    print(f"Wrote {len(pairs)} pairs ({n_amb} ambiguous, {n_clr} clear) -> {path}")


def sample_for_verification(
    pairs_path: str | Path = PREFERENCE_DATA_PATH,
    out_path: str | Path = HAND_VERIFIED_PATH,
    k: int = HAND_VERIFY_SAMPLE,
    seed: int = DATA_SEED,
) -> None:
    """Samples k pairs into a hand-verification file with a blank 'verdict' field.

    Open the file and set each pair's "verdict" to "accept" or "reject" by hand
    (a pair is a reject if the chosen/rejected labels are wrong, the prompt is
    mislabeled ambiguous/clear, or either response is low quality). Then run
    --report.
    """
    pairs = [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(k, len(pairs)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            rec = dict(rec)
            rec["verdict"] = ""  # Fill with "accept" or "reject" by hand.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} pairs to verify -> {out_path}")
    print("Fill in each 'verdict' (accept/reject) by hand, then run --report.")


def rejection_rate(path: str | Path = HAND_VERIFIED_PATH) -> float:
    """Computes the rejection rate over the hand-verified sample."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]

    unfilled = [i for i, v in enumerate(verdicts) if v not in {"accept", "reject"}]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} verdicts are not 'accept'/'reject' "
            f"(first at row {unfilled[0]}). Finish hand-verifying before reporting."
        )

    n_reject = sum(1 for v in verdicts if v == "reject")
    rate = n_reject / len(rows)

    print(f"Hand-verified {len(rows)} pairs: {n_reject} rejected -> rejection rate {rate:.1%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize + write pairs, then sample for verification",
    )

    parser.add_argument(
        "--n", type=int, default=MIN_PREFERENCE_PAIRS, help="how many pairs to synthesize"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute the rejection rate from the hand-verified file",
    )

    args = parser.parse_args()

    if args.generate:
        pairs = synthesize_pairs(args.n)
        write_pairs(pairs)
        sample_for_verification()

    if args.report:
        rejection_rate()

    if not (args.generate or args.report):
        parser.print_help()


if __name__ == "__main__":
    main()
