
#!/usr/bin/env python3

"""Fix for the 'short prompt => ask' shortcut found in run v1: generate SHORT, fully-specified

requests (chosen = direct answer, rejected = needless clarifying question) with the 8B and

APPEND them to data/preference_pairs.jsonl (existing ids unchanged)."""

import json, os, random, re, sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_preference_data as g

from configs.train_config import DATA_SEED

from configs.paths_config import PREFERENCE_DATA_PATH

N = int(os.environ.get("A2_N_SHORT", "800"))

KINDS = ["a factual question with one correct answer", "an arithmetic or percentage calculation with the numbers given",

 "a unit conversion with the value and both units given", "a request to define a specific term", "a request to translate a given short phrase into a named language",

 "a request to fix the grammar of a given sentence", "a multiple-choice question that lists options A) B) C) D)", "a yes/no factual question",

 "a request for a haiku or two-line poem about a named topic", "a request to list a stated number of examples of something specific",

 "a request for a one-line code snippet in a named language doing a specific thing", "a question about a date, person, or place in history",

 "a request for a synonym or antonym of a given word", "a short science question", "a request to summarize a given one-sentence text in a few words"]

SYS = "You write training data. Reply with ONE valid JSON object and nothing else."

USR = """Write one SHORT user request: {kind}, in the area of {domain}.

It must be FULLY specified (3-25 words): an expert could answer immediately, nothing is missing.

JSON keys:

"prompt": the request.

"direct_answer": the ideal reply: answers directly and correctly, no questions, under 60 words. For multiple choice start with the letter.

"needless_question": a bad reply that asks an unnecessary clarifying question about something the request already made clear; one sentence ending with "?".

Reply with ONLY the JSON object."""

def parse(raw):

    try:

        obj, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])

        if all(isinstance(obj.get(k), str) and obj[k].strip() for k in ("prompt", "direct_answer", "needless_question")):

            return {k: obj[k].strip() for k in ("prompt", "direct_answer", "needless_question")}

    except (ValueError, AttributeError):

        pass

    return None

def bad(r):

    if not (2 <= g._words(r["prompt"]) <= 40): return "prompt length"

    if not g._is_question(r["needless_question"]) or g._words(r["needless_question"]) > 40: return "needless_question"

    if r["direct_answer"].rstrip().endswith("?") or g._words(r["direct_answer"]) > 90: return "direct_answer"

    return None

existing = [json.loads(l) for l in open(PREFERENCE_DATA_PATH, encoding="utf-8") if l.strip()]

seen = {g._norm(p["prompt"]) for p in existing} | g._eval_prompt_blocklist()

rng = random.Random(DATA_SEED + 1)

g.SYNTH_MAX_NEW_TOKENS = 220

gen = g._LocalHFGenerator(g.SYNTH_MODEL)

new, drops, raw_out = [], {}, open(Path(PREFERENCE_DATA_PATH).with_name("raw_short_clear.jsonl"), "w", encoding="utf-8")

while len(new) < N:

    seeds = [{"kind": rng.choice(KINDS), "domain": rng.choice(g.DOMAINS)} for _ in range(64)]

    outs = gen.generate([[{"role": "system", "content": SYS}, {"role": "user", "content": USR.format(**s)}] for s in seeds])

    for s, raw in zip(seeds, outs):

        r = parse(raw); why = "unparseable" if r is None else bad(r)

        if why is None and g._norm(r["prompt"]) in seen: why = "duplicate/eval"

        raw_out.write(json.dumps({"seed": s, "raw": raw, "dropped": why}, ensure_ascii=False) + "\n")

        if why: drops[why] = drops.get(why, 0) + 1; continue

        seen.add(g._norm(r["prompt"]))

        new.append({"class": "clear", "prompt": r["prompt"], "chosen": r["direct_answer"], "rejected": r["needless_question"], "meta": {**s, "source": "short_clear_v2"}})

    print(f"[short] {len(new)}/{N} kept; drops {drops}", flush=True)

new = new[:N]

all_pairs = existing + new

g.write_pairs([{k: p[k] for k in ("class", "prompt", "chosen", "rejected")} for p in all_pairs])

with Path(PREFERENCE_DATA_PATH).with_name("pair_meta.jsonl").open("a", encoding="utf-8") as f:

    for i, p in enumerate(new, start=len(existing)):

        f.write(json.dumps({"id": f"pair_{i:06d}", "class": "clear", **p["meta"]}, ensure_ascii=False) + "\n")

print(f"appended {len(new)} short clear pairs (hand_verified.jsonl untouched)")

