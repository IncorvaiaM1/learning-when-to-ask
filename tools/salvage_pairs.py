
#!/usr/bin/env python3

"""Rebuild preference pairs from data/raw_generations.jsonl with a lenient parser (no GPU)."""

import collections, json, random, re, sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_preference_data as g

from configs.train_config import DATA_SEED, MIN_PREFERENCE_PAIRS

from configs.paths_config import PREFERENCE_DATA_PATH

KEYS = g.REQUIRED_FIELDS

def lenient(raw):

    r = g._parse_json(raw)

    if r: return r, "strict"

    try:

        obj, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])

        if all(isinstance(obj.get(k), str) and obj[k].strip() for k in KEYS):

            return {k: obj[k].strip() for k in KEYS}, "first-object"

    except (ValueError, AttributeError):

        pass


    pos = []

    for k in KEYS:

        m = re.search(r'"%s"\s*:\s*' % k, raw)

        if not m: return None, "missing key " + k

        pos.append((m.start(), m.end(), k))

    pos.sort(); out = {}

    for i, (s, e, k) in enumerate(pos):

        end = pos[i + 1][0] if i + 1 < len(pos) else len(raw)

        v = raw[e:end].strip()

        if i + 1 == len(pos): v = re.split(r'"\s*\}', v, maxsplit=1)[0]

        v = v.rstrip().rstrip(',').strip()

        if v.startswith('"'): v = v[1:]

        if v.endswith('"'): v = v[:-1]

        v = v.replace('\\n', '\n').replace('\\"', '"').replace('\\t', ' ').strip()

        if not v: return None, "empty " + k

        out[k] = v

    return out, "lenient"

rows = [json.loads(l) for l in open(g.RAW_GENERATIONS_PATH, encoding="utf-8") if l.strip()]

block, seen, pairs = g._eval_prompt_blocklist(), set(), []

how, drops = collections.Counter(), collections.Counter()

for row in rows:

    rec, mode = lenient(row["raw"])

    reason = mode if rec is None else g._reject_reason(rec)

    if reason is None:

        ka, kc = g._norm(rec["ambiguous_prompt"]), g._norm(rec["clear_prompt"])

        if ka in seen or kc in seen: reason = "duplicate prompt"

        elif ka in block or kc in block: reason = "matches an eval prompt"

    if reason is not None:

        drops[reason.split(" ")[0] if reason.startswith(("missing", "empty")) else reason] += 1; continue

    how[mode] += 1; seen.update({ka, kc}); sd = row["seed"]

    meta = {"domain": sd["domain"], "task": sd["task"], "ambiguity": sd["ambiguity"], "missing_info": rec["missing_info"], "parse": mode}

    pairs.append({"class": "ambiguous", "prompt": rec["ambiguous_prompt"], "chosen": rec["clarifying_question"], "rejected": rec["confident_guess"], "meta": meta})

    pairs.append({"class": "clear", "prompt": rec["clear_prompt"], "chosen": rec["direct_answer"], "rejected": rec["needless_question"], "meta": meta})

print(f"{len(rows)} raw generations -> {len(pairs)//2} kept ({dict(how)}); drops: {dict(drops)}")

if len(pairs) < MIN_PREFERENCE_PAIRS: sys.exit(f"Only {len(pairs)} pairs -- paste this output to Claude.")

random.Random(DATA_SEED).shuffle(pairs)

with Path(PREFERENCE_DATA_PATH).with_name("pair_meta.jsonl").open("w", encoding="utf-8") as f:

    for i, p in enumerate(pairs):

        f.write(json.dumps({"id": f"pair_{i:06d}", "class": p["class"], **p["meta"]}, ensure_ascii=False) + "\n")

g.write_pairs(pairs)

g.sample_for_verification()

