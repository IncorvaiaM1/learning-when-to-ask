"""Print base vs DPO beta=0.5 greedy responses for the report transcripts.
Run from the repo root on PACE (GPU):  python tools/transcripts.py > transcripts.txt
Uses the same greedy decode as evaluate.py. T3 candidates = ambiguous prompts where
beta=0.5 output has no '?' (likely answered instead of asking); pick one and confirm."""
import json, os, sys
sys.path.insert(0, os.getcwd())
from evaluate import load_base_policy, load_adapter_policy, generate_responses, _free
from utils.model_utils import load_tokenizer
amb = [json.loads(l) for l in open("eval/held_out_ambiguous.jsonl")]
clr = [json.loads(l) for l in open("eval/held_out_clear.jsonl")]
prompts = [x["prompt"] for x in amb] + [x["prompt"] for x in clr]
tok = load_tokenizer()
out = {}
m = load_base_policy(); out["base"] = generate_responses(m, tok, prompts); _free(m)
m = load_adapter_policy("checkpoints/dpo/beta_0.5"); out["dpo"] = generate_responses(m, tok, prompts); _free(m)
json.dump({"prompts": prompts, **out}, open("output/transcripts.json", "w"), indent=1)
def show(i, tag):
    print(f"\n===== {tag}\nUSER: {prompts[i]}\n--- BASE:\n{out['base'][i]}\n--- DPO beta=0.5:\n{out['dpo'][i]}")
show(prompts.index("How much paint do I need for my room?"), "T1 (ambiguous)")
show(prompts.index("What is 15% of 240?"), "T2 (clear)")
for i in range(len(amb)):
    if "?" not in out["dpo"][i]:
        show(i, "T3 candidate (ambiguous, beta=0.5 may not ask)")
