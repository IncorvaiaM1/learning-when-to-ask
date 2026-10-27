#!/bin/bash
# Queue + the tail of the most recent log of each stage + headline numbers if available.
cd "$(dirname "$0")/.."
squeue -u "$USER" -o "%.10i %.14j %.8T %.10M %.20R" 2>/dev/null
for s in a2-smoke a2-gen a2-sft a2-dpo a2-eval; do
  f=$(ls -t slurm/logs/${s}-*.out 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  echo; echo "=== $f"; tail -n 4 "$f"
  e="${f%.out}.err"; [ -s "$e" ] && grep -iE "error|traceback|killed|oom|denied" "$e" | tail -3
done
if [ -f output/results.json ]; then
  echo; python - <<'PY'
import json; r = json.load(open("output/results.json"))["runs"]
rows = [("base", r["base"]), ("sft", r["sft"])] + [(f"dpo b={d['beta']}", d) for d in r["dpo"]]
f = lambda v: "  -  " if v is None else f"{100*v:5.1f}"
print(f"{'run':12s} ask%  ans%  mmlu%")
for n, m in rows: print(f"{n:12s} {f(m['ask_rate'])} {f(m['answer_rate'])} {f(m['mmlu_acc'])}")
b = r["base"]
print("targets: ask +25 vs base, answer >= base-5, mmlu >= base-3")
PY
fi
