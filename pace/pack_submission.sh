#!/bin/bash
# Builds the Gradescope zip: code + data + adapters (no base weights) + results.json.
#   bash pace/pack_submission.sh   ->  ~/CAI_A2_submission.zip
cd "$(dirname "$0")/.."
OUT=~/CAI_A2_submission.zip
rm -f "$OUT"
for f in data/preference_pairs.jsonl data/hand_verified.jsonl checkpoints/sft/adapter_config.json output/results.json; do
  [ -f "$f" ] || echo "[warn] missing $f"
done
zip -rq "$OUT" . -x "*/__pycache__/*" "slurm/logs/*" "output/judge_cache.json" "*.pyc" "*.ipynb_checkpoints*"
du -h "$OUT"; unzip -l "$OUT" | grep -E "adapter_model|results.json|preference_pairs|hand_verified"
