#!/bin/bash
# Submits the whole pipeline as a SLURM dependency chain:
#   [smoke] -> gen_data -> train_sft -> train_dpo (3 betas, array) -> evaluate
#
#   bash pace/run_all.sh                # full chain from data generation
#   bash pace/run_all.sh --smoke        # run the 2-minute GPU smoke test first
#   bash pace/run_all.sh --from sft     # reuse existing data/ ; start at SFT
#   bash pace/run_all.sh --from dpo     # reuse checkpoints/sft ; start at DPO
#   bash pace/run_all.sh --from eval    # just re-evaluate
#
# Hand verification (tools/review_pairs.py) is NOT in the chain -- do it while SFT/DPO run.
set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

FROM=gen; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --from) FROM="$2"; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac; shift
done

submit() {  # submit <dep-jobid-or-empty> <sbatch file>
  local dep=""; [ -n "$1" ] && dep="--dependency=afterok:$1 --kill-on-invalid-dep=yes"
  sbatch --parsable $dep "$2"
}

PREV=""
if [ $SMOKE = 1 ]; then PREV=$(submit "" slurm/smoke.sbatch); echo "smoke     $PREV"; fi
case "$FROM" in
  gen)  PREV=$(submit "$PREV" slurm/gen_data.sbatch);  echo "gen_data  $PREV" ;&
  sft)  [ "$FROM" = sft ] && [ ! -f data/preference_pairs.jsonl ] && { echo "no data/preference_pairs.jsonl"; exit 1; }
        PREV=$(submit "$PREV" slurm/train_sft.sbatch); echo "train_sft $PREV" ;&
  dpo)  [ "$FROM" = dpo ] && [ ! -f checkpoints/sft/adapter_config.json ] && { echo "no checkpoints/sft"; exit 1; }
        PREV=$(submit "$PREV" slurm/train_dpo.sbatch); echo "train_dpo $PREV (array: 3 betas)" ;&
  eval) PREV=$(submit "$PREV" slurm/evaluate.sbatch);  echo "evaluate  $PREV" ;;
  *) echo "--from must be gen|sft|dpo|eval"; exit 1 ;;
esac
echo
echo "Watch:  bash pace/status.sh      (logs in slurm/logs/)"
