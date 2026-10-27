#!/bin/bash
# ~2 min sanity check of the env + code on a GPU node: tiny random model, offline data,
# full gen -> SFT -> DPO -> evaluate. Writes nothing into the real data/ or checkpoints/.
#SBATCH --job-name=a2-smoke
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -e
source pace/env.sh

W=$(mktemp -d "$HOME/scratch/a2_smoke.XXXX")
cp -r configs eval utils tools *.py "$W/"
rm -f "$W"/eval/judge_cache.json
cd "$W"
bash eval/use_dev_sets.sh
python tools/make_tiny_model.py "$W/tiny"
export A2_MODEL_ID_OVERRIDE="$W/tiny" A2_JUDGE_BACKEND=mock A2_SYNTH_BACKEND=template
python generate_preference_data.py --generate
python train_sft.py
for b in 0.05 0.1 0.5; do python train_dpo.py --beta $b; done
python evaluate.py | grep -E "validates|Saved"
rm -rf "$W"
echo "SMOKE OK"
