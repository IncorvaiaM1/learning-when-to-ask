#!/bin/bash
#
# Part 1 (final v4 pipeline): gen_v2.py (8B minimal pairs, cached in data/v2/) then
# tools/build_v3.py (base-1B self-distillation + Alpaca + ARC) -> data/preference_pairs.jsonl.
# With the shipped data/v2 caches, gen_v2 skips generation; it never overwrites a filled
# data/hand_verified.jsonl. The original v1 generator is generate_preference_data.py (see runs/).
# Usage: sbatch slurm/gen_data.sbatch

#SBATCH --job-name=a2-gen
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -e
source pace/env.sh
VLLM_PY="$HOME/scratch/conda/envs/vllm/bin/python"
if [ -x "$VLLM_PY" ]; then "$VLLM_PY" gen_v2.py; else python gen_v2.py; fi   # HF fallback if no vLLM env
python tools/build_v3.py
python generate_preference_data.py --report
