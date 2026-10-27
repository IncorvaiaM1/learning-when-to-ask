#!/bin/bash
# Sourced by every sbatch job (and by setup.sh). One place for env/caches on PACE-ICE.
A2_ENV="${A2_ENV:-$HOME/scratch/conda/envs/CAI_A2}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/scratch/conda/pkgs}"
export TOKENIZERS_PARALLELISM=false
# Smoke-test overrides must never leak into a real run.
unset A2_MODEL_ID_OVERRIDE A2_JUDGE_BACKEND A2_SYNTH_BACKEND

module load anaconda3 >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate "$A2_ENV"

# Jobs start in the submit dir; make that explicit so relative paths always work.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
echo "[env] $(hostname) | python $(which python) | HF_HOME=$HF_HOME | cwd=$(pwd)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
