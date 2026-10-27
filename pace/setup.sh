#!/bin/bash
# ONE-TIME setup on the PACE-ICE login node. Safe to re-run (skips finished steps).
#   cd ~/CAI_Assignment_2 && bash pace/setup.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)

echo "== 0. normalize line endings (files came from Windows)"
find . -type f \( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' -o -name '*.yml' \) -exec sed -i 's/\r$//' {} +
chmod +x pace/*.sh eval/*.sh tools/*.py 2>/dev/null || true
mkdir -p slurm/logs

echo "== 1. scratch dirs + ~/.bashrc exports"
mkdir -p ~/scratch/conda/envs ~/scratch/conda/pkgs ~/scratch/hf_cache
grep -q "A2: HF cache" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
# A2: HF cache + conda pkgs on scratch (home quota is small)
export HF_HOME=~/scratch/hf_cache
export CONDA_PKGS_DIRS=~/scratch/conda/pkgs
export PIP_CACHE_DIR=~/scratch/pip_cache
RC
export HF_HOME=~/scratch/hf_cache CONDA_PKGS_DIRS=~/scratch/conda/pkgs PIP_CACHE_DIR=~/scratch/pip_cache
mkdir -p ~/scratch/pip_cache

echo "== 2. conda env on scratch"
module load anaconda3
eval "$(conda shell.bash hook)"
ENV=~/scratch/conda/envs/CAI_A2
if [ ! -x "$ENV/bin/python" ]; then
  conda env create -f env.yml -p "$ENV"
else
  echo "   env exists at $ENV (delete it to rebuild)"
fi
conda activate "$ENV"
python -c "import torch, transformers, peft, datasets; print('   torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

echo "== 3. Hugging Face auth (gated Llama models)"
if ! python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  echo "   Not logged in. Paste a READ token from https://huggingface.co/settings/tokens"
  echo "   (first accept the licenses on the model pages for Llama-3.2-1B-Instruct AND Llama-3.1-8B-Instruct)"
  huggingface-cli login
fi
python -c "from huggingface_hub import whoami; print('   logged in as', whoami()['name'])"

echo "== 4. prefetch models into scratch cache (so compute nodes never stall on downloads)"
POLICY=$(python -c "from configs.train_config import MODEL_ID; print(MODEL_ID)")
JUDGE=$(python -c "from configs.evaluation_config import JUDGE_MODEL; print(JUDGE_MODEL)")
for M in "$POLICY" "$JUDGE"; do
  echo "   $M"
  huggingface-cli download "$M" --exclude "original/*" >/dev/null
done

echo "== 5. dev eval sets + dev MMLU slice"
bash eval/use_dev_sets.sh
[ -f eval/mmlu_slice.jsonl ] || python eval/build_mmlu_slice.py

echo "== 6. GPU visible to torch?"
python -c "import torch; print('   cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU (fine on a login node)')"

echo "== 7. masking check against the REAL Llama chat template"
python tools/check_masking_real.py

echo
echo "Setup done. Next:  bash pace/run_all.sh        (or: bash pace/run_all.sh --smoke)"
