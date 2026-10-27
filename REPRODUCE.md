# Reproduction

Run commands from the repository root. Recorded artifacts are in `results/`;
runtime evaluation writes to `output/`. Adapters and datasets are not in Git.
The prepared local folder retains these files from the original experiment.

## Setup

Accept the Hugging Face terms for meta-llama/Llama-3.2-1B-Instruct and
meta-llama/Llama-3.1-8B-Instruct, then authenticate with the Hugging Face CLI.

```bash
conda env create -f env.yml
conda activate CAI_A2
bash eval/use_dev_sets.sh
python eval/build_mmlu_slice.py --seed 6604
```

The two dev prompt JSONL files are local-only. On a fresh clone, supply your
own disjoint ambiguous and clear evaluation prompts at
`eval/dev/dev_ambiguous.jsonl` and `eval/dev/dev_clear.jsonl` before running
the copy script; see eval/README.md for schema. Different prompts produce
different results. The original dev prompts are retained in the local copy.

## Data and training

If local final `data/preference_pairs.jsonl` is already present, skip to training.
For new data:

```bash
python gen_v2.py
python tools/review_pairs.py
python generate_preference_data.py --report
python tools/build_v3.py
```

Generation uses Llama-3.1-8B-Instruct (vLLM when available, HF fallback).
Alpaca and ARC inputs are downloaded by the construction script. Cached
generations and manually reviewed verdicts are local-only; fresh generation
and manual review will not recreate the original dataset byte for byte.

```bash
python train_sft.py --seed 42
python train_dpo.py --beta 0.05 --seed 42
python train_dpo.py --beta 0.1 --seed 42
python train_dpo.py --beta 0.5 --seed 42
python evaluate.py
```

SFT: lr 1e-4, 2 epochs. DPO: lr 1e-5, 1 epoch, effective batch 8,
base-model reference, NLL anchor 0.5. See configs/train_config.py.
Data/evaluation seed: 6604; training seed: 42. The original results JSON
reports 6604 for training because the evaluator reused EVAL_SEED in that field;
training logs record 42. GPU kernels can change greedy outputs slightly.

## Report figures

Install matplotlib separately for plotting (`pip install matplotlib`).
After training, from `report/`:

```bash
python make_figures.py --repo ..
```

Existing plots and their recorded inputs are also in `report_figures/`.
Prior runs and final training logs are preserved under `results/`.
PACE-ICE users can use pace/setup.sh and pace/run_all.sh instead of manual setup.
