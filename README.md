# Learning When to Ask

Teaching Llama-3.2-1B-Instruct to ask a clarifying question for ambiguous
requests, answer clear requests directly, and retain general capabilities.
Developed by Michael Incorvaia for Conversational AI.

The pipeline builds preference pairs, applies LoRA supervised fine-tuning,
and compares DPO with beta values 0.05, 0.1, and 0.5. Training includes
response-only loss masking, a base-model reference, and an NLL anchor.

## Recorded results

Dev evaluation: 50 ambiguous prompts, 50 clear prompts, and 200 MMLU questions.
These are development results, not hidden-test performance.

| Model | Ask on ambiguous (%) | Answer on clear (%) | MMLU (%) |
|---|---:|---:|---:|
| Base | 54 | 98 | 41.0 |
| SFT | 88 | 94 | 38.0 |
| DPO beta=0.05 | 88 | 98 | 38.0 |
| DPO beta=0.1 | 88 | 96 | 38.5 |
| DPO beta=0.5 | 88 | 94 | 41.5 |

Earlier runs expose over-asking and capability loss; later data revisions add
self-distillation, ARC, and Alpaca examples. The final set has 2,805 pairs.
The manually reviewed sample rejected 13 of 100 pairs.

## Run the project

See [REPRODUCE.md](REPRODUCE.md) for setup, data construction, training, and evaluation.
The base and judge models require Hugging Face access and license acceptance.
Training was performed on an H100; it has not been benchmarked on smaller GPUs.

## Layout

- Root scripts: preference generation, SFT, DPO, and evaluation.
- `configs/`, `utils/`: model settings, paths, loading, and response masking.
- `tools/`: data construction, hand review, and diagnostic utilities.
- `eval/`: evaluation code and schema; local prompt files are excluded from Git.
- `results/`: recorded final metrics, training logs, and prior-run summaries.
- `report/`, `report_figures/`: report source, PDF, plots, and plotting inputs.
- `pace/`, `slurm/`: optional PACE-ICE cluster scripts.

Datasets, adapters, generation caches, and runtime output are excluded from
Git. They remain available in the prepared local folder. The report describes
the original experiment; use REPRODUCE.md for this repo's layout.
See [ATTRIBUTION.md](ATTRIBUTION.md).
