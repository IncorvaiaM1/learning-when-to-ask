# Base model -- your choice between these two small instruct checkpoints. Both run
# LoRA on a single PACE-ICE GPU in minutes per epoch. Set MODEL_ID to whichever you
# use and record it in results.json's metadata.base_model.
#
# NOTE: these are *-Instruct checkpoints. Unlike a from-scratch base model, they can
# already follow instructions -- your job is not to create that ability but to *shift
# behavior*: ask when the request is ambiguous, answer when it is clear, without
# forgetting everything else. See README.md.

ALLOWED_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
import os

# A2_MODEL_ID_OVERRIDE exists only for the offline smoke test (tools/make_tiny_model.py).
MODEL_ID = os.environ.get("A2_MODEL_ID_OVERRIDE", "meta-llama/Llama-3.2-1B-Instruct")


# LoRA config -- shared by Part A (SFT) and Part B (DPO).
RANK = 16
ALPHA = 32
DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # present in both model families.


# --- Preference data construction (what generate_preference_data.py must produce). ---
# You build the data yourself over two input classes: ambiguous requests (where a
# clarifying question is preferred) and fully-specified requests (where a direct answer
# is preferred). See README.md Part 1.

MIN_PREFERENCE_PAIRS = 1000  # Spec floor: at least 1,000 pairs total.
HAND_VERIFY_SAMPLE = 100  # Spec: hand-verify 100 pairs and report the rejection rate.
DATA_SEED = 6604  # Fix every sampling/shuffle step for reproducibility.


# --- Part A: SFT-LoRA (warm-start the target behavior). ---
# SFT trains on the PREFERRED response of each pair you built
# (data_loader.load_sft_examples()). Hyperparameters below are yours to tune.

SFT_BATCH_SIZE = 8
SFT_GRAD_ACCUM = 1
SFT_EPOCHS = 2
# LoRA wants a much larger LR than full fine-tuning; 2e-5 barely moves a r=16 adapter.
SFT_LEARNING_RATE = 1e-4
SFT_WARMUP_RATIO = 0.05
SFT_MAX_SEQ_LEN = 1024


# --- Part B: DPO (align on your preference pairs). ---
# You must sweep beta over at least three values (spec). Each value trains its own
# adapter; `train_dpo.py --beta <value>` picks one and checkpoints it separately.
# These three are the course default sweep -- you may change them, but keep at least
# three distinct values and report all of them in results.json.

DPO_BETAS = [0.05, 0.1, 0.5]
DPO_BETA = 0.1  # Default used when --beta is not passed on the CLI.
DPO_BATCH_SIZE = 4  # pairs per micro-batch (=> 8 sequences per forward pass)
DPO_GRAD_ACCUM = 2  # effective batch = 8 pairs

DPO_EPOCHS = 1
DPO_LEARNING_RATE = 1e-5
DPO_WARMUP_RATIO = 0.1
DPO_MAX_SEQ_LEN = 1024

# Reference model for the KL leash:
#   "base" -- README recipe: policy.disable_adapter() => the untouched instruct model.
#   "sft"  -- textbook DPO: the SFT policy itself (log-probs precomputed once, same model copy).
# With "base" the leash pulls toward the original model (helps MMLU retention) but the
# implicit reward starts non-zero, since SFT already moved the policy. Worth an ablation.
DPO_REFERENCE = "base"

# Optional NLL-on-chosen anchor added to the DPO loss (0 = pure DPO). Small values (~0.1-0.2)
# counter the known DPO failure where BOTH chosen and rejected log-probs fall.
DPO_SFT_NLL_WEIGHT = 0.5


# --- Shared optimizer / logging. ---
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10  # optimizer steps
