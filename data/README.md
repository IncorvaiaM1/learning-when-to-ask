# Preference data (local, excluded from Git)

The prepared local folder retains the working data, including final
preference_pairs.jsonl and hand_verified.jsonl. These are excluded from Git.
Fresh clones can regenerate data following REPRODUCE.md; manual verification
must be repeated when the reviewed sample is unavailable.

The final set contains 2,805 pairs (1,123 ambiguous and 1,682 clear), including
synthetic generations, Alpaca instructions, and ARC examples. Their source
terms still apply. See gen_v2.py and tools/build_v3.py for construction.
