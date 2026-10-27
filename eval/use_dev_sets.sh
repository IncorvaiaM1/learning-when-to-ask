#!/bin/bash
# Puts OUR dev prompts at the paths evaluate.py reads -- only if nothing is there yet, so it
# can never clobber the graders' hidden test set. Use --force to refresh after editing eval/dev/.
# Nothing here is trained on; generate_preference_data.py also drops any matching training prompt.
cd "$(dirname "$0")"
for kind in ambiguous clear; do
  if [ "$1" == "--force" ] || [ ! -f "held_out_${kind}.jsonl" ]; then
    cp "dev/dev_${kind}.jsonl" "held_out_${kind}.jsonl"
  fi
done
echo "eval sets: $(wc -l < held_out_ambiguous.jsonl) ambiguous, $(wc -l < held_out_clear.jsonl) clear"
[ -f mmlu_slice.jsonl ] || echo "no mmlu_slice.jsonl yet -> python3 eval/build_mmlu_slice.py"
