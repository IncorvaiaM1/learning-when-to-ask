#!/usr/bin/env python3
"""Offline smoke-test fixture: a tiny random Llama + a Llama-3-style chat-template tokenizer.

Lets the whole pipeline (gen -> SFT -> DPO -> evaluate) run on CPU in a minute without
Hugging Face access. The outputs are gibberish; this only checks the plumbing.

    python3 tools/make_tiny_model.py /tmp/tiny-llama
    A2_MODEL_ID_OVERRIDE=/tmp/tiny-llama A2_JUDGE_BACKEND=mock A2_SYNTH_BACKEND=template ...
"""
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

TEMPLATE = (
    "{{ bos_token }}"
    "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n<|eot_id|>"
    "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] | trim }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)
SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def main(out):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["The quick brown fox jumps over the lazy dog? What which how many, answer: A B C D 0123456789."] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                 pad_token="<|end_of_text|>", model_input_names=["input_ids", "attention_mask"])
    hf.chat_template = TEMPLATE
    hf.save_pretrained(out)

    cfg = LlamaConfig(vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      bos_token_id=hf.bos_token_id, eos_token_id=hf.eos_token_id, pad_token_id=hf.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(out)
    print(f"tiny model + tokenizer -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-llama")
