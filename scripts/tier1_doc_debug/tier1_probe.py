#!/usr/bin/env python3
"""Reproduce the Tier 1 document failure outside lm_eval's evaluation loop.

Run from the mt-evaluation root with the venv active and the usual env vars set:

    cd /gpfs/projects/bsc88/mt_translation/mt-evaluation
    source venv/bin/activate
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export HF_HOME=/gpfs/projects/bsc88/mt_translation/hf_cache
    export HF_DATASETS_CACHE=$HF_HOME
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python tier1_probe.py

Needs a GPU, so run it inside an allocation:
    salloc -A bsc88 -q acc_bscaii -p acc --gres=gpu:1 -c 20 -t 00:30:00

The earlier interactive test used the FIRST row of the file, a 388-token
document, and it translated correctly. But lm_eval sorts requests DESCENDING by
token length, so the outputs seen in the job logs are the LONGEST documents,
3,600-4,000 tokens. This script runs both, so the length axis is visible in one
output.
"""

import json

import sentencepiece as spm
from lm_eval.models.huggingface import HFLM

CKPT = (
    "/gpfs/projects/bsc88/mt_translation/train_nemo/hf_cks/"
    "cpt_2b_adapted_vocab_256K_gaussian_salamandraTA_2b_cp3_treatment/"
    "megatron_gpt_cpt_2b_adapted_vocab_256K_gaussian_salamandraTA_2b_cp3_treatment"
    "--val_loss_2.14-step_29225-consumed_samples_14963200.0-last_hf"
)
SETS = (
    "/gpfs/projects/bsc88/mt_translation/data/2B_v2_doc-level/"
    "cpt3_doc_only/eval/sets/doc/cy-en.jsonl"
)
OVERLENGTH = (
    "/gpfs/projects/bsc88/mt_translation/data/2B_v2_doc-level/"
    "cpt3_doc_only/eval/sets/overlength/cy-en.json"
)
TOKENIZER = (
    "/gpfs/projects/bsc88/mt_translation/langtech_tokenizers/tokenizers/"
    "salamandraTA_extended/salamandraTA_extended/salamandraTA_extended.model"
)

MAX_GEN_TOKS = 4096
CONTEXT = 8192


def build_prompt(row):
    return (
        "Translate the following text from Welsh to English.\n"
        f"Welsh: {row['src']}\nEnglish:"
    )


def main():
    rows = [json.loads(line) for line in open(SETS, encoding="utf-8")]
    excluded = {
        entry["identity_key"]
        for entry in json.load(open(OVERLENGTH, encoding="utf-8"))["excluded"]
    }
    kept = [row for row in rows if row["identity_key"] not in excluded]

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    ranked = sorted(
        ((len(sp.encode(build_prompt(row))), row) for row in kept),
        key=lambda pair: -pair[0],
    )

    # The two ends of the length range: what the job hits first, and what the
    # earlier interactive test happened to use.
    cases = [("LONGEST", ranked[0]), ("2nd LONGEST", ranked[1])]
    shortest_first = next(
        (length, row) for length, row in ranked if row is kept[0]
    )
    cases.append(("FILE ROW 0 (tested OK earlier)", shortest_first))

    lm = HFLM(
        pretrained=CKPT,
        dtype="bfloat16",
        max_length=CONTEXT,
        trust_remote_code=True,
        batch_size=1,
    )

    for label, (length, row) in cases:
        prompt = build_prompt(row)
        ids, attn = lm.tok_batch_encode(
            [prompt], left_truncate_len=CONTEXT - MAX_GEN_TOKS
        )
        print("=" * 72)
        print(f"{label}: {length} prompt tokens, encoded {ids.shape[1]}")
        print(f"  truncated: {ids.shape[1] < length}")
        print(f"  ref starts: {row['ref'][:90]!r}")

        out = lm._model_generate(
            context=ids.to(lm.model.device),
            attention_mask=attn.to(lm.model.device),
            max_length=ids.shape[1] + MAX_GEN_TOKS,
            stop=["</s>", "<end_of_turn>"],
            do_sample=False,
            num_beams=1,
        )
        generated = out[0, ids.shape[1]:]
        raw = lm.tokenizer.decode(generated, skip_special_tokens=False)
        clean = lm.tokenizer.decode(generated, skip_special_tokens=True)

        print(f"  generated {len(generated)} tokens")
        print(f"  hit cap  : {len(generated) >= MAX_GEN_TOKS}")
        print(f"  RAW  [:300]: {raw[:300]!r}")
        print(f"  CLEAN[:300]: {clean[:300]!r}")
        print()


if __name__ == "__main__":
    main()
