# Tier 1 document-level generation debug

Scratch investigation, opened 2026-08-11. Delete once the Tier 1 document
evaluation produces sane output.

## The symptom

`*_tier1_doc` jobs return degenerate output: single-line fragments, repetition
loops (`CBD isolatis CBD isolatis ...`), and empty strings, on a checkpoint that
translates the same documents correctly in a notebook.

Job logs: `launch_evaluations/slurm_logs_audrey/tier_1_eval_treatment_4450*.err`

## What has been ruled out

Each of these was tested and is **not** the cause:

| hypothesis | evidence against |
|---|---|
| batch padding | `batch_size=1` reproduces it identically |
| missing BOS | `add_bos_token=True` verified to work (389 tokens, first id `1`); job output unchanged |
| `pad_token_id` = `<unk>` (0) | sentence-level tasks share it and score BLEU 53.6 |
| prompt truncation | longest kept prompt is 3,982 tokens against a 4,096 budget; nothing truncates |
| `until: []` / stop sequences | resolves to `['</s>','</s>','</s>','<end_of_turn>']`, no newline; bouquet uses the same successfully |
| `max_gen_toks` vs `max_new_tokens` | job rerun with bouquet's `max_new_tokens=800` still fails |
| bf16 / the checkpoint itself | same checkpoint scores BLEU 47.3 on `bouquet_paragraph` in this harness |

Critically, `_model_generate` — the harness's own generation function, with its
stopping criteria — produces a correct full-document translation when called
directly on the harness's own tokenized tensor.

## The open lead

lm_eval sorts requests **descending by token length**
(`lm_eval/models/huggingface.py`, `_collate`). The outputs at the top of a job
log are therefore the *longest* documents, 3,600-4,000 tokens — not the first
row of the file.

The earlier interactive test that succeeded used file row 0, which is only 388
tokens. So the working test and the failing job were not comparing the same
documents.

`tier1_probe.py` runs both ends of the length range through the same code path,
so the length axis is visible in one output.

## Running it

Needs a GPU:

```bash
salloc -A bsc88 -q acc_bscaii -p acc --gres=gpu:1 -c 20 -t 00:30:00

cd /gpfs/projects/bsc88/mt_translation/mt-evaluation
source venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/gpfs/projects/bsc88/mt_translation/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/tier1_doc_debug/tier1_probe.py
```

`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is required — without it the
Llama tokenizer fails to load with a protobuf descriptor error. The launch
scripts export it; an interactive shell does not.

## Reading the output

- **Long documents degenerate, row 0 fine** — the fault is length-dependent and
  lives in generation, not in the dataset loader or the prompt.
- **All three fine** — the fault is in the evaluation loop above
  `_model_generate`: the task layer, the dataset loader, or `build_all_requests`.
  Those are the parts not yet covered by any test.
