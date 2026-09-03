#!/usr/bin/env python3
"""Compare instruct prompt variants on one direction, outside lm_eval.

Why this exists
---------------
The instructed 2B SFT checkpoints score lower on FLORES+ than their CPT base
while scoring much higher on bouquet paragraph. Before spending GPU hours on a
full re-evaluation we want to know whether any part of that gap is attributable
to the evaluation prompt rather than to the model.

The evaluation prompt (mt_prompts.yaml: salamandraTA7B_instruct) sends an
*empty but present* system block:

    <|im_start|>system\n<|im_end|>\n<|im_start|>user\n...

SFT used FastChat's ``bsc_chat_template_system_0.5``, whose system_message_prob
is 0.5, so training rows are either

    <s><|im_start|>system\n<REAL MESSAGE><|im_end|>\n<|im_start|>user\n...   (~50%)
    <s><|im_start|>user\n...                                                (~50%)

Neither is the empty-system form, and lm_eval does not prepend BOS by default.

Note the earlier 2B instruct models were trained with the *default* template
``bsc_chat_template``, which has system_message="" and prob 1.0 -- so the system
block is never emitted and those models saw ONE uniform format. They tolerated
this eval prompt. These new checkpoints are the first trained on a 50/50 split
of two formats, so the older runs are not a control for them.

This probe is a measurement, not a proof of that hypothesis: it reports what
each variant actually scores.

Variants
--------
  eval_empty_sys   exactly what mt_prompts.yaml sends today (the baseline)
  eval_plus_bos    same, with <s> prepended
  train_no_sys     the p=0.5 branch that emits no system block
  train_with_sys   the p=0.5 branch with a real system message
  chat_template    the model's own tokenizer.apply_chat_template

Usage
-----
    python3 probe_instruct_prompt.py --model MODEL \
        --src-lang Spanish --tgt-lang English \
        --from-results results/.../results_es_en_flores+_devtest.json --n 64

``--from-results`` reads sources/targets/translations straight out of the
results JSON, so it needs no dataset access and reuses exactly the segments the
recorded score was computed over. The recorded hypotheses are rescored on the
same subset as an anchor: eval_empty_sys should land close to it, and a large
gap means this probe is not comparable to the harness.
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Matches the decoding in the launch scripts, so results are comparable to the
# recorded runs rather than to some other setting.
GEN = dict(do_sample=False, num_beams=5, early_stopping=True, max_new_tokens=800)

USER = "Translate the following text from {src} into {tgt}.\n{src}: {ctx} \n{tgt}: "

# One of the seven English system messages in bsc_chat_template_system_0.5.
# Any is representative; the contrast is "a real one" vs "none" vs "empty".
SYS = "You are a helpful assistant."

VARIANTS = ["eval_empty_sys", "eval_plus_bos", "train_no_sys", "train_with_sys", "chat_template"]


def build(variant, tok, src, tgt, ctx):
    """Return the exact prompt string for one variant."""
    user = USER.format(src=src, tgt=tgt, ctx=ctx)
    if variant == "eval_empty_sys":
        return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if variant == "eval_plus_bos":
        return f"<s><|im_start|>system\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if variant == "train_no_sys":
        return f"<s><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if variant == "train_with_sys":
        return f"<s><|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if variant == "chat_template":
        return tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(variant)


def load_from_results(path):
    """Recover sources, references and recorded hypotheses from a results JSON."""
    j = json.load(open(path))
    results = j.get("results", {})
    if not results:
        sys.exit(f"no results block in {path}")
    task, block = next(iter(results.items()))
    try:
        srcs = block["sources,none"]
        refs = block["targets,none"]
    except KeyError:
        sys.exit(
            f"{path} has no sources/targets arrays (task {task}).\n"
            "Pass --src-file and --ref-file instead."
        )
    return srcs, refs, block.get("translations,none"), block.get("bleu,none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src-lang", default="English")
    ap.add_argument("--tgt-lang", default="Spanish")
    ap.add_argument("--src-file")
    ap.add_argument("--ref-file")
    ap.add_argument("--from-results")
    ap.add_argument("--n", type=int, default=64, help="segments to score per variant")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", help="write per-variant hypotheses here as JSON")
    args = ap.parse_args()

    recorded_hyps = recorded_bleu = None
    if args.from_results:
        srcs, refs, recorded_hyps, recorded_bleu = load_from_results(args.from_results)
    elif args.src_file and args.ref_file:
        srcs = [l.rstrip("\n") for l in open(args.src_file) if l.strip()]
        refs = [l.rstrip("\n") for l in open(args.ref_file) if l.strip()]
    else:
        sys.exit("need --from-results, or both --src-file and --ref-file")

    srcs, refs = srcs[: args.n], refs[: args.n]
    if recorded_hyps:
        recorded_hyps = recorded_hyps[: args.n]
    if not srcs:
        sys.exit("no source segments loaded")
    print(f"{len(srcs)} segments, {args.src_lang} -> {args.tgt_lang}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model)
    # Left padding: decoder-only batched generation. Right padding would place
    # pad tokens between the prompt and the first generated token.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    import sacrebleu

    # Anchor: rescore the harness's own stored hypotheses over this same subset.
    subset_anchor = None
    if recorded_hyps and len(recorded_hyps) == len(refs):
        subset_anchor = sacrebleu.corpus_bleu(recorded_hyps, [refs]).score
        note = f" (full-set recorded {recorded_bleu:.2f})" if recorded_bleu else ""
        print(f"recorded hypotheses on this subset: BLEU {subset_anchor:.2f}{note}", file=sys.stderr)

    dump, rows = {}, []
    for variant in args.variants.split(","):
        prompts = [build(variant, tok, args.src_lang, args.tgt_lang, s) for s in srcs]
        hyps = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            # add_special_tokens=False: each variant states its own BOS above,
            # so the tokenizer must not add a second one.
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
                model.device
            )
            with torch.no_grad():
                out = model.generate(**enc, **GEN, pad_token_id=tok.pad_token_id)
            for j in range(len(batch)):
                gen = out[j][enc["input_ids"].shape[1] :]
                text = tok.decode(gen, skip_special_tokens=True)
                # lm_eval stops at the first blank line (until: ["\n\n"]).
                hyps.append(text.split("\n\n")[0].strip())
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        empty = sum(1 for h in hyps if not h)
        rows.append((variant, bleu, empty))
        dump[variant] = hyps
        print(f"  {variant:16s} BLEU {bleu:6.2f}   empty {empty}", file=sys.stderr)

    print(f"\n{'variant':16s} {'BLEU':>7} {'vs eval':>8}  empty")
    if subset_anchor is not None:
        print(f"{'[recorded]':16s} {subset_anchor:7.2f} {'':>8}  -")
    ref = dict((v, b) for v, b, _ in rows).get("eval_empty_sys")
    for variant, bleu, empty in rows:
        delta = f"{bleu - ref:+8.2f}" if ref is not None else "       -"
        print(f"{variant:16s} {bleu:7.2f} {delta}  {empty}")

    if subset_anchor is not None and ref is not None:
        gap = abs(ref - subset_anchor)
        verdict = "OK" if gap <= 1.5 else "DIVERGENT -- do not trust the rows below it"
        print(f"\neval_empty_sys vs recorded: {ref - subset_anchor:+.2f} BLEU  [{verdict}]")
    print(
        "\neval_empty_sys is what the launch scripts send today. If the train_* "
        "variants beat it clearly, the prompt is implicated; if all variants sit "
        "within ~1 BLEU, the prompt is exonerated."
    )

    if args.out:
        json.dump(dump, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
