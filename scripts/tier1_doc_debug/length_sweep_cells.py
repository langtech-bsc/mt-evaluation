"""Notebook cells for finding the generation-length cliff.

Paste CELL 1 once, then CELL 2 repeatedly, changing IDX each time.

Two corpora are offered. DocHPLT (`cy-en`) is mined web text, so poor source
quality confounds the length signal. Project Gutenberg is clean literary prose
and spans 288-4,157 tokens in a single file, which makes it the better ladder.
Its `en-es.jsonl` is actually Spanish source -> English target, so the output is
English and directly assessable by eye.

Known so far: both CPT2 and CPT3-treatment degenerate on a 3,649-token DocHPLT
document and both handle a ~390-token one. The cliff is somewhere between.
"""

# ============================== CELL 1 (once) ==============================
import json

MANIFESTS = "/gpfs/projects/bsc88/mt_translation/salamandraTA_extended_instructions/manifests"

# Project Gutenberg, Spanish source -> English target. Clean literary prose.
GUTENBERG = f"{MANIFESTS}/proj_gutenberg/outputs/en-es.jsonl"

gutenberg = [json.loads(line) for line in open(GUTENBERG, encoding="utf-8")]

# index -> approximate source tokens, measured with the salamandraTA_extended
# tokenizer. Source language is Spanish despite the file name.
LADDER = {
    580: 288,
    345: 569,
    94: 739,
    306: 916,
    381: 1523,
    199: 1958,
    392: 2623,
    564: 2999,
    107: 3499,
    530: 4001,
}

SRC_LANG = "Spanish"
TGT_LANG = "English"


def prompt_for(idx):
    """Build the evaluation prompt, same template the harness uses."""
    src = gutenberg[idx]["src_text"]
    return (f"Translate the following text from {SRC_LANG} to {TGT_LANG}.\n"
            f"{SRC_LANG}: {src}\n{TGT_LANG}:")


def show(idx, output, head=400):
    """Print a compact verdict for one document."""
    src = gutenberg[idx]["src_text"]
    words = output.split()
    unique = len(set(words)) / len(words) if words else 0.0
    verdict = "DEGENERATE" if unique < 0.35 else "looks ok"
    print(f"idx {idx}  ~{LADDER.get(idx, '?')} source tokens")
    print(f"  src chars {len(src)}  |  out chars {len(output)}")
    print(f"  out/src char ratio : {len(output) / max(len(src), 1):.2f}"
          f"   (healthy es->en is roughly 0.9-1.1)")
    print(f"  unique-word ratio  : {unique:.3f}   ({verdict})")
    print(f"  SRC[:150]: {src[:150]!r}")
    print(f"  OUT[:{head}]: {output[:head]!r}")


print(f"{len(gutenberg)} Gutenberg documents loaded ({SRC_LANG} -> {TGT_LANG})")
for idx, approx in sorted(LADDER.items(), key=lambda kv: kv[1]):
    print(f"  idx {idx:4}  ~{approx:5} tokens  "
          f"{gutenberg[idx]['src_text'][:55]!r}")


# ======================= CELL 2 (repeat, change IDX) =======================
# Work UP the ladder until the output degenerates. That length is the cliff.
#
# Keep max_new_tokens in generate_text_base above the source length or the
# translation is cut off for length reasons rather than degeneration:
# at idx 530 (~4,000 source tokens) use at least 4,096.

IDX = 580        # 580, 345, 94, 306, 381, 199, 392, 564, 107, 530

text = prompt_for(IDX)
output = generate_text_base(text)      # noqa: F821  (defined in the notebook)
show(IDX, output)
