#!/usr/bin/env python3
"""Assert an evaluation prompt tokenises as a prefix of a training row.

The 2026-07-28 defect was a trailing space in `salamandraTA_extended_base`,
which SentencePiece encoded as a dangling U+2581 that never precedes a target
in training. Comparing strings would not have caught it; comparing token ids
does. Run this before submitting an evaluation.

    python3 tools/assert_prompt_prefix.py --style salamandraTA_extended_base

Exit status is 0 when every checked direction is a clean token prefix.
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = (
    REPO
    / "langtech_tokenizers/tokenizers/salamandraTA_extended/salamandraTA_extended"
    / "salamandraTA_extended.model"
)
DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "lm_eval/prompts/mt_prompts.yaml"
DEFAULT_SUBSAMPLE = REPO / "train_nemo/salamandraTA_extended_cp2/subsample.py"

# Directions to check. Includes the two carrying base-model name overrides.
CASES = [("eng_Latn", "spa_Latn"), ("eng_Latn", "hun_Latn"), ("eng_Latn", "nob_Latn")]
SOURCE = "The committee approved the proposal on Tuesday morning."
TARGET = "El comité aprobó la propuesta el martes por la mañana."


def training_templates(path):
    """PROMPT_TEMPLATES from subsample.py without importing it."""
    namespace = {}
    source = path.read_text()
    start = source.index("PROMPT_TEMPLATES = [")
    end = source.index("\n]", start) + 2
    exec(source[start:end], namespace)  # noqa: S102 - literal list of str
    return namespace["PROMPT_TEMPLATES"]


def resolve(prompt_yaml, style, mappings, src, tgt, context):
    entry = yaml.safe_load(prompt_yaml.read_text())["prompt_structures"][style]
    if entry.get("language_map"):
        table = getattr(mappings, entry["mapping_type"])
        src, tgt = table[src], table[tgt]
    return entry["prompt"].format(src=src, tgt=tgt, context=context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="salamandraTA_extended_base")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--subsample", type=Path, default=DEFAULT_SUBSAMPLE)
    args = ap.parse_args()

    sys.path.insert(0, str(DEFAULT_PROMPTS.parent))
    import mappings  # noqa: E402 - path set above

    tok = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    templates = training_templates(args.subsample)
    failures = []

    for src, tgt in CASES:
        prompt = resolve(args.prompts, args.style, mappings, src, tgt, SOURCE)
        prompt_ids = tok.encode(prompt)
        table = getattr(mappings, yaml.safe_load(args.prompts.read_text())
                        ["prompt_structures"][args.style]["mapping_type"])
        rows = [
            template.format(
                source_lang=table[src], target_lang=table[tgt],
                source_sent=SOURCE, target_sent=TARGET,
            )
            for template in templates
        ]
        matched = [r for r in rows if tok.encode(r)[: len(prompt_ids)] == prompt_ids]
        status = "OK" if matched else "FAIL"
        print(f"{status} {src}->{tgt}  {len(prompt_ids)} tokens, "
              f"{len(matched)}/{len(rows)} training templates matched")
        if not matched:
            failures.append((src, tgt, prompt, prompt_ids))

    for src, tgt, prompt, prompt_ids in failures:
        print(f"\n--- {src}->{tgt} is not a token prefix of any training row", file=sys.stderr)
        print(f"prompt   {prompt!r}", file=sys.stderr)
        print(f"tail ids {prompt_ids[-6:]} = {[tok.id_to_piece(i) for i in prompt_ids[-6:]]}",
              file=sys.stderr)
        best = max(
            (r for r in [
                t.format(source_lang="X", target_lang="Y", source_sent=SOURCE, target_sent=TARGET)
                for t in templates]),
            key=lambda r: sum(1 for a, b in zip(tok.encode(r), prompt_ids) if a == b),
        )
        print(f"closest  {best!r}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
