"""Make unbabel-comet work with transformers >= 5.

COMET (unmaintained since v2.2.7, Sept 2025) reaches into transformers internals
that changed in v5, so this rewrites the installed package in two places:

1. The encoders call the HF model with ``return_dict=False`` and unpack a
   3-tuple ``(last_hidden_state, pooler_output, hidden_states)``. Transformers 5
   no longer returns the pooler output, raising
   ``ValueError: not enough values to unpack (expected 3, got 2)``. Affects every
   COMET model.

2. ``Encoder.concat_sequences`` calls
   ``tokenizer.build_inputs_with_special_tokens``, which transformers 5 removed,
   raising ``AttributeError: XLMRobertaTokenizer has no attribute
   build_inputs_with_special_tokens``. Only the unified_metric models take this
   path (CometKiwi, XCOMET), which is why reference-based wmt22-comet-da works
   without it. The fallback rebuilds the sequence using the encoder's own
   ``size_separator``, so it stays correct for both XLM-R style (``</s></s>``)
   and BERT style (``[SEP]``) encoders.

Both rewrites try the original call first and only fall back when it is missing,
so behaviour on transformers 4 is unchanged.

Run once after every ``pip install`` that (re)installs unbabel-comet:

    python tools/patch_comet_transformers5.py
"""

import pathlib
import sys


XLMR_OLD = """        last_hidden_states, _, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
"""

XLMR_NEW = """        _out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_states = _out.last_hidden_state
        all_layers = _out.hidden_states
"""

BERT_OLD = """        last_hidden_states, pooler_output, all_layers = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
"""

BERT_NEW = """        _out = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_states = _out.last_hidden_state
        all_layers = _out.hidden_states
        pooler_output = getattr(_out, "pooler_output", None)
        if pooler_output is None:
            pooler_output = last_hidden_states[:, 0, :]
"""

CONCAT_OLD = """                new_sequence = self.tokenizer.build_inputs_with_special_tokens(
                    new_sequence[1:-1], concat_input_ids[j][i][1:-1]
                )
"""

CONCAT_NEW = """                new_sequence = self._build_inputs_with_special_tokens(
                    new_sequence[1:-1], concat_input_ids[j][i][1:-1]
                )
"""

# Inserted into the Encoder class, immediately before concat_sequences.
CONCAT_ANCHOR = "    def concat_sequences("

CONCAT_HELPER = '''    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        """Join two token sequences with this encoder's special tokens.

        transformers 5 removed Tokenizer.build_inputs_with_special_tokens, so
        build the sequence here when it is absent. size_separator is 2 for
        XLM-R style encoders (</s></s>) and 1 for BERT style ([SEP]).
        """
        build = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
        if build is not None:
            return build(token_ids_0, token_ids_1)
        cls = [self.tokenizer.cls_token_id]
        sep = [self.tokenizer.sep_token_id]
        return cls + token_ids_0 + sep * self.size_separator + token_ids_1 + sep

'''


def patch_replace(path: pathlib.Path, old: str, new: str, marker: str) -> bool | None:
    """Return True if patched, None if already patched, False if the pattern is gone."""
    source = path.read_text()
    if marker in source:
        return None
    if old not in source:
        return False
    path.write_text(source.replace(old, new, 1))
    return True


def patch_base(path: pathlib.Path) -> bool | None:
    """Swap the call site and insert the fallback method into the Encoder class."""
    source = path.read_text()
    if "_build_inputs_with_special_tokens" in source:
        return None
    if CONCAT_OLD not in source or CONCAT_ANCHOR not in source:
        return False
    source = source.replace(CONCAT_OLD, CONCAT_NEW, 1)
    source = source.replace(CONCAT_ANCHOR, CONCAT_HELPER + CONCAT_ANCHOR, 1)
    path.write_text(source)
    return True



def purge_pycache(root: pathlib.Path) -> None:
    """Drop cached bytecode for the patched package.

    Python normally invalidates a .pyc from the source mtime, but hash-based
    caches (PEP 552) and shared filesystems can leave a stale one in place, so
    the patched source would be shown in tracebacks while the old bytecode
    still runs.
    """
    import shutil

    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def main() -> int:
    try:
        import comet.encoders.base
        import comet.encoders.bert
        import comet.encoders.xlmr
    except ImportError:
        print("unbabel-comet is not installed in this environment; nothing to do.")
        return 0

    # XLMREncoder is the base class for XLMRXLEncoder (XCOMET), MiniLMEncoder
    # and RemBERTEncoder, so patching it covers those too.
    jobs = [
        (comet.encoders.xlmr, lambda p: patch_replace(p, XLMR_OLD, XLMR_NEW, XLMR_NEW)),
        (comet.encoders.bert, lambda p: patch_replace(p, BERT_OLD, BERT_NEW, BERT_NEW)),
        (comet.encoders.base, patch_base),
    ]

    failed = False
    for module, patch in jobs:
        path = pathlib.Path(module.__file__)
        result = patch(path)
        if result is None:
            print(f"already patched: {path.name}")
        elif result:
            print(f"patched: {path.name}")
        else:
            print(f"PATTERN NOT FOUND (comet version changed?): {path}")
            failed = True

    purge_pycache(pathlib.Path(comet.encoders.base.__file__).parent.parent)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
