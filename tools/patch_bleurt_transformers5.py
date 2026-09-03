"""Make bleurt-pytorch work with transformers >= 5.

bleurt-pytorch (0.0.1, unmaintained) reaches into transformers internals that
changed in v5, so importing it - and therefore BLEURT scoring - fails:

1. ``transformers.pytorch_utils.find_pruneable_heads_and_indices`` was removed
   (``apply_chunking_to_forward`` and ``prune_linear_layer`` survive). It is
   only used by the attention-head pruning path, which BLEURT never calls, so
   the implementation is vendored from transformers v4.57.1.

2. ``transformers.models.bert.tokenization_bert_fast`` was merged into
   ``tokenization_bert``, which still defines ``BertTokenizerFast`` as an alias
   of ``BertTokenizer``; only the module path changed.

Both rewrites keep the original import first and fall back only when it fails,
so the package behaves identically on transformers 4.

Run once after every ``pip install`` that (re)installs bleurt-pytorch:

    python tools/patch_bleurt_transformers5.py
"""

import importlib.util
import pathlib
import sys


PRUNE_OLD = (
    "from transformers.pytorch_utils import apply_chunking_to_forward, "
    "find_pruneable_heads_and_indices, prune_linear_layer"
)

PRUNE_NEW = '''from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

try:  # removed in transformers 5
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    import torch as _torch

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        """Vendored from transformers v4.57.1 (removed in transformers 5).

        Only used by the attention-head pruning path, which BLEURT scoring
        never calls; kept so the module imports on both major versions.
        """
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = _torch.arange(len(mask))[mask].long()
        return heads, index'''

TOK_OLD = "from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast"

TOK_NEW = """try:
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
except ModuleNotFoundError:  # transformers 5 merged the fast tokenizer module
    from transformers.models.bert.tokenization_bert import BertTokenizerFast"""

HEAD_MASK_OLD = (
    "        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)"
)

HEAD_MASK_NEW = """        # transformers 5 removed PreTrainedModel.get_head_mask
        _get_head_mask = getattr(self, "get_head_mask", None)
        if _get_head_mask is not None:
            head_mask = _get_head_mask(head_mask, self.config.num_hidden_layers)
        elif head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers
        else:
            raise NotImplementedError(
                "an explicit head_mask needs PreTrainedModel.get_head_mask, "
                "which transformers 5 removed"
            )"""

EXT_MASK_OLD = "        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)"

EXT_MASK_NEW = """        # get_extended_attention_mask is deprecated and slated for removal;
        # this fallback is dormant while transformers still provides it
        _get_extended = getattr(self, "get_extended_attention_mask", None)
        if _get_extended is not None:
            extended_attention_mask: torch.Tensor = _get_extended(attention_mask, input_shape)
        else:
            _dtype = self.dtype
            extended_attention_mask = attention_mask[:, None, None, :].to(_dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(_dtype).min"""

DISPATCH_OLD = """        try:
            return BleurtTokenizerFast.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        except (OSError, TypeError):
            return BleurtSPTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)"""

DISPATCH_NEW = '''        # BLEURT-20 needs the SentencePiece tokenizer. The original code relied on
        # BleurtTokenizerFast raising when the WordPiece vocabulary is absent, but
        # transformers 5's BertTokenizer returns an empty tokenizer instead, so the
        # fallback never fired and every token became [UNK] - scores came out
        # identical for good and garbage translations. Pick SentencePiece up front,
        # and validate the fast tokenizer before trusting it.
        spm_file = os.path.join(str(pretrained_model_name_or_path), "spm.model")
        if os.path.isfile(spm_file):
            return BleurtSPTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        try:
            tokenizer = BleurtTokenizerFast.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        except (OSError, TypeError):
            return BleurtSPTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        probe = tokenizer("hello world", add_special_tokens=False)["input_ids"]
        if unk_id is not None and probe and all(i == unk_id for i in probe):
            return BleurtSPTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return tokenizer'''

# (relative path inside the package, old text, new text, marker proving it is applied)
PATCHES = [
    ("bleurt/modeling_bleurt.py", PRUNE_OLD, PRUNE_NEW, "removed in transformers 5"),
    ("bleurt/tokenization_bleurt_fast.py", TOK_OLD, TOK_NEW, "merged the fast tokenizer module"),
    ("bleurt/modeling_bleurt.py", HEAD_MASK_OLD, HEAD_MASK_NEW, "_get_head_mask = getattr"),
    ("bleurt/modeling_bleurt.py", EXT_MASK_OLD, EXT_MASK_NEW, "_get_extended = getattr"),
    ("bleurt/tokenization_bleurt.py", DISPATCH_OLD, DISPATCH_NEW, "Pick SentencePiece up front"),
]


def package_dir() -> pathlib.Path | None:
    # find_spec does not execute the package, so this works even while the
    # package is unimportable because of the very bug being fixed.
    try:
        spec = importlib.util.find_spec("bleurt_pytorch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(list(spec.submodule_search_locations)[0])



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
    root = package_dir()
    if root is None:
        print("bleurt-pytorch is not installed; nothing to do.")
        return 0

    failed = False
    for rel, old, new, marker in PATCHES:
        path = root / rel
        if not path.exists():
            print(f"missing file (bleurt-pytorch version changed?): {path}")
            failed = True
            continue
        source = path.read_text()
        if marker in source:
            print(f"already patched: {rel}")
            continue
        if old not in source:
            print(f"PATTERN NOT FOUND (bleurt-pytorch version changed?): {rel}")
            failed = True
            continue
        path.write_text(source.replace(old, new, 1))
        print(f"patched: {rel}")

    purge_pycache(root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
