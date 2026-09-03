from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

# CPT3 Tier 1 DocHPLT holdout.
#
# Document level is BIDIRECTIONAL: the evaluation sets are direction
# independent, so each of the 17 pairs is evaluated both ways. 34 tasks:
#     cy_en_tier1_doc   en_cy_tier1_doc   bg_en_tier1_doc   en_bg_tier1_doc ...
# Data comes from the self-contained TSV archive data/tier1_holdout/.
#
# Sentence level is UNIDIRECTIONAL and unchanged, 17 tasks:
#     cy_en_tier1_sent  bg_en_tier1_sent ...
# It reads the JSONL sets in place via data/tier1_holdout_sent/, which is the
# code path the recorded results_*_tier1_sent.json runs used. Left alone
# deliberately so those results stay comparable.
#
# Scores are NOT comparable across directions (CPT3_PLAN.md §5.1): DocHPLT's
# top-10% filter is a per-language-pair percentile. Only within-direction,
# across-arm comparisons are valid, and no single score should be aggregated
# over the directions.

# Pair order matches the TSV filenames and the sentence JSONL names. For the
# document tasks this is a column order, not a translation direction; for the
# sentence tasks it is the direction, as before.
PAIRS = [
    "cy-en", "en-is", "en-gl", "en-sr", "en-hr", "en-et", "en-sl", "en-lv",
    "en-lt", "en-hi", "en-sk", "bg-en", "en-uk", "ar-en", "en-fi", "en-ko",
    "en-ja",
]

# FLORES+ codes, used only for MTask's target-side tokenizer selection
# (Chinese/Japanese/Korean get special sacrebleu tokenizers). These must match
# build_cpt3_arms.py LANG_TABLE. Note arb_Arab ("Arabic"), NOT arz_Arab
# ("Egyptian Arabic") which the FLORES+ scripts use for ar.
FLORES_CODES = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "cy": "cym_Latn",
    "en": "eng_Latn",
    "et": "ekk_Latn",
    "fi": "fin_Latn",
    "gl": "glg_Latn",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "is": "isl_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "uk": "ukr_Cyrl",
}


class _Tier1Task(MTask):
    OUTPUT_TYPE = "generate_until"
    GENERATION_KWARGS = None

    def doc_to_text(self, doc):
        return doc["src"]

    def doc_to_target(self, doc):
        return doc["ref"]


class _Tier1DocTask(_Tier1Task):
    VERSION = 2
    DATASET_PATH = "tier1_holdout"

    # Document sources and references are newline-joined anchors, so the
    # document tasks must opt out of the newline stop sequences the HF wrapper
    # adds unless until == [] (lm_eval/models/huggingface.py). A newline stop
    # would truncate every document after its first sentence.
    GENERATION_KWARGS = {"until": [], "do_sample": False}

    # DATASET_NAME is set per registered subclass to the direction's builder
    # config, so only that direction is built. Each config exposes a single
    # "test" split.
    def __init__(self, config=None):
        super().__init__(
            config={
                "target_delimiter": "",
                "test_split": "test",
                "generation_kwargs": dict(self.GENERATION_KWARGS),
            }
        )


class _Tier1SentTask(_Tier1Task):
    VERSION = 1
    DATASET_PATH = "tier1_holdout_sent"

    # Sentence text has no newlines, so the default stop sequences are correct
    # here. Unchanged from the original task definition.
    def __init__(self, config=None):
        super().__init__(config={"target_delimiter": "", "test_split": "test"})


def _register_doc(source_code, target_code):
    """One document task, source_code -> target_code."""
    direction = f"{source_code}_{target_code}"
    task_name = f"{direction}_tier1_doc"
    task_class = type(
        task_name.upper(),
        (_Tier1DocTask,),
        {
            "DATASET_NAME": direction,
            "get_target": lambda self, code=FLORES_CODES[target_code]: code,
        },
    )
    register_task(task_name)(task_class)


def _register_sent(direction):
    """One sentence task, unchanged from the original definition."""
    source_code, target_code = direction.split("-")
    task_name = f"{source_code}_{target_code}_tier1_sent"
    task_class = type(
        task_name.upper(),
        (_Tier1SentTask,),
        {
            "DATASET_NAME": f"sent_{direction}",
            "get_target": lambda self, code=FLORES_CODES[target_code]: code,
        },
    )
    register_task(task_name)(task_class)


for pair in PAIRS:
    l1, l2 = pair.split("-")
    _register_doc(l1, l2)
    _register_doc(l2, l1)
    _register_sent(pair)
