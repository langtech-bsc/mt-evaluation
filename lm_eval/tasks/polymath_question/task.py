from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask


class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "polymath_question"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        # generation_kwargs.until defaults to [fewshot_delimiter="\n\n"] in
        # ConfigurableTask. PolyMath source contains multi-paragraph problems,
        # so we must not stop at "\n\n" or translations get truncated at the
        # first paragraph. Empty list = stop only at EOS / max_new_tokens.
        super().__init__(config={
            'target_delimiter': '',
            'validation_split': 'test',
            'generation_kwargs': {'until': []},
        })


dataset_name = 'polymath_question'

languages = [
    "ar", "bn", "ca", "de", "en", "es", "fr", "id", "it", "ja",
    "ko", "ms", "pt", "ru", "sw", "te", "th", "vi", "zh",
]

MAPPING_POLYMATH = {lang: lang for lang in languages}


task_definitions = []
for l1 in languages:
    for l2 in languages:
        if l1 != l2:
            item = (
                f'{l1}_{l2}_{dataset_name}',
                f'sentence_{MAPPING_POLYMATH[l1]}',
                f'sentence_{MAPPING_POLYMATH[l2]}',
                MAPPING_POLYMATH[l2],
            )
            task_definitions.append(item)

for task_name, source_field, target_field, target_lang in task_definitions:
    class_name = task_name.upper()

    task_class = type(
        class_name,
        (_MTask,),
        {
            'doc_to_text': (lambda self, doc, source_field=source_field: doc[source_field]),
            'doc_to_target': (lambda self, doc, target_field=target_field: doc[target_field]),
            'get_target': (lambda self, target_lang=target_lang: target_lang),
        },
    )
    register_task(task_name)(task_class)
