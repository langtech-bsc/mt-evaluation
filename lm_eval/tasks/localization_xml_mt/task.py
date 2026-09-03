from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "localization_xml_mt"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split': self.get_split()})

dataset_name = 'localization_xml_mt'

all_langs = [ ['en', 'ja'],
              ['en', 'zh'],
              ['en', 'nl'],
              ['en', 'fi'],
              ['en', 'fr'],
              ['en', 'de'],
              ['en', 'ru']
            ]


task_definitions = []
for languages in all_langs:
    for l1 in languages:
        for l2 in languages:
            if l1 != l2:
                item = (l1, l2, f'{l1}_{l2}_{dataset_name}', 'src', 'ref', l2)
                task_definitions.append(item)

for l1, l2, task_name, source_field, target_field, target_lang in task_definitions:
    class_name = task_name.upper()

    split = f"{l1}_{l2}"

    task_class = type(
        class_name,
        (_MTask,),
        {   
            'get_split': (lambda self, split=split: split),
            'doc_to_text': (lambda self, doc, source_field=source_field: doc[source_field]),
            'doc_to_target': (lambda self, doc, target_field=target_field: doc[target_field]),
            'get_target': (lambda self, target_lang=target_lang: target_lang),
        }
    ) 

    register_task(task_name)(task_class)
