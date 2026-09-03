from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "abstracts"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split': self.get_split()})

dataset_name = 'abstracts'
#languages = ['ca', "en"]
#all_langs = [ ['ca', "en"], ['ca', 'es'], ['ca', 'de'], ['ca', 'eu'], ['ca', 'fr'], ['ca', 'it'], ['ca', 'pt'],
#              ['de', 'en'], ['de', 'eu'], ['de', 'fr'], ['de', 'it'], ['de', 'pt'],
#              ['en', 'es'], ['en', 'eu'], ['en', 'fr'], ['en', 'it'], ['en', 'pt'],
#              ['es', 'eu'], ['es', 'fr'], ['es', 'it'], ['es', 'pt'], 
#              ['eu', 'fr'], ['eu', 'it'], ['eu', 'pt'],
#              ['fr', 'it'], ['fr', 'pt'],
#              ['it', 'pt']
#            ]

all_langs = [ ['ca', "en"],
              ['de', 'en'], 
              ['en', 'es'], 
              ['en', 'fr'], 
              ['en', 'it'], 
              ['en', 'pt'],
              ['ca', 'es'],
              ['de', 'fr'],
              ['en', 'eu'],
              ['es', 'eu'],
              ['es', 'fr'],
              ['es', 'gl'],
              ['es', 'pt'],
              ['fr', 'it'],
              ['fr', 'pt']
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