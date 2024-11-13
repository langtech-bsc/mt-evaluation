from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "ntrex"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'test'})

dataset_name = 'ntrex'
# TO DO: Add missing ntrex languages
languages = ['af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'et', 'fi', 'fr', 'ff', 'gd', 'ga', 'gl', 'gu', 'ht', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'ig', 'ilo', 'id', 'is', 'it', 'jv', 'ja', 'kn', 'ka', 'kk', 'km', 'ko', 'lo', 'ln', 'lt', 'lb', 'lg', 'lv', 'ml', 'mr', 'mk', 'mg', 'mn', 'my', 'nl', 'no', 'ne', 'ns', 'oc', 'or', 'pa', 'fa', 'pl', 'pt', 'ps', 'ro', 'ru', 'si', 'sk', 'sl', 'sd', 'so', 'es', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'tl', 'th', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'ms', 'zu', 'eu', 'mt', 'dv', 'dz', 'fo', 'fj', 'fil', 'fuc', 'gsw', 'hmn', 'ku', 'kmr', 'mey', 'sna', 'shi', 'nqo', 'ton']

MAPPING_NTREX = {'af': 'afr', 'am': 'amh', 'ar': 'arb', 'ast': 'ast', 'az': 'azj', 'ba': 'bak', 
                 'be': 'bel', 'bn': 'ben', 'bs': 'bos', 'bg': 'bul', 'ca': 'cat', 'ceb': 'ceb', 
                 'cs': 'ces', 'cy': 'cym', 'da': 'dan', 'de': 'deu', 'el': 'ell', 'en': 'eng', 
                 'et': 'est', 'fi': 'fin', 'fr': 'fra', 'ff': 'fuv', 'gd': 'gla', 'ga': 'gle', 
                 'gl': 'glg', 'gu': 'guj', 'ht': 'hat', 'ha': 'hau', 'he': 'heb', 'hi': 'hin', 
                 'hr': 'hrv', 'hu': 'hun', 'hy': 'hye', 'ig': 'ibo', 'ilo': 'ilo', 'id': 'ind', 
                 'is': 'isl', 'it': 'ita', 'jv': 'jav', 'ja': 'jpn', 'kn': 'kan', 'ka': 'kat', 
                 'kk': 'kaz', 'km': 'khm', 'ko': 'kor', 'lo': 'lao', 'ln': 'lin', 'lt': 'lit', 
                 'lb': 'ltz', 'lg': 'lug', 'lv': 'lav', 'ml': 'mal', 'mr': 'mar', 'mk': 'mkd', 
                 'mg': 'plt', 'mn': 'khk', 'my': 'mya', 'nl': 'nld', 'no': 'nob', 'ne': 'npi', 
                 'ns': 'nso', 'oc': 'oci', 'or': 'ory', 'pa': 'pan', 'fa': 'pes', 'pl': 'pol', 
                 'pt': 'por', 'ps': 'pbt', 'ro': 'ron', 'ru': 'rus', 'si': 'sin', 'sk': 'slk', 
                 'sl': 'slv', 'sd': 'snd', 'so': 'som', 'es': 'spa', 'sq': 'als', 'sr': 'srp', 
                 'ss': 'ssw', 'su': 'sun', 'sv': 'swe', 'sw': 'swh', 'ta': 'tam', 'tl': 'tgl', 
                 'th': 'tha', 'tn': 'tsn', 'tr': 'tur', 'uk': 'ukr', 'ur': 'urd', 'uz': 'uzn', 
                 'vi': 'vie', 'wo': 'wol', 'xh': 'xho', 'yi': 'ydd', 'yo': 'yor', 'zh': 'zho-CN', 
                 'ms': 'zsm', 'zu': 'zul', 'eu': 'eus', 'mt': 'mlt', 'dv': 'div', 'dz': 'dzo', 
                 'fo': 'fao', 'fj': 'fij', 'fil': 'fil', 'fuc': 'fuc', 'gsw': 'gsw-ZH', 'hmn': 'hmn', 
                 'ku': 'ckb-Arab', 'kmr': 'kmr', 'mey': 'mey', 'sna': 'sna-Latn', 'shi': 'shi', 
                 'nqo': 'nqo', 'ton': 'ton'}



task_definitions = []
for l1 in languages:
  for l2 in languages:
    if l1 != l2:
      item = (f'{l1}_{l2}_{dataset_name}', f'sentence_{MAPPING_NTREX[l1]}', f'sentence_{MAPPING_NTREX[l2]}', MAPPING_NTREX[l2])
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
        }
    ) 
    register_task(task_name)(task_class)