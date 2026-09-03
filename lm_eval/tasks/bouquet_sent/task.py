from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "bouquet_sent"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split': 'test'})

dataset_name = 'bouquet_sent'
# No nob_Latn -> Norwegian Bokmål， 'ast': 'ast_Latn', 'arg': 'arg_Latn',

languages = [
    'bg', 'ca', 'cs', 'cy',
    'da', 'de', 'el', 'en',
    'es', 'et', 'eu', 'fi',
    'fr', 'ga', 'gl', 'hr',
    'hu', 'it', 'lt', 'lv',
    'mt', 'nl', 'nn', 'oc',
    'pl', 'pt', 'ro', 'ru',
    'sl', 'sk', 'sr', 'sv',
    'uk', 'ar',
    'ja', 'hi', 'ko', 'zh', 'is', 
    'zh_TW', 'vi', 'arz', 'hy', 'be', 'id', 'kk', 'lij', 'se', 'th'
    
]

MAPPING_BOUQUET = {
    'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'cs': 'ces_Latn', 'cy': 'cym_Latn',
    'da': 'dan_Latn', 'de': 'deu_Latn', 'el': 'ell_Grek', 'en': 'eng_Latn',
    'es': 'spa_Latn', 'et': 'ekk_Latn', 'eu': 'eus_Latn', 'fi': 'fin_Latn',
    'fr': 'fra_Latn', 'ga': 'gle_Latn', 'gl': 'glg_Latn', 'hr': 'hrv_Latn',
    'hu': 'hun_Latn', 'it': 'ita_Latn', 'lt': 'lit_Latn', 'lv': 'lvs_Latn',
    'mt': 'mlt_Latn', 'nl': 'nld_Latn', 'nn': 'nno_Latn', 'oc': 'oci_Latn',
    'pl': 'pol_Latn', 'pt': 'por_Latn_braz1246', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl',
    'sl': 'slv_Latn', 'sk': 'slk_Latn', 'sr': 'srp_Cyrl', 'sv': 'swe_Latn',
    'uk': 'ukr_Cyrl', 'ar': 'arb_Arab',
    'ja': 'jpn_Jpan', 'hi': 'hin_Deva', 'ko': 'kor_Kore', 'zh': 'cmn_Hans', 'is': 'isl_Latn',
    'zh_TW': 'cmn_Hant', 'vi': 'vie_Latn', 'arz': 'arz_Arab', 'hy': 'hye_Armn', 'be': 'bel_Cyrl', 'id' : 'ind_Latn', 'kk' : 'kaz_Cyrl', 'lij' : 'lij_Latn', 'se' : 'sme_Latn', 'th' : 'tha_Thai', 
    
}

task_definitions = []
for l1 in languages:
    for l2 in languages:
        if l1 != l2:
            item = (f'{l1}_{l2}_{dataset_name}', f'sentence_{MAPPING_BOUQUET[l1]}', f'sentence_{MAPPING_BOUQUET[l2]}', MAPPING_BOUQUET[l2])
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
