from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "flores+_devtest"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'devtest'})

dataset_name = 'flores+_devtest'
# TO DO: Add missing flores languages
languages = ['bg', 'ca', 'eu', 'gl', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv', 'zh', 'ast', 'arg', 'arn']

MAPPING_FLORES = {'af': 'afr_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab', 'ast': 'ast_Latn', 'az': 'azj_Latn', 'ba': 'bak_Cyrl', 
                  'be': 'bel_Cyrl', 'bn': 'ben_Beng', 'bs': 'bos_Latn', 'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'ceb': 'ceb_Latn', 
                  'cs': 'ces_Latn', 'cy': 'cym_Latn', 'da': 'dan_Latn', 'de': 'deu_Latn', 'el': 'ell_Grek', 'en': 'eng_Latn', 
                  'eu': 'eus_Latn', 'et': 'est_Latn', 'fi': 'fin_Latn', 'fr': 'fra_Latn', 'ff': 'fuv_Latn', 'gd': 'gla_Latn', 
                  'ga': 'gle_Latn', 'gl': 'glg_Latn', 'gu': 'guj_Gujr', 'ht': 'hat_Latn', 'ha': 'hau_Latn', 'he': 'heb_Hebr', 
                  'hi': 'hin_Deva', 'hr': 'hrv_Latn', 'hu': 'hun_Latn', 'hy': 'hye_Armn', 'ig': 'ibo_Latn', 'ilo': 'ilo_Latn', 
                  'id': 'ind_Latn', 'is': 'isl_Latn', 'it': 'ita_Latn', 'jv': 'jav_Latn', 'ja': 'jpn_Jpan', 'kn': 'kan_Knda', 
                  'ka': 'kat_Geor', 'kk': 'kaz_Cyrl', 'km': 'khm_Khmr', 'ko': 'kor_Hang', 'lo': 'lao_Laoo', 'ln': 'lin_Latn', 
                  'lt': 'lit_Latn', 'lb': 'ltz_Latn', 'lg': 'lug_Latn', 'lv': 'lvs_Latn', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 
                  'mk': 'mkd_Cyrl', 'mg': 'plt_Latn', 'mn': 'khk_Cyrl', 'my': 'mya_Mymr', 'nl': 'nld_Latn', 'no': 'nob_Latn', 
                  'ne': 'npi_Deva', 'ns': 'nso_Latn', 'oc': 'oci_Latn', 'or': 'ory_Orya', 'pa': 'pan_Guru', 'fa': 'pes_Arab', 
                  'pl': 'pol_Latn', 'pt': 'por_Latn', 'ps': 'pbt_Arab', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'si': 'sin_Sinh', 
                  'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sd': 'snd_Arab', 'so': 'som_Latn', 'es': 'spa_Latn', 'sq': 'als_Latn', 
                  'sr': 'srp_Cyrl', 'ss': 'ssw_Latn', 'su': 'sun_Latn', 'sv': 'swe_Latn', 'sw': 'swh_Latn', 'ta': 'tam_Taml', 
                  'tl': 'tgl_Latn', 'th': 'tha_Thai', 'tn': 'tsn_Latn', 'tr': 'tur_Latn', 'uk': 'ukr_Cyrl', 'ur': 'urd_Arab', 
                  'uz': 'uzn_Latn', 'vi': 'vie_Latn', 'wo': 'wol_Latn', 'xh': 'xho_Latn', 'yi': 'ydd_Hebr', 'yo': 'yor_Latn', 
                  'zh': 'zho_Hans', 'ms': 'zsm_Latn', 'zu': 'zul_Latn', 'mt': 'mlt_Latn', 'arg':'arg_Latn', 'arn':'arn_Latn'}


task_definitions = []
for l1 in languages:
  for l2 in languages:
    if l1 != l2:
      item = (f'{l1}_{l2}_{dataset_name}', f'sentence_{MAPPING_FLORES[l1]}', f'sentence_{MAPPING_FLORES[l2]}', MAPPING_FLORES[l2])
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