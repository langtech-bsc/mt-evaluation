from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "wmt24pp"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'test'})


dataset_name = 'wmt24pp'
# TO DO: Add missing flores languages
languages = [
	"ar", "ar_SA", "bg", "bn", "ca", "cs", "da", 
	"de", "el", "es", "et", "fa", "fi", "fil", "fr_CA", 
	"fr", "gu", "he", "hi", "hr", "hu", "id", "is", 
	"it", "ja", "kn", "ko", "lt", "lv", "ml", "mr", 
	"nl", "no", "pa", "pl", "pt_BR", "pt", "ro", "ru", 
	"sk", "slI", "sr", "sv", "sw", "sw_TZ", "ta", "te", 
	"th", "tr", "uk", "ur", "vi", "zh", "zh_TW", "zu",
	"en", "sh"
]

MAPPING_WMT24PP = {
    
    "ar": "ar_EG", "ar_SA": "ar_SA", "bg": "bg_BG", "bn": "bn_IN", "ca": "ca_ES", "cs": "cs_CZ", "da": "da_DK",
    "de": "de_DE", "el": "el_GR", "es": "es_MX", "et": "et_EE", "fa": "fa_IR", "fi": "fi_FI", "fil": "fil_PH",
    "fr_CA": "fr_CA", "fr": "fr_FR", "gu": "gu_IN", "he": "he_IL", "hi": "hi_IN", "hr": "hr_HR", "hu": "hu_HU",
    "id": "id_ID", "is": "is_IS", "it": "it_IT", "ja": "ja_JP", "kn": "kn_IN", "ko": "ko_KR", "lt": "lt_LT",
    "lv": "lv_LV", "ml": "ml_IN", "mr": "mr_IN", "nl": "nl_NL", "no": "no_NO", "pa": "pa_IN", "pl": "pl_PL",
    "pt_BR": "pt_BR", "pt": "pt_PT", "ro": "ro_RO", "ru": "ru_RU", "sk": "sk_SK", "slI": "sl_SI", "sr": "sr_RS",
    "sv": "sv_SE", "sw": "sw_KE", "sw_TZ": "sw_TZ", "ta": "ta_IN", "te": "te_IN", "th": "th_TH", "tr": "tr_TR",
    "uk": "uk_UA", "ur": "ur_PK", "vi": "vi_VN", "zh": "zh_CN", "zh_TW": "zh_TW", "zu": "zu_ZA",
    "en": "en", "sh": "sr_Ltn"
}


task_definitions = []
for l1 in languages:
  for l2 in languages:
    if l1 != l2:
      item = (f'{l1}_{l2}_{dataset_name}', f'sentence_{MAPPING_WMT24PP[l1]}', f'sentence_{MAPPING_WMT24PP[l2]}', MAPPING_WMT24PP[l2])
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