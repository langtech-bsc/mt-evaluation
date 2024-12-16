from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "facebook/flores"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'devtest'})


dataset_name = 'flores_devtest'
# TO DO: Add missing flores languages
languages = ['ac', 'ace', 'acm', 'acq', 'aeb', 'af', 'ajp', 'sq', 'am', 'apc', 'ar', 'arb', 'ars', 'ary','arz', 'as', 'ast', 'awa', 'ay', 'aze', 'az', 'ba', 'be', 'bm', 'ban', 'bem', 'bn', 'bho', 'bj', 'bjn', 'bo', 'bs', 'bug', 'bg', 'ca', 'ceb', 'cs', 'cjk', 'ckb', 'crh', 'cy', 'da', 'de', 'dik', 'dyu', 'dz', 'el', 'en', 'eo', 'eu', 'et', 'ee', 'fo', 'fi', 'fj', 'fon', 'fr', 'fur', 'ff', 'om', 'gd', 'ga', 'gl', 'gn', 'gu', 'ht', 'ha', 'he', 'hi', 'hne', 'hr', 'hu', 'hy', 'ig', 'ilo', 'id', 'is', 'it', 'jv', 'ja', 'kab', 'kac', 'kam','kn', 'ks', 'kas', 'ka', 'kk', 'kbp', 'kea', 'mn', 'km', 'ki', 'rw', 'ky', 'kmb', 'kmr', 'kr', 'knc', 'kg','ko', 'lo', 'lij', 'li', 'ln', 'lt', 'lmo', 'ltg', 'lb', 'lua', 'lg', 'luo', 'lus', 'lv', 'mag', 'mai', 'ml', 'mr', 'min', 'min_l', 'mk', 'mt', 'mni', 'mos', 'mi', 'my', 'nl', 'nn', 'no', 'ne', 'ns', 'nus', 'nya', 'oc', 'or', 'pag', 'pa', 'pap', 'ps', 'fa', 'mg', 'pl', 'pt', 'prs', 'qu', 'ro', 'rn', 'ru', 'sg', 'sa', 'sat', 'scn', 'shn', 'si', 'sk', 'sl', 'sm', 'sn', 'sd', 'so', 'st', 'es', 'sc', 'sr', 'ss', 'su', 'sv', 'sw', 'szl', 'ta', 'tq', 'taq','tt', 'te', 'tg', 'tl', 'th', 'ti', 'tpi', 'tn', 'ts', 'tk', 'tum', 'tr', 'tw', 'tzm', 'ug', 'uk', 'umb', 'ur', 'uz', 'vec', 'vi', 'war', 'wo', 'xh', 'yi', 'yo', 'yue', 'zh', 'zho', 'ms', 'zu', 'arg', 'arn', 'vl']

MAPPING_FLORES = {'ac':'ace_Arab', 'ace':'ace_Latn', 'acm':'acm_Arab', 'acq': 'acq_Arab', 'aeb':'aeb_Arab', 'af': 'afr_Latn', 'ajp':'ajp_Arab',
                  'sq': 'als_Latn', 'am': 'amh_Ethi', 'apc': 'apc_Arab', 'ar': 'arb_Arab', 'arb': 'arb_Latn', 'ars': 'ars_Arab', 
                  'ary': 'ary_Arab','arz': 'arz_Arab', 'as':'asm_Beng', 'ast': 'ast_Latn', 'awa': 'awa_Deva', 'ay': 'ayr_Latn', 
                  'aze': 'azb_Arab', 'az': 'azj_Latn', 'ba': 'bak_Cyrl', 'be': 'bel_Cyrl', 'bm': 'bam_Latn', 'ban': 'ban_Latn', 
                  'bem': 'bem_Latn', 'bn': 'ben_Beng', 'bho': 'bho_Deva', 'bj': 'bjn_Arab', 'bjn':'bjn_Latn', 
                  'bo': 'bod_Tibt', 'bs': 'bos_Latn', 'bug': 'bug_Latn', 'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'ceb': 'ceb_Latn', 
                  'cs': 'ces_Latn', 'cjk': 'cjk_Latn', 'ckb': 'ckb_Arab', 'crh': 'crh_Latn', 'cy': 'cym_Latn', 'da': 'dan_Latn', 
                  'de': 'deu_Latn', 'dik': 'dik_Latn', 'dyu': 'dyu_Latn', 'dz': 'dzo_Tibt', 'el': 'ell_Grek', 'en': 'eng_Latn', 
                  'eo': 'epo_Latn', 'eu': 'eus_Latn', 'et': 'est_Latn', 'ee':'ewe_Latn', 'fo': 'fao_Latn', 'fi': 'fin_Latn', 
                  'fj': 'fij_Latn', 'fon': 'fon_Latn', 'fr': 'fra_Latn', 'fur': 'fur_Latn', 'ff': 'fuv_Latn', 'om': 'gaz_Latn', 
                  'gd': 'gla_Latn', 'ga': 'gle_Latn', 'gl': 'glg_Latn', 'gn': 'grn_Latn', 'gu': 'guj_Gujr', 'ht': 'hat_Latn', 
                  'ha': 'hau_Latn', 'he': 'heb_Hebr', 'hi': 'hin_Deva', 'hne': 'hne_Deva', 'hr': 'hrv_Latn', 'hu': 'hun_Latn', 
                  'hy': 'hye_Armn', 'ig': 'ibo_Latn', 'ilo': 'ilo_Latn', 'id': 'ind_Latn', 'is': 'isl_Latn', 'it': 'ita_Latn', 
                  'jv': 'jav_Latn', 'ja': 'jpn_Jpan', 'kab': 'kab_Latn', 'kac': 'kac_Latn', 'kam': 'kam_Latn','kn': 'kan_Knda', 
                  'ks': 'kas_Arab', 'kas': 'kas_Deva', 'ka': 'kat_Geor', 'kk': 'kaz_Cyrl', 'kbp': 'kbp_Latn', 'kea': 'kea_Latn', 
                  'mn': 'khk_Cyrl', 'km': 'khm_Khmr', 'ki': 'kik_Latn', 'rw': 'kin_Latn', 'ky': 'kir_Cyrl', 'kmb': 'kmb_Latn', 
                  'kmr': 'kmr_Latn', 'kr': 'knc_Arab', 'knc': 'knc_Latn', 'kg': 'kon_Latn','ko': 'kor_Hang', 'lo': 'lao_Laoo', 
                  'lij': 'lij_Latn', 'li': 'lim_Latn', 'ln': 'lin_Latn', 'lt': 'lit_Latn', 'lmo': 'lmo_Latn', 'ltg': 'ltg_Latn', 
                  'lb': 'ltz_Latn', 'lua': 'lua_Latn', 'lg': 'lug_Latn', 'luo': 'luo_Latn', 'lus': 'lus_Latn', 'lv': 'lvs_Latn', 
                  'mag': 'mag_Deva', 'mai': 'mai_Deva', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'min': 'min_Arab', 'min_l': 'min_Latn', 
                  'mk': 'mkd_Cyrl', 'mt': 'mlt_Latn', 'mni': 'mni_Beng', 'mos': 'mos_Latn', 'mi': 'mri_Latn', 'my': 'mya_Mymr', 
                  'nl': 'nld_Latn', 'nn': 'nno_Latn', 'no': 'nob_Latn', 'ne': 'npi_Deva', 'ns': 'nso_Latn', 'nus': 'nus_Latn', 
                  'nya': 'nya_Latn', 'oc': 'oci_Latn', 'or': 'ory_Orya', 'pag': 'pag_Latn', 'pa': 'pan_Guru', 'pap': 'pap_Latn', 
                  'ps': 'pbt_Arab', 'fa': 'pes_Arab', 'mg': 'plt_Latn', 'pl': 'pol_Latn', 'pt': 'por_Latn', 'prs': 'prs_Arab', 
                  'qu': 'quy_Latn', 'ro': 'ron_Latn', 'rn': 'run_Latn', 'ru': 'rus_Cyrl', 'sg': 'sag_Latn', 'sa': 'san_Deva', 
                  'sat': 'sat_Olck', 'scn': 'scn_Latn', 'shn': 'shn_Mymr', 'si': 'sin_Sinh', 'sk': 'slk_Latn', 'sl': 'slv_Latn', 
                  'sm': 'smo_Latn', 'sn': 'sna_Latn', 'sd': 'snd_Arab', 'so': 'som_Latn', 'st': 'sot_Latn', 'es': 'spa_Latn', 
                  'sc': 'srd_Latn', 'sr': 'srp_Cyrl', 'ss': 'ssw_Latn', 'su': 'sun_Latn', 'sv': 'swe_Latn', 'sw': 'swh_Latn', 
                  'szl': 'szl_Latn', 'ta': 'tam_Taml', 'tq': 'taq_Latn', 'taq': 'taq_Tfng','tt': 'tat_Cyrl', 'te': 'tel_Telu', 
                  'tg': 'tgk_Cyrl', 'tl': 'tgl_Latn', 'th': 'tha_Thai', 'ti': 'tir_Ethi', 'tpi': 'tpi_Latn',  'tn': 'tsn_Latn', 
                  'ts': 'tso_Latn', 'tk': 'tuk_Latn', 'tum': 'tum_Latn', 'tr': 'tur_Latn', 'tw': 'twi_Latn', 'tzm': 'tzm_Tfng', 
                  'ug': 'uig_Arab', 'uk': 'ukr_Cyrl', 'umb': 'umb_Latn', 'ur': 'urd_Arab', 'uz': 'uzn_Latn', 'vec': 'vec_Latn', 
                  'vi': 'vie_Latn', 'war': 'war_Latn', 'wo': 'wol_Latn', 'xh': 'xho_Latn', 'yi': 'ydd_Hebr', 'yo': 'yor_Latn', 
                  'yue': 'yue_Hant', 'zh': 'zho_Hans', 'zho': 'zho_Hant', 'ms': 'zsm_Latn', 'zu': 'zul_Latn', 'arg':'arg_Latn', 
                  'arn':'arn_Latn', 'vl': 'val_Latn'}


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