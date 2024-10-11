from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

from lm_eval.extra_metrics.toxicity.etox import etox_single

import pandas as pd

class HOLISTIC_BIAS_TASK(MTask):
    VERSION = 1
    DATASET_PATH = "holistic_bias"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__( config={'target_delimiter': '', 'validation_split':self.get_split()} )

    def doc_to_text(self, doc):
        return doc["text"]
    
    def doc_to_target(self, doc):
        return None

    def get_row_values(self, doc):
        return  {
                "text": doc["text"],
                "axis": doc["axis"],
                "bucket": doc["bucket"],
                "descriptor": doc["descriptor"],
                "descriptor_gender": doc["descriptor_gender"],
                "descriptor_preference": doc["descriptor_preference"],
                "noun": doc["noun"],
                "plural_noun": doc["plural_noun"],
                "noun_gender": doc["noun_gender"],
                "noun_phrase": doc["noun_phrase"],
                "plural_noun_phrase": doc["plural_noun_phrase"],
                "noun_phrase_type": doc["noun_phrase_type"],
                "template": doc["template"],
                "first_turn_only": doc["first_turn_only"],
                "must_be_noun": doc["must_be_noun"]
                }

    # modify create_dicts to account only for qe-mt 
    # metrics as HolisticBias does not have references
    def create_dicts(self, source, result, row_value):

        res, dict_aggregated = {}, {}
        
        res['toxicity'] = ( result, row_value )
        res['matched_toxicity_list'] = (None)

        dict_aggregated['toxicity'] = self.toxicity
        dict_aggregated['matched_toxicity_list'] = self.matched_toxicity_list

        if self.metric_configs['comet_kiwi']['compute']:
            res["comet_kiwi"] = (source, result)
            res["comet_kiwi_segments"] = (None)

            dict_aggregated["comet_kiwi"] = self.comet_kiwi_corpus
            dict_aggregated["comet_kiwi_segments"] = self.comet_kiwi_segments

        if self.metric_configs['metricx_qe']['compute']:
            res["metricx_qe"] = (source, result)
            res["metricx_qe_segments"] = (None)

            dict_aggregated["metricx_qe"] = self.metricx_qe_corpus
            dict_aggregated["metricx_qe_segments"] = self.metricx_qe_segments

        res["sources"] = (source)
        res["translations"] = (result)

        dict_aggregated["translations"] = self.get_translations
        dict_aggregated["sources"] = self.get_sources

        self.res = res
        self.dict_aggregated = dict_aggregated

    def process_results(self, doc, results):
        # load yaml config
        if self.metric_configs is None:
            self.load_yaml_config()

        source = self.doc_to_text(doc)
        result = results[0]
        row_value = self.get_row_values(doc)

        self.create_dicts(source, result, row_value)
        return self.res

    def toxicity(self, arr):

        results = [i[0] for i in arr]
        row_values = [i[1] for i in arr]
        target_language = self.get_target()
        toxicity_list_path = f'./lm_eval/extra_metrics/toxicity/NLLB-200_TWL/{target_language}_twl.txt'
        
        text_df = pd.DataFrame(results, columns = ['string_raw'])
        text_df.index.name = 'Dataset_ID'

        etox_output = etox_single(text_df, toxicity_list_path)
        df_Eval, _, _, _, _, _ = etox_output
        
        n_toxic_sentences = df_Eval['toxic_phrase_count'].sum()
        self.matched_toxicity_list = list(df_Eval['matched_toxicity_list'].values)
        self.selected_indices = [i for i, l in enumerate(self.matched_toxicity_list) if len(l) > 0]
        return n_toxic_sentences

    def matched_toxicity_list(self, aux=None):
        if len(self.selected_indices) > 0:
            return [self.matched_toxicity_list[i] for i in self.selected_indices]
        return []

    def comet_kiwi_segments(self, aux=None):
        if len(self.selected_indices) > 0:
            return [self.comet_kiwi_segments_list[i] for i in self.selected_indices]
        return []

    def metricx_qe_segments(self, aux=None):
        if len(self.selected_indices) > 0:
            return [self.metricxqe_segments_list[i] for i in self.selected_indices]
        return []

    def get_translations(self, arr):
        translations = [i for i in arr]
        if len(self.selected_indices) > 0:
            return [translations[i] for i in self.selected_indices]
        return []
    
    def get_targets(self, arr):
        targets = [i for i in arr]
        if len(self.selected_indices) > 0:
            return [targets[i] for i in self.selected_indices]
        return []
    
    def get_sources(self, arr):
        sources = [i for i in arr]
        if len(self.selected_indices) > 0:
            return [sources[i] for i in self.selected_indices]
        return []


languages = ['bg', 'ca', 'eu', 'gl', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv', 'zh']

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
                  'zh': 'zho_Hans', 'ms': 'zsm_Latn', 'zu': 'zul_Latn', 'mt': 'mlt_Latn'}

splits = ['others', 'ability', 'age', 'body_type',
          'characteristics', 'cultural', 'gender_and_sex',
          'nationality', 'nonce', 'political_ideologies',
          'race_ethnicity', 'religion', 'sexual_orientation',
          'socioeconomic_class']

task_definitions = []
for split in splits:
    for l1 in languages:
        if l1 != 'en':
            item = (f'en_{l1}_{split}_hb', split, MAPPING_FLORES[l1])
            task_definitions.append(item)

for task_name, split_dt, target_lang in task_definitions:
    class_name = task_name.upper()

    task_class = type(
        class_name,
        (HOLISTIC_BIAS_TASK,),
        {
            'get_split': (lambda self, split_dt=split_dt: split_dt),
            'get_target': (lambda self, target_lang=target_lang: target_lang),
        }
    )
    register_task(task_name)(task_class)