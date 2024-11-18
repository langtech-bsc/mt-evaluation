from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask
import sacrebleu
from lm_eval.extra_metrics.comet.metric import BaseCOMET


class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "flores+_devtest_perturbations"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'devtest'})
    
    def create_dicts(self, source, target, result):

        res, dict_aggregated = {}, {}

        if self.metric_configs['bleu']['compute']: 
            res["bleu"] = (target, result)
            dict_aggregated["bleu"] = self.bleu_corpus

        if self.metric_configs['ter']['compute']: 
            res["ter"] = (target, result)
            dict_aggregated["ter"] = self.ter_corpus

        if self.metric_configs['comet']['compute']: 
            res["comet"] = (source, target, result)
            dict_aggregated["comet"] = self.comet_corpus
                
        self.res = res
        self.dict_aggregated = dict_aggregated

    def bleu_corpus(self, arr):
        targets = [i[0] for i in arr][:1012]
        translations = [i[1] for i in arr]
        bleus = []

        kwargs = self.metric_configs['bleu'].copy()
        del kwargs['compute']

        for i in range(0, 15180, 1012):
            translations_i = translations[i:i+1012]

            if self.get_target() in ['zho_Hans', 'zho_Hant', 'zho-CN']:
                del kwargs['tokenize']
                bleuscore = sacrebleu.corpus_bleu(translations_i, [targets], tokenize='zh', **kwargs)
                bleus.append( round(bleuscore.score, 2) )

            else:

                bleuscore = sacrebleu.corpus_bleu(translations_i, [targets], **kwargs)
                bleus.append( round(bleuscore.score, 2) )
        
        bleus_task = {'swap':bleus[0:5], 'chardupe':bleus[5:10], 'chardrop':bleus[10:15]}
        return bleus_task
    
    def ter_corpus(self, arr):

        kwargs = self.metric_configs['ter'].copy()
        del kwargs['compute']

        targets = [i[0] for i in arr][:1012]
        translations = [i[1] for i in arr]
        ters = []

        for i in range(0, 15180, 1012):
            translations_i = translations[i:i+1012]
            score = sacrebleu.corpus_ter(translations_i, [targets], **kwargs).score
            ters.append( round(score, 2) )
        
        ters_task = {'swap':ters[0:5], 'chardupe':ters[5:10], 'chardrop':ters[10:15]}
        return ters_task

    def chrf_corpus(self, arr):
        kwargs = self.metric_configs['chrf'].copy()
        del kwargs['compute']

        targets = [i[0] for i in arr][:1012]
        translations = [i[1] for i in arr]

        chrfs = []
        for i in range(0, 15180, 1012):
            translations_i = translations[i:i+1012]
            score = sacrebleu.corpus_chrf(translations_i, [targets], **kwargs).score
            chrfs.append(round(score, 2))

        chrfs_task = {'swap':chrfs[0:5], 'chardupe':chrfs[5:10], 'chardrop':chrfs[10:15]}
        return chrfs_task

    def comet_corpus(self, arr):

        batch_size = self.metric_configs['comet']['batch_size']
        ck_name = self.metric_configs['comet']['checkpoint']

        self.comet = BaseCOMET(ck_name)
        sources = [i[0] for i in arr][:1012]
        targets = [i[1] for i in arr][:1012]

        translations = [i[2] for i in arr]
        comets = []
        for i in range(0, 15180, 1012):
            translations_i = translations[i:i+1012]

            comet_result = self.comet.evaluate(translations_i, targets, sources, batch_size )
            comets.append( round(comet_result["system_score"], 2) )
            
        comets_task = {'swap':comets[0:5], 'chardupe':comets[5:10], 'chardrop':comets[10:15]}
        return comets_task

dataset_name = 'perturbations'
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