from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

from sacrebleu.metrics import METRICS as sacreBLEU_metrics

import pandas as pd
import json

class MULTILINGUAL_HOLISTIC_BIAS_TASK(MTask):
    VERSION = 1
    DATASET_PATH = "multilingual_holistic_bias"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def get_row_values(self, doc):
        return  {
                "sentence_eng": doc["sentence_eng"],
                "both": doc["both"],
                "feminine": doc["feminine"],
                "masculine": doc["masculine"],
                "lang": doc["lang"],
                "gender_group": doc["gender_group"]
                }

    def create_dicts(self, source, result, row_value):

        res, dict_aggregated = {}, {}
        
        res['chrfs_both'] = ( result, row_value )
        dict_aggregated['chrfs_both'] = self.chrfs_both

        res['chrfs_masculine'] = ( None )
        dict_aggregated['chrfs_masculine'] = self.chrfs_masculine

        res['chrfs_feminine'] = ( None )
        dict_aggregated['chrfs_feminine'] = self.chrfs_feminine

        res['chrfs_masculine_segments'] = ( None )
        dict_aggregated['chrfs_masculine_segments'] = self.chrfs_masculine_segments

        res['chrfs_feminine_segments'] = ( None )
        dict_aggregated['chrfs_feminine_segments'] = self.chrfs_feminine_segments

        res['chrfs_both_segments'] = ( None )
        dict_aggregated['chrfs_both_segments'] = self.chrfs_both_segments
       
        res['masculine_translation'] = ( None )
        dict_aggregated['masculine_translation'] = self.masculine_translation

        res['feminine_translation'] = ( None )
        dict_aggregated['feminine_translation'] = self.feminine_translation
        
        res['both_translation'] = ( None )
        dict_aggregated['both_translation'] = self.both_translation

        res['masculine_source'] = ( None )
        dict_aggregated['masculine_source'] = self.masculine_source

        res['feminine_source'] = ( None )
        dict_aggregated['feminine_source'] = self.feminine_source
        
        res['both_source'] = ( None )
        dict_aggregated['both_source'] = self.both_source

        res['masculine_ref'] = ( None )
        dict_aggregated['masculine_ref'] = self.masculine_ref

        res['feminine_ref'] = ( None )
        dict_aggregated['feminine_ref'] = self.feminine_ref
        
        res['both_ref'] = ( None )
        dict_aggregated['both_ref'] = self.both_ref

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

    def chrfs_both(self, arr):

        results = [i[0] for i in arr]
        row_values = [i[1] for i in arr]

        masculine_translations = [i for i, info in zip(results, row_values) if info['masculine'] != '' ]
        masculine_refs = [eval(info['masculine']) for i, info in zip(results, row_values) if info['masculine'] != '' ]

        feminine_translations = [i for i, info in zip(results, row_values) if info['feminine'] != '' ]
        feminine_refs = [eval(info['feminine']) for i, info in zip(results, row_values) if info['feminine'] != '' ]

        both_translations = [i for i, info in zip(results, row_values) if info['both'] != '' ]
        both_refs = [eval(info['both']) for i, info in zip(results, row_values) if info['both'] != '' ]
        
        kwargs = self.metric_configs['chrf'].copy()
        del kwargs['compute']

        self.chrf_masculine, self.chrf_masculine_segments = self.get_chrf_and_segments(masculine_translations, masculine_refs, kwargs)
        self.chrf_feminine, self.chrf_feminine_segments = self.get_chrf_and_segments(feminine_translations, feminine_refs, kwargs)
        self.chrf_both, self.chrf_both_segments = self.get_chrf_and_segments(both_translations, both_refs, kwargs)
        
        self.masculine_translations = masculine_translations
        self.masculine_refs = masculine_refs
        self.masculine_sources = [info['sentence_eng'] for i, info in zip(results, row_values) if info['masculine'] != '' ]

        self.feminine_translations = feminine_translations
        self.feminine_refs = feminine_refs
        self.feminine_sources = [info['sentence_eng'] for i, info in zip(results, row_values) if info['feminine'] != '' ]

        self.both_translations = both_translations
        self.both_refs = both_refs
        self.both_sources = [info['sentence_eng'] for i, info in zip(results, row_values) if info['both'] != '' ]

        return self.chrf_both

    def get_chrf_and_segments(self, translations, refs, kwargs):

        scorer = sacreBLEU_metrics["CHRF"]()
        segment_scores = []
       
        for h, r in zip(translations, refs):
            
            try:
                score = scorer.sentence_score(h, r)
                score_in_json = json.loads(score.format(signature=str(scorer.get_signature()), is_json=True))
                segment_score = score_in_json["score"]

                segment_scores.append(segment_score)
            except:
                print('Omitting references: ', r)
                pass
            
        score = sum(segment_scores) / len(segment_scores)
        return score, segment_scores

    def chrfs_masculine(self, aux=None):
        return self.chrf_masculine

    def chrfs_feminine(self, aux=None):
        return self.chrf_feminine

    # SEGMENTS

    def chrfs_masculine_segments(self, aux=None):
        return self.chrf_masculine_segments

    def chrfs_feminine_segments(self, aux=None):
        return self.chrf_feminine_segments

    def chrfs_both_segments(self, aux=None):
        return self.chrf_both_segments


    # TRANSLATIONS

    def masculine_translation(self, aux=None):
        return self.masculine_translations

    def feminine_translation(self, aux=None):
        return self.feminine_translations

    def both_translation(self, aux=None):
        return self.both_translations

    # SOURCES 

    def masculine_source(self, aux=None):
        return self.masculine_sources

    def feminine_source(self, aux=None):
        return self.feminine_sources

    def both_source(self, aux=None):
        return self.both_sources

    # REFERENCES

    def masculine_ref(self, aux=None):
        return self.masculine_refs

    def feminine_ref(self, aux=None):
        return self.feminine_refs

    def both_ref(self, aux=None):
        return self.both_refs

    def doc_to_target(self, doc):
        return None


########### DEV SPLIT ###########

@register_task("en_es_mmhb_dev")
class MMHB_EN_ES_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'spa_dev'})

    def get_target(self):
        return 'spa_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_fr_mmhb_dev")
class MMHB_EN_FR_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'fra_dev'})

    def get_target(self):
        return 'fra_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_it_mmhb_dev")
class MMHB_EN_IT_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'ita_dev'})

    def get_target(self):
        return 'ita_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_hi_mmhb_dev")
class MMHB_EN_HI_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'hin_dev'})

    def get_target(self):
        return 'hin_Deva'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_id_mmhb_dev")
class MMHB_EN_ID_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'ind_dev'})

    def get_target(self):
        return 'ind_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]


@register_task("en_pt_mmhb_dev")
class MMHB_EN_PT_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'por_dev'})

    def get_target(self):
        return 'por_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_vi_mmhb_dev")
class MMHB_EN_VI_DEV(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'vie_dev'})

    def get_target(self):
        return 'vie_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

########### DEVTEST SPLIT ###########

@register_task("en_es_mmhb_devtest")
class MMHB_EN_ES_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'spa_devtest'})

    def get_target(self):
        return 'spa_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_fr_mmhb_devtest")
class MMHB_EN_FR_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'fra_devtest'})

    def get_target(self):
        return 'fra_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_it_mmhb_devtest")
class MMHB_EN_IT_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'ita_devtest'})

    def get_target(self):
        return 'ita_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_hi_mmhb_devtest")
class MMHB_EN_HI_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'hin_devtest'})

    def get_target(self):
        return 'hin_Deva'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_id_mmhb_devtest")
class MMHB_EN_ID_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'ind_devtest'})

    def get_target(self):
        return 'ind_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]


@register_task("en_pt_mmhb_devtest")
class MMHB_EN_PT_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'por_devtest'})

    def get_target(self):
        return 'por_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_vi_mmhb_devtest")
class MMHB_EN_VI_DEVTEST(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'vie_devtest'})

    def get_target(self):
        return 'vie_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]


########### TRAIN SPLIT ###########

@register_task("en_es_mmhb_train")
class MMHB_EN_ES_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'spa_train'})

    def get_target(self):
        return 'spa_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_fr_mmhb_train")
class MMHB_EN_FR_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'fra_train'})

    def get_target(self):
        return 'fra_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_it_mmhb_train")
class MMHB_EN_IT_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'ita_train'})

    def get_target(self):
        return 'ita_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_hi_mmhb_train")
class MMHB_EN_HI_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'hin_train'})

    def get_target(self):
        return 'hin_Deva'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_id_mmhb_train")
class MMHB_EN_ID_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'ind_train'})

    def get_target(self):
        return 'ind_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]


@register_task("en_pt_mmhb_train")
class MMHB_EN_PT_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'por_train'})

    def get_target(self):
        return 'por_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]

@register_task("en_vi_mmhb_train")
class MMHB_EN_VI_TRAIN(MULTILINGUAL_HOLISTIC_BIAS_TASK):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'vie_train'})

    def get_target(self):
        return 'vie_Latn'

    def doc_to_text(self, doc):
        return doc["sentence_eng"]