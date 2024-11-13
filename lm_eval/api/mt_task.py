import numpy as np
from lm_eval.extra_metrics.bleurt.metric import BLEURT
from lm_eval.extra_metrics.comet.metric import BaseCOMET
from lm_eval.extra_metrics.comet_kiwi.metric import COMETKiwi
from lm_eval.extra_metrics.xcomet.metric import XCOMET
from lm_eval.extra_metrics.metricx.metric import RefMetricX, QEMetricX
from lm_eval.api.task import ConfigurableTask
from lm_eval import utils
import sacrebleu
import random
import yaml

METRICS_MT = [  "bleu", "ter", "chrf", "comet", "comet_kiwi", "bleurt", 
                "xcomet", "bleu_segments", "ter_segments", "chrf_segments", "comet_kiwi_segments", "comet_segments", "xcomet_segments", 
                "xcomet_error_spans", "metricx", "metricx_segments", "metricx_qe", "metricx_qe_segments", 
                "translations", "targets", "sources"]

class MTask(ConfigurableTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_configs = None

    ############## METRICS ##############
    def bleu_corpus(self, arr):
        """
        Computes the BLEU score for the corpus.
        Args:
            arr (list): A list of tuples containing target and translation pairs.

        Returns:
            float: The BLEU score.
        """
        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        kwargs = self.metric_configs['bleu'].copy()
        del kwargs['compute']

        if self.get_target() in ['zho_Hans', 'zho_Hant', 'zho-CN']:
            del kwargs['tokenize']
            bleuscore = sacrebleu.corpus_bleu(translations, [targets], tokenize='zh', **kwargs)
            return bleuscore.score

        bleuscore = sacrebleu.corpus_bleu(translations, [targets], **kwargs)
        return bleuscore.score

    def ter_corpus(self, arr):
        """
        Computes the TER score for the corpus.
        Args:
            arr (list): A list of tuples containing target and translation pairs.

        Returns:
            float: The TER score.
        """
        kwargs = self.metric_configs['ter'].copy()
        del kwargs['compute']

        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        score = sacrebleu.corpus_ter(translations, [targets], **kwargs).score
        return score

    def chrf_corpus(self, arr):
        """
        Computes the CHRF score for the corpus.
        Args:
            arr (list): A list of tuples containing target and translation pairs.

        Returns:
            float: The CHRF score.
        """
        kwargs = self.metric_configs['chrf'].copy()
        del kwargs['compute']

        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]

        score = sacrebleu.corpus_chrf(translations, [targets], **kwargs).score
        return score

    def comet_corpus(self, arr):
        """
        Computes the COMET score for the corpus.
        Args:
            arr (list): A list of tuples containing source, target, and translation triples.

        Returns:
            float: The COMET score.
        """

        batch_size = self.metric_configs['comet']['batch_size']
        ck_name = self.metric_configs['comet']['checkpoint']

        self.comet = BaseCOMET(ck_name)
        sources = [i[0] for i in arr]
        targets = [i[1] for i in arr]
        translations = [i[2] for i in arr]
        comet_result = self.comet.evaluate(translations, targets, sources, batch_size )
        self.comet_segments_list = comet_result["segments_scores"]
        return comet_result["system_score"]

    def comet_kiwi_corpus(self, arr):
        """
        Computes the COMET KIWI score for the corpus.
        Args:
            arr (list): A list of tuples containing source and translation tuples.

        Returns:
            float: The COMET KIWI score.
        """

        batch_size = self.metric_configs['comet_kiwi']['batch_size']
        ck_name = self.metric_configs['comet_kiwi']['checkpoint']

        self.comet = COMETKiwi(ck_name)
        sources = [i[0] for i in arr]

        translations = [i[1] for i in arr]
        comet_result = self.comet.evaluate(translations, sources, batch_size )
        self.comet_kiwi_segments_list = comet_result["segments_scores"]
        return comet_result["system_score"]

    def bleurt_corpus(self, arr):
        """
        Computes the BLEURT score for the corpus.
        Args:
            arr (list): A list of tuples containing target and translation pairs.

        Returns:
            float: The BLEURT score.
        """

        batch_size = self.metric_configs['bleurt']['batch_size']
        ck_name = self.metric_configs['bleurt']['checkpoint']

        self.bleurt = BLEURT(ck_name)
        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        bleurt_result = self.bleurt.evaluate(translations, targets, batch_size)
        return bleurt_result["system_score"]
    
    def xcomet_corpus(self, arr):
        """
        Computes the XCOMET-XL score for the corpus.
        Args:
            arr (list): A list of tuples containing source, target, and translation triples.

        Returns:
            float: The XCOMET score.
        """

        batch_size = self.metric_configs['xcomet']['batch_size']
        ck_name = self.metric_configs['xcomet']['checkpoint']
        self.xcometxl = XCOMET(ck_name)
        sources = [i[0] for i in arr]
        targets = [i[1] for i in arr]
        translations = [i[2] for i in arr]
        xcometxl_result = self.xcometxl.evaluate(translations, targets, sources, batch_size )
        self.xcomet_error_spans_list = xcometxl_result["error_spans"]
        self.xcomet_segments_list = xcometxl_result['segments_scores']
        return xcometxl_result['system_score']

    def metricx_corpus(self, arr):
        """
        Computes the MetricX score for the corpus.
        Args:
            arr (list): A list of tuples containing target and translation pairs.

        Returns:
            float: The METRICX score [0, 25] where lower is better.
        """
        ck_name = self.metric_configs['metricx']['checkpoint']
        tk_name = self.metric_configs['metricx']['tokenizer']

        self.metricx = RefMetricX( tk_name, ck_name)

        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]

        metricx_result = self.metricx.evaluate(sources = [], hypotheses = translations, references = targets)
        self.metricx_segments_list = metricx_result['segments_scores']
        return metricx_result["system_score"]

    def metricx_qe_corpus(self, arr):
        """
        Computes the MetricX-QE score for the corpus.
        Args:
            arr (list): A list of tuples containing source and translation tuples.

        Returns:
            float: The MetricX-QE score.
        """
        ck_name = self.metric_configs['metricx_qe']['checkpoint']
        tk_name = self.metric_configs['metricx_qe']['tokenizer']

        self.metricx_qe = QEMetricX( tk_name, ck_name)

        sources = [i[0] for i in arr]
        translations = [i[1] for i in arr]

        metricxqe_result = self.metricx_qe.evaluate(sources = sources, hypotheses = translations, references = [])
        self.metricxqe_segments_list = metricxqe_result['segments_scores']
        return metricxqe_result["system_score"]

    ############## SEGMENTS ##############
    def bleu_segments(self, arr):
        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        segment_scores = []
        for h, r in zip(translations, targets):
            segment_score = sacrebleu.corpus_bleu([h], [[r]])
            segment_scores.append(segment_score.score)
        return segment_scores

    def ter_segments(self, arr):
        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        segment_scores = []
        for h, r in zip(translations, targets):
            segment_score = sacrebleu.corpus_ter([h], [[r]])
            segment_scores.append(segment_score.score)
        return segment_scores
    
    def chrf_segments(self, arr):
        targets = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        segment_scores = []
        for h, r in zip(translations, targets):
            segment_score = sacrebleu.corpus_chrf([h], [[r]])
            segment_scores.append(segment_score.score)
        return segment_scores

    def comet_segments(self, aux=None):
        return self.comet_segments_list
    
    def comet_kiwi_segments(self, aux=None):
        return self.comet_kiwi_segments_list

    def xcomet_segments(self, aux=None):
        return self.xcomet_segments_list
    
    def xcomet_error_spans(self, aux=None):
        return self.xcomet_error_spans_list

    def metricx_segments(self, aux=None):
        return self.metricx_segments_list
    
    def metricx_qe_segments(self, aux=None):
        return self.metricxqe_segments_list

    def get_translations(self, arr):
        translations = [i for i in arr]
        return translations
    
    def get_targets(self, arr):
        targets = [i for i in arr]
        return targets
    
    def get_sources(self, arr):
        sources = [i for i in arr]
        return sources

    def load_yaml_config(self):
        YAML_PATH = './lm_eval/extra_metrics/mt_metrics_config.yaml'
        
        with open(YAML_PATH, 'r') as file:
            config = yaml.safe_load(file)

        mt_metrics = config.get('mt_metrics', {})

        metric_configs = {}
        for metric_name, metric_info in mt_metrics.items():
            metric_configs[metric_name] = metric_info

        self.metric_configs = metric_configs

    def create_dicts(self, source, target, result):

        res, dict_aggregated = {}, {}

        if self.metric_configs['bleu']['compute']: 
            res["bleu"] = (target, result)
            res["bleu_segments"] = (target, result)

            dict_aggregated["bleu"] = self.bleu_corpus
            dict_aggregated["bleu_segments"] = self.bleu_segments

        if self.metric_configs['ter']['compute']: 
            res["ter"] = (target, result)
            res["ter_segments"] = (target, result)

            dict_aggregated["ter"] = self.ter_corpus
            dict_aggregated["ter_segments"] = self.ter_segments

        if self.metric_configs['chrf']['compute']: 
            res["chrf"] = (target, result)
            res["chrf_segments"] = (target, result)

            dict_aggregated["chrf"] = self.chrf_corpus
            dict_aggregated["chrf_segments"] = self.chrf_segments

        if self.metric_configs['comet']['compute']: 
            res["comet"] = (source, target, result)
            res["comet_segments"] = (None)

            dict_aggregated["comet"] = self.comet_corpus
            dict_aggregated["comet_segments"] = self.comet_segments

                
        if self.metric_configs['comet_kiwi']['compute']:
            res["comet_kiwi"] = (source, result)
            res["comet_kiwi_segments"] = (None)

            dict_aggregated["comet_kiwi"] = self.comet_kiwi_corpus
            dict_aggregated["comet_kiwi_segments"] = self.comet_kiwi_segments

        if self.metric_configs['bleurt']['compute']:
            res["bleurt"] = (target, result)

            dict_aggregated["bleurt"] = self.bleurt_corpus
        
        if self.metric_configs['xcomet']['compute']:
            res["xcomet"] = (source, target, result)
            res["xcomet_segments"] = (None)
            res["xcomet_error_spans"] = (None)

            dict_aggregated["xcomet"] = self.xcomet_corpus
            dict_aggregated["xcomet_segments"] = self.xcomet_segments
            dict_aggregated["xcomet_error_spans"] = self.xcomet_error_spans

        if self.metric_configs['metricx']['compute']:
            res["metricx"] = (target, result)
            res["metricx_segments"] = (None)

            dict_aggregated["metricx"] = self.metricx_corpus
            dict_aggregated["metricx_segments"] = self.metricx_segments

        if self.metric_configs['metricx_qe']['compute']:
            res["metricx_qe"] = (source, result)
            res["metricx_qe_segments"] = (None)

            dict_aggregated["metricx_qe"] = self.metricx_qe_corpus
            dict_aggregated["metricx_qe_segments"] = self.metricx_qe_segments

        res["sources"] = (source)
        res["targets"] = (target)
        res["translations"] = (result)

        dict_aggregated["translations"] = self.get_translations
        dict_aggregated["targets"] = self.get_targets
        dict_aggregated["sources"] = self.get_sources


        self.res = res
        self.dict_aggregated = dict_aggregated

    def process_results(self, doc, results):

        # load yaml config
        if self.metric_configs is None:
            self.load_yaml_config()

        source = self.doc_to_text(doc)
        target = self.doc_to_target(doc)
        result = results[0]

        self.create_dicts(source, target, result)

        return self.res

    def aggregation(self):
        """
        Returns a dictionary of aggregation functions for metrics.
        Returns:
            dict: A dictionary where keys are metric names and values are functions that aggregate metric scores.
        """ 
        return self.dict_aggregated

    def higher_is_better(self):
        """
        Indicates whether higher values for each metric are better.
        Returns:
            dict: A dictionary where keys are metric names and values are booleans indicating if higher values are better.
        """
        return {k: True for k in METRICS_MT}
    
    def get_target(self):
        return None