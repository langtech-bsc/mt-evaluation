import numpy as np
from lm_eval.extra_metrics.bleurt.metric import BLEURT
from lm_eval.extra_metrics.comet.metric import BaseCOMET
from lm_eval.extra_metrics.comet_kiwi.metric import COMETKiwi
from lm_eval.extra_metrics.xcomet.metric import XCOMET, XCOMET_QE
from lm_eval.extra_metrics.metricx.metric import RefMetricX, QEMetricX
from lm_eval.api.task import ConfigurableTask
from lm_eval import utils
import sacrebleu
import random
import yaml

from lm_eval.caching.cache import load_from_cache, save_to_cache
from tqdm import tqdm
from lm_eval.prompts.mappings import *

from collections.abc import Callable
import logging

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

METRICS_MT = [  "bleu", "ter", "chrf", "comet", "comet_kiwi", "bleurt", 
                "xcomet", "xcomet_qe", "bleu_segments", "ter_segments", "chrf_segments", "comet_kiwi_segments", "comet_segments", "xcomet_segments", "xcomet_qe_segments", 
                "xcomet_error_spans", "xcomet_qe_error_spans", "metricx", "metricx_segments", "metricx_qe", "metricx_qe_segments", 
                "translations", "targets", "sources"]

eval_logger = logging.getLogger("lm-eval")

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

    def xcomet_qe_corpus(self, arr):
        """
        Computes the XCOMET-XL score for the corpus.
        Args:
            arr (list): A list of tuples containing source, and translation triples.

        Returns:
            float: The XCOMET QE score.
        """

        batch_size = self.metric_configs['xcomet_qe']['batch_size']
        ck_name = self.metric_configs['xcomet_qe']['checkpoint']
        self.xcometxl_qe = XCOMET_QE(ck_name)
        sources = [i[0] for i in arr]
        translations = [i[1] for i in arr]
        xcometxl_qe_result = self.xcometxl_qe.evaluate(translations, [], sources, batch_size )
        self.xcomet_qe_error_spans_list = xcometxl_qe_result["error_spans"]
        self.xcomet_qe_segments_list = xcometxl_qe_result['segments_scores']
        return xcometxl_qe_result['system_score']

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

    def xcomet_qe_segments(self, aux=None):
        return self.xcomet_qe_segments_list

    def xcomet_qe_error_spans(self, aux=None):
        return self.xcomet_qe_error_spans_list

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

        if self.metric_configs['xcomet_qe']['compute']: 
            res["xcomet_qe"] = (source, result)
            res["xcomet_qe_segments"] = (None)
            res["xcomet_qe_error_spans"] = (None)

            dict_aggregated["xcomet_qe"] = self.xcomet_qe_corpus
            dict_aggregated["xcomet_qe_segments"] = self.xcomet_qe_segments
            dict_aggregated["xcomet_qe_error_spans"] = self.xcomet_qe_error_spans

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

    @utils.positional_deprecated
    def fewshot_context(
        self,
        doc: str,
        num_fewshot: int,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        mt_kwargs = None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.
        """
        if mt_kwargs is None:
            return self.doc_to_text(doc)

        rnd = None
        if rnd is None:
            if self.fewshot_rnd is not None:
                rnd = self.fewshot_rnd
            else:
                raise ValueError(
                    "A `random.Random` generator argument must be provided to `rnd`"
                )

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]


            labeled_examples = (
                "\n\n".join(
                    [
                        self.apply_mt_template( self.doc_to_text(doc), mt_kwargs ) + ' ' + self.doc_to_target(doc) for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.apply_mt_template( self.doc_to_text(doc), mt_kwargs )
        return labeled_examples + example

    def load_yaml_prompts(self):
        YAML_PATH = './lm_eval/prompts/mt_prompts.yaml'
        # Load the YAML file
        with open(YAML_PATH, 'r') as file:
            self.config_prompts = yaml.safe_load(file)

    def apply_mt_template(self, ctx, mt_kwargs):
        if mt_kwargs['prompt_style'] not in self.config_prompts['prompt_structures']:
            raise ValueError(f"Invalid prompt_style '{ mt_kwargs['prompt_style'] }' not found in YAML file.")
        
        self.prompt_structure = self.config_prompts['prompt_structures'][ mt_kwargs['prompt_style'] ]
        prompt_template = self.prompt_structure['prompt']

        if self.prompt_structure['language_map']:
            language_map = globals()[self.prompt_structure['mapping_type']]
            src = language_map[ mt_kwargs['src_language'] ]
            tgt = language_map[ mt_kwargs['tgt_language'] ]
        else:
            src = mt_kwargs['src_language']
            tgt = mt_kwargs['tgt_language']

        prompt = prompt_template.format(src=src, tgt=tgt, context=ctx)
        return prompt

    def build_all_requests(
        self,
        *,
        limit: Union[int, None] = None,
        rank: int = 0,
        world_size: int = 1,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        tokenizer_name: str = "",
        mt_kwargs = None
    ) -> None:
        """Build a set of Instances for a task, and store them in task.instances"""

        self.load_yaml_prompts()

        # used with caching
        og_limit = limit

        cache_key = f"requests-{self._config.task}-{self.config.num_fewshot}shot-rank{rank}-world_size{world_size}"
        cache_key += "-chat_template" if apply_chat_template else ""
        cache_key += "-fewshot_as_multiturn" if fewshot_as_multiturn else ""
        cache_key += (
            f"-system_prompt_hash{utils.hash_string(system_instruction)}"
            if system_instruction is not None
            else ""
        )
        cache_key += f"-tokenizer{tokenizer_name}"

        cached_instances = load_from_cache(file_name=cache_key)

        if cache_requests and cached_instances and not rewrite_requests_cache:
            cached_instances = cached_instances[:limit]

            flattened_instances = [
                instance
                for instance_group in cached_instances
                for instance in instance_group
            ]

            self._instances = flattened_instances
            return

        eval_logger.info(f"Building contexts for {self.config.task} on rank {rank}...")

        instances = []

        # process all documents when caching is specified for simplicity
        if (
            cache_requests
            and (not cached_instances or rewrite_requests_cache)
            and limit is not None
        ):
            limit = None

        doc_id_docs = list(
            self.doc_iterator(rank=rank, limit=limit, world_size=world_size)
        )

        num_docs = len(doc_id_docs)

        for doc_id, doc in tqdm(
            doc_id_docs,
            total=num_docs,
        ):

            doc_copy = doc.copy()

            # sample fewshot context #TODO: need to offset doc_id by rank now!
            if mt_kwargs is not None:
                fewshot_ctx = self.fewshot_context(
                    doc_copy,
                    0 if self.config.num_fewshot is None else self.config.num_fewshot,
                    system_instruction,
                    apply_chat_template,
                    fewshot_as_multiturn,
                    chat_template,
                    mt_kwargs
                )
            else:
                raise ValueError(f"You need to specify translation arguments.")

            # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute

            inst = self.construct_requests(
                doc=doc_copy,
                ctx=fewshot_ctx,
                metadata=(self.config["task"], doc_id, self.config.repeats),
            )

            if not isinstance(inst, list):
                inst = [inst]

            instances.append(inst)

        # now flatten, this is to allow slicing to work with pickles
        sliced_instances = instances[:og_limit]

        flattened_instances = [
            instance
            for instance_group in sliced_instances
            for instance in instance_group
        ]

        self._instances = flattened_instances

        if len(self._instances) == 0:
            raise ValueError("task.build_requests() did not find any docs!")

        if cache_requests and (not cached_instances or rewrite_requests_cache):
            save_to_cache(file_name=cache_key, obj=instances)