from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

import os
import string
import argparse

class GENEVAL_CONTEXTUAL(MTask):
    VERSION = 1
    DATASET_PATH = "geneval_contextual"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def doc_to_text(self, doc):
        return doc['SRC']
    
    def doc_to_target(self, doc):
        return doc['REF']

    def get_row_values(self, doc):
        return  {
                'REF': doc['REF'],
                'WRONG-REF': doc['WRONG-REF']
                }

    def process_results(self, doc, results):

        # load yaml config
        if self.metric_configs is None:
            self.load_yaml_config()

        source = self.doc_to_text(doc)
        target = self.doc_to_target(doc)
        result = results[0]
        row_value = self.get_row_values(doc)

        self.create_dicts(source, target, result)
        self.res['gender_from_context'] = ( result, row_value )
        return self.res

    def aggregation(self):
        """
        Returns a dictionary of aggregation functions for metrics.
        Returns:
            dict: A dictionary where keys are metric names and values are functions that aggregate metric scores.
        """ 

        self.dict_aggregated['gender_from_context'] = self.geneval_scores
        return self.dict_aggregated

    def geneval_scores(self, arr):

        results = [i[0] for i in arr]
        row_values = [i[1] for i in arr]

        self.accuracy_scores = self.accuracy_metric( results, row_values )

        return  self.accuracy_scores 

    def extract_target(self, line):
        parts = line.split("<sep>")
        return parts[1] if len(parts) > 1 else line

    def get_words(self, line):
        STRIP_PUNCT = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        line = line.lower().translate(STRIP_PUNCT).strip()
        return set(line.strip().split())

    def get_trg_correct_incorrect(self, results, right_ref, wrong_ref):

        # get words for each segment
        trg_words, orig_words, ctf_words = self.get_words(results), self.get_words(right_ref), self.get_words(wrong_ref)
        # get unique words in each of the references
        orig_unique = orig_words - ctf_words
        ctf_unique = ctf_words - orig_words
        # now check the words in the target sentence for overlap with incorrect unique words
        trg_correct = trg_words & orig_unique 
        trg_incorrect = trg_words & ctf_unique
        return trg_correct, trg_incorrect 


    def gender_decision(self, results, right_ref, wrong_ref):

        trg_correct, trg_incorrect = self.get_trg_correct_incorrect(results, right_ref, wrong_ref)

        if trg_incorrect:
            decision = 'Incorrect'
        else:
            decision = 'Correct'

        return [decision, trg_correct, trg_incorrect]


    def accuracy_metric(self, results, row_values):

        metric_annot_mapped = []
        count, correct = 0, 0
        for trg_line, references in zip(results, row_values):
            try:
                target = self.extract_target(trg_line)
                [decision, trg_correct, trg_incorrect] = self.gender_decision(target, references['REF'], references['WRONG-REF'])
                metric_annot_mapped.append((decision))
                count +=1
                if decision == "Correct":
                    correct += 1
            except:
                continue
        accuracy = correct / count 

        return accuracy


@register_task("en_ar_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_ar'})

@register_task("en_de_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_de'})

@register_task("en_es_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_es'})

@register_task("en_fr_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_fr'})

@register_task("en_hi_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_hi'})

@register_task("en_it_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_it'})

@register_task("en_nl_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_nl'})

@register_task("en_pt_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_pt'})

@register_task("en_ru_geneval_contextual")
class GENEVAL_CONTEXTUAL_EN_ES(GENEVAL_CONTEXTUAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_ru'})