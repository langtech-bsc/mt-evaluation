from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

import os
import string
import argparse

class GENEVAL(MTask):
    VERSION = 1
    DATASET_PATH = "geneval"
    DATASET_NAME = "all"
    OUTPUT_TYPE = "generate_until"

    def doc_to_text(self, doc):
        return doc['SRC']
    
    def doc_to_target(self, doc):
        return doc['REF']

    def get_row_values(self, doc):
        return  {
                'REF': doc['REF'],
                'WRONG-REF': doc['WRONG-REF'], 
                'GENDER': doc['GENDER'],
                'ID': doc['ID']
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
        self.res['geneval_scores'] = ( result, row_value )
        return self.res

    def aggregation(self):
        """
        Returns a dictionary of aggregation functions for metrics.
        Returns:
            dict: A dictionary where keys are metric names and values are functions that aggregate metric scores.
        """ 

        self.dict_aggregated['geneval_scores'] = self.geneval_scores
        return self.dict_aggregated

    def geneval_scores(self, arr):

        results = [i[0] for i in arr]
        row_values = [i[1] for i in arr]

        self.accuracy_scores = self.accuracy_metric( results, row_values )

        return [ self.accuracy_scores ]

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
        masc_correct, masc_total, fem_correct, fem_total = 0,0,0,0
        id_grouped = {}    
        for trg_line, references in zip(results, row_values):
            print(trg_line)
            [decision, trg_correct, trg_incorrect] = self.gender_decision(trg_line, references['REF'], references['WRONG-REF'])
            metric_annot_mapped.append((decision,references['GENDER'],references['ID'])) 
            if references['GENDER'] == "MASC":
                masc_total += 1
                if decision == "Correct":
                    masc_correct += 1
            elif references['GENDER'] == "FEM":
                fem_total += 1
                if decision == "Correct":
                    fem_correct += 1

                # Group by id_num for both_gender accuracy
            if references['ID'] not in id_grouped:
                id_grouped[references['ID']] = []
            id_grouped[references['ID']].append(decision)

        # Calculate the both_gender-based accuracy
        both_gender_correct = sum(
            1 for decisions in id_grouped.values() if all(d == "Correct" for d in decisions)
        )
        both_gender_total = len(id_grouped)

        # Calculate both_gender accuracies
        masc_accuracy = masc_correct / masc_total if masc_total > 0 else 0
        fem_accuracy = fem_correct / fem_total if fem_total > 0 else 0
        both_gender_accuracy = both_gender_correct / both_gender_total if both_gender_total > 0 else 0

        accuracies = {
            "MASC": masc_accuracy,
            "FEM": fem_accuracy,
            "PAIR": both_gender_accuracy,
        }
        return accuracies


@register_task("en_es_geneval")
class GENEVAL_EN_ES(GENEVAL):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_es'})
