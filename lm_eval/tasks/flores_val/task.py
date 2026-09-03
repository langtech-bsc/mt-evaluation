from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

import os
import string
import argparse

class FLORES_VAL(MTask):
	VERSION = 1
	DATASET_PATH = "flores_val"
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
		self.res['valencian_scores'] = ( result, row_value )
		return self.res

	def aggregation(self):
		"""
		Returns a dictionary of aggregation functions for metrics.
		Returns:
			dict: A dictionary where keys are metric names and values are functions that aggregate metric scores.
		""" 

		self.dict_aggregated['valencian_scores'] = self.valencian_scores
		return self.dict_aggregated

	def valencian_scores(self, arr):

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


	def valencian_decision(self, results, right_ref, wrong_ref):

		trg_correct, trg_incorrect = self.get_trg_correct_incorrect(results, right_ref, wrong_ref)

		if trg_incorrect:
			decision = 'Incorrect'
		elif not trg_correct:
			decision = 'Invalid'
		else:
			decision = 'Correct'

		return [decision, trg_correct, trg_incorrect]


	def accuracy_metric(self, results, row_values):

		metric_annot_mapped = []
		val_correct = 0
		val_invalid = 0
		for trg_line, references in zip(results, row_values):
			[decision, trg_correct, trg_incorrect] = self.valencian_decision(trg_line, references['REF'], references['WRONG-REF'])

			if decision == "Correct":
				val_correct += 1
			elif decision == 'Invalid':
				val_invalid += 1

		# Calculate both_gender accuracies
		val_accuracy = val_correct / (323 - val_invalid)
		accuracies = {
			"ACC": val_accuracy,
			"INVALID": val_invalid
		}
		return accuracies


@register_task("en_vl_flores_val")
class FLORES_VAL_EN_VL(FLORES_VAL):
	def __init__(self, config=None):
		super().__init__(config={'target_delimiter': '', 'validation_split':'en_vl'})

	def get_target(self):
		return 'val_Latn'
