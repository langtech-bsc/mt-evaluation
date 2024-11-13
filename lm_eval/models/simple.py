import random
from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import os
import shutil
import random

eval_logger = utils.eval_logger

@register_model("simplegenerator")
class SimpleSentenceGenerator(LM):
    """
    A simple class that loads sentences from a file
    and uses them for generating responses.
    """
    def __init__(self, model_name, sentence_file_path, batch_size=1):
        super().__init__()

        path_translation = sentence_file_path

        # Load sentences from the provided file path
        eval_logger.info(f"Using sentences already generated. File path: {path_translation}")
        with open(path_translation, 'r') as file:
            self.sentences = [line.strip() for line in file.readlines()]

    def generate_until(self, requests):
        # For each request, return the next sentence in the list
        return [self.sentences[i % len(self.sentences)] for i in range(len(requests))]

    def loglikelihood(self, requests):
        return None

    def loglikelihood_rolling(self, requests):
        return None