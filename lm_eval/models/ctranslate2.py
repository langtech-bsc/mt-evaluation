import random
from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import ctranslate2
import pyonmttok
import torch
import os
import shutil

eval_logger = utils.eval_logger

class CTranslateMAIN(LM):
    """
    An abstracted Ctranslate model class.
    """

    def __init__(self, model, batch_size=1) -> None:
        super().__init__()
        eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if self._device == "cuda":
            device_count = torch.cuda.device_count()
            device_index = list(range(device_count))
            eval_logger.info(f"Using {device_count} GPUs: {device_index}")
            self._model = ctranslate2.Translator(model,  device=self._device, device_index=device_index)
        else:
            device_index = None
            eval_logger.info("Using CPU")
            self._model = ctranslate2.Translator(model)
        
        self._tokenizer = pyonmttok.Tokenizer(
            mode="none", 
            sp_model_path=f"{model}/spm.model"
        )

    def generate_until(self, requests):
        res = []

        for item in requests:
            ctx = item.args[0]
            generation_params = item.args[1]

            beam_size=generation_params.pop('num_beams', 5)
            length_penalty=generation_params.pop('length_penalty', 1)
            no_repeat_ngram_size=generation_params.pop('no_repeat_ngram_size', 0)
            max_decoding_length=generation_params.pop('max_length', 500)

            tokenized = self._tokenizer.tokenize(ctx)
            translated = self._model.translate_batch([tokenized[0]], beam_size=beam_size, length_penalty=length_penalty, 
                                                                     no_repeat_ngram_size=no_repeat_ngram_size,
                                                                     max_decoding_length=max_decoding_length )
            translated = self._tokenizer.detokenize(translated[0][0]['tokens'])
            res.append(translated)
            
        return res

    def loglikelihood(self, requests):
        return None

    def loglikelihood_rolling(self, requests):
        return None

@register_model("ctranslate")
class CTranslate(CTranslateMAIN):

    def __init__(self, model, batch_size=1) -> None:
        super().__init__(model, batch_size)

@register_model("fairseq")
class Fairseq(CTranslateMAIN):
    """
    An abstracted Fairseq model class, which converts a Fairseq model 
    to CTranslate. This class, inherits from the CTranslate class
    """

    def __init__(self, model_name, model_fairseq, data_dir, spm_path, batch_size=1) -> None:

        PATH_CTRANSLATE_MODELS = './ctranslate_models'
        path_converted_model = os.path.join(PATH_CTRANSLATE_MODELS, model_name)

        if not os.path.exists(path_converted_model):
            os.system(f"ct2-fairseq-converter --model_path {model_fairseq} --data_dir {data_dir} --output_dir {path_converted_model}")

            # Move spm_path to path_converted_model
            spm_dest_path = os.path.join(path_converted_model, 'spm.model')
            shutil.copy(spm_path, spm_dest_path)
            
        else:
            eval_logger.info(f"Model already converted to ctranslate2. Re-using converted model: {path_converted_model}")
        super().__init__(path_converted_model)