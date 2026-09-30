# coding=utf-8
"""NTEU dataset"""

import os
import sys
import datasets

from typing import Union, List, Optional


_CITATION = """
"""

_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = ""

_LANGUAGES = ['bg', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']


_URL = 'nteudataset'

_SPLITS = ["test"]

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join("nteu_dataset", f"NTEU_TestSet.{lang}")
        for split in _SPLITS
    } for lang in _LANGUAGES
}

from itertools import permutations

def _pairings(iterable, r=2):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p


class NTEUConfig(datasets.BuilderConfig):
    """BuilderConfig for the NTEU dataset."""
    def __init__(self, lang: str, lang2: str = None, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.lang = lang
        self.lang2 = lang2


class NTEU(datasets.GeneratorBasedBuilder):
    """NTEU dataset."""

    BUILDER_CONFIGS = [
        NTEUConfig(
            name=lang,
            description=f"NTEU: {lang} subset.",
            lang=lang
        )
        for lang in _LANGUAGES
    ] +  [
        NTEUConfig(
            name="all",
            description=f"NTEU: all language pairs",
            lang=None
        )
    ] +  [
        NTEUConfig(
            name=f"{l1}-{l2}",
            description=f"NTEU: {l1}-{l2} aligned subset.",
            lang=l1,
            lang2=l2
        ) for (l1,l2) in _pairings(_LANGUAGES)
    ]

    def _info(self):
        features = {
            "id": datasets.Value("int32"),
            "URL": datasets.Value("string"),
            "domain": datasets.Value("string"),
            "topic": datasets.Value("string"),
            "has_image": datasets.Value("int32"),
            "has_hyperlink": datasets.Value("int32")
        }
        if self.config.name != "all" and "-" not in self.config.name:
            features["sentence"] = datasets.Value("string")
        elif "-" in self.config.name:
            for lang in [self.config.lang, self.config.lang2]:
                features[f"sentence_{lang}"] = datasets.Value("string")
        else:
            for lang in _LANGUAGES:
                features[f"sentence_{lang}"] = datasets.Value("string")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)
        
        def _get_sentence_paths(split):
            sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in _LANGUAGES]
            return sentence_paths
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "sentence_paths": _get_sentence_paths(split)
                }
            ) for split in _SPLITS
        ]

    def _generate_examples(self, sentence_paths: Union[str, List[str]]):
        """Yields examples as (key, example) tuples."""
        
        sentences = {}
        if len(sentence_paths) == len(_LANGUAGES):
            langs = _LANGUAGES
        else:
            langs = [self.config.lang, self.config.lang2]
        N = None
        for path, lang in zip(sentence_paths, langs):
            with open(path, "r") as sent_file:
                sentences[lang] = [l.strip() for l in sent_file.readlines()]
                if N is None:
                    N = len(sentences[lang])
        #with open(metadata_path, "r") as metadata_file:
        #    metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
        for id_ in range(N):
            
            yield id_, {
                **{
                    f"sentence_{lang}": sentences[lang][id_]
                    for lang in langs
                }
            }