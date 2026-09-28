# coding=utf-8
"""NTREX Evaluation Benchmark """

import os
import sys
import datasets

from typing import Union, List, Optional


_CITATION = """
@inproceedings{federmann-etal-2022-ntrex,
    title = "{NTREX}-128 {--} News Test References for {MT} Evaluation of 128 Languages",
    author = "Federmann, Christian and Kocmi, Tom and Xin, Ying",
    booktitle = "Proceedings of the First Workshop on Scaling Up Multilingual Evaluation",
    month = "nov",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sumeval-1.4",
    pages = "21--24",
}
"""

_DESCRIPTION = """
NTREX -- News Test References for MT Evaluation from English into a total of 128 target languages with document-level information.
"""

_HOMEPAGE = "https://github.com/MicrosoftTranslator/NTREX"

_LICENSE = "CC BY-SA 4.0 license"

_LANGUAGES = ['cat', 'ita', 'fra', 'deu', 'glg', 'eus', 'zho-CN', 'eng', 'por', 'spa', 'bul', 'ces', 'lit', 'hrv', 'nld', 'ron', 'dan', 'ell', 'fin', 'hun', 'slk', 'slv', 'est', 'pol', 'lav', 'swe', 'mlt', 'gle']


#_URL = "https://github.com/MicrosoftTranslator/NTREX"
_URL = 'ntrexdataset'

_SPLITS = ["test"]

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join("ntrex_dataset", f"newstest2019-ref.{lang}.txt")
        for split in _SPLITS
    } for lang in _LANGUAGES
}

_METADATA_PATHS = {
    split: os.path.join("ntrex_dataset", f"LANGUAGES.tsv")
    for split in _SPLITS
}

from itertools import permutations

def _pairings(iterable, r=2):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p


class NTREXConfig(datasets.BuilderConfig):
    """BuilderConfig for the NTREX dataset."""
    def __init__(self, lang: str, lang2: str = None, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.lang = lang
        self.lang2 = lang2


class NTREX(datasets.GeneratorBasedBuilder):
    """NTREX dataset."""

    BUILDER_CONFIGS = [
        NTREXConfig(
            name=lang,
            description=f"NTREX: {lang} subset.",
            lang=lang
        )
        for lang in _LANGUAGES
    ] +  [
        NTREXConfig(
            name="all",
            description=f"NTREX: all language pairs",
            lang=None
        )
    ] +  [
        NTREXConfig(
            name=f"{l1}-{l2}",
            description=f"NTREX: {l1}-{l2} aligned subset.",
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