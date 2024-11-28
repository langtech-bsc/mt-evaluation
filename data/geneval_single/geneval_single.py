# coding=utf-8
"""Gender-Bias Evaluation Dataset"""

import os
import datasets

_CITATION = """
"""

_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = ""

_URL = "geneval_single"  # Replace with the actual path or URL to your data

_SPLITS = ["en_ar","en_de", "en_es", "en_fr", "en_hi", "en_it", "en_pt", "en_ru"]

_TSV_FILES = {
    "en_ar": os.path.join("geneval_single_dataset", "geneval_single_en_ar.tsv"),
    "en_de": os.path.join("geneval_single_dataset", "geneval_single_en_de.tsv"),
    "en_es": os.path.join("geneval_single_dataset", "geneval_single_en_es.tsv"),
    "en_fr": os.path.join("geneval_single_dataset", "geneval_single_en_fr.tsv"),
    "en_hi": os.path.join("geneval_single_dataset", "geneval_single_en_hi.tsv"),
    "en_it": os.path.join("geneval_single_dataset", "geneval_single_en_it.tsv"),
    "en_pt": os.path.join("geneval_single_dataset", "geneval_single_en_pt.tsv"),
    "en_ru": os.path.join("geneval_single_dataset", "geneval_single_en_ru.tsv"),
}


class GenderBiasEvaluation(datasets.GeneratorBasedBuilder):
    """Gender-Bias Evaluation Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features({
            "SRC": datasets.Value("string"),
            "REF": datasets.Value("string"),
            "WRONG-REF": datasets.Value("string"),
            "GENDER": datasets.Value("string"),
            "ID": datasets.Value("string")
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        
        dl_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, _TSV_FILES[split]),
                    "split": split,
                },
            )
            for split in _SPLITS
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            headers = f.readline().strip().split('\t')
            for idx, line in enumerate(f):
                values = line.strip().split('\t')
                if len(values) != len(headers):
                    continue  # Skip lines with missing values
                data = dict(zip(headers, values))
                yield idx, data