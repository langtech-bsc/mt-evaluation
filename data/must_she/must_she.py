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

_URL = "must_she"  # Replace with the actual path or URL to your data

_SPLITS = ["en_ca", "en_es"]

_TSV_FILES = {
    "en_ca": os.path.join("must_she_dataset", "MuST-SHE_en-ca.tsv"),
    "en-es": os.path.join("must_she_dataset", "MuST-SHE_en-es.tsv"),
}


class GenderBiasEvaluation(datasets.GeneratorBasedBuilder):
    """Gender-Bias Evaluation Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features({
            "ID": datasets.Value("string"),
            "LANG": datasets.Value("string"),
            "TALK": datasets.Value("string"),
            "SRC": datasets.Value("string"),
            "REF": datasets.Value("string"),
            "WRONG-REF": datasets.Value("string"),
            "SPEAKER": datasets.Value("string"),
            "GENDER": datasets.Value("string"),
            "CATEGORY": datasets.Value("string"),
            "TEXT-CATEGORY": datasets.Value("string"),
            "GENDERTERMS": datasets.Value("string"),
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