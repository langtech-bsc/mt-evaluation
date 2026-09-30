# coding=utf-8
"""HolisticBias Evaluation Dataset"""

import os
import datasets

_CITATION = """
@inproceedings{smith2022m,
  title={“I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset},
  author={Smith, Eric Michael and Hall, Melissa and Kambadur, Melanie and Presani, Eleonora and Williams, Adina},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={9180--9211},
  year={2022}
}
"""

_DESCRIPTION = """
This dataset contains the source data of the Holistic Bias dataset as described by Smith et. al. (2022). 
The dataset contains noun phrases and sentences used to measure the likelihood bias of various models. 
The original dataset is released on GitHub.
"""

_HOMEPAGE = ""

_LICENSE = """
This re-release of the dataset is not associated with the original authors. 
The dataset is released under the CC-BY-SA-4.0 license.
"""

_URL = "holistic_bias"

_SPLITS = [ 'ability', 'age', 'body_type', 'characteristics', 'cultural',
            'gender_and_sex', 'nationality', 'nonce',
            'political_ideologies', 'race_ethnicity', 'religion',
            'sexual_orientation', 'socioeconomic_class', 'others']

_CSV_FILES = {
    "others": os.path.join("holistic_bias_dataset", "others_mutox.csv"),
    "ability": os.path.join("holistic_bias_dataset", "ability_mutox.csv"),
    "age": os.path.join("holistic_bias_dataset", "age_mutox.csv"),
    "body_type": os.path.join("holistic_bias_dataset", "body_type_mutox.csv"),
    "characteristics": os.path.join("holistic_bias_dataset", "characteristics_mutox.csv"),
    "cultural": os.path.join("holistic_bias_dataset", "cultural_mutox.csv"),
    "gender_and_sex": os.path.join("holistic_bias_dataset", "gender_and_sex_mutox.csv"),
    "nationality": os.path.join("holistic_bias_dataset", "nationality_mutox.csv"),
    "nonce": os.path.join("holistic_bias_dataset", "nonce_mutox.csv"),
    "political_ideologies": os.path.join("holistic_bias_dataset", "political_ideologies_mutox.csv"),
    "race_ethnicity": os.path.join("holistic_bias_dataset", "race_ethnicity_mutox.csv"),
    "religion": os.path.join("holistic_bias_dataset", "religion_mutox.csv"),
    "sexual_orientation": os.path.join("holistic_bias_dataset", "sexual_orientation_mutox.csv"),
    "socioeconomic_class": os.path.join("holistic_bias_dataset", "socioeconomic_class_mutox.csv")
}


class HolisticBiasEvaluation(datasets.GeneratorBasedBuilder):
    """HolisticBias Evaluation Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features({
            "text": datasets.Value("string"),
            "axis": datasets.Value("string"),
            "bucket": datasets.Value("string"),
            "descriptor": datasets.Value("string"),
            "descriptor_gender": datasets.Value("string"),
            "descriptor_preference": datasets.Value("string"),
            "noun": datasets.Value("string"),
            "plural_noun": datasets.Value("string"),
            "noun_gender": datasets.Value("string"),
            "noun_phrase": datasets.Value("string"),
            "plural_noun_phrase": datasets.Value("string"),
            "noun_phrase_type": datasets.Value("string"),
            "template": datasets.Value("string"),
            "first_turn_only": datasets.Value("string"),
            "must_be_noun": datasets.Value("string")
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
                    "filepath": os.path.join(dl_dir, _CSV_FILES[split]),
                    "split": split,
                },
            )
            for split in _SPLITS
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            headers = f.readline().strip().split(',')
            for idx, line in enumerate(f):
                values = line.strip().split(',')
                if len(values) != len(headers):
                    continue  # Skip lines with missing values
                data = dict(zip(headers, values))
                yield idx, data