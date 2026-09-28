# coding=utf-8
"""MultilingualHolisticBias Evaluation Dataset"""

import os
import datasets

_CITATION = """
"""

_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = """
This re-release of the dataset is not associated with the original authors. 
The dataset is released under the CC-BY-SA-4.0 license.
"""

_URL = "multilingual_holistic_bias.tar.gz"

_SPLITS = ['fra_dev', 'hin_dev', 'ind_dev', 'ita_dev', 'por_dev', 'spa_dev', 'vie_dev',
           'fra_train', 'hin_train', 'ind_train', 'ita_train', 'por_train', 'spa_train', 'vie_train',
           'fra_devtest', 'hin_devtest', 'ind_devtest', 'ita_devtest', 'por_devtest', 'spa_devtest', 'vie_devtest']

_CSV_FILES = {
    "fra_dev": os.path.join("mmhb_dataset", "fra", "dev.csv"),
    "hin_dev": os.path.join("mmhb_dataset", "hin", "dev.csv"),
    "ind_dev": os.path.join("mmhb_dataset", "ind", "dev.csv"),
    "ita_dev": os.path.join("mmhb_dataset", "ita", "dev.csv"),
    "por_dev": os.path.join("mmhb_dataset", "por", "dev.csv"),
    "spa_dev": os.path.join("mmhb_dataset", "spa", "dev.csv"),
    "vie_dev": os.path.join("mmhb_dataset", "vie", "dev.csv"),

    "fra_devtest": os.path.join("mmhb_dataset", "fra", "devtest.csv"),
    "hin_devtest": os.path.join("mmhb_dataset", "hin", "devtest.csv"),
    "ind_devtest": os.path.join("mmhb_dataset", "ind", "devtest.csv"),
    "ita_devtest": os.path.join("mmhb_dataset", "ita", "devtest.csv"),
    "por_devtest": os.path.join("mmhb_dataset", "por", "devtest.csv"),
    "spa_devtest": os.path.join("mmhb_dataset", "spa", "devtest.csv"),
    "vie_devtest": os.path.join("mmhb_dataset", "vie", "devtest.csv"),

    "fra_train": os.path.join("mmhb_dataset", "fra", "train.csv"),
    "hin_train": os.path.join("mmhb_dataset", "hin", "train.csv"),
    "ind_train": os.path.join("mmhb_dataset", "ind", "train.csv"),
    "ita_train": os.path.join("mmhb_dataset", "ita", "train.csv"),
    "por_train": os.path.join("mmhb_dataset", "por", "train.csv"),
    "spa_train": os.path.join("mmhb_dataset", "spa", "train.csv"),
    "vie_train": os.path.join("mmhb_dataset", "vie", "train.csv"),
}


class MultilingualHolisticBiasEvaluation(datasets.GeneratorBasedBuilder):
    """MultilingualHolisticBias Evaluation Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features({
            "sentence_eng": datasets.Value("string"),
            "both": datasets.Value("string"),
            "feminine": datasets.Value("string"),
            "masculine": datasets.Value("string"),
            "lang": datasets.Value("string"),
            "gender_group": datasets.Value("string")
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
        dl_dir = dl_manager.extract(_URL)

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
            headers = f.readline().strip().split('\t')
            for idx, line in enumerate(f):
                values = line.strip().split('\t')
                if len(values) != len(headers):
                    continue  # Skip lines with missing values
                data = dict(zip(headers, values))
                selected_cols = ['sentence_eng', 'both', 'feminine', 'masculine', 'lang', 'gender_group']
                data = {k: v for k, v in data.items() if k in selected_cols}
                
                yield idx, data
