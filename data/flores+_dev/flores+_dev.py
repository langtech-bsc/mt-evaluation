# coding=utf-8
"""Flores+"""

import os
import sys
import datasets

from typing import Union, List, Optional


_CITATION = """
@article{nllb-22,
    title = {No Language Left Behind: Scaling Human-Centered Machine Translation},
    author = {{NLLB Team} and Costa-jussà, Marta R. and Cross, James and Çelebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Mejia-Gonzalez, Gabriel and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzmán, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff},
    year = {2022},
    eprint = {arXiv:1902.01382},
}
"""

_DESCRIPTION = """
This repository hosts the open source FLORES+ machine translation evaluation benchmark, released under 
CC BY-SA 4.0. This dataset was originally released by FAIR researchers at Meta under the name FLORES. 
Further information about these initial releases can be found in the papers below. The data is now being 
managed by OLDI, the Open Language Data Initiative. The + has been added to the name to disambiguate 
between the original datasets and this new actively developed version.
"""

_HOMEPAGE = "https://github.com/openlanguagedata/flores?tab=readme-ov-file"

_LICENSE = "CC BY-SA 4.0."

_LANGUAGES = ['ace_Arab', 'bho_Deva', 'epo_Latn', 'ilo_Latn', 'lao_Laoo', 'nld_Latn', 'shn_Mymr', 'tsn_Latn',
 'ace_Latn', 'bjn_Arab', 'eus_Latn', 'ind_Latn', 'lij_Latn', 'nno_Latn', 'sin_Sinh', 'tso_Latn',
 'acm_Arab', 'bjn_Latn', 'ewe_Latn', 'isl_Latn', 'lim_Latn', 'nob_Latn', 'slk_Latn', 'tuk_Latn',
 'acq_Arab', 'bod_Tibt', 'fao_Latn', 'ita_Latn', 'lin_Latn', 'npi_Deva', 'slv_Latn', 'tum_Latn',
 'aeb_Arab', 'bos_Latn', 'fij_Latn', 'jav_Latn', 'lit_Latn', 'nqo_Nkoo', 'smo_Latn', 'tur_Latn',
 'afr_Latn', 'brx_Deva', 'fil_Latn', 'jpn_Jpan', 'lmo_Latn', 'nso_Latn', 'sna_Latn', 'twi_Latn_akua1239',
 'als_Latn', 'bug_Latn', 'fin_Latn', 'kab_Latn', 'ltg_Latn', 'nus_Latn', 'snd_Arab', 'twi_Latn_asan1239',
 'amh_Ethi', 'bul_Cyrl', 'fon_Latn', 'kac_Latn', 'ltz_Latn', 'nya_Latn', 'snd_Deva', 'uig_Arab',
 'apc_Arab_nort3139', 'cat_Latn', 'fra_Latn', 'kam_Latn', 'lua_Latn', 'oci_Latn', 'som_Latn', 'ukr_Cyrl',
 'apc_Arab_sout3123', 'ceb_Latn', 'fur_Latn', 'kan_Knda', 'lug_Latn', 'ory_Orya', 'sot_Latn', 'umb_Latn',
 'arb_Arab', 'ces_Latn', 'fuv_Latn', 'kas_Arab', 'luo_Latn', 'pag_Latn', 'spa_Latn', 'urd_Arab',
 'arb_Latn', 'chv_Cyrl', 'gaz_Latn', 'kas_Deva', 'lus_Latn', 'pan_Guru', 'srd_Latn', 'uzn_Latn',
 'ars_Arab', 'cjk_Latn', 'gla_Latn', 'kat_Geor', 'lvs_Latn', 'pap_Latn', 'srp_Cyrl', 'vec_Latn',
 'ary_Arab', 'ckb_Arab', 'gle_Latn', 'kaz_Cyrl', 'mag_Deva', 'pbt_Arab', 'ssw_Latn', 'vie_Latn',
 'arz_Arab', 'cmn_Hans', 'glg_Latn', 'kbp_Latn', 'mai_Deva', 'pes_Arab', 'sun_Latn', 'war_Latn',
 'asm_Beng', 'cmn_Hant', 'gom_Deva', 'kea_Latn', 'mal_Mlym', 'plt_Latn', 'swe_Latn', 'wol_Latn',
 'ast_Latn', 'crh_Latn', 'gug_Latn', 'khk_Cyrl', 'mar_Deva', 'pol_Latn', 'swh_Latn', 'xho_Latn',
 'awa_Deva', 'cym_Latn', 'guj_Gujr', 'khm_Khmr', 'mhr_Cyrl', 'por_Latn', 'szl_Latn', 'ydd_Hebr',
 'ayr_Latn', 'dan_Latn', 'hat_Latn', 'kik_Latn', 'min_Arab', 'prs_Arab', 'tam_Taml', 'yor_Latn',
 'azb_Arab', 'deu_Latn', 'hau_Latn', 'kin_Latn', 'min_Latn', 'quy_Latn', 'taq_Latn', 'yue_Hant',
 'azj_Latn', 'dgo_Deva', 'heb_Hebr', 'kir_Cyrl', 'mkd_Cyrl', 'ron_Latn', 'taq_Tfng', 'zgh_Tfng',
 'bak_Cyrl', 'dik_Latn', 'hin_Deva', 'kmb_Latn', 'mlt_Latn', 'run_Latn', 'tat_Cyrl', 'zsm_Latn',
 'bam_Latn', 'dyu_Latn', 'hne_Deva', 'kmr_Latn', 'mni_Beng', 'rus_Cyrl', 'tel_Telu', 'zul_Latn',
 'ban_Latn', 'dzo_Tibt', 'hrv_Latn', 'knc_Arab', 'mni_Mtei', 'sag_Latn', 'tgk_Cyrl',
 'bel_Cyrl', 'ekk_Latn', 'hun_Latn', 'knc_Latn', 'mos_Latn', 'san_Deva', 'tha_Thai',
 'bem_Latn', 'ell_Grek', 'hye_Armn', 'kor_Hang', 'mri_Latn', 'sat_Olck', 'tir_Ethi',
 'ben_Beng', 'eng_Latn', 'ibo_Latn', 'ktu_Latn', 'mya_Mymr', 'scn_Latn', 'tpi_Latn', 'arg_Latn', 'arn_Latn']


_URL = "flores+_dataset_dev"

_SPLITS = ["dev"]

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join(split, f"{split}.{lang}")
        for split in _SPLITS
    } for lang in _LANGUAGES
}

_METADATA_PATHS = {
    split: os.path.join("dev", f"metadata_{split}.tsv")
    for split in _SPLITS
}

from itertools import permutations

def _pairings(iterable, r=2):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p


class Flores200Config(datasets.BuilderConfig):
    """BuilderConfig for the FLORES-200 dataset."""
    def __init__(self, lang: str, lang2: str = None, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.lang = lang
        self.lang2 = lang2


class Flores200(datasets.GeneratorBasedBuilder):
    """FLORES-200 dataset."""

    BUILDER_CONFIGS = [
        Flores200Config(
            name=lang,
            description=f"FLORES-200: {lang} subset.",
            lang=lang
        )
        for lang in _LANGUAGES
    ] +  [
        Flores200Config(
            name="all",
            description=f"FLORES-200: all language pairs",
            lang=None
        )
    ] +  [
        Flores200Config(
            name=f"{l1}-{l2}",
            description=f"FLORES-200: {l1}-{l2} aligned subset.",
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
            features=datasets.Features(features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)

        def _get_sentence_paths(split):
            if isinstance(self.config.lang, str) and isinstance(self.config.lang2, str):
                sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in (self.config.lang, self.config.lang2)]
            elif isinstance(self.config.lang, str):
                sentence_paths = os.path.join(dl_dir, _SENTENCES_PATHS[self.config.lang][split])
            else:
                sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in _LANGUAGES]
            return sentence_paths
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "sentence_paths": _get_sentence_paths(split),
                    "metadata_path": os.path.join(dl_dir, _METADATA_PATHS[split]),
                }
            ) for split in _SPLITS
        ]

    def _generate_examples(self, sentence_paths: Union[str, List[str]], metadata_path: str, langs: Optional[List[str]] = None):
        """Yields examples as (key, example) tuples."""
        if isinstance(sentence_paths, str):
            with open(sentence_paths, "r") as sentences_file:
                with open(metadata_path, "r") as metadata_file:
                    metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
                    for id_, (sentence, metadata) in enumerate(
                        zip(sentences_file, metadata_lines)
                    ):
                        sentence = sentence.strip()
                        metadata = metadata.split("\t")
                        yield id_, {
                            "id": id_ + 1,
                            "sentence": sentence,
                            "URL": metadata[0],
                            "domain": metadata[1],
                            "topic": metadata[2],
                            "has_image": 1 if metadata == "yes" else 0,
                            "has_hyperlink": 1 if metadata == "yes" else 0
                        }
        else:
            sentences = {}
            if len(sentence_paths) == len(_LANGUAGES):
                langs = _LANGUAGES
            else:
                langs = [self.config.lang, self.config.lang2]
            for path, lang in zip(sentence_paths, langs):
                with open(path, "r") as sent_file:
                    sentences[lang] = [l.strip() for l in sent_file.readlines()]
            with open(metadata_path, "r") as metadata_file:
                metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
            for id_, metadata in enumerate(metadata_lines):
                metadata = metadata.split("\t")
                yield id_, {
                    **{
                        "id": id_ + 1,
                        "URL": metadata[0],
                        "domain": metadata[1],
                        "topic": metadata[2],
                        "has_image": 1 if metadata == "yes" else 0,
                        "has_hyperlink": 1 if metadata == "yes" else 0
                    }, **{
                        f"sentence_{lang}": sentences[lang][id_]
                        for lang in langs
                    }
                }