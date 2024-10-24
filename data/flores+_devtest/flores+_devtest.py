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

_LANGUAGES = ['ace_Arab', 'ceb_Latn', 'heb_Hebr', 'lij_Latn', 'pan_Guru', 'taq_Tfng', 'ace_Latn', 'ces_Latn', 'hin_Deva', 
 'lim_Latn', 'pap_Latn', 'tat_Cyrl', 'acm_Arab', 'cjk_Latn', 'hne_Deva', 'lin_Latn', 'pbt_Arab', 'tel_Telu', 
 'acq_Arab', 'ckb_Arab', 'hrv_Latn', 'lit_Latn', 'pes_Arab', 'tgk_Cyrl', 'aeb_Arab', 'cmn_Hans', 'hun_Latn', 
 'lmo_Latn', 'plt_Latn', 'tha_Thai', 'afr_Latn', 'cmn_Hant', 'hye_Armn', 'ltg_Latn', 'pol_Latn', 'tir_Ethi', 
 'als_Latn', 'crh_Latn', 'ibo_Latn', 'ltz_Latn', 'por_Latn', 'tpi_Latn', 'amh_Ethi', 'cym_Latn', 'ilo_Latn', 
 'lua_Latn', 'prs_Arab', 'tsn_Latn', 'apc_Arab_nort3139', 'dan_Latn', 'ind_Latn', 'lug_Latn', 'quy_Latn', 
 'tso_Latn', 'apc_Arab_sout3123', 'deu_Latn', 'isl_Latn', 'luo_Latn', 'ron_Latn', 'tuk_Latn', 'arb_Arab', 
 'dik_Latn', 'ita_Latn', 'lus_Latn', 'run_Latn', 'tum_Latn', 'arb_Latn', 'dyu_Latn', 'jav_Latn', 'lvs_Latn', 
 'rus_Cyrl', 'tur_Latn', 'ars_Arab', 'dzo_Tibt', 'jpn_Jpan', 'mag_Deva', 'sag_Latn', 'twi_Latn_akua1239', 
 'ary_Arab', 'ekk_Latn', 'kab_Latn', 'mai_Deva', 'san_Deva', 'twi_Latn_asan1239', 'arz_Arab', 'ell_Grek', 
 'kac_Latn', 'mal_Mlym', 'sat_Olck', 'uig_Arab', 'asm_Beng', 'eng_Latn', 'kam_Latn', 'mar_Deva', 'scn_Latn', 
 'ukr_Cyrl', 'ast_Latn', 'epo_Latn', 'kan_Knda', 'min_Arab', 'shn_Mymr', 'umb_Latn', 'awa_Deva', 'eus_Latn', 
 'kas_Arab', 'min_Latn', 'sin_Sinh', 'urd_Arab', 'ayr_Latn', 'ewe_Latn', 'kas_Deva', 'mkd_Cyrl', 'slk_Latn', 
 'uzn_Latn', 'azb_Arab', 'fao_Latn', 'kat_Geor', 'mlt_Latn', 'slv_Latn', 'vec_Latn', 'azj_Latn', 'fij_Latn', 
 'kaz_Cyrl', 'mni_Beng', 'smo_Latn', 'vie_Latn', 'bak_Cyrl', 'fil_Latn', 'kbp_Latn', 'mos_Latn', 'sna_Latn', 
 'war_Latn', 'bam_Latn', 'fin_Latn', 'kea_Latn', 'mri_Latn', 'snd_Arab', 'wol_Latn', 'ban_Latn', 'fon_Latn', 
 'khk_Cyrl', 'mya_Mymr', 'som_Latn', 'xho_Latn', 'bel_Cyrl', 'fra_Latn', 'khm_Khmr', 'nld_Latn', 'sot_Latn', 
 'ydd_Hebr', 'bem_Latn', 'fur_Latn', 'kik_Latn', 'nno_Latn', 'spa_Latn', 'yor_Latn', 'ben_Beng', 'fuv_Latn', 
 'kin_Latn', 'nob_Latn', 'srd_Latn', 'yue_Hant', 'bho_Deva', 'gaz_Latn', 'kir_Cyrl', 'npi_Deva', 'srp_Cyrl', 
 'zgh_Tfng', 'bjn_Arab', 'gla_Latn', 'kmb_Latn', 'nqo_Nkoo', 'ssw_Latn', 'zsm_Latn', 'bjn_Latn', 'gle_Latn', 
 'kmr_Latn', 'nso_Latn', 'sun_Latn', 'zul_Latn', 'bod_Tibt', 'glg_Latn', 'knc_Arab', 'nus_Latn', 'swe_Latn', 
 'bos_Latn', 'gug_Latn', 'knc_Latn', 'nya_Latn', 'swh_Latn', 'bug_Latn', 'guj_Gujr', 'kor_Hang', 'oci_Latn', 
 'szl_Latn', 'bul_Cyrl', 'hat_Latn', 'ktu_Latn', 'ory_Orya', 'tam_Taml', 'cat_Latn', 'hau_Latn', 'lao_Laoo', 
 'pag_Latn', 'taq_Latn', 'arg_Latn', 'arn_Latn']


_URL = "flores+_dataset_devtest"

_SPLITS = ["devtest"]

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join(split, f"{split}.{lang}")
        for split in _SPLITS
    } for lang in _LANGUAGES
}

_METADATA_PATHS = {
    split: os.path.join("devtest", f"metadata_{split}.tsv")
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