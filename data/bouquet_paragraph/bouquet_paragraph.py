# coding=utf-8
"""Bouquet Paragragh-Level Evaluation Benchmark"""

import os
import datasets

from typing import Union, List
from itertools import permutations


_CITATION = """
@inproceedings{andrews-etal-2025-bouquet,
    title = "{BOUQ}u{ET} : dataset, Benchmark and Open initiative for Universal Quality Evaluation in Translation",
    author = "Andrews, Pierre  and
      Artetxe, Mikel  and
      Meglioli, Mariano Coria  and
      Costa-juss{\`a}, Marta R.  and
      Chuang, Joe  and
      Dale, David  and
      Duppenthaler, Mark  and
      Ekberg, Nathanial Paul  and
      Gao, Cynthia  and
      Licht, Daniel Edward  and
      Maillard, Jean  and
      Mourachko, Alexandre  and
      Ropers, Christophe  and
      Saleem, Safiyyah  and
      S{\'a}nchez, Eduardo  and
      Tsiamas, Ioannis  and
      Turkatenko, Arina  and
      Ventayol-Boada, Albert  and
      Yates, Shireen",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1400/",
    doi = "10.18653/v1/2025.emnlp-main.1400",
    pages = "27515--27535",
    ISBN = "979-8-89176-332-6",
}
"""

_DESCRIPTION = """
Bouquet paragraph-level evaluation dataset.
Contains parallel paragraphs in 276 languages.
"""

_HOMEPAGE = "https://huggingface.co/datasets/facebook/bouquet"

_LICENSE = "CC BY-SA 4.0"

_LANGUAGES = [
    'aar_Latn', 'abl_Latn', 'afr_Latn', 'agr_Latn', 'aiq_Arab', 'als_Latn', 'amh_Ethi',
    'ami_Latn', 'ane_Latn', 'apc_Arab', 'arh_Latn', 'arn_Latn', 'arz_Arab', 'arz_Latn',
    'asm_Beng', 'ayr_Latn', 'ayz_Latn', 'azb_Arab', 'azj_Latn', 'azm_Latn', 'azz_Latn',
    'bak_Cyrl', 'bam_Latn', 'bas_Latn', 'bba_Latn', 'bel_Cyrl', 'ben_Beng', 'ben_Latn',
    'bft_Arab', 'bhb_Deva', 'bho_Deva', 'bod_Tibt', 'bos_Latn', 'bre_Latn', 'brh_Arab',
    'brx_Deva', 'bsh_Arab', 'bsk_Arab', 'bul_Cyrl', 'cak_Latn', 'cat_Latn', 'ceb_Latn',
    'ces_Latn', 'che_Cyrl', 'chr_Cher', 'chv_Cyrl', 'cja_Arab', 'cjk_Latn', 'ckb_Arab',
    'ckl_Latn', 'cmn_Hans', 'cmn_Hant', 'crk_Cans', 'crk_Latn', 'cux_Latn', 'cym_Latn',
    'dan_Latn', 'daq_Deva', 'deu_Latn', 'dgo_Deva', 'dik_Latn', 'diq_Latn', 'div_Thaa',
    'djc_Latn', 'dje_Latn', 'dtm_Latn', 'dts_Latn', 'dua_Latn', 'dzo_Tibt', 'ekk_Latn',
    'ell_Grek', 'enb_Latn', 'eng_Latn', 'enl_Latn', 'eto_Latn', 'eus_Latn', 'ewo_Latn',
    'fao_Latn', 'fia_Copt', 'fin_Latn', 'fra_Latn', 'fry_Latn', 'fuc_Latn', 'fuv_Latn',
    'fvr_Latn', 'gax_Latn', 'gaz_Latn', 'gil_Latn', 'gkp_Latn', 'gla_Latn', 'gle_Latn',
    'glg_Latn', 'gom_Deva', 'guc_Latn', 'gug_Latn', 'guj_Gujr', 'guz_Latn', 'gxx_Latn',
    'hat_Latn', 'hau_Latn', 'heb_Hebr', 'heh_Latn', 'hin_Deva', 'hin_Latn', 'hne_Deva',
    'hrv_Latn', 'hun_Latn', 'hve_Latn', 'hye_Armn', 'ibo_Latn', 'ijc_Latn', 'ilo_Latn',
    'ind_Latn', 'irk_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jmc_Latn', 'jnj_Latn',
    'jpn_Jpan', 'kaa_Cyrl', 'kac_Latn', 'kai_Latn', 'kal_Latn', 'kam_Latn', 'kan_Knda',
    'kat_Geor', 'kaz_Cyrl', 'kea_Latn', 'kek_Latn', 'khk_Cyrl', 'khm_Khmr', 'khq_Latn',
    'khw_Arab', 'kin_Latn', 'kir_Cyrl', 'kls_Arab', 'kmb_Latn', 'kmr_Latn', 'knc_Arab',
    'knw_Latn', 'kor_Kore', 'krt_Latn', 'kru_Deva', 'ksf_Latn', 'ktu_Latn', 'kuj_Latn',
    'kwy_Latn', 'kxp_Arab', 'lao_Laoo', 'led_Latn', 'lgg_Latn', 'lij_Latn', 'lim_Latn',
    'lin_Latn', 'lir_Latn', 'lit_Latn', 'loa_Latn', 'loh_Latn', 'lug_Latn', 'luo_Latn',
    'lvs_Latn', 'maf_Latn', 'mai_Deva', 'mal_Mlym', 'mam_Latn', 'mar_Deva', 'mas_Latn',
    'mey_Latn', 'mie_Latn', 'min_Arab', 'miq_Latn', 'mkd_Cyrl', 'mlt_Latn', 'mos_Latn',
    'mri_Latn', 'mtq_Latn', 'mya_Mymr', 'mzl_Latn', 'naq_Latn', 'nhe_Latn', 'nld_Latn',
    'nlv_Latn', 'nno_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'ory_Orya',
    'pan_Guru', 'pbs_Latn', 'pbt_Arab', 'pcm_Latn', 'pes_Arab', 'plt_Latn', 'pol_Latn',
    'por_Latn_braz1246', 'quc_Latn', 'quh_Latn', 'quz_Latn', 'rob_Latn', 'roh_Latn',
    'ron_Latn', 'rus_Cyrl', 'sat_Olck', 'sba_Latn', 'scn_Latn', 'sgc_Latn', 'shn_Mymr',
    'sif_Latn', 'sin_Sinh', 'skr_Arab', 'slk_Latn', 'slv_Latn', 'sme_Latn', 'sna_Latn',
    'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'sro_Latn', 'srp_Cyrl', 'ssw_Latn',
    'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Latn', 'tam_Taml', 'taq_Latn',
    'taq_Tfng', 'tat_Cyrl', 'tda_Latn', 'tel_Latn', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn',
    'tha_Thai', 'tir_Ethi', 'toc_Latn', 'tpi_Latn', 'tpl_Latn', 'tsg_Latn', 'tsn_Latn',
    'tso_Latn', 'tsz_Latn', 'tui_Latn', 'tur_Latn', 'twi_Latn', 'tzh_Latn', 'tzm_Tfng',
    'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'urd_Latn', 'uzn_Latn', 'ven_Latn',
    'vie_Latn', 'vmw_Latn', 'war_Latn', 'wlv_Latn', 'wol_Latn', 'wuu_Hans', 'xho_Latn',
    'xuu_Latn', 'ydd_Hebr', 'ydg_Arab', 'yor_Latn', 'yua_Latn', 'yue_Hant', 'zai_Latn',
    'zne_Latn', 'zsm_Latn', 'zul_Latn',
]


def _pairings(iterable, r=2):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p

_SPLITS = ["test"]

_URL = "bouquet_paragraph"

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join("bouquet_paragraph_dataset", f"{lang}.txt")
        for split in _SPLITS
    } for lang in _LANGUAGES
}


class BouquetSentConfig(datasets.BuilderConfig):
    """BuilderConfig for the Bouquet paragraph-level dataset."""
    def __init__(self, lang: str, lang2: str = None, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.lang = lang
        self.lang2 = lang2


class BouquetSent(datasets.GeneratorBasedBuilder):
    """Bouquet paragraph-level evaluation dataset."""

    BUILDER_CONFIGS = [
        BouquetSentConfig(
            name=lang,
            description=f"Bouquet: {lang} subset.",
            lang=lang
        )
        for lang in _LANGUAGES
    ] + [
        BouquetSentConfig(
            name="all",
            description="Bouquet: all language pairs",
            lang=None
        )
    ] + [
        BouquetSentConfig(
            name=f"{l1}-{l2}",
            description=f"Bouquet: {l1}-{l2} aligned subset.",
            lang=l1,
            lang2=l2
        )
        for (l1, l2) in _pairings(_LANGUAGES)
    ]

    def _info(self):
        features = {}
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

        if self.config.name == "all":
            langs = _LANGUAGES
        elif "-" in self.config.name:
            langs = [self.config.lang, self.config.lang2]
        else:
            langs = [self.config.lang]

        def _get_sentence_paths(split):
            return [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in langs]

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "sentence_paths": _get_sentence_paths(split),
                    "langs": langs,
                }
            ) for split in _SPLITS
        ]

    def _generate_examples(self, sentence_paths: List[str], langs: List[str]):
        """Yields examples as (key, example) tuples."""
        sentences = {}
        N = None
        for path, lang in zip(sentence_paths, langs):
            with open(path, "r") as f:
                # Our extended model takes \n
                sentences[lang] = [line.strip().replace('\\n', '\n') for line in f.readlines()]
                # MADLAD takes \\n
                #sentences[lang] = [line.strip() for line in f.readlines()]
                if N is None:
                    N = len(sentences[lang])
        for id_ in range(N):
            if len(langs) == 1:
                yield id_, {"sentence": sentences[langs[0]][id_]}
            else:
                yield id_, {
                    f"sentence_{lang}": sentences[lang][id_]
                    for lang in langs
                }
