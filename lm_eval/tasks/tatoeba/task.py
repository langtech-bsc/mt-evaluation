from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class _MTask(MTask):
    VERSION = 1
    DATASET_PATH = "Helsinki-NLP/tatoeba_mt"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None):
        super().__init__( config={'target_delimiter': '', 'validation_split': 'test', 'dataset_kwargs': {'language_pair':self.get_langpair()} } )

    def doc_to_text(self, doc):
        return doc["sourceString"]
    
    def doc_to_target(self, doc):
        return doc["targetString"]

_LanguagePairs = [ "afr-deu", "afr-eng", "afr-epo", "afr-nld", "afr-rus", "afr-spa", "ain-fin", "ara-ber", "ara-ber_Latn", "ara-deu", "ara-ell", "ara-eng", "ara-epo", "ara-fra", "ara-heb", "ara-ita", "ara-jpn", "ara-jpn_Hira", "ara-pol", "ara-rus", "ara-spa", "ara-tur", "arq-eng", "avk-fra", "avk-spa", "awa-eng", "aze-eng", "aze-spa", "aze-tur", "aze_Latn-tur", "bel-deu", "bel-eng", "bel-epo", "bel-fra", "bel-ita", "bel-lat", "bel-nld", "bel-pol", "bel-rus", "bel-spa", "bel-ukr", "bel-zho", "ben-eng", "ber-deu", "ber-eng", "ber-epo", "ber-fra", "ber-spa", "ber_Latn-deu", "ber_Latn-eng", "ber_Latn-epo", "ber_Latn-fra", "bre-eng", "bre-fra", "bua-rus", "bua_Cyrl-rus", "bul-bul", "bul-cmn_Hans", "bul-deu", "bul-eng", "bul-epo", "bul-fra", "bul-ita", "bul-jpn", "bul-jpn_Hira", "bul-rus", "bul-spa", "bul-tur", "bul-ukr", "bul-zho", "cat-deu", "cat-eng", "cat-epo", "cat-fra", "cat-ita", "cat-nld", "cat-por", "cat-spa", "cat-ukr", "cbk-eng", "ceb-deu", "ceb-eng", "ces-deu", "ces-eng", "ces-epo", "ces-fra", "ces-hun", "ces-ita", "ces-lat", "ces-pol", "ces-rus", "ces-slv", "ces-spa", "ces-ukr", "cha-eng", "chm-rus", "chv-eng", "chv-rus", "chv-tur", "cmn_Hans-wuu", "cor-deu", "cor-eng", "cor-epo", "cor-fra", "cor-ita", "cor-rus", "cor-spa", "crh-tur", "cym-eng", "dan-dan", "dan-deu", "dan-eng", "dan-epo", "dan-fin", "dan-fra", "dan-ita", "dan-jpn", "dan-jpn_Hira", "dan-nld", "dan-nob", "dan-nor", "dan-por", "dan-rus", "dan-spa", "dan-swe", "dan-tur", "deu-cmn_Hans", "deu-cmn_Hant", "deu-deu", "deu-dsb", "deu-ell", "deu-eng", "deu-epo", "deu-est", "deu-eus", "deu-fas", "deu-fin", "deu-fra", "deu-frr", "deu-gos", "deu-hbs", "deu-heb", "deu-hrv", "deu-hrx", "deu-hsb", "deu-hun", "deu-ido", "deu-ile", "deu-ina", "deu-ind", "deu-isl", "deu-ita", "deu-jbo", "deu-jpn", "deu-jpn_Hani", "deu-jpn_Hira", "deu-jpn_Kana", "deu-kab", "deu-kor", "deu-kor_Hang", "deu-kur", "deu-kur_Latn", "deu-lad", "deu-lat", "deu-lfn", "deu-lfn_Latn", "deu-lit", "deu-ltz", "deu-msa", "deu-nds", "deu-nld", "deu-nob", "deu-nor", "deu-pol", "deu-por", "deu-ron", "deu-run", "deu-rus", "deu-slv", "deu-spa", "deu-srp_Latn", "deu-swe", "deu-swg", "deu-tat", "deu-tgl", "deu-tlh", "deu-toki", "deu-tur", "deu-ukr", "deu-vie", "deu-vol", "deu-yid", "deu-zho", "dsb-hsb", "dsb-slv", "dtp-eng", "dtp-jpn", "dtp-jpn_Hira", "dtp-msa", "dtp-zsm_Latn", "egl-ita", "ell-ell", "ell-eng", "ell-epo", "ell-fra", "ell-ita", "ell-nld", "ell-por", "ell-rus", "ell-spa", "ell-swe", "ell-tur", "eng-bos_Latn", "eng-cmn_Hans", "eng-cmn_Hant", "eng-eng", "eng-epo", "eng-est", "eng-eus", "eng-fao", "eng-fas", "eng-fin", "eng-fra", "eng-fry", "eng-gla", "eng-gle", "eng-glg", "eng-gos", "eng-got", "eng-grc", "eng-gsw", "eng-hbs", "eng-heb", "eng-hin", "eng-hoc", "eng-hoc_Latn", "eng-hrv", "eng-hrx", "eng-hun", "eng-hye", "eng-ido", "eng-ido_Latn", "eng-ile", "eng-ilo", "eng-ina", "eng-ind", "eng-isl", "eng-ita", "eng-jav", "eng-jbo", "eng-jbo_Latn", "eng-jpn", "eng-jpn_Hani", "eng-jpn_Hira", "eng-jpn_Kana", "eng-kab", "eng-kat", "eng-kaz", "eng-kaz_Cyrl", "eng-kha", "eng-khm", "eng-kor", "eng-kor_Hang", "eng-kur", "eng-kur_Latn", "eng-kzj", "eng-lad", "eng-lad_Latn", "eng-lat", "eng-lav", "eng-lfn", "eng-lfn_Cyrl", "eng-lfn_Latn", "eng-lit", "eng-ltz", "eng-mal", "eng-mar", "eng-mkd", "eng-mlt", "eng-mon", "eng-mri", "eng-msa", "eng-mya", "eng-nds", "eng-nld", "eng-nno", "eng-nob", "eng-nor", "eng-nov", "eng-nst", "eng-oci", "eng-orv", "eng-ota", "eng-ota_Arab", "eng-ota_Latn", "eng-pam", "eng-pes", "eng-pms", "eng-pol", "eng-por", "eng-prg", "eng-que", "eng-rom", "eng-ron", "eng-run", "eng-rus", "eng-slv", "eng-spa", "eng-sqi", "eng-srp_Cyrl", "eng-srp_Latn", "eng-swa", "eng-swe", "eng-tam", "eng-tat", "eng-tel", "eng-tgl", "eng-tha", "eng-tlh", "eng-toki", "eng-tuk", "eng-tuk_Latn", "eng-tur", "eng-tzl", "eng-tzl_Latn", "eng-uig", "eng-uig_Arab", "eng-ukr", "eng-urd", "eng-uzb", "eng-uzb_Latn", "eng-vie", "eng-vol", "eng-war", "eng-xal", "eng-yid", "eng-yue_Hans", "eng-yue_Hant", "eng-zho", "eng-zsm_Latn", "eng-zza", "epo-cmn_Hans", "epo-cmn_Hant", "epo-epo", "epo-fas", "epo-fin", "epo-fra", "epo-glg", "epo-hbs", "epo-heb", "epo-hrv", "epo-hun", "epo-ido", "epo-ile", "epo-ile_Latn", "epo-ina", "epo-isl", "epo-ita", "epo-jbo", "epo-jpn", "epo-jpn_Hani", "epo-jpn_Hira", "epo-lad", "epo-lad_Latn", "epo-lat", "epo-lfn", "epo-lfn_Latn", "epo-lit", "epo-nds", "epo-nld", "epo-nob", "epo-nor", "epo-oci", "epo-pol", "epo-por", "epo-ron", "epo-rus", "epo-slv", "epo-spa", "epo-srp_Cyrl", "epo-srp_Latn", "epo-swe", "epo-tgl", "epo-tlh", "epo-toki", "epo-tur", "epo-ukr", "epo-vie", "epo-vol", "epo-yid", "epo-zho", "est-rus", "eus-jpn", "eus-rus", "eus-spa", "fas-fra", "fin-fin", "fin-fkv", "fin-fra", "fin-heb", "fin-hun", "fin-ita", "fin-jpn", "fin-jpn_Hani", "fin-jpn_Hira", "fin-jpn_Kana", "fin-kor", "fin-kor_Hang", "fin-kur", "fin-lat", "fin-nld", "fin-nno", "fin-nob", "fin-nor", "fin-pol", "fin-por", "fin-rus", "fin-spa", "fin-swe", "fin-tur", "fin-zho", "fra-cmn_Hans", "fra-cmn_Hant", "fra-fra", "fra-gcf", "fra-hbs", "fra-heb", "fra-hrv", "fra-hun", "fra-ido", "fra-ile", "fra-ina", "fra-ind", "fra-ita", "fra-jbo", "fra-jpn", "fra-jpn_Hani", "fra-jpn_Hira", "fra-kab", "fra-kor", "fra-kor_Hang", "fra-lat", "fra-lfn", "fra-lfn_Latn", "fra-msa", "fra-nds", "fra-nld", "fra-nob", "fra-nor", "fra-oci", "fra-pcd", "fra-pol", "fra-por", "fra-ron", "fra-run", "fra-rus", "fra-slv", "fra-spa", "fra-swe", "fra-tat", "fra-tgl", "fra-tlh", "fra-tlh_Latn", "fra-toki", "fra-toki_Latn", "fra-tur", "fra-uig", "fra-uig_Arab", "fra-ukr", "fra-vie", "fra-wuu", "fra-yid", "fra-zho", "fry-nld", "gcf-gcf", "gla-spa", "glg-por", "glg-spa", "gos-nld", "grn-por", "grn-spa", "hbs-ita", "hbs-jpn", "hbs-nor", "hbs-pol", "hbs-rus", "hbs-spa", "hbs-ukr", "hbs-zho", "heb-cmn_Hans", "heb-cmn_Hant", "heb-heb", "heb-hun", "heb-ina", "heb-ita", "heb-jpn", "heb-jpn_Hira", "heb-lad", "heb-lat", "heb-lfn", "heb-lfn_Latn", "heb-nld", "heb-pol", "heb-por", "heb-rus", "heb-spa", "heb-tur", "heb-ukr", "heb-yid", "heb-zho", "hin-urd", "hin-zho", "hrv-jpn_Hira", "hrv-pol", "hrv-spa", "hrv-ukr", "hsb-slv", "hun-cmn_Hans", "hun-hun", "hun-ita", "hun-jpn", "hun-jpn_Hani", "hun-jpn_Hira", "hun-kor", "hun-kor_Hang", "hun-lat", "hun-nld", "hun-pol", "hun-por", "hun-rus", "hun-spa", "hun-swe", "hun-tur", "hun-ukr", "hun-zho", "hye-rus", "ido-ina", "ido-ita", "ido-lfn", "ido-spa", "ido-yid", "ido_Latn-lfn_Latn", "ina-ita", "ina-lad", "ina-lat", "ina-lfn", "ina-nld", "ina-por", "ina-rus", "ina-spa", "ina-tlh", "ina-tur", "ina-yid", "ina_Latn-lad_Latn", "ina_Latn-lfn_Latn", "ina_Latn-tlh_Latn", "ind-zsm_Latn", "isl-ita", "isl-jpn", "isl-jpn_Hira", "isl-spa", "ita-cmn_Hans", "ita-cmn_Hant", "ita-ind", "ita-ita", "ita-jpn", "ita-jpn_Hani", "ita-jpn_Hira", "ita-lat", "ita-lit", "ita-msa", "ita-nds", "ita-nld", "ita-nor", "ita-pms", "ita-pol", "ita-por", "ita-ron", "ita-rus", "ita-spa", "ita-swe", "ita-toki", "ita-tur", "ita-ukr", "ita-vie", "ita-yid", "ita-zho", "jbo-jpn", "jbo-rus", "jbo-spa", "jbo-swe", "jbo-zho", "jbo_Latn-cmn_Hans", "jbo_Latn-cmn_Hant", "jbo_Latn-jpn_Hira", "jpn-jpn", "jpn-kor", "jpn-lit", "jpn-mar", "jpn-msa", "jpn-nds", "jpn-nld", "jpn-nor", "jpn-pol", "jpn-por", "jpn-rus", "jpn-spa", "jpn-swe", "jpn-tlh", "jpn-toki", "jpn-tur", "jpn-ukr", "jpn-vie", "jpn-zho", "jpn_Hani-cmn_Hans", "jpn_Hani-nld", "jpn_Hani-pol", "jpn_Hani-por", "jpn_Hani-rus", "jpn_Hani-spa", "jpn_Hira-cmn_Hans", "jpn_Hira-cmn_Hant", "jpn_Hira-ind", "jpn_Hira-jpn_Hira", "jpn_Hira-kor_Hang", "jpn_Hira-lit", "jpn_Hira-mar", "jpn_Hira-nds", "jpn_Hira-nld", "jpn_Hira-nob", "jpn_Hira-pol", "jpn_Hira-por", "jpn_Hira-rus", "jpn_Hira-spa", "jpn_Hira-swe", "jpn_Hira-tlh_Latn", "jpn_Hira-tur", "jpn_Hira-ukr", "jpn_Hira-vie", "jpn_Kana-rus", "jpn_Kana-spa", "kab-kab", "kab-rus", "kab-spa", "kat-rus", "kaz-rus", "kaz_Cyrl-rus", "khm-spa", "kor-rus", "kor-spa", "kor-zho", "kor_Hang-cmn_Hans", "kor_Hang-rus", "kor_Hang-spa", "kzj-msa", "kzj_Latn-zsm_Latn", "lad-lat", "lad-lfn", "lad-spa", "lad-yid", "lad_Latn-lfn_Latn", "lad_Latn-spa", "lad_Latn-yid", "lat-lat", "lat-lfn", "lat-nld", "lat-nor", "lat-pol", "lat-por", "lat-rus", "lat-spa", "lat-tlh", "lat-ukr", "lat-yid", "lat_Latn-lfn_Latn", "lat_Latn-por", "lav-rus", "lfn-por", "lfn-rus", "lfn-spa", "lfn-yid", "lfn_Cyrl-por", "lfn_Latn-por", "lfn_Latn-yid", "lit-pol", "lit-rus", "lit-spa", "lit-tur", "ltz-nld", "mkd-spa", "msa-msa", "msa-spa", "msa-zho", "nds-nld", "nds-por", "nds-rus", "nds-spa", "nld-cmn_Hant", "nld-nld", "nld-nor", "nld-pol", "nld-por", "nld-ron", "nld-rus", "nld-spa", "nld-toki", "nld-tur", "nld-ukr", "nld-zho", "nob-nno", "nob-rus", "nob-spa", "nob-swe", "nor-nor", "nor-pol", "nor-por", "nor-rus", "nor-spa", "nor-swe", "nor-ukr", "nor-zho", "orv-ukr", "ota-tur", "pol-cmn_Hans", "pol-cmn_Hant", "pol-por", "pol-rus", "pol-spa", "pol-swe", "pol-tur", "pol-ukr", "pol-zho", "por-cmn_Hans", "por-cmn_Hant", "por-por", "por-ron", "por-rus", "por-spa", "por-swe", "por-tgl", "por-toki", "por-tur", "por-ukr", "por-zho", "ron-rus", "ron-spa", "ron-tur", "run-rus", "run-spa", "rus-cmn_Hans", "rus-cmn_Hant", "rus-rus", "rus-sah", "rus-slv", "rus-spa", "rus-swe", "rus-tat", "rus-tlh", "rus-toki", "rus-toki_Latn", "rus-tur", "rus-uig", "rus-uig_Arab", "rus-ukr", "rus-vie", "rus-xal", "rus-yue_Hans", "rus-zho", "slv-cmn_Hans", "slv-ukr", "slv-zho", "spa-cmn_Hans", "spa-cmn_Hant", "spa-spa", "spa-swe", "spa-tat", "spa-tgl", "spa-tlh", "spa-toki", "spa-tur", "spa-ukr", "spa-vie", "spa-yid", "spa-zho", "srp_Cyrl-rus", "srp_Cyrl-ukr", "srp_Latn-ita", "srp_Latn-nob", "srp_Latn-rus", "srp_Latn-ukr", "swe-cmn_Hans", "swe-cmn_Hant", "swe-swe", "swe-tur", "swe-zho", "tat-tur", "tat-vie", "tlh-yid", "tlh-zho", "tlh_Latn-cmn_Hans", "tlh_Latn-cmn_Hant", "tlh_Latn-yid", "tur-cmn_Hans", "tur-cmn_Hant", "tur-tur", "tur-uig", "tur-ukr", "tur-uzb", "tur-zho", "uig-zho", "uig_Arab-cmn_Hans", "uig_Arab-cmn_Hant", "ukr-cmn_Hans", "ukr-cmn_Hant", "ukr-ukr", "ukr-zho", "vie-cmn_Hans", "vie-vie", "vie-zho", "yid-yid", "zho-zho" ]

task_definitions = []
for lang_pair in _LanguagePairs:
    task_definitions.append( (f'{lang_pair}_tatoeba', lang_pair, lang_pair.split('-')[1]) )


for task_name, pair, target_lang in task_definitions:
    class_name = task_name.upper()

    task_class = type(
        class_name,
        (_MTask,),
        {
            'get_langpair': (lambda self, pair=pair: pair),
            'get_target': (lambda self, target_lang=target_lang: target_lang),
        }
    ) 
    register_task(task_name)(task_class)

# add directions reversed
task_definitions_reversed = []
for lang_pair in _LanguagePairs:
    srclang = lang_pair.split('-')[0]
    tgtlang = lang_pair.split('-')[1]
    lang_pair_reversed = '{}-{}'.format( tgtlang, srclang )

    if srclang!=tgtlang:
        task_definitions_reversed.append( (f'{lang_pair_reversed}_tatoeba', lang_pair, lang_pair.split('-')[0]) )


for task_name, pair, target_lang in task_definitions_reversed:
    class_name = task_name.upper()

    task_class = type(
        class_name,
        (_MTask,),
        {
            'get_langpair': (lambda self, pair=pair: pair),
            'get_target': (lambda self, target_lang=target_lang: target_lang),

            'doc_to_text': (lambda self, doc: doc["targetString"]),
            'doc_to_target': (lambda self, doc: doc["sourceString"])
            
        }
    ) 
    register_task(task_name)(task_class)