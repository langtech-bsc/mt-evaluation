dataset_paths = {
        "nteu":"data/nteu/nteu.py",
        "ntrex":"data/ntrex/ntrex.py",
        "facebook/flores": "data/flores/flores.py",
        "holistic_bias": "data/holistic_bias/holistic_bias.py",
        "must_she": "data/must_she/must_she.py",
        "flores+_dev":"data/flores+_dev/flores+_dev.py",
        "flores+_devtest":"data/flores+_devtest/flores+_devtest.py",
        "multilingual_holistic_bias": "data/multilingual_holistic_bias/multilingual_holistic_bias.py",
        "flores+_devtest_perturbations":"data/flores+_devtest_perturbations/flores+_devtest_perturbations.py",
        "geneval_single":"data/geneval_single/geneval_single.py",
        "geneval_contextual":"data/geneval_contextual/geneval_contextual.py",
        "geneval_contextual_trailing":"data/geneval_contextual_trailing/geneval_contextual_trailing.py",
		"european_comission":"data/european_comission/european_comission.py",
		"acpd_eval":"data/acpd_eval/acpd_eval.py",
		"tacon":"data/tacon/tacon.py",
		"cybersecurity":"data/cybersecurity/cybersecurity.py",
		"aapp_ca-es":"data/aapp_ca-es/aapp_ca-es.py",
		"aapp_ca-esn":"data/aapp_ca-en/aapp_ca-en.py",
		"wmt19_biomed":"data/wmt19_biomed/wmt19_biomed.py",
		"un_eval":"data/un_eval/un_eval.py",
		"eupress":"data/eupress/eupress.py",
		"wmt24pp":"data/wmt24pp/wmt24pp.py",
		"abstracts":"data/abstracts/abstracts.py",
		"act_single":"data/act_single/act_single.py",
		"act_dev":"data/act_dev/act_dev.py",
		"localization_xml_mt":"data/localization_xml_mt/localization_xml_mt.py",
		"flores_val":"data/flores_val/flores_val.py",
		"alia_eval":"data/alia_eval/alia_eval.py",
  		"commonvoice_valencian":"data/commonvoice_valencian/commonvoice_valencian.py",
		"acpd_eval":"data/acpd_eval/acpd_eval.py",
		"tacon":"data/tacon/tacon.py",
		"cybersecurity":"data/cybersecurity/cybersecurity.py",
		"aapp_ca-es":"data/aapp_ca-es/aapp_ca-es.py",
		"aapp_ca-en":"data/aapp_ca-en/aapp_ca-en.py",
		"wmt19_biomed":"data/wmt19_biomed/wmt19_biomed.py",
		"un_eval":"data/un_eval/un_eval.py",
		"eupress":"data/eupress/eupress.py",
		"bouquet_sent":"data/bouquet_sent/bouquet_sent.py",
		"bouquet_paragraph":"data/bouquet_paragraph/bouquet_paragraph.py",
		"polymath_question":"data/polymath_question/polymath_question.py",
		"polymath_question_jsonl":"data/polymath_question_jsonl/polymath_question_jsonl.py",
		"tier1_holdout":"data/tier1_holdout/tier1_holdout.py",
		"tier1_holdout_sent":"data/tier1_holdout_sent/tier1_holdout_sent.py",

  }
def resolve_dataset_path(dataset_path):
    """Redirect known MT dataset names to their local loader scripts.

    Returns the absolute path to the local dataset script when `dataset_path`
    is one of the BSC MT datasets, otherwise returns `dataset_path` unchanged.
    """
    import os

    if dataset_path not in dataset_paths:
        return dataset_path

    relative_data_path = dataset_paths[dataset_path]

    # repo root when running from a source checkout / editable install
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    candidate = os.path.join(repo_root, relative_data_path)
    if os.path.exists(candidate):
        print(f"Dataset loaded from local path: {candidate}")
        return candidate

    # fall back to walking up from the current directory
    current = os.getcwd()
    while True:
        candidate = os.path.join(current, relative_data_path)
        if os.path.exists(candidate):
            print(f"Dataset loaded from local path: {candidate}")
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                f"Local dataset script '{relative_data_path}' for '{dataset_path}' "
                "not found relative to the mt-evaluation repository or the current directory."
            )
        current = parent
