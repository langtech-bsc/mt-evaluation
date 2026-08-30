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
        "geneval_contextual":"data/geneval_contextual/geneval_contextual.py"
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
