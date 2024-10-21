import numpy as np
import os
from collections import defaultdict
from sacrebleu.significance import Result, _compute_p_value, estimate_ci

def compute_bootstrap_test(system1_scores, system2_scores, metric_name, paired_bs_n=2000):
    """
    Computes the bootstrap test between two systems using their segment-level scores.

    :param system1_scores: List of scores for system 1
    :param system2_scores: List of scores for system 2
    :param metric_name: Name of the metric (e.g., 'bleu_segments', 'comet_segments')
    :param paired_bs_n: Number of bootstrap samples
    :return: Dictionary with results
    """
    metric_sentence_scores = {metric_name: [system1_scores, system2_scores]}
    results = paired_bs(metric_sentence_scores, paired_bs_n)
    return results

def paired_bs(
    metric_sentence_scores: dict,
    paired_bs_n: int = 2000,
):
    """
    :param metric_sentence_scores: A dictionary of metric_name to a list of lists of sentence-level scores
        where each item corresponds to the results of one system.
    :param paired_bs_n: Number of bootstrap samples.
    :return: A dictionary with keys as metrics and values as lists of Result objects containing the test results.
    """
    seed = 89477
    rng = np.random.default_rng(seed)
    dataset_size = len(next(iter(metric_sentence_scores.values()))[0])
    idxs = rng.choice(dataset_size, size=(paired_bs_n, dataset_size), replace=True)

    results = defaultdict(list)
    for metric_name, all_sys_scores in metric_sentence_scores.items():
        all_sys_scores = np.array(all_sys_scores)
        scores_bl, all_sys_scores = all_sys_scores[0], all_sys_scores[1:]
        # Baseline
        real_mean_bl = scores_bl.mean().item()
        bs_scores_bl = scores_bl[idxs].mean(axis=1)
        bs_bl_mean, bl_ci = estimate_ci(bs_scores_bl)

        results[metric_name].append(Result(score=real_mean_bl, p_value=None, mean=bs_bl_mean, ci=bl_ci))

        for scores_sys in all_sys_scores:
            # Real mean score for the system
            real_mean_sys = scores_sys.mean().item()
            diff = abs(real_mean_bl - real_mean_sys)
            bs_scores_sys = scores_sys[idxs].mean(axis=1)
            bs_mean_sys, sys_ci = estimate_ci(bs_scores_sys)
            sample_diffs = np.abs(bs_scores_sys - bs_scores_bl)
            stats = sample_diffs - sample_diffs.mean()
            p = _compute_p_value(stats, diff)
            results[metric_name].append(Result(score=real_mean_sys, p_value=p, mean=bs_mean_sys, ci=sys_ci))

    return dict(results)