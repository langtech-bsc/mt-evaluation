"""Completion-rate and quality table for the CPT3 2B arms.

The CPT3 arms differ from the CPT2 base mainly in how often they produce an
output of plausible length at all: base was trained on single-line rows and
stops early at paragraph length. Corpus chrF/COMET/BLEU therefore mostly track
completion rather than per-segment translation quality, so this reports
completion first and the quality metrics beside it.

Reads the lm_eval result JSONs directly (they carry sources, targets,
translations and per-segment scores under --write_out), so it does not wait on
results_summary.csv extraction.

Usage:
    python scripts/cpt3_completion_table.py
    python scripts/cpt3_completion_table.py --dataset bouquet_paragraph
    python scripts/cpt3_completion_table.py --per-direction
    python scripts/cpt3_completion_table.py --arm treatment --per-direction
    python scripts/cpt3_completion_table.py --csv out.csv

"In scope" means the direction is one of the 22 DocHPLT pairs CPT3 trained
document data on (CPT3_PLAN.md section 3), in either direction. The in-scope
vs out-of-scope split is the experiment's main contrast.
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

RESULTS_ROOT = "/gpfs/projects/bsc88/mt_translation/mt-evaluation/results"
ARMS = ["base", "treatment", "control", "pseudo"]
PREFIX = "salamandraTA_2B_v2_"

# A hypothesis under this fraction of the reference length counts as a failure
# to complete; over RUNAWAY it counts as over-generation.
TRUNCATED = 0.5
RUNAWAY = 1.5

# The 22 DocHPLT pairs of CPT3 (CPT3_PLAN.md section 3). Both directions of
# each pair received document-level data, so scope is symmetric.
DOCHPLT_PAIRS = [
    ("ca", "en"), ("en", "ar"), ("en", "bg"), ("en", "cy"), ("en", "et"),
    ("en", "eu"), ("en", "fi"), ("en", "ga"), ("en", "gl"), ("en", "hi"),
    ("en", "hr"), ("en", "is"), ("en", "ja"), ("en", "ko"), ("en", "lt"),
    ("en", "lv"), ("en", "mt"), ("en", "nn"), ("en", "sk"), ("en", "sl"),
    ("en", "sr"), ("en", "uk"),
]
IN_SCOPE = set(DOCHPLT_PAIRS) | {(t, s) for s, t in DOCHPLT_PAIRS}

QUALITY = ["chrf", "comet", "bleu"]


def load(results_root, dataset):
    """arm -> (src, tgt) -> record with per-segment length ratios."""
    data = defaultdict(dict)
    for arm in ARMS:
        pattern = os.path.join(results_root, PREFIX + arm,
                               f"results_*_{dataset}.json")
        for path in glob.glob(pattern):
            m = re.match(rf"results_(\w+?)_(\w+?)_{re.escape(dataset)}\.json$",
                         os.path.basename(path))
            if not m:
                continue
            src, tgt = m.group(1), m.group(2)
            try:
                block = json.load(open(path))["results"][f"{src}_{tgt}_{dataset}"]
            except (KeyError, ValueError, OSError):
                continue
            rec = {k.split(",")[0]: v for k, v in block.items() if k != "alias"}
            if not rec.get("translations") or not rec.get("targets"):
                continue
            rec["ratios"] = [len(h) / max(len(g), 1)
                             for h, g in zip(rec["translations"], rec["targets"])]
            data[arm][(src, tgt)] = rec
    return data


def truncated(rec):
    return sum(x < TRUNCATED for x in rec["ratios"]) / len(rec["ratios"])


def runaway(rec):
    return sum(x > RUNAWAY for x in rec["ratios"]) / len(rec["ratios"])


def summary_block(data, dirs, label):
    """Mean completion and quality per arm over a set of directions."""
    if not dirs:
        return
    print(f"\n== {label} (n={len(dirs)}) ==")
    print(f'{"":18s}' + "".join(f"{a[:9]:>11s}" for a in ARMS))
    stats = [
        ("truncated <%.1fx" % TRUNCATED, truncated, True),
        ("runaway >%.1fx" % RUNAWAY, runaway, True),
        ("median len ratio", lambda r: st.median(r["ratios"]), False),
    ] + [(m, (lambda k: lambda r: r[k])(m), False) for m in QUALITY]
    for name, fn, pct in stats:
        line = f"{name:18s}"
        for arm in ARMS:
            vals = [fn(data[arm][d]) for d in dirs if d in data[arm]]
            if not vals:
                line += f"{'-':>11s}"
                continue
            v = st.mean(vals)
            line += f"{v * 100:10.1f}%" if pct else f"{v:11.3f}"
        print(line)


def scope_contrast(data, arm, baseline="base"):
    """The experiment's main contrast: does the effect concentrate in scope?"""
    pair = sorted(set(data[baseline]) & set(data[arm]))
    if not pair:
        return
    print(f"\n== {arm} vs {baseline}, in-scope contrast ==")
    header = (f'{"":14s}{"n":>5s}{"base trunc":>12s}{arm[:9] + " trunc":>16s}'
              f'{"delta":>9s}{"chrF delta":>12s}')
    print(header)
    for label, subset in (("in scope", [d for d in pair if d in IN_SCOPE]),
                          ("out of scope", [d for d in pair if d not in IN_SCOPE])):
        if not subset:
            continue
        bt = st.mean(truncated(data[baseline][d]) for d in subset)
        at = st.mean(truncated(data[arm][d]) for d in subset)
        dc = st.mean(data[arm][d]["chrf"] - data[baseline][d]["chrf"]
                     for d in subset)
        print(f"{label:14s}{len(subset):5d}{bt:11.1%}{at:15.1%}{at - bt:+9.1%}{dc:+12.2f}")


def per_direction(data, arm, baseline, csv_path=None):
    """One row per direction, in-scope first, sorted by truncation change."""
    pair = sorted(set(data[baseline]) & set(data[arm]))
    if not pair:
        print(f"\nno directions shared by {baseline} and {arm}")
        return
    rows = []
    for d in pair:
        b, a = data[baseline][d], data[arm][d]
        rows.append({
            "direction": f"{d[0]}-{d[1]}",
            "in_scope": d in IN_SCOPE,
            "n_segments": len(a["ratios"]),
            "base_truncated": truncated(b),
            "arm_truncated": truncated(a),
            "truncated_delta": truncated(a) - truncated(b),
            "arm_runaway": runaway(a),
            "base_median_ratio": st.median(b["ratios"]),
            "arm_median_ratio": st.median(a["ratios"]),
            **{f"base_{m}": b[m] for m in QUALITY if m in b},
            **{f"arm_{m}": a[m] for m in QUALITY if m in a},
        })
    rows.sort(key=lambda r: (not r["in_scope"], r["truncated_delta"]))

    print(f"\n== {arm} vs {baseline}, per direction "
          f"(* = DocHPLT in scope) ==")
    print(f'{"dir":10s}{"n":>5s}{"base tr":>9s}{"arm tr":>9s}{"delta":>9s}'
          f'{"chrF b":>9s}{"chrF a":>9s}{"delta":>8s}{"COMET b":>9s}{"COMET a":>9s}')
    for r in rows:
        mark = "*" if r["in_scope"] else " "
        print(f'{mark}{r["direction"]:<9s}{r["n_segments"]:5d}'
              f'{r["base_truncated"]:8.1%}{r["arm_truncated"]:9.1%}'
              f'{r["truncated_delta"]:+9.1%}'
              f'{r.get("base_chrf", float("nan")):9.2f}{r.get("arm_chrf", float("nan")):9.2f}'
              f'{r.get("arm_chrf", 0) - r.get("base_chrf", 0):+8.2f}'
              f'{r.get("base_comet", float("nan")):9.3f}{r.get("arm_comet", float("nan")):9.3f}')

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {csv_path}")


def correlation(data, arm, baseline):
    """How much of the quality change is explained by the completion change."""
    pair = sorted(set(data[baseline]) & set(data[arm]))
    xs = [truncated(data[arm][d]) - truncated(data[baseline][d]) for d in pair]
    ys = [data[arm][d]["chrf"] - data[baseline][d]["chrf"] for d in pair]
    if len(xs) < 3 or st.pstdev(xs) == 0 or st.pstdev(ys) == 0:
        return
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
    r = cov / (st.pstdev(xs) * st.pstdev(ys))
    print(f"\ncorr(truncation change, chrF change) = {r:.3f}  n={len(xs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=RESULTS_ROOT,
                    help="directory holding the salamandraTA_2B_v2_* result dirs")
    ap.add_argument("--dataset", default="bouquet_paragraph",
                    help="dataset suffix of the result files")
    ap.add_argument("--arm", default="treatment", choices=ARMS,
                    help="arm to contrast against the baseline")
    ap.add_argument("--baseline", default="base", choices=ARMS)
    ap.add_argument("--per-direction", action="store_true",
                    help="also print one row per direction")
    ap.add_argument("--csv", help="write the per-direction rows to this path")
    args = ap.parse_args()

    data = load(args.results_root, args.dataset)
    if not any(data.values()):
        sys.exit(f"no {args.dataset} results under {args.results_root}")

    print(f"directions with per-segment output ({args.dataset}):")
    for arm in ARMS:
        n_scope = sum(d in IN_SCOPE for d in data[arm])
        print(f"  {arm:10s} {len(data[arm]):3d}  ({n_scope} in DocHPLT scope)")

    common = sorted(set.intersection(*(set(data[a]) for a in ARMS))) \
        if all(data[a] for a in ARMS) else []
    print(f"\ncommon to all four arms: {len(common)}")
    summary_block(data, common, "all four arms, all common directions")
    summary_block(data, [d for d in common if d in IN_SCOPE], "all four arms, in scope")
    summary_block(data, [d for d in common if d not in IN_SCOPE],
                  "all four arms, out of scope")

    scope_contrast(data, args.arm, args.baseline)
    correlation(data, args.arm, args.baseline)
    if args.per_direction or args.csv:
        per_direction(data, args.arm, args.baseline, args.csv)


if __name__ == "__main__":
    main()
