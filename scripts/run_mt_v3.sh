#!/usr/bin/env bash
set -euo pipefail
: "${GEN_PYTHON:?Set GEN_PYTHON to the v5 environment python}"
: "${METRIC_PYTHON:?Set METRIC_PYTHON to the v4 environment python}"
: "${GEN_MODULES:=intel impi mkl hdf5 python/3.12.1}"
: "${METRIC_MODULES:=intel impi mkl hdf5 python/3.11.5-gcc}"

if (( $# < 2 )); then
    echo 'Usage: run_mt_v3.sh OUTPUT.json [lm_eval arguments]' >&2
    exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
output="$1"
shift
[[ "$output" == *.json ]] || { echo 'OUTPUT must end in .json' >&2; exit 2; }
for arg in "$@"; do
    case "$arg" in
        --output_path*|--predict_only*)
            echo 'Remove output_path/predict_only: the wrapper controls MT output.' >&2
            exit 2 ;;
        *holistic*)
            echo 'Holistic/toxicity aggregation requires a separate adapter.' >&2
            exit 2 ;;
    esac
done

# Invoked only inside stage subshells; does not alter the parent shell.
load_stage_modules() {
    local module_names="$1"
    local -a requested_modules
    type module >/dev/null 2>&1 || {
        echo 'The module command is unavailable; initialize Lmod in the launch script.' >&2
        return 1
    }
    # Lmod scripts may reference unset variables.
    set +u
    if declare -F deactivate >/dev/null; then deactivate; fi
    module purge || return 1
    read -r -a requested_modules <<< "$module_names"
    module load "${requested_modules[@]}" || return 1
    unset PYTHONHOME PYTHONPATH
    set -u
}

mkdir -p "$(dirname "$output")"
if [[ ! -f "$output" ]]; then
    (
        load_stage_modules "$GEN_MODULES"
        "$GEN_PYTHON" -c 'import sys, torch, transformers; print(sys.executable, sys.version, transformers.__version__, torch.__version__, flush=True); assert transformers.__version__.split(".")[0] == "5", "Generation requires Transformers v5"'
        MT_DEFER_NEURAL_METRICS=1 "$GEN_PYTHON" -m lm_eval "$@" --output_path "$output" --write_out
    )
else
    echo "Reusing saved generation: $output"
fi

(
    load_stage_modules "$METRIC_MODULES"
    "$METRIC_PYTHON" -c 'import sys, torch, transformers; print(sys.executable, sys.version, transformers.__version__, torch.__version__, flush=True); assert transformers.__version__.split(".")[0] == "4", "Scoring requires Transformers v4"'
    "$METRIC_PYTHON" -m neural_scoring.score --input "$output" --validate-only
    "$METRIC_PYTHON" -m neural_scoring.score --input "$output" \
        --output "$output" \
        --config "${MT_METRICS_CONFIG:-$repo_dir/lm_eval/extra_metrics/mt_metrics_config.yaml}"
)
