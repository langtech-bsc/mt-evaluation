#!/usr/bin/env bash
# LLM judge stage (GEMBA-ESA, neural_scoring/llm_judge.py) for saved MT results, in the vLLM / Transformers-5
# environment. Called at the end of scripts/run_mt_v3.sh; can also be run on its own to add the judge later
# (several results files share one model load):
#     GEN_PYTHON=venv-v5/bin/python bash scripts/run_llm_judge.sh [--on] CONFIG.yaml results/a.json [results/b.json ...]
# Runs only if the config has  mt_metrics: llm_judge: compute: True,  or with --on (judge existing results:
# launch_evaluation/example_launch/llm_judge.sh).  Needs tensor_parallel_size GPUs
# (gemma-4-31B-it: 2 in bf16, 1 with quantization: fp8); with fewer visible GPUs it prints a warning and
# skips without failing. Files already scored by this judge are skipped (llm_judge.py; --force to redo).
set -euo pipefail
: "${GEN_PYTHON:?Set GEN_PYTHON to the v5 environment python}"
: "${GEN_MODULES:=intel impi mkl hdf5 python/3.12.1}"
force_on=0
if [[ "${1:-}" == --on ]]; then force_on=1; shift; fi
(( $# >= 2 )) || { echo 'Usage: run_llm_judge.sh [--on] CONFIG.yaml RESULTS.json [RESULTS.json ...]' >&2; exit 2; }
config="$1"; shift
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

type module >/dev/null 2>&1 || {
    echo 'The module command is unavailable; initialize Lmod in the launch script.' >&2
    exit 1
}
set +u  # Lmod scripts may reference unset variables
if declare -F deactivate >/dev/null; then deactivate; fi
module purge
read -r -a mods <<< "$GEN_MODULES"
module load "${mods[@]}"
set -u
unset PYTHONHOME PYTHONPATH

read -r judge_on judge_gpus judge_name < <("$GEN_PYTHON" - "$config" "$force_on" <<'PY'
import sys, yaml
c = ((yaml.safe_load(open(sys.argv[1])) or {}).get('mt_metrics') or {}).get('llm_judge')
if c is None:
    if sys.argv[2] == '1':
        sys.exit(f'No mt_metrics: llm_judge: block in {sys.argv[1]}')
    c = {}
if sys.argv[2] == '1':  # --on
    c['compute'] = True
name = '-'
if c.get('compute', False):  # the name the results will use (name: auto + precision suffix)
    from neural_scoring.llm_judge import apply_precision
    c.setdefault('stage2_input', 'raw'); name = apply_precision(dict(c))['name']
print(int(bool(c.get('compute', False))), int(c.get('tensor_parallel_size', 1)), name)
PY
)
if (( ! judge_on )); then
    echo "LLM judge: off (llm_judge.compute is not True in $config)"
    exit 0
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"; visible=${#gpu_ids[@]}
else
    visible=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
fi
if (( visible < judge_gpus )); then
    echo "WARNING: LLM judge skipped: it needs $judge_gpus GPU(s) (tensor_parallel_size) but $visible are visible." >&2
    echo "         Request --gres=gpu:$judge_gpus, or set quantization: fp8 + tensor_parallel_size: 1, then add it later with:" >&2
    echo "         GEN_PYTHON=$GEN_PYTHON bash scripts/run_llm_judge.sh $config $*" >&2
    exit 0
fi

# vLLM in this environment needs a newer SQLite, gcc for its JIT kernels, and no FlashInfer sampler
# (as in launch_evaluation/example_launch/vllm_smoke_test.sh).
export LD_LIBRARY_PATH=/apps/ACC/EASYBUILD/software/SQLite/3.45.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}
export CC=gcc CXX=g++
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
echo "LLM judge: $judge_name on $visible visible GPU(s), config $config"
"$GEN_PYTHON" -m neural_scoring.llm_judge --config "$config" --input "$@"
