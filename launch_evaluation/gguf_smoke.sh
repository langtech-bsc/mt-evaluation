#!/usr/bin/env bash
#SBATCH --job-name=gguf_mt_smoke
#SBATCH --output=slurm_logs_xixian/gguf_%j.out
#SBATCH --error=slurm_logs_xixian/gguf_%j.err
#SBATCH -q acc_debug
#SBATCH -A bsc88
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -eo pipefail
: "${LLAMA_SERVER:?Set absolute path to CUDA-enabled llama-server}"
: "${GGUF_FILE:?Set absolute path to a GGUF file}"
: "${SLURM_JOB_ID:?Run inside a GPU allocation or use sbatch}"
cd /gpfs/projects/bsc88/mt_translation/mt-evaluation-v3
[[ -x "$LLAMA_SERVER" && -f "$GGUF_FILE" ]] || { echo 'Check LLAMA_SERVER and GGUF_FILE paths'; exit 1; }
GEN_PYTHON="$PWD/venv-v5/bin/python"
METRIC_PYTHON="$PWD/venv-neural-v4/bin/python"
export HF_HOME=/gpfs/projects/bsc88/mt_translation/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
run_dir="$PWD/results/gguf_$(basename "$GGUF_FILE" .gguf)_${SLURM_JOB_ID}_$(date +%s)"
mkdir -p "$run_dir"
output="$run_dir/results_en_zh_flores+_devtest.json"
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
module purge
module load intel impi mkl hdf5 python/3.12.1
unset PYTHONHOME PYTHONPATH
# Add any modules required by your existing llama.cpp build, if needed.
if [[ -n "${LLAMA_MODULES:-}" ]]; then
    read -r -a extra_modules <<< "$LLAMA_MODULES"
    module load "${extra_modules[@]}"
    unset PYTHONHOME PYTHONPATH
fi
port="${LLAMA_PORT:-18080}"
"$GEN_PYTHON" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$port"
"$LLAMA_SERVER" --version > "$run_dir/llama-version.txt" 2>&1
"$LLAMA_SERVER" --model "$GGUF_FILE" --host 127.0.0.1 --port "$port" \
    --n-gpu-layers 99 --ctx-size 8192 --parallel 1 --threads 20 \
    > "$run_dir/llama-server.log" 2>&1 &
server_pid=$!
"$GEN_PYTHON" - "$port" "$server_pid" <<'PY'
import os,sys,time,urllib.request
port,pid=sys.argv[1],int(sys.argv[2])
for _ in range(300):
    os.kill(pid,0)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
            if r.status==200: break
    except Exception: pass
    time.sleep(2)
else: raise SystemExit('Server was not ready after 10 minutes; inspect llama-server.log')
PY
# /v1/completions receives the MT prompt verbatim; no second chat template.
MT_DEFER_NEURAL_METRICS=1 "$GEN_PYTHON" -m lm_eval \
    --model gguf --model_args "base_url=http://127.0.0.1:${port},max_length=8192" \
    --tasks en_zh_flores+_devtest --num_fewshot 0 --batch_size 1 \
    --translation_kwargs 'src_language=eng_Latn,tgt_language=zho_Hans,prompt_style=salamandraTA7B_instruct' \
    --gen_kwargs 'max_gen_toks=800,num_beams=1,do_sample=False' \
    --limit 5 --output_path "$output" --write_out
# Stop llama.cpp before scoring so its GPU memory is released.
cleanup
server_pid=''
module purge
module load intel impi mkl hdf5 python/3.11.5-gcc
unset PYTHONHOME PYTHONPATH
"$METRIC_PYTHON" -m neural_scoring.score --input "$output" --output "$output"
"$METRIC_PYTHON" - "$output" "$GGUF_FILE" "$run_dir/llama-version.txt" <<'PY'
import json,sys
from pathlib import Path
from neural_scoring.score import atomic_write
p=Path(sys.argv[1]);data=json.loads(p.read_text())
f=Path(sys.argv[2]).resolve()
data['gguf_inference']={'file':str(f),'size_bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'engine':'llama.cpp','engine_version':Path(sys.argv[3]).read_text(),'decoding':'greedy','seed':1234,'max_tokens':800,'ctx_size':8192}
atomic_write(p,data)
PY
printf 'Completed: %s\n' "$output"
