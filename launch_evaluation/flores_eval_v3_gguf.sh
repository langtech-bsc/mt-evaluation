#!/usr/bin/env bash
#SBATCH --job-name=gguf_mt_full
#SBATCH --output=slurm_logs_xixian/gguf_%j.out
#SBATCH --error=slurm_logs_xixian/gguf_%j.err
#SBATCH -q acc_bscaii
#SBATCH -A bsc88
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -eo pipefail

export LLAMA_SERVER="/gpfs/scratch/bsc88/quantization/llama.cpp/build/bin/llama-server"
#export GGUF_FILE="/gpfs/projects/bsc88/mt_translation/instructed_models/salamandraTA_7b_v3_mixture1_final_GGUF/salamandraTA_7B_v3_q4_k_m.gguf"

export GGUF_FILE="/gpfs/projects/bsc88/mt_translation/instructed_models/salamandraTA_7b_v3_mixture1_final_GGUF/salamandraTA_7B_v3_q8_0.gguf"


export LLAMA_MODULES="cuda/12.2"


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
run_dir="$PWD/results/gguf_full_$(basename "$GGUF_FILE" .gguf)_${SLURM_JOB_ID}_$(date +%s)"
mkdir -p "$run_dir"
printf 'Results directory: %s\n' "$run_dir"
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

# Keep both directions, but evaluate each task only once.
declare -a language_pairs=(
    "eng_Latn:zho_Hans:en:zh"
    "spa_Latn:eus_Latn:es:eu"
    "spa_Latn:eng_Latn:es:en"
    "ast_Latn:spa_Latn:ast:es"
    "spa_Latn:arz_Arab:es:ar"
    "spa_Latn:hin_Deva:es:hi"
    "spa_Latn:zho_Hans:es:zh"
    "spa_Latn:jpn_Jpan:es:ja"
    "spa_Latn:kor_Hang:es:ko"
    "spa_Latn:isl_Latn:es:is"
    "glg_Latn:spa_Latn:gl:es"
    "spa_Latn:arg_Latn:es:arg"
    "pol_Latn:spa_Latn:pl:es"
    "oci_Latn:spa_Latn:oc:es"
    "por_Latn:spa_Latn:pt:es"
    "cat_Latn:spa_Latn:ca:es"
    "eng_Latn:dan_Latn:en:da"
    "cat_Latn:ukr_Cyrl:ca:uk"
    "deu_Latn:spa_Latn:de:es"
    "spa_Latn:arn_Latn:es:arn"
    "spa_Latn:ces_Latn:es:cs"
    "spa_Latn:deu_Latn:es:de"
    "spa_Latn:ell_Grek:es:el"
    "spa_Latn:ekk_Latn:es:et"
    "spa_Latn:eng_Latn:es:en"
    "spa_Latn:fin_Latn:es:fi"
    "spa_Latn:fra_Latn:es:fr"
    "spa_Latn:gle_Latn:es:ga"
    "spa_Latn:hrv_Latn:es:hr"
    "spa_Latn:ita_Latn:es:it"
    "spa_Latn:lit_Latn:es:lt"
    "spa_Latn:lvs_Latn:es:lv"
    "spa_Latn:mlt_Latn:es:mt"
    "spa_Latn:nld_Latn:es:nl"
    "spa_Latn:nno_Latn:es:nn"
    "spa_Latn:ron_Latn:es:ro"
    "spa_Latn:rus_Cyrl:es:ru"
    "spa_Latn:slk_Latn:es:sk"
    "spa_Latn:slv_Latn:es:sl"
    "spa_Latn:srp_Cyrl:es:sr"
    "spa_Latn:ukr_Cyrl:es:uk"
    "spa_Latn:dan_Latn:es:da"
    "spa_Latn:cym_Latn:es:cy"
    "spa_Latn:bul_Cyrl:es:bg"
    "spa_Latn:swe_Latn:es:sv"
    "spa_Latn:nob_Latn:es:no"
    "spa_Latn:hun_Latn:es:hu"
    "spa_Latn:val_Latn:es:vl"
    "cat_Latn:isl_Latn:ca:is"
    "cat_Latn:arz_Arab:ca:ar"
    "cat_Latn:hin_Deva:ca:hi"
    "cat_Latn:zho_Hans:ca:zh"
    "cat_Latn:jpn_Jpan:ca:ja"
    "cat_Latn:kor_Hang:ca:ko"
    "cat_Latn:ces_Latn:ca:cs"
    "cat_Latn:eng_Latn:ca:en"
    "cat_Latn:deu_Latn:ca:de"
    "cat_Latn:ell_Grek:ca:el"
    "cat_Latn:spa_Latn:ca:es"
    "cat_Latn:eng_Latn:ca:en"
    "cat_Latn:ekk_Latn:ca:et"
    "cat_Latn:eus_Latn:ca:eu"
    "cat_Latn:fin_Latn:ca:fi"
    "cat_Latn:fra_Latn:ca:fr"
    "cat_Latn:gle_Latn:ca:ga"
    "cat_Latn:glg_Latn:ca:gl"
    "cat_Latn:hrv_Latn:ca:hr"
    "cat_Latn:ita_Latn:ca:it"
    "cat_Latn:lit_Latn:ca:lt"
    "cat_Latn:lvs_Latn:ca:lv"
    "cat_Latn:mlt_Latn:ca:mt"
    "cat_Latn:nld_Latn:ca:nl"
    "cat_Latn:nno_Latn:ca:nn"
    "cat_Latn:oci_Latn:ca:oc"
    "cat_Latn:pol_Latn:ca:pl"
    "cat_Latn:por_Latn:ca:pt"
    "cat_Latn:ron_Latn:ca:ro"
    "cat_Latn:rus_Cyrl:ca:ru"
    "cat_Latn:slk_Latn:ca:sk"
    "cat_Latn:slv_Latn:ca:sl"
    "cat_Latn:srp_Cyrl:ca:sr"
    "cat_Latn:dan_Latn:ca:da"
    "cat_Latn:cym_Latn:ca:cy"
    "cat_Latn:bul_Cyrl:ca:bg"
    "cat_Latn:swe_Latn:ca:sv"
    "cat_Latn:nob_Latn:ca:no"
    "cat_Latn:ast_Latn:ca:ast"
    "cat_Latn:arg_Latn:ca:arg"
    "cat_Latn:arn_Latn:ca:arn"
    "cat_Latn:hun_Latn:ca:hu"
    "cat_Latn:val_Latn:ca:vl"
    "cat_Latn:nob_Latn:ca:no"
    "eng_Latn:isl_Latn:en:is"
    "ces_Latn:ukr_Cyrl:cs:uk"
    "ces_Latn:deu_Latn:cs:de"
    "jpn_Jpan:zho_Hans:ja:zh"
    "eng_Latn:arz_Arab:en:ar"
    "eng_Latn:hin_Deva:en:hi"
    "eng_Latn:jpn_Jpan:en:ja"
    "eng_Latn:kor_Hang:en:ko"
    "eng_Latn:ces_Latn:en:cs"
    "eng_Latn:deu_Latn:en:de"
    "eng_Latn:ell_Grek:en:el"
    "eng_Latn:ekk_Latn:en:et"
    "eng_Latn:eus_Latn:en:eu"
    "eng_Latn:fin_Latn:en:fi"
    "eng_Latn:fra_Latn:en:fr"
    "eng_Latn:gle_Latn:en:ga"
    "eng_Latn:glg_Latn:en:gl"
    "eng_Latn:hrv_Latn:en:hr"
    "eng_Latn:ita_Latn:en:it"
    "eng_Latn:lit_Latn:en:lt"
    "eng_Latn:lvs_Latn:en:lv"
    "eng_Latn:mlt_Latn:en:mt"
    "eng_Latn:nld_Latn:en:nl"
    "eng_Latn:nno_Latn:en:nn"
    "eng_Latn:oci_Latn:en:oc"
    "eng_Latn:pol_Latn:en:pl"
    "eng_Latn:por_Latn:en:pt"
    "eng_Latn:ron_Latn:en:ro"
    "eng_Latn:rus_Cyrl:en:ru"
    "eng_Latn:slk_Latn:en:sk"
    "eng_Latn:slv_Latn:en:sl"
    "eng_Latn:srp_Cyrl:en:sr"
    "eng_Latn:ukr_Cyrl:en:uk"
    "eng_Latn:dan_Latn:en:da"
    "eng_Latn:cym_Latn:en:cy"
    "eng_Latn:bul_Cyrl:en:bg"
    "eng_Latn:swe_Latn:en:sv"
    "eng_Latn:nob_Latn:en:no"
    "eng_Latn:ast_Latn:en:ast"
    "eng_Latn:arg_Latn:en:arg"
    "eng_Latn:arn_Latn:en:arn"
    "eng_Latn:hun_Latn:en:hu"
    "eng_Latn:val_Latn:en:vl"
    "glg_Latn:eus_Latn:gl:eu"
    "eus_Latn:glg_Latn:eu:gl"
)
declare -A seen_tasks=()
declare -a outputs=()
for pair in "${language_pairs[@]}"; do
    IFS=: read -r src tgt src_short tgt_short <<< "$pair"
    for direction in "$src:$tgt:$src_short:$tgt_short" "$tgt:$src:$tgt_short:$src_short"; do
        IFS=: read -r source target source_short target_short <<< "$direction"
        task="${source_short}_${target_short}_flores+_devtest"
        [[ -n "${seen_tasks[$task]:-}" ]] && continue
        seen_tasks[$task]=1
        output="$run_dir/results_${task}.json"
        printf 'Generating full task: %s\n' "$task"
        MT_DEFER_NEURAL_METRICS=1 "$GEN_PYTHON" -m lm_eval \
            --model gguf --model_args "base_url=http://127.0.0.1:${port},max_length=8192" \
            --tasks "$task" --num_fewshot 0 --batch_size 1 --verbosity INFO \
            --translation_kwargs "src_language=${source},tgt_language=${target},prompt_style=salamandraTA7B_instruct" \
            --gen_kwargs 'max_gen_toks=800,num_beams=1,do_sample=False' \
            --output_path "$output" --write_out
        outputs+=("$output")
    done
done
# Release the GPU before loading neural metrics.
cleanup
server_pid=''
module purge
module load intel impi mkl hdf5 python/3.11.5-gcc
unset PYTHONHOME PYTHONPATH
for output in "${outputs[@]}"; do
    printf 'Scoring: %s\n' "$output"
    "$METRIC_PYTHON" -m neural_scoring.score --input "$output" --output "$output" \
        --config "${MT_METRICS_CONFIG:-$PWD/lm_eval/extra_metrics/mt_metrics_config.yaml}"
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
done
printf 'All tasks completed: %s\n' "$run_dir"
