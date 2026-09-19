#!/bin/bash
#SBATCH --job-name=gemma4_31b_evaluation
#SBATCH --error=slurm_logs_xixian/gemma4_31b_%j.err
#SBATCH --output=slurm_logs_xixian/gemma4_31b_%j.out
#SBATCH -q acc_debug
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=60
#SBATCH -N 1
#SBATCH --account bsc88
#SBATCH --time=00-02:00:00


set -euo pipefail
cd /gpfs/projects/bsc88/mt_translation/mt-evaluation-v3

export GEN_PYTHON="$PWD/venv-v5/bin/python"
export METRIC_PYTHON="$PWD/venv-neural-v4/bin/python"
export GEN_MODULES="intel impi mkl hdf5 python/3.12.1"
export METRIC_MODULES="intel impi mkl hdf5 python/3.11.5-gcc"

export HF_HOME=/gpfs/projects/bsc88/mt_translation/hf_cache
export TRANSFORMERS_CACHE=/gpfs/projects/bsc88/mt_translation/hf_cache
export HF_DATASETS_CACHE=/gpfs/projects/bsc88/mt_translation/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


model='/gpfs/projects/bsc88/hf-models/gemma-4-31B-it'
model_name_dir='gemma-4-31B-it'
prompt_style='gemma4-it'
batch_size=2
num_fewshot=0
verbosity='INFO'

declare -a language_pairs=(

"spa_Latn:eus_Latn:es:eu"
"spa_Latn:eng_Latn:es:en"
"ast_Latn:spa_Latn:ast:es"
"eng_Latn:zho_Hans:en:zh"

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

# Each direction runs once per job, even if listed more than once.
declare -A seen_directions=()
run_direction() {
    local src_language="$1" tgt_language="$2" src_abrv="$3" tgt_abrv="$4"
    local direction="${src_abrv}_${tgt_abrv}"
    if [[ -n "${seen_directions[$direction]:-}" ]]; then
        return 0
    fi
    seen_directions[$direction]=1
    local output_file="$PWD/results/${model_name_dir}/results_${direction}_flores+_devtest.json"

    # The wrapper reuses saved translations and merges neural scores in place.
    bash scripts/run_mt_v3.sh "$output_file" --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16,parallelize=True" \
        --tasks "${direction}_flores+_devtest" \
        --num_fewshot "$num_fewshot" \
        --batch_size "$batch_size" \
        --verbosity "$verbosity" \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}" \
        --gen_kwargs "max_gen_toks=800"
}

for pair in "${language_pairs[@]}"; do
    IFS=':' read -r src_language tgt_language src_abrv tgt_abrv <<< "$pair"
    run_direction "$src_language" "$tgt_language" "$src_abrv" "$tgt_abrv"
    run_direction "$tgt_language" "$src_language" "$tgt_abrv" "$src_abrv"
done
