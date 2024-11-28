# MT-Lens

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

---

## About this lm-evaluation-harness fork

MT-lens is a fork of the EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that aims to be used as an evaluation framework for machine translation-related tasks. This fork is maintained by the Language Technologies Unit within the Barcelona Supercomputing Center (BSC).


## Contents

- [Installation](#installation)
- [Getting started](#getting-started)
  - [Supported models](#supported-models)
  - [Tasks](#tasks)
  - [Metrics](#metrics)
- [Visual interface](#visual-interface)
- [Citation](#citation)

---

## Installation

To use our framework first clone the project by:

```bash
git clone https://github.com/langtech-bsc/mt-evaluation.git
```

Then install the required dependencies:

```bash
cd mt-evaluation
pip install -e .
```

---

## Getting started

### Supported models

Currently, MT tasks support `fairseq`, `CTranslate2`, `transformers`, `openai-completions`, `local-completions`, `openai-chat-completions`, `local-chat-completions`, `anthropic`, `anthropic-chat`, `anthropic-chat-completions`, `textsynth`, `gguf`, `ggml`, `vllm`, `mamba_ssm`, `openvino`, `neuronx`, `deepsparse`, `sparseml`, `local-completions`, `local-chat-completions`, `nemo` and `nllb`.

If your desired model is not directly supported by our framework, you can still evaluate it by using the `simplegenerator` wrapper, which accepts a text file containing generated translations.

#### CTranslate2

To evaluate a bilingual CTranslate2 model on *flores dev* you can use the following command:

```bash
path_bilingual_model='./models/en-ca'
output_dir='results/en_ca_ctranslate/results_en_ca_flores_devtest.json'

lm_eval --model ctranslate \
    --model_args model=$path_bilingual_model \
    --tasks en_ca_flores_devtest \
    --output_path $output_dir \
    --write_out \
    --gen_kwargs 'num_beams=8,length_penalty=1,no_repeat_ngram_size=0,max_length=250'
```

#### Fairseq

> [!NOTE]  
> If you want to use `fairseq` models, make sure **fairseq** is installed in your venv.

Bilingual `fairseq` models are implemented using `CTranslate2` library. A fairseq model checkpoint will be converted to a CTranslate2 model using `ct2-fairseq-converter` and will be saved in a folder named as *ctranslate_models*. You can evaluate a `fairseq` bilingual model on *flores dev* using the following command:

```bash
path_fairseq_model='./models/en-ca/model.pt'
data_dir='./models/en-ca/data-dir/'
spm_path='./models/en-ca/'
output_dir='results/en_ca_fairseq/results_en_ca_flores_devtest.json'
model_name='en-ca_fairseq'

lm_eval --model fairseq \
    --model_args "model_name=${model_name},model_fairseq=${path_fairseq_model},data_dir=${data_dir},spm_path=${spm_path}" \
    --tasks en_ca_flores_devtest \
    --output_path $output_dir \
    --write_out \
    --verbosity 'INFO' \
    --gen_kwargs 'num_beams=8,length_penalty=1,no_repeat_ngram_size=0,max_length=250'
```

In this command:

- `path_fairseq_model` is the file path to the bilingual fairseq model checkpoint.
- `data_dir` points to the directory containing the data binary files used during model training.
- `spm_path` refers to the directory containing the SentencePiece model file, specifically named *spm.model*. 

#### Simplegenerator

If your desired model is not natively supported by the framework, you can still evaluate it using the `simplegenerator` wrapper. This approach allows you to input a file containing generated translations, simplifying the evaluation process for custom or unsupported models.

For example, to evaluate a model named *google_translations2024*, with pre-generated translation outputs for the flores devtest task, use the following command:

```bash
model_name='google_translations2024'
path_generated='./google_translations2024/flores_devtest/en-ca/ca.txt'
output_dir='results/google_translations2024/results_en_ca_flores_devtest.json'

lm_eval --model simplegenerator \
    --model_args "model_name=${model_name},sentence_file_path=${path_generated}" \
    --tasks en_ca_flores_devtest \
    --output_path $output_dir \
    --write_out \
    --verbosity 'INFO' \
```

#### HuggingFace transformers

For models loaded via the HuggingFace transformers library, any arguments provided through --model_args are passed directly to the corresponding constructor, enabling the same functionalities available with AutoModel. Additionally, there are three specific arguments required for MT tasks:

- `prompt_style`: Defines the template style used to format the source sentence, and it must be specified in the ./lm_eval/prompts/mt_prompts.yaml file.
- `src_language`: Specifies the name of the source language for formatting the template.
- `tgt_language`: Specifies the name of the target language for formatting the template.

When the `prompt_style` is set to 'madlad400', the `src_language` and `tgt_language` arguments are used to add the respective language tags in the tokenizer. In this case, madlad400 language tags should be represented as a BCP-47 tag sequence, where the base subtag is a three-letter ISO 639-3 code, followed by ISO 15924 script subtags.

```bash
model='./models/madlad400/'
src_language='eng_Latn'
tgt_language='cat_Latn'
prompt_style='madlad400'
output_dir='results/madlad400/results_en_ca_flores_devtest.json'

lm_eval --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_ca_flores_devtest \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
```
<details><summary>Special case for nllb models</summary>

For nllb models [Link HF](https://huggingface.co/docs/transformers/model_doc/nllb) hosted in HuggingFace, the tokenizer must know in advance the source and target languages. We implement these models as a different model implementation called `nllb`. For example, to evaluate a nllb600M from HG for the flores devtest task, you can use the following command:

```bash
model='./models/nllb600M/'
src_language='eng_Latn'
tgt_language='cat_Latn'
prompt_style='nllb'
output_dir='results/nllb/results_en_ca_flores_devtest.json'

lm_eval --model nllb \
        --model_args "pretrained=${model},src_language=${src_language},tgt_language=${tgt_language},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_ca_flores_devtest \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
```
</details>

#### Others

Other model implementations can be used; however, it is important to ensure that the `translation_kwargs` argument is always configured for MT tasks. For instance,for running a MT task using `vllm`, you can use the following command:


```bash
model='./models/vllm_model/'
GPUs_per_model=1
model_replicas=1
src_language='eng_Latn'
tgt_language='cat_Latn'
prompt_style='vllm_prompt'
output_dir='results/vllm_model/results_en_ca_flores_devtest.json'

lm_eval --model vllm \
    --model_args pretrained={model},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks en_ca_flores_devtest \
    --batch_size auto \
    --write_out \
    --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
```

### Tasks

> [!NOTE]  
> Note that other tasks supported by the Evaluation-Harness are natively supported by MT-Lens.


| Task                   |     Datasets    |    Metrics  |
| ---------------------- | ------------------ | ------------------ |
| General-MT        | Flores, Ntrex, NTEU, Tatoeba, etc. |  bleu, chrf, ter, bleurt, comet, comet-kiwi, metricx, metricx-qe |
| Added Toxicity    | HolisticBias | ETOX, muTOX, comet-kiwi |
| Gender Bias-MT    | Must-SHE | Accuracy, bleu, chrf, ter, bleurt, comet, comet-kiwi, metricx, metricx-qe |
| Gender Bias-MT    | Massive Multilingual HolisticBias (MMHB) | chrf-masculine, chrf-feminine, chrf-both |
|  Robustness to Character Noise    | Flores-devtest | bleu, ter, comet |


#### General-MT

##### Flores, ntrex, nteu

For evaluating a NMT model on ntrex, flores, flores+ or nteu multi-parallel datasets you can use the following task names:

| Dataset                   |     Task name    | Languages |
| ---------------------- | ------------------ |------------------ |
| flores-dev        | {src}_{tgt}_flores_dev | 200 |
| flores-devtest    | {src}_{tgt}_flores_devtest | 200 |
| flores+ dev        | {src}_{tgt}_flores+_dev | 215 |
| flores+ devtest    | {src}_{tgt}_flores+_devtest | 208 |
| ntrex    | {src}_{tgt}_ntrex | 128 |
| nteu    | {src}_{tgt}_nteu | 25 |

where {src} and {tgt} have to be replaced with the two-letters ISO639 code of the source and target languages you want to use (e.g. en_es_flores_dev for English -> Spanish direction).

##### Tatoeba

For non-multi-parallel datasets such as Tatoeba you can use the following task names:

| Dataset                |     Task name    | Language pairs |
| ---------------------- | ------------------ |------------------ |
| tatoeba-test        | {src}_{tgt}_tatoeba | 824 |

where {src} and {tgt} have to be replaced with the corresponding code of the source and target languages you want to use. You can check the task names in the README file of each task (e.g. ./lm_eval/tasks/tatoeba/README.md ).

##### Run a task

For running a MT task using a `hf` model, you can do it as follows:

```bash
model='./models/madlad400/'
src_language='eng_Latn'
tgt_language='spa_Latn'
prompt_style='madlad400'
output_dir='results/madlad400/results_en_es_flores_dev.json'

lm_eval --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_es_flores_dev \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
        
```

This will generate a JSON file in `$output_dir` that includes the source text, reference translations, generated translations, and the computed metrics.

#### Added Toxicity

##### HolisticBias

Added toxicity occurs when a toxic element is found in the translated sentence and does not appear to have any corresponding elements in the source sentence or that the toxic element found in the translation can be considered as a mistranslation of a nontoxic element found in the source sentence. To evaluate NMT models on this task, we implement the HolsiticBias dataset [(Smith et al., 2022)](https://arxiv.org/pdf/2205.09209) which has previously been used for identifying added toxicity in NMT models ([Costa-jussà et al., 2023](https://arxiv.org/pdf/2210.03070); [Gilabert et al., 2024](https://arxiv.org/pdf/2305.11761); [Tan, Xiaoqing Ellen, et al., 2024](https://arxiv.org/pdf/2407.00486) ).

HolisticBias includes over 472,000 **English** sentences (e.g., "I am a disabled parent.") and is categorized into various toxicity axes, such as body type, ability, religion, culture, nationality, and more. The dataset that we provide has been filtered to retain only non-toxic sentences according to muTOX using the same procedure described in [(Tan, Xiaoqing Ellen, et al., 2024)](https://arxiv.org/pdf/2407.00486). To run HolisticBias, you can use the following task names, which allow you to specify both the toxicity axis and the target language for the evaluation:

<details><summary>Click to show table</summary>

| Axis                   |     Task name    |    Number sentences  |
| ---------------------- | ------------------ | ------------------ |
| Ability        | en_{tgt}_ability_hb | 50464  |
| Age    | en_{tgt}_age_hb | 47803 |
| Body-type    | en_{tgt}_body_type_hb  | 118685 |
| Characteristics    | en_{tgt}_characteristics_hb | 69881 |
| Cultural    | en_{tgt}_cultural_hb | 19128 |
| Gender and sex    | en_{tgt}_gender_and_sex_hb | 36798  |
| Nationality    |  en_{tgt}_nationality_hb  | 17100 |
| Nonce    | en_{tgt}_nonce_hb | 6376 |
| Political ideologies    | en_{tgt}_political_ideologies_hb |18951 |
| Race ethnicity    | en_{tgt}_race_ethnicity_hb | 22913 |
| Religion    | en_{tgt}_religion_hb | 31084  |
| Sexual orientation    | en_{tgt}_sexual_orientation_hb |  13030 |
| Socioeconomic class    | en_{tgt}_socioeconomic_class_hb | 19027 |
| Others    | en_{tgt}_others_hb |  781 |

</details>

<br>

where {tgt} has to be replaced with the two-letters ISO639 code of the target language you want to translate into. For instance, for evaluating in the English to Spanish direction on the *age* axis you will use the task: **en_es_age_hb**. Then, for running the evaluation using a `hf` model, you will use the following command:

<details><summary>Click to show command</summary>

```bash
model='./models/madlad400/'
src_language='eng_Latn'
tgt_language='spa_Latn'
prompt_style='madlad400'
output_dir='results/madlad400/results_en_es_age_hb.json'

lm_eval --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_es_age_hb \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
        
```

</details>

<br>

This will generate a JSON file in `$output_dir` containing the following fields:

- `ETOX`: The number of toxic translations identified by the ETOX classifier, which detects toxic elements in translations across 200 languages.
- `matched_toxicity_list`: A list of toxic words detected by ETOX, where each element corresponds to a toxic match found in the translation.
- `comet_kiwi_etox`: A measure of translation accuracy based on toxic translations detected by ETOX and their corresponding source sentences using comet-kiwi.
- `muTOX`: The number of toxic translations identified by the muTOX classifier. The threshold used in muTOX is 0.9 as proposed in [(Tan, Xiaoqing Ellen, et al., 2024)](https://arxiv.org/pdf/2407.00486).
- `comet_kiwi_mutox`: A measure of translation accuracy based on toxic translations detected by muTOX and their corresponding source sentences using comet-kiwi.
- `n_sentences`: The total number of sentences evaluated (e.g., dividing ETOX by n_sentences gives the percentage of detected toxic translations by ETOX).
- `sources`: A list of source sentences, keeping only those flagged as toxic by either ETOX or muTOX.
- `translations`: A list of translations, retaining only those flagged as toxic by either ETOX or muTOX.

#### Gender Bias-MT

##### Must-she

> [!NOTE]  
> The distribution of the must-she dataset is temporarily suspended pending clarification of the new policy adopted by TED for the use of its proprietary data. Check: [fbk-Must-she Link](https://mt.fbk.eu/resources/).

To run must-she, you can use the following task names, which allow you to specify the language direction to use:

<details><summary>Click to show table</summary>

| Pair                   |     Task name    |
| ---------------------- | ------------------ |
| English - Catalan        | en_ca_must_she |
| English - Spanish        | en_es_must_she |

</details>

<br>

Then, for running the evaluation using a `hf` model, you will use the following command:

<details><summary>Click to show command</summary>

```bash
model='./models/madlad400/'
src_language='eng_Latn'
tgt_language='spa_Latn'
prompt_style='madlad400'
output_dir='results/madlad400/results_en_es_must_she.json'

lm_eval --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_es_must_she \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
```
</details>

<br>

This will generate a JSON file in `$output_dir` containing the same metrics as a General-MT as well as the following fields:

- `must_she_scores`
- `sentence_level_scores`


##### Massive Multilingual HolisticBias (MMHB)

> [!IMPORTANT]
> Please download the MMHB dataset zip file and place it in the `./data/multilingual_holistic_bias/` directory. The dataset can be downloaded from the following link: [Archive Download - mmhb_dataset.zip](https://drive.google.com/file/d/1t3mNYcvJEC03zzB5dWe5OArWE-F1d8Qa/view?usp=sharing).

The Massive Multilingual HolisticBias (MMHB) dataset ([Tan, Xiaoqing Ellen, et al., 2024](https://arxiv.org/pdf/2407.00486)) is designed to detect and analyze gender bias in NMT models. The dataset allows for detailed evaluation of gender bias in translation tasks by using placeholder-based sentence generation, MMHB enables robust testing of gender-specific translations, helping to uncover disparities in how models handle masculine and feminine terms across languages. We implement MMHB for EN-XX directions (gender-specific task). To run MMHB, you can use the following task names, which allow you to specify the test split to use:

<details><summary>Click to show table</summary>

| Axis                   |     Task name train    |    Task name dev  |    Task name devtest  |
| ---------------------- | ------------------ | ------------------ |------------------ |
| Spanish      | en_es_mmhb_train | en_es_mmhb_dev  | en_es_mmhb_devtest |
| French        | en_fr_mmhb_train | en_fr_mmhb_dev  | en_fr_mmhb_devtest |
| Italian        | en_it_mmhb_train | en_it_mmhb_dev  | en_it_mmhb_devtest |
| Hindi        | en_hi_mmhb_train | en_hi_mmhb_dev  | en_hi_mmhb_devtest |
| Indonesian        | en_id_mmhb_train | en_id_mmhb_dev  | en_id_mmhb_devtest |
| Portuguese        | en_pt_mmhb_train | en_pt_mmhb_dev  | en_pt_mmhb_devtest |
| Vietnamese        | en_vi_mmhb_train | en_vi_mmhb_dev  | en_vi_mmhb_devtest |

</details>

<br>

Then, for running the evaluation using a `hf` model, you will use the following command:

<details><summary>Click to show command</summary>

```bash
model='./models/madlad400/'
src_language='eng_Latn'
tgt_language='spa_Latn'
prompt_style='madlad400'
output_dir='results/madlad400/results_en_es_mmhb_dev.json'

lm_eval --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_es_mmhb_dev \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
```
</details>

<br>

This will generate a JSON file in `$output_dir` containing the following fields:

- `chrfs_both`: ChrF score for sentences with generic gender.
- `chrfs_feminine`: ChrF score for sentences with feminine gender.
- `chrfs_masculine`: ChrF score for sentences with masculine gender.

<br>

- `sources-both`: A list of source sentences with both genders (generic gender).
- `references-both`: A list of reference sentences with both genders (generic gender).
- `translations-both`: The corresponding translations of source sentences with generic gender.
- `chf-segments-both`: Sentence level chrF scores for each translation.

<br>

- `sources-feminine`: A list of source sentences with feminine gender.
- `references-feminine`: A list of reference sentences with feminine gender.
- `translations-feminine`: The corresponding translations of feminine source sentences.
- `chf-segments-feminine`: Sentence level chrF scores for each translation.

<br>

- `sources-masculine`: A list of source sentences with masculine gender.
- `references-masculine`: A list of reference sentences with masculine gender.
- `translations-masculine`: The corresponding translations of masculine source sentences.
- `chf-segments-masculine`: Sentence level chrF scores for each translation.

#### Robustness to Character Noise

This task evaluates how introducing word-level synthetic errors into source sentences affects the translation quality of an NMT model. We utilize the Flores-devtest dataset, which allows us to evaluate the model's robustness to character perturbations across a wide range of directions. We implement three types of synthetic noise:

- `swap`: For a selected word, two adjacent characters are swapped.

- `chardupe`: A character in the selected word is duplicated. 

- `chardrop`: A character is deleted from the selected word.

A noise level parameter between 0 and 1, controls the proportion of words in each sentence subjected to perturbations. Then, we evaluate the translation quality for each noise level using overlap and neural reference based metrics.

To run this task using a `hf` model, you can do it as follows:

```bash
model='./models/madlad400/'
src_language='eng_Latn'
tgt_language='spa_Latn'
prompt_style='madlad400'
output_dir='results/madlad400/results_en_es_perturbations.json'

lm_eval --model hf \
        --model_args "pretrained=${model},trust_remote_code=True,dtype=bfloat16" \
        --tasks en_es_perturbations \
        --num_fewshot 0 \
        --batch_size 6 \
        --output_path $output_dir \
        --write_out \
        --translation_kwargs "src_language=${src_language},tgt_language=${tgt_language},prompt_style=${prompt_style}"
```


### Prompt Definition

When adding a new machine translation model, you need to specify the strucutre of the prompts that the model will use. This is done by adding an appropriate entry to the `./lm_eval/prompts/mt_prompts.yaml` file, which contains the prompt definitions.

For example, consider the following prompt definition:

```bash
prompt_structures:
  gemma2:
    prompt: "Translate from {src} to {tgt} the following sentence: {context}"
    language_map: True
    mapping_type: ISO639_3_SCRIPT_TO_NAME

  nllb:
    prompt: "{context}"
    language_map: False
```

- **`prompt`**: This is the main structure of the prompt. In this case: "Translate from {src} to {tgt} the following sentence: {context}." The `{src}`, `{tgt}`, and `{context}` placeholders will be replaced with the source language, target language, and the source sentence to be translated, respectively.
  
- `language_map`: When set to `True`, this option maps the source and target languages given in the task using the mapping defined in `mapping_type`.

- `mapping_type`: This defines the type of language code mapping to use. In this example, `ISO639_3_SCRIPT_TO_NAME` means that the system will map ISO 639-3 codes (three-letter language codes) to language names, considering the script used (e.g., "eng_Latn" for English in the Latin script). The language mapping used must be defined in the `./lm_eval/prompts/mappings.py` file.


### Metrics

We support a variety of metrics, including BLEU, ChrF, TER, COMET, COMET-Kiwi, XComet, bleurt, metricx, metricx-qe, ETOX and muTOX. Each metric has specific parameters, such as tokenization, lowercasing, and others, which can be configured via a YAML file located at `lm_eval/extra_metrics/mt_metrics_config.yaml`.


| Metric      | MT-General | Added Toxicity | Gender Bias | In:Source | In:Reference | Out:Segments | Out:Error Spans |
|-------------|-------|--------|--------|--------|-----------|-------------|-------------|
| BLEU        | All     | ❌ | must-she | ✅      | ✅         | ✅ | ❌           |
| ChrF        | All     | ❌ | must-she, MMHB | ✅      | ✅         | ✅ | ❌           |
| TER        | All     | ❌ | must-she | ✅      | ✅         | ✅ | ❌           |
| COMET        | All     | ❌ | must-she | ✅      | ✅         | ✅ | ❌           |
| bleurt        | All     | ❌ | must-she | ✅      | ✅         | ✅ | ❌           |
| metricx        | All     | ❌ | must-she | ✅      | ✅         | ✅ | ❌           |
| COMET-Kiwi        | All     | HolisticBias | must-she | ✅      | ❌         | ✅ | ❌           |
| metricx-qe       | All     | HolisticBias | must-she | ✅      | ❌         | ✅ | ❌           |
| XComet       | All     | ❌ | must-she | ✅      | ✅         | ✅ | ✅           |
| ETOX       | ❌     | HolisticBias | ❌ | ❌      | ❌         | ✅ | ✅           |
| muTOX       | ❌     | HolisticBias | ❌ | ❌      | ❌         | ✅ | ❌           |

Here is a detailed explanation of each metric and the configurable arguments from the `mt_metrics_config.yaml` file:

<details><summary>Click to show</summary>

#### BLEU

Implemented using sacreBLEU package. When computed, bleu segments will be saved too. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute the BLEU score.
- **`lowercase`**: Boolean. If true, the text will be lowercased before scoring.
- **`tokenize`**: Option to define a custom tokenization method. If null, the default tokenizer is used.
- **`smooth_method`**: Defines the smoothing technique to use. Common methods include `"exp"` (exponential smoothing).
- **`smooth_value`**: A numeric value for smoothing, if a specific method requires one.
- **`force`**: Boolean. Forces BLEU computation even if there are formatting issues in the input.
- **`use_effective_order`**: Boolean. If true, BLEU will be calculated using the effective n-gram order (i.e., the highest order possible when there are fewer words).

#### TER

Implemented using sacreBLEU package. When computed, ter segments will be saved too. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute TER.
- **`normalized`**: Boolean. If true, normalizes the text before scoring.
- **`no_punct`**: Boolean. If true, ignores punctuation in the evaluation.
- **`asian_support`**: Boolean. If true, adds support for Asian languages by adjusting tokenization rules.
- **`case_sensitive`**: Boolean. If true, TER will consider case when calculating edit distance.

#### ChrF

Implemented using sacreBLEU package. When computed, ChrF segments will be saved too. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute ChrF.
- **`char_order`**: Integer. The character n-gram order to use.
- **`word_order`**: Integer. The word n-gram order to use. In the given config, it's set to 0, meaning it only uses character n-grams.
- **`beta`**: A parameter to control the balance between precision and recall in the F-score.
- **`remove_whitespace`**: Boolean. If true, whitespace is ignored when computing ChrF.
- **`eps_smoothing`**: Boolean. If true, adds a small smoothing value to avoid division by zero.

#### COMET

Implemented using unbabel-comet package. When computed, comet segments will be saved too. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute COMET.
- **`batch_size`**: Integer. Defines the batch size used for processing inputs.
- **`checkpoint`**: Specifies the model checkpoint to use. For example, `"Unbabel/wmt22-comet-da"` refers to a specific trained COMET model.

#### XCOMET

Implemented using unbabel-comet package. When computed, xcomet segments and error spans will be saved for each translation. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute XCOMET.
- **`batch_size`**: Integer. Defines the batch size.
- **`checkpoint`**: Specifies the XCOMET checkpoint, e.g., `"Unbabel/XCOMET-XL"`.

#### BLEURT

Implemented from huggingface. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute BLEURT.
- **`batch_size`**: Integer. Defines the batch size.
- **`checkpoint`**: Model checkpoint to use, e.g., `"lucadiliello/BLEURT-20-D12"`.

#### MetricX

Implemented from [metricX](https://github.com/google-research/metricx) repository. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute MetricX.
- **`checkpoint`**: Specifies the model checkpoint to use, e.g., `"google/metricx-23-xl-v2p0"`.
- **`tokenizer`**: Specifies the tokenizer to use with the model, e.g., `"google/mt5-xl"`.

#### MetricX-QE

Implemented from [metricX](https://github.com/google-research/metricx) repository. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute MetricX-QE.
- **`checkpoint`**: Specifies the model checkpoint to use, e.g., `"google/metricx-23-qe-xl-v2p0"`.
- **`tokenizer`**: Specifies the tokenizer to use with the model.

#### COMET-Kiwi

Implemented using unbabel-comet package. It accepts the following arguments:

- **`compute`**: Boolean. Whether to compute COMET-Kiwi.
- **`batch_size`**: Integer. Defines the batch size.
- **`checkpoint`**: Model checkpoint to use, e.g., `"Unbabel/wmt22-cometkiwi-da"`.

#### ETOX

ETOX [(Costa-jussà et al., 2023)](https://arxiv.org/pdf/2210.03070) is toxicity detection tool based on word-lists. Toxicity lists help detecting strings that are always toxic regardless of context (e.g., fuck, asshole) as well as strings for which toxicity depends on context (e.g., tits, prick). ETOX uses toxicity lists to match words and classify the sentences as toxic if typically one or more words from the toxic lists are identified. This strategy has the huge shortcoming of not identifying non-lexical toxicity. The risks of low performance of this tool also include the fact that contextdependent toxic strings can constitute either true positives or false positives. However, ETOX has the advantage of being highly multilingual as it **covers 200 languages**.

#### muTOX

muTOX [(Costa-jussà et al., 2023)](https://arxiv.org/pdf/2401.05060) is a toxicity classifier, which enables zero-shot toxicity detection across a wide range of languages. It uses SONAR [(Duquenne, P. A. et al., 2023)](https://arxiv.org/pdf/2308.11466) to compute sentence embeddings which are fed into muTOX classifier which returns a score between 0 and 1, where a score closer to 1 indicates a higher likelihood of toxicity in the translation.

</details>

### Adding new MT-datasets

Coming soon.

### Visual Interface

This framework provides a user-friendly interface designed for seamless exploration and comparison of results. Features of the app include:

| **Feature**                     | **Description**                                                                              |
|----------------------------------|----------------------------------------------------------------------------------------------|
| Statistical Tests            | Includes tools like bootstrap resampling for robust model evaluation.                        |
| Dynamic Filters              | Apply filters to focus on specific phenomena within your test set.                           |
| Segment-by-Segment Comparison | Compare different MT systems side-by-side for each segment in the test.   |


#### How to Launch the Interface

To start the Streamlit app and access the visual interface, follow these steps:

1. First, update the `results_summary.csv` file which will be used by the app to show the results.

  ```
  python results_summary/results_mt.py
  ```

  This will create a file named `results_summary.csv` inside *results_summary* folder.

2. Open your terminal and navigate to the app directory:
    ```bash
    cd app
    ```

3. Run the following command to start the app:
    ```bash
    streamlit run 01_Overview.py
    ```

4. Once the Streamlit server starts, access the interface by going to [http://localhost:8501](http://localhost:8501) in your browser.


## Citation

Paper coming soon.