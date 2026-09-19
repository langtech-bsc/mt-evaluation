from vllm import LLM, SamplingParams

# ← 直接用本地已有的文件夹,不下载、不联网
model_dir  = "/gpfs/projects/bsc88/mt_translation/instructed_models/salamandraTA_7b_v3_mixture1_final_GGUF"
model_file = model_dir + "/salamandraTA_7B_v3_q4_k_m.gguf"

llm = LLM(model=model_file, tokenizer=model_dir)

source = "Spanish"
target = "English"
sentence = ("Ayer se fue, tomó sus cosas y se puso a navegar. Una camisa, un pantalón vaquero "
            "y una canción, dónde irá, dónde irá. Se despidió, y decidió batirse en duelo con el mar. "
            "Y recorrer el mundo en su velero. Y navegar, nai-na-na, navegar.")

prompt = f"Translate the following text from {source} into {target}.\n{source}: {sentence} \n{target}:"
messages = [{"role": "user", "content": prompt}]

outputs = llm.chat(
    messages,
    sampling_params=SamplingParams(
        temperature=0.1,
        stop_token_ids=[5],
        max_tokens=200,
    ),
)[0].outputs

print("=" * 40)
print("TRANSLATION:")
print(outputs[0].text)
print("=" * 40)