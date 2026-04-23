import os
os.environ["HF_TOKEN"] = "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"
import numpy as np
from datasets import load_from_disk
import rrg_metric
import torch
from transformers import AutoModelForImageTextToText
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

# module load openjdk/17.0.8.1_1
# module load gcc/12.3.0
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

def setup_nltk():
    import ssl
    import nltk
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
setup_nltk()

dataset = load_from_disk("/fs/ess/PAS3128/ruoxiyang/data/processed_mimic_dataset")

def transform_for_grpo(example):
    return {
        "prompt": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this X-ray"}
            ]}
        ],
        "images": [example["image"]], 
        "answer": example["report"]
    }

train_dataset = dataset.map(
    transform_for_grpo,
    remove_columns=dataset.column_names,
    load_from_cache_file=False
)

def nlp_reward_func(completions, answer, **kwargs):
    gen_reports = [c[0]['content'] for c in completions]
    gts = [answer] * len(gen_reports)

    b_res = rrg_metric.compute(metric="bleu", preds=gen_reports, gts=gts, per_sample=True)
    r_res = rrg_metric.compute(metric="rouge", preds=gen_reports, gts=gts, per_sample=True)
    m_res = rrg_metric.compute(metric="meteor", preds=gen_reports, gts=gts, per_sample=True)

    b = np.array(b_res['per_sample_results'])
    r = np.array(r_res['per_sample_results'])
    m = np.array(m_res['per_sample_results'])

    rewards = (5/11 * b + 1/11 * m + 5/11 * r)
    return rewards.tolist()

def chexbert_reward_func(completions, answer, **kwargs):
    gen_reports = [c[0]['content'] for c in completions]
    if isinstance(answer, list):
        actual_answer = str(answer[0])
    else:
        actual_answer = str(answer)
    gts = [actual_answer] * len(gen_reports)

    res = rrg_metric.compute(metric="chexbert", preds=gen_reports, gts=gts, per_sample=True)
    
    rewards = res['f1chexbert_micro_f1_14']
    
    if not isinstance(rewards, (list, np.ndarray)):
        return [float(rewards)] * len(gen_reports)
    
    return [float(r) for r in rewards]

base = "google/medgemma-1.5-4b-it"
output_dir="/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-rl"

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,           
    num_generations=4,                      
    max_completion_length=100,
    max_steps=1000, 
    logging_steps=10,
    save_steps=100,
    report_to="tensorboard",
    use_vllm=True,                           
    vllm_mode="colocate",                    
    vllm_gpu_memory_utilization=.30,         
    bf16=True,                               
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False               
    }
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=64,
    target_modules="all-linear",
)

base_model = AutoModelForImageTextToText.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

adapter = "/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-sft-lora"
model = PeftModel.from_pretrained(base_model, adapter)
model = model.merge_and_unload()

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[nlp_reward_func, chexbert_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
)
trainer.train()
trainer.save_model(output_dir=training_args.output_dir)