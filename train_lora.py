import os
os.environ["HF_TOKEN"] = "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"
import json
import torch
from transformers import (AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig)
from datasets import load_from_disk, Dataset, Features, Image, Value
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

json_path = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/mimic_annotation.json"
img_root = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/files"
model_id = "google/medgemma-1.5-4b-it"
output_dir = "/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-sft-lora"

with open(json_path, 'r') as f:
    full_data = json.load(f)

train_raw = full_data['train']
data_list = []

for item in train_raw:
    rel_path = item['image_path'][0]
    full_img_path = os.path.join(img_root, rel_path)
    
    if os.path.exists(full_img_path):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a professional radiologist."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image",},
                    {"type": "text", "text": "Please provide a radiology report of the following chest X-ray image. Keep only the 'Findings' and 'Impression' sections in your report."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item['report']}
                ]
            }
        ]
        
        data_list.append({
            "image": full_img_path,
            "messages": messages
        })

features = Features({
    "image": Image(),
    "messages": [
        {
            "role": Value("string"),
            "content": [
                {
                    "type": Value("string"),
                    "text": Value("string") 
                }
            ]
        }
    ]
})

dataset = Dataset.from_list(data_list, features=features)
train_data = dataset 
print(f"Total samples: {len(train_data)}")

model_kwargs = dict(
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map={"": torch.cuda.current_device()},
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
    ensure_weight_tying=True,
)

def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        images.append([example["image"]])
        texts.append(processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False))

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == 262144] = -100 
    
    batch["labels"] = labels
    return batch

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,      
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    ddp_find_unused_parameters=False,
    optim="adamw_torch_fused", 
    logging_steps=10,    
    save_strategy="steps",
    save_steps=1000,    
    learning_rate=2e-4,                 
    bf16=True,     
    max_grad_norm=0.3,       
    warmup_ratio=0.03,     
    lr_scheduler_type="linear",          
    eval_strategy="no",              
    do_eval=False,                       
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},          
    label_names=["labels"],
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,   
    peft_config=peft_config,
    data_collator=collate_fn,
)
trainer.train()
trainer.save_model(output_dir)