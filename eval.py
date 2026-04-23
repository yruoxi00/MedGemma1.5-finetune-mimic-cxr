import pandas as pd
import torch
import os
os.environ["HF_TOKEN"] = "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"
import json
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import rrg_metric

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

base_path = "google/medgemma-1.5-4b-it"
sft_adapter = "/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-sft-lora"
# rl_adapter = "/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-rl"
json_path = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/mimic_annotation.json"
img_root = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/files"

def load_test():
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    test = full_data.get('test', [])
    print(f"Total test samples: {len(test)}")
    return test

def run_evaluation(model, processor, samples):
    model.eval()
    preds, gts, paths = [], [], []
    
    for s in samples:
        img_path = s['image_path']        
        rel_path = img_path[0] if isinstance(img_path, list) else img_path
        full_path = os.path.join(img_root, rel_path)
        image = Image.open(full_path)
        
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
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=False,
                num_beams=1
            )
        
        decoded = processor.decode(output_ids[0], skip_special_tokens=True)
        report = decoded.split("assistant\n")[-1].strip()
        
        preds.append(report)
        gts.append(s['report'])
        paths.append(full_path)
        
    
    bleu = rrg_metric.compute(metric="bleu", preds=preds, gts=gts)
    rouge = rrg_metric.compute(metric="rouge", preds=preds, gts=gts)
    meteor = rrg_metric.compute(metric="meteor", preds=preds, gts=gts)
    chexbert = rrg_metric.compute(metric="chexbert", preds=preds, gts=gts)
    radgraph = rrg_metric.compute(metric="f1radgraph", preds=preds, gts=gts)
    green = rrg_metric.compute(metric="green", preds=preds, gts=gts)
    
    return {
        "BLEU-4": bleu.get('total_results', 0.0),
        "ROUGE-L": rouge.get('total_results', 0.0),
        "METEOR": meteor.get('total_results', 0.0),
        "CheXBert": chexbert.get('f1chexbert_micro_f1_14', 0.0), 
        "F1-RadGraph": radgraph.get("total_results", 0.0),
        "GREEN": green.get("total_results", 0.0)
    }

def main():
    full_test_samples = load_test()
    samples = full_test_samples[:2]
    # samples = load_test()
    results_metrics[]
    
    export_data = pd.DataFrame()
    path_set = False 
    
    configs = [
        ("Base", base_path, []),
        ("SFT (Base+LoRA)", base_path, [sft_adapter])
        # ("RL (Base+LoRA+RL)", base_path, [sft_adapter, rl_adapter]) 
    ]

    for name, m_path, adapters in configs:
        processor = AutoProcessor.from_pretrained(base_path)
        model = AutoModelForImageTextToText.from_pretrained(
            m_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        for adapter in adapters:
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
            
        metrics, preds, gts, paths = run_evaluation(model, processor, samples)
        results_metrics[name] = metrics
        
        if not path_set:
            export_data['image_path'] = paths
            export_data['ground_truth'] = gts
            path_set = True
        
        export_data[f'{name}_generated_report'] = preds
        
        del model
        torch.cuda.empty_cache()
    
    output_excel = "eval_results.xlsx"
    export_data.to_excel(output_excel, index=False)

    table = "\n" + "="*120 + "\n"
    table += f"{'Model':<25} | {'BLEU-4':<10} | {'ROUGE-L':<10} | {'METEOR':<10} | {'CheXBert':<10} | {'F1-RadGraph':<12} | {'GREEN':<10}\n"
    table += "-" * 120 + "\n"
    for name, m in results.items():
        table += f"{name:<25} | {m['BLEU-4']:<10.4f} | {m['ROUGE-L']:<10.4f} | {m['METEOR']:<10.4f} | {m['CheXBert']:<10.4f} | {m['F1-RadGraph']:<12.4f} | {m['GREEN']:<10.4f}\n"
    table += "="*120 + "\n"
    print(table)    
    
    viz_sample = samples[2]
    print("\n" + " VISUAL COMPARISON ".center(85, "#"))
    print(f"Image Path: {viz_sample['image_path']}")
    print(f"Ground Truth Report: {viz_sample['report']}\n")

    for name, m_path, adapters in configs:
        processor = AutoProcessor.from_pretrained(base_path)
        model = AutoModelForImageTextToText.from_pretrained(
            m_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        for adapter in adapters:
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        
        path_data = viz_sample['image_path']
        rel_path = path_data[0] if isinstance(path_data, list) else path_data
        img_path = os.path.join(img_root, rel_path)
        image = Image.open(img_path)
        
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
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=[image], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100)
        
        report = processor.decode(output_ids[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
        print(f"[{name}] Generated Report:\n{report}\n")
        
        del model
        torch.cuda.empty_cache()
main()