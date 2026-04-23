import pandas as pd
import torch
import os
os.environ["HF_TOKEN"] = "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"
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

base_path = "google/medgemma-1.5-4b-it"
sft_adapter = "/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-sft-lora"
# rl_adapter = "/fs/ess/PAS3128/ruoxiyang/models/medgemma-mimic-rl"
json_path = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/mimic_annotation.json"
img_root = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/files"
output_json = "/fs/ess/PAS3128/ruoxiyang/result/base_lora_reports.json"
output_xlsx = "/fs/ess/PAS3128/ruoxiyang/result/base_lora_reports.xlsx"

def load_test():
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    test = full_data.get('test', [])
    print(f"Total test samples: {len(test)}")
    return test

def generate_reports(model, processor, samples):
    model.eval()
    preds, gts, paths = [], [], []

    for i, s in enumerate(samples, start=1):
        img_path = s["image_path"]
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
        gts.append(s["report"])
        paths.append(full_path)

        print(f"[{i}/{len(samples)}] Done")

    return preds, gts, paths

def main():
    samples = load_test()

    configs = [
        ("Base", base_path, []),
        ("SFT (Base+LoRA)", base_path, [sft_adapter]),
        # ("RL (Base+LoRA+RL)", base_path, [sft_adapter, rl_adapter]),
    ]

    export_data = pd.DataFrame()
    path_set = False

    all_outputs = {
        "metadata": {
            "num_samples": len(samples),
            "base_model": base_path,
            "configs": [name for name, _, _ in configs],
        },
        "rows": []
    }
    
    processor = AutoProcessor.from_pretrained(base_path)

    for name, m_path, adapters in configs:
        print(f"\nRunning config: {name}")

        model = AutoModelForImageTextToText.from_pretrained(
            m_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        for adapter in adapters:
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()

        preds, gts, paths = generate_reports(model, processor, samples)

        if not path_set:
            export_data["image_path"] = paths
            export_data["ground_truth"] = gts
            path_set = True

            for p, g in zip(paths, gts):
                all_outputs["rows"].append({
                    "image_path": p,
                    "ground_truth": g
                })

        export_data[f"{name}_generated_report"] = preds

        for idx, pred in enumerate(preds):
            all_outputs["rows"][idx][f"{name}_generated_report"] = pred

        del model
        torch.cuda.empty_cache()

    export_data.to_excel(output_xlsx, index=False)
    with open(output_json, "w") as f:
        json.dump(all_outputs, f, indent=2)

    print(f"\nSaved reports to: {output_json}")
    print(f"Saved spreadsheet to: {output_xlsx}")

main()