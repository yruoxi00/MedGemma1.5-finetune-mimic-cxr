import json
import os
from datasets import Dataset, Features, Image, Value

json_path = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/mimic_annotation.json"
img_root = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/files"
save_path = "/fs/ess/PAS3128/ruoxiyang/data/processed_mimic_dataset_all"

with open(json_path, 'r') as f:
    full_data = json.load(f)

# train_raw = full_data['train'][:1000]
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

# dataset.save_to_disk(save_path)

print(f"Total samples: {len(dataset)}")
# print(dataset[0])