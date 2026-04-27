import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
import torch
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

img_root = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/files"
excel_path = "/fs/ess/PAS3128/ruoxiyang/result/base_lora_reports.xlsx"
df_results = pd.read_excel(excel_path)
# df_results = df_results.head(10)

#Default CLIP feature extractor
#Load the pretrained Model
model_name = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(model_name)

weights=torch.load('/fs/ess/PAS3128/ruoxiyang/code/radclip/Models/RadCLIP.pth',map_location=device)
model.load_state_dict(weights)
print("Model Weights loaded")

model= model.to(device)
model.eval()

def get_image_features_clip_batch(image, model, processor, device):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", do_rescale=True).to(device)
        outputs = model.vision_model(**inputs)
        vision_features = outputs.pooler_output
        vision_projections = model.visual_projection(vision_features)
    return vision_features.detach().cpu(),vision_projections.detach().cpu()

def get_text_features(text):
    text = tokenizer(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text).to("cpu")
    return text_features

def process_image(file_path):
    # Determine file type
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Load image
    if ext in ['.jpg', '.jpeg']:
        image = Image.open(file_path)
    elif ext == '.png':
        image = Image.open(file_path)
    elif ext == '.dicom':
        dicom_image = pydicom.dcmread(file_path)
        image_array = dicom_image.pixel_array
        if len(image_array.shape) == 2:  # Single-channel
            image_array = np.stack((image_array,)*3, axis=-1)
        image = Image.fromarray(image_array)
    else:
        raise ValueError("Unsupported file format")

    return image

def get_score(text, img_proj, model, tokenizer, device):
    if not text.strip(): 
        return 0.0
    text_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        t_feat = model.get_text_features(**text_input)
    t_feat = F.normalize(t_feat, p=2, dim=-1)
    return torch.mm(img_proj, t_feat.t()).item()

gt_scores, base_scores, sft_scores = [], [], []

for idx, row in tqdm(df_results.iterrows(), total=len(df_results), desc="Scoring Reports"):
    try:
        full_path = str(row['image_path'])
        img_obj = process_image(full_path)
        
        _, img_proj = get_image_features_clip_batch(img_obj, model, processor, device)
        img_proj = F.normalize(img_proj.to(device), p=2, dim=-1)

        t_gt = str(row['ground_truth']) if pd.notna(row['ground_truth']) else ""
        t_base = str(row['Base_generated_report']) if pd.notna(row['Base_generated_report']) else ""
        t_sft = str(row['SFT (Base+LoRA)_generated_report']) if pd.notna(row['SFT (Base+LoRA)_generated_report']) else ""

        gt_scores.append(get_score(t_gt, img_proj, model, tokenizer, device))
        base_scores.append(get_score(t_base, img_proj, model, tokenizer, device))
        sft_scores.append(get_score(t_sft, img_proj, model, tokenizer, device))

    except Exception as e:
        print(f"Error processing {row.get('image_path', idx)}: {e}")
        gt_scores.append(np.nan)
        base_scores.append(np.nan)
        sft_scores.append(np.nan)

df_results['RadCLIP_Score_GT'] = gt_scores
df_results['RadCLIP_Score_Base'] = base_scores
df_results['RadCLIP_Score_SFT'] = sft_scores

metrics = {
    "Avg_GT_Score": df_results['RadCLIP_Score_GT'].mean(),
    "Avg_Base_Score": df_results['RadCLIP_Score_Base'].mean(),
    "Avg_SFT_Score": df_results['RadCLIP_Score_SFT'].mean()
}

print("\n" + "="*30)
print("FINAL METRICS:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("="*30)

output_path = "/fs/ess/PAS3128/ruoxiyang/result/base_lora_radclip_scores.xlsx"
df_results.to_excel(output_path, index=False)