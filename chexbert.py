import pandas as pd
import os
import torch
import types
import numpy as np
import f1chexbert
from tqdm import tqdm
from f1chexbert import F1CheXbert

def patched_tokenize(impressions, tokenizer):
    res = []
    for imp in impressions:
        tokenized_imp = tokenizer.tokenize(imp)
        if not tokenized_imp: 
             res.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
             continue
        ids = tokenizer.convert_tokens_to_ids(tokenized_imp)
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        res.append(ids)
    return res

f1chexbert.f1chexbert.tokenize = patched_tokenize

report = "/fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/report_results3/all.csv"
df = pd.read_csv(report)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_probs(self, report):
    self.model.eval()
    if pd.isna(report) or str(report).strip() == "":
        return [{"pos_prob": 0.0, "uncertain_prob": 0.0}] * 14
    
    impressions = pd.Series([str(report)])
    out_tokens = patched_tokenize(impressions, self.tokenizer)
    batch = torch.LongTensor([o for o in out_tokens])
    src_len = [len(o) for o in out_tokens]
    
    attn_mask = f1chexbert.f1chexbert.generate_attention_masks(batch, src_len, self.device)    
    
    with torch.no_grad():
        out = self.model(batch.to(self.device), attn_mask)
    
    res = []
    for j in range(14):
        softmax_res = torch.softmax(out[j], dim=1)
        
        if j < 13:
            p = softmax_res[0, 1].item()
            u = softmax_res[0, 3].item()
        else:
            p = softmax_res[0, 1].item()
            u = 0.0
            
        res.append({
            "pos_prob": p,
            "uncertain_prob": u
        })
    return res

classifier = F1CheXbert(device=device)
classifier.get_probs = types.MethodType(get_probs, classifier)

if not hasattr(classifier.tokenizer, 'encode_plus'):
    classifier.tokenizer.encode_plus = classifier.tokenizer._encode_plus
    
label_names = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

def get_chexbert_labels(reports):
    results = []
    for report in tqdm(reports, desc="Labeling reports"):
        if pd.isna(report) or str(report).strip() == "":
            results.append([0] * 14)
        else:
            labels = classifier.get_label(str(report), mode="rrg")
            results.append(labels)
    return results

gt_labels = get_chexbert_labels(df['ground_truth'].tolist())
mg_labels = get_chexbert_labels(df['medgemma_15'].tolist())

gt_arr = np.array(gt_labels)
mg_arr = np.array(mg_labels)

def compare_labels(gt, mg):
    if gt == 1 and mg == 1:
        return "agree"
    elif gt == 0 and mg == 0:
        return "N/A"
    else:
        return "disagree"

structured_data = []

for i in tqdm(range(len(df)), desc="Processing final table"):
    current_filename = df.iloc[i]['filename']
    gt_report = df.iloc[i]['ground_truth']
    mg_report = df.iloc[i]['medgemma_15']
    
    gt_probs = classifier.get_probs(df.iloc[i]['ground_truth'])
    mg_probs = classifier.get_probs(df.iloc[i]['medgemma_15'])
    
    row_gt = {"file_name": current_filename, "type": "ground truth"}
    for j, label in enumerate(label_names):
        row_gt[label] = gt_arr[i, j]
        row_gt[f"{label}_status"] = ""
        row_gt[f"{label}_pos_prob"] = round(gt_probs[j]['pos_prob'], 4)
        row_gt[f"{label}_uncert_prob"] = round(gt_probs[j]['uncertain_prob'], 4)
    structured_data.append(row_gt)

    row_mg = {"file_name": current_filename, "type": "medgemma1.5"}
    for j, label in enumerate(label_names):
        row_mg[label] = mg_arr[i, j]
        row_mg[f"{label}_status"] = compare_labels(gt_arr[i, j], mg_arr[i, j])
        row_mg[f"{label}_pos_prob"] = round(mg_probs[j]['pos_prob'], 4)
        row_mg[f"{label}_uncert_prob"] = round(mg_probs[j]['uncertain_prob'], 4)
    structured_data.append(row_mg)

final_df = pd.DataFrame(structured_data)

ordered_cols = ["file_name", "type"]
for label in label_names:
    ordered_cols.append(label)
    ordered_cols.append(f"{label}_status")
    ordered_cols.append(f"{label}_pos_prob")    
    ordered_cols.append(f"{label}_uncert_prob")

final_df = pd.DataFrame(structured_data)
final_df = final_df[ordered_cols]

output_path = "/fs/ess/PAS3128/ruoxiyang/result/chexbert100_eval.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"Finished!")