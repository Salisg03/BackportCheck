import pandas as pd
import requests
import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from tqdm import tqdm

DATA_FILE = "dataset_for_ai.csv"
RAW_FILE = "openstack_all_backport_usage.jsonl"
OLLAMA_API = "http://localhost:11434/api/generate"
# Only testing Few-Shot because Zero-Shot is known to be bad
MODELS = ["llama3:latest", "gemma2:latest", "mistral:latest", "phi3:latest"]

# THE GOOD PROMPT
REAL_WORLD_PROMPT = """You are a strict OpenStack Release Manager. 
Your job is to REJECT changes unless they are critical bug fixes or necessary updates.

TRAINING DATA:
1. Subject: "ansible-lint: fix unnamed-task" -> REJECT (Style fix)
2. Subject: "Add TLS support" -> REJECT (New Feature)
3. Subject: "Correct lock path" -> REJECT (Minor follow-up)
4. Subject: "Add Debian 12 setup" -> ACCEPT (OS Compatibility)
5. Subject: "Support pagination for list API" -> ACCEPT (Fixes broken API)
6. Subject: "Ensure services stay disabled" -> ACCEPT (Fixes regression)

TASK:
Subject: "{subject}"
Message: "{message}"

Instructions:
1. Linter/Style/Typo -> NO
2. New Feature (unless OS compat) -> NO
3. Fixes Crash/Breakage/Failure -> YES

Respond in JSON: {{ "decision": "YES" or "NO" }}
"""

class TextCleaner:
    def __init__(self):
        self.patterns = [(re.compile(r'Change-Id:.*', re.IGNORECASE), '')]
    def clean(self, text):
        if not text: return ""
        cleaned = text
        for p, r in self.patterns: cleaned = p.sub(r, cleaned)
        return cleaned.strip()

def get_llm_pred(model, subject, message):
    cleaner = TextCleaner()
    prompt = REAL_WORLD_PROMPT.format(subject=cleaner.clean(subject), message=cleaner.clean(message)[:400])
    
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.0, "num_predict": 50}}
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=25)
        if resp.status_code == 200:
            js = json.loads(resp.json()['response'])
            return 1 if js.get('decision') == 'YES' else 0
    except: return 0
    return 0

def calc_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp+tn)>0 else 0
    g_mean = np.sqrt((tp/(tp+fn))*(tn/(tn+fp)))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred),
        "G-Mean": g_mean
    }

def run():
    print("Loading Data...")
    df_labels = pd.read_csv(DATA_FILE)
    split_idx = int(len(df_labels) * 0.8)
    test_df = df_labels.iloc[split_idx:].copy()
    y_true = test_df['target'].values
    
    id_to_text = {}
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ch = json.loads(line)
                curr = ch.get('current_revision')
                msg = ch['revisions'][curr]['commit']['message']
                id_to_text[ch['id']] = {'s': ch['subject'], 'm': msg}
            except: continue

    results = []
    for model in MODELS:
        print(f"Testing {model} (Few-Shot)...")
        preds = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            cid = row['change_id']
            if cid in id_to_text:
                p = get_llm_pred(model, id_to_text[cid]['s'], id_to_text[cid]['m'])
            else: p = 0
            preds.append(p)
        
        m = calc_metrics(y_true, preds)
        m['Model'] = f"{model} (Few-Shot)"
        results.append(m)

    res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print("\n" + "="*80)
    print(" FINAL LLM BENCHMARK RESULTS")
    print("="*80)
    print(res_df.to_string(index=False))
    res_df.to_csv("llm_benchmark_final.csv", index=False)

if __name__ == "__main__":
    run()