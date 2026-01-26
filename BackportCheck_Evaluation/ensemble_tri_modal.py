#XGBoost + CodeBERT + Llama 3 
import pandas as pd
import numpy as np
import torch
import json
import re
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
XGB_MODEL_FILE = "backport_predictor_optimized.json"
BERT_MODEL_PATH = "best_cnn_model.pth" 
DATA_CSV = "dataset_for_ai.csv"        
DATA_JSONL = "openstack_data_with_diffs.jsonl" 
LLM_RESULTS_FILE = "llm_full_results.csv"      
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512

# ==========================================
# 1. MODEL ARCHITECTURE (Must match trained CNN)
# ==========================================
class CodeBertCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = 768
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
        permuted = last_hidden_state.permute(0, 2, 1)
        conv_out = self.conv1d(permuted) 
        conv_out = self.relu(conv_out)
        pooled, _ = torch.max(conv_out, dim=2)
        logits = self.classifier(pooled)
        return logits

# ==========================================
# 2. DATA PREP HELPER
# ==========================================
class SmartDiffCompressor:
    def __init__(self):
        self.git_headers = re.compile(r'^(diff --git|index |new file|deleted file|similarity|---|\+\+\+)')
        self.func_pattern = re.compile(r'@@.*@@\s*(.+)$')

    def compress(self, diff_text, subject, message):
        if not diff_text: diff_text = ""
        changes = []
        curr_func = None
        for line in diff_text.split('\n'):
            if self.git_headers.match(line): continue
            if line.startswith('@@'):
                m = self.func_pattern.search(line)
                if m: curr_func = m.group(1).strip()[:50]
                continue
            if line.startswith('+') or line.startswith('-'):
                if len(line) > 3:
                    changes.append(f"FUNC: {curr_func} {line[:100]}")
        
        diff_str = '\n'.join(changes[:30]) # Top 30 changes
        return f"{subject}. {message}\n\n[DIFF]\n{diff_str}"

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def run_tri_ensemble():
    print("1. Loading Models...")
    
    # --- XGBOOST FIX ---
    # Load as native Booster to avoid sklearn TypeErrors
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_FILE)
    
    # Load CodeBERT (CNN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    bert_model = CodeBertCNN()
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH))
    bert_model.to(device)
    bert_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("2. Loading & Merging Data Sources...")
    
    # A. Load XGBoost Data (Base)
    df_features = pd.read_csv(DATA_CSV)
    
    # B. Load Diffs
    json_map = {}
    with open(DATA_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ch = json.loads(line)
                json_map[ch['id']] = {
                    'subject': ch.get('subject', ''),
                    'message': ch.get('revisions', {}).get(ch.get('current_revision', ''), {}).get('commit', {}).get('message', ''),
                    'diff_text': ch.get('diff_text', '')
                }
            except: continue
            
    # C. Load LLM Results
    try:
        df_llm = pd.read_csv(LLM_RESULTS_FILE)
        llm_map = dict(zip(df_llm['change_id'], df_llm['predicted']))
    except FileNotFoundError:
        print("   Warning: LLM results file not found. Skipping LLM.")
        llm_map = {}

    # D. Prepare Test Set (Last 20%)
    split_idx = int(len(df_features) * 0.8)
    df_test = df_features.iloc[split_idx:].copy()
    
    print(f"   Test Set Size: {len(df_test)}")

    # ==========================================
    # 4. GENERATE PREDICTIONS
    # ==========================================
    
    # --- XGBoost Prediction (Native) ---
    print("   > Running XGBoost...")
    drops = ['change_id', 'project', 'author_name', 'target', 'file_list', 'created_date']
    X_xgb = df_test.drop(columns=[c for c in drops if c in df_test.columns], errors='ignore')
    
    # Align Columns with Booster
    model_cols = xgb_model.feature_names
    for c in model_cols:
        if c not in X_xgb.columns: X_xgb[c] = 0
    X_xgb = X_xgb[model_cols]
    
    # Convert to DMatrix for native prediction
    dtest = xgb.DMatrix(X_xgb)
    prob_xgb = xgb_model.predict(dtest) # Returns Class 1 prob directly

    # --- CodeBERT Prediction ---
    print("   > Running CodeBERT (CNN)...")
    prob_bert = []
    compressor = SmartDiffCompressor()
    
    batch_text = []
    ids = df_test['change_id'].values
    
    for cid in ids:
        if cid in json_map:
            data = json_map[cid]
            text = compressor.compress(data['diff_text'], data['subject'], data['message'])
        else:
            text = ""
        batch_text.append(text)
        
    # Batch processing
    bs = 16
    for i in range(0, len(batch_text), bs):
        batch = batch_text[i:i+bs]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = bert_model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.softmax(logits, dim=1)[:, 1]
            prob_bert.extend(probs.cpu().numpy())
    
    prob_bert = np.array(prob_bert)

    # --- Llama 3 ---
    print("   > Aligning Llama 3...")
    prob_llm = []
    for cid in ids:
        prob_llm.append(float(llm_map.get(cid, 0.5)))
    prob_llm = np.array(prob_llm)

    # ==========================================
    # 5. FUSION & EVALUATION
    # ==========================================
    y_true = df_test['target'].values
    
    print("\n" + "="*40)
    print(" INDIVIDUAL PERFORMANCE")
    print("="*40)
    print(f"XGBoost:  {accuracy_score(y_true, [1 if p>0.5 else 0 for p in prob_xgb]):.4f}")
    print(f"CodeBERT: {accuracy_score(y_true, [1 if p>0.5 else 0 for p in prob_bert]):.4f}")
    print(f"Llama 3:  {accuracy_score(y_true, [1 if p>0.5 else 0 for p in prob_llm]):.4f}")

    print("\n" + "="*40)
    print(" TRI-MODAL ENSEMBLE RESULTS")
    print("="*40)
    
    # Weighted Average
    final_prob = (0.45 * prob_xgb) + (0.35 * prob_bert) + (0.20 * prob_llm)
    final_pred = [1 if p > 0.5 else 0 for p in final_prob]
    
    print(f"Accuracy: {accuracy_score(y_true, final_pred):.4f}")
    print(classification_report(y_true, final_pred))
    
    cm = confusion_matrix(y_true, final_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    run_tri_ensemble()