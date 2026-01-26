import json
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 100  # <--- FULL 100 EPOCHS (No Early Stopping)
LEARNING_RATE = 2e-5
INPUT_FILE = "openstack_data_with_diffs.jsonl"
LOG_FILE = "training_log.txt"  # <--- Output file for your professor

# ==========================================
# 1. LOGGING HELPER
# ==========================================
def log_message(message):
    """Prints to console AND appends to text file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ==========================================
# 2. SMART PREPROCESSING
# ==========================================
class SmartDiffCompressor:
    def __init__(self):
        self.patterns = [
            (re.compile(r'Signed-off-by:.*', re.IGNORECASE), ''),
            (re.compile(r'Co-authored-by:.*', re.IGNORECASE), ''),
            (re.compile(r'Change-Id:.*', re.IGNORECASE), ''),
            (re.compile(r'(Closes-Bug|Related-Bug|Bug):\s*#?(\d+)', re.IGNORECASE), r'\1: ID'),
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 'EMAIL'),
            (re.compile(r'https?://[^\s]+'), 'URL'),
        ]
        project_names = ['nova', 'neutron', 'cinder', 'glance', 'keystone', 
                        'horizon', 'swift', 'kolla', 'ansible', 'oslo', 'openstack']
        self.project_regex = re.compile(r'\b(' + '|'.join(project_names) + r')\b', re.IGNORECASE)
        self.git_headers = re.compile(r'^(diff --git|index |new file|deleted file|similarity|---|\+\+\+)')
        self.func_pattern = re.compile(r'@@.*@@\s*(.+)$')
    
    def clean_text(self, text):
        if not text: return ""
        cleaned = text
        for pattern, replacement in self.patterns:
            cleaned = pattern.sub(replacement, cleaned)
        cleaned = self.project_regex.sub('PROJECT', cleaned)
        return cleaned.strip()
    
    def compress_diff(self, diff_text, max_changes=30):
        if not diff_text: return ""
        changes = []
        current_func = None
        for line in diff_text.split('\n'):
            if self.git_headers.match(line): continue
            if line.startswith('@@'):
                func_match = self.func_pattern.search(line)
                if func_match: current_func = func_match.group(1).strip()[:50]
                continue
            if line.startswith('+') or line.startswith('-'):
                clean_line = line
                for pattern, replacement in self.patterns:
                    clean_line = pattern.sub(replacement, clean_line)
                clean_line = self.project_regex.sub('PROJECT', clean_line)
                if len(clean_line.strip()) < 3 or clean_line.strip() in ['+', '-']: continue
                changes.append({'func': current_func or 'unknown', 'line': clean_line[:100]})
        
        if len(changes) > max_changes: changes = changes[:max_changes]
        if not changes: return "[NO_CODE_CHANGES]"
        
        output = []
        for change in changes:
            output.append(f"FUNC: {change['func']}")
            output.append(change['line'])
        return '\n'.join(output)

class UnifiedDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.compressor = SmartDiffCompressor()
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        s = self.compressor.clean_text(row['subject'])
        m = self.compressor.clean_text(row['message'])
        d = self.compressor.compress_diff(row['diff_text'])
        full_text = f"{s}. {m}\n\n[DIFF]\n{d}"
        encoding = self.tokenizer(
            full_text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

# ==========================================
# 3. MODEL ARCHITECTURE (CNN)
# ==========================================
class CodeBertCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = 768
        
        # CNN
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        
        # Output
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
# 4. TRAINING LOOP
# ==========================================
def main():
    # Initialize Log File
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    
    # Load Data
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ch = json.loads(line)
                label = 0
                bc = ch.get('labels', {}).get('Backport-Candidate', {})
                if 'approved' in bc or (bc.get('all') and max([v.get('value', 0) for v in bc.get('all')]) >= 1): label = 1
                curr = ch.get('current_revision')
                rev = ch.get('revisions', {}).get(curr, {})
                records.append({
                    'subject': ch.get('subject', ""),
                    'message': rev.get('commit', {}).get('message', ""),
                    'diff_text': ch.get('diff_text', ""),
                    'label': label,
                    'created': ch.get('created', '')
                })
            except: continue
    
    df = pd.DataFrame(records)
    df['created'] = pd.to_datetime(df['created'])
    df = df.sort_values('created').reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = UnifiedDataset(train_df, tokenizer)
    test_ds = UnifiedDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = CodeBertCNN()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler optimized for 100 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Tracking
    best_f1 = 0.0
    history = {'epoch': [], 'train_loss': [], 'test_acc': [], 'test_f1': []}

    log_message(f"\nStarting 100 Epoch Run with Log File: {LOG_FILE}...")
    log_message("-" * 50)
    
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)
            
            optimizer.zero_grad()
            out = model(ids, mask)
            loss = criterion(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # TEST
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                out = model(ids, mask)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                trues.extend(batch['label'].numpy())
                
        acc = accuracy_score(trues, preds)
        _, _, f1, _ = precision_recall_fscore_support(trues, preds, average='binary')
        
        # Log Logic
        status_msg = f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f}"
        
        # SAVE BEST MODEL (Always save the winner)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_cnn_model.pth")
            status_msg += "\n >>> New Best F1! Model Saved."
        else:
            status_msg += "\n ... No improvement."
            
        log_message(status_msg)
        log_message("-" * 30)

        # HISTORY
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        history['test_f1'].append(f1)

    # SAVE FINAL STATE
    torch.save(model.state_dict(), "final_model_100.pth")
    log_message("Finished 100 Epochs. Final model saved.")

    # PLOT
    print("\nGenerating 100 Epoch Plot...")
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['epoch'], history['train_loss'], 'b-o', label='Training Loss')
        plt.title('Training Loss (100 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epoch'], history['test_acc'], 'g-o', label='Test Acc')
        plt.plot(history['epoch'], history['test_f1'], 'r--x', label='Test F1')
        plt.title('Test Performance (100 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('marathon_100_analysis.png')
        print("Analysis saved to marathon_100_analysis.png")
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    main()