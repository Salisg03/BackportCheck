import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = r"data/raw_data/openstack_all_backport_usage.jsonl"
MODEL_SAVE_PATH = r"final_extention/backend_server/deepjit_model.pth"
VOCAB_SAVE_PATH = r"final_extention/backend_server/vocab.json"
MAX_VOCAB_SIZE = 10000
MAX_LEN_MSG = 50   # On garde les 50 premiers mots du message
MAX_LEN_CODE = 100 # On garde les 100 premiers mots des chemins de fichiers
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

# --- 1. PRÉPARATION DES DONNÉES ---

def tokenizer(text):
    """Nettoyage simple et tokenisation"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\/\._-]', '', text) # On garde les chars de fichiers
    return text.split()

print("1. Chargement et nettoyage des données...")
dataset = []
all_words = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
            # Target
            labels = entry.get('labels', {}).get('Backport-Candidate', {})
            # Logique simplifiée pour l'exemple (à adapter selon ta rigueur)
            is_backport = 0
            if 'all' in labels:
                votes = [int(v.get('value', 0)) for v in labels['all'] if 'value' in v]
                if votes and max(votes) > 0: is_backport = 1
            elif int(labels.get('value', 0)) > 0: is_backport = 1
            
            if entry.get('status') != 'MERGED': continue

            # Extraction Message
            rev = list(entry.get('revisions', {}).values())[0]
            msg = rev.get('commit', {}).get('message', '')
            
            # Extraction "Code" (Ici : chemins de fichiers)
            files = list(rev.get('files', {}).keys())
            code_proxy = " ".join([f for f in files if f != "/COMMIT_MSG"])
            
            tokens_msg = tokenizer(msg)
            tokens_code = tokenizer(code_proxy)
            
            dataset.append({
                'msg': tokens_msg,
                'code': tokens_code,
                'label': is_backport
            })
            all_words.extend(tokens_msg + tokens_code)
        except: continue

# Construction du Vocabulaire
print("2. Construction du vocabulaire...")
word_counts = Counter(all_words)
common_words = word_counts.most_common(MAX_VOCAB_SIZE - 2)
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in common_words:
    vocab[word] = len(vocab)

# Sauvegarde du vocabulaire pour l'app
with open(VOCAB_SAVE_PATH, 'w') as f:
    json.dump(vocab, f)

# Conversion en tenseurs
def encode_text(tokens, max_len):
    encoded = [vocab.get(w, 1) for w in tokens[:max_len]]
    padding = [0] * (max_len - len(encoded))
    return encoded + padding

X_msg = torch.tensor([encode_text(d['msg'], MAX_LEN_MSG) for d in dataset])
X_code = torch.tensor([encode_text(d['code'], MAX_LEN_CODE) for d in dataset])
y = torch.tensor([d['label'] for d in dataset], dtype=torch.float32)

# Split Train/Test
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

class OpenStackDataset(Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        real_i = self.idx[i]
        return X_msg[real_i], X_code[real_i], y[real_i]

train_loader = DataLoader(OpenStackDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(OpenStackDataset(val_idx), batch_size=BATCH_SIZE)

# --- 2. ARCHITECTURE DU MODÈLE (DeepJIT-Like) ---

class DeepJITModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super(DeepJITModel, self).__init__()
        
        # Couche d'embedding (transforme les mots en vecteurs)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Branche Message (CNN)
        self.conv_msg = nn.Conv1d(embed_dim, 32, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Branche Code (CNN)
        self.conv_code = nn.Conv1d(embed_dim, 32, kernel_size=3)
        
        # Couche Dense de fusion
        self.fc1 = nn.Linear(32 + 32, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, msg, code):
        # Embeddings: [Batch, Len] -> [Batch, Len, Dim] -> [Batch, Dim, Len] (pour Conv1d)
        emb_msg = self.embedding(msg).permute(0, 2, 1)
        emb_code = self.embedding(code).permute(0, 2, 1)
        
        # Convolution + Pooling
        feat_msg = self.pool(torch.relu(self.conv_msg(emb_msg))).squeeze(2)
        feat_code = self.pool(torch.relu(self.conv_code(emb_code))).squeeze(2)
        
        # Fusion
        combined = torch.cat((feat_msg, feat_code), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x)).squeeze(1)

# --- 3. ENTRAÎNEMENT ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"3. Lancement de l'entraînement sur {device}...")

model = DeepJITModel(len(vocab)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for b_msg, b_code, b_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        b_msg, b_code, b_y = b_msg.to(device), b_code.to(device), b_y.to(device)
        
        optimizer.zero_grad()
        prediction = model(b_msg, b_code)
        loss = criterion(prediction, b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_msg, b_code, b_y in val_loader:
            b_msg, b_code, b_y = b_msg.to(device), b_code.to(device), b_y.to(device)
            preds = model(b_msg, b_code)
            predicted_class = (preds > 0.5).float()
            correct += (predicted_class == b_y).sum().item()
            total += b_y.size(0)
            
    val_acc = correct / total
    print(f"   Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

# Sauvegarde
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Modèle DeepJIT sauvegardé sous '{MODEL_SAVE_PATH}'")