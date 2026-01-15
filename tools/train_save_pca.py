import json
import pickle
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


INPUT_FILE = r"data\raw_data\openstack_all_backport_usage.jsonl" 
OUTPUT_PCA = "pca_model.pkl"

print("1. Chargement des messages...")
messages = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                change = json.loads(line)
                revs = change.get('revisions', {})
                target_rev = next((r for r in revs.values() if r.get('_number') == 1), list(revs.values())[0])
                msg = target_rev.get('commit', {}).get('message', '')
                clean_lines = [l for l in msg.split('\n') if not re.match(r'^(Change-Id|Signed-off-by):', l)]
                messages.append("\n".join(clean_lines).strip())
            except: continue
except FileNotFoundError:
    print(f"ERREUR: Fichier {INPUT_FILE} introuvable.")
    exit()

print(f"2. Encodage de {len(messages)} messages (CodeBERT)")
model = SentenceTransformer('microsoft/codebert-base')
embeddings = model.encode(messages, batch_size=32, show_progress_bar=True)

print("3. PCA (768 -> 15)...")
pca = PCA(n_components=15)
pca.fit(embeddings)

with open(OUTPUT_PCA, 'wb') as f:
    pickle.dump(pca, f)
print("pca_model.pkl généré.")