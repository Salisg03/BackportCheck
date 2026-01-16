import os
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
dest_path = os.path.join(current_dir, '..', 'backend_server', 'model_cache')

print(f"⬇ Downloading CodeBERT to: {dest_path}")

# Download
model = SentenceTransformer('microsoft/codebert-base')
model.save(dest_path)

print("Model downloaded successfully.")
print("The application will now use this local version (faster, no network errors).")