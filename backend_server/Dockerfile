FROM python:3.10-slim
WORKDIR /app
# 1. System Installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Update pip and base tools (To avoid typing-extensions conflict)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel typing-extensions

# 3. Install PyTorch CPU (Secure >= 2.6.0)
# we Do it ALONE to prevent pip from getting confused with the rest
RUN pip install --no-cache-dir "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

# 4. Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. AI Model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('microsoft/codebert-base'); \
    model.save('/app/model_cache')"

# 6. Source Code
COPY . .
# Expose the Flask port
EXPOSE 5000
CMD ["python", "app.py"]