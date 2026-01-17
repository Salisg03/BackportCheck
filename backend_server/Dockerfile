FROM python:3.10-slim

WORKDIR /app

# 1. System Installation
# build-essential is needed for some math/text libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Update pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Source Code & Models
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]