# BackportCheck: An AI-Powered Decision Support System for OpenStack

**Version 2.2**

## Project Overview

BackportCheck is a machine learning-based tool designed to optimize the maintenance workflow of the OpenStack cloud computing platform. It assists maintainers by predicting the eligibility of code changes for backporting in real-time within the Gerrit review interface.

This system bridges the gap between historical data analysis and daily operational decisions by leveraging a hybrid architecture: a Gradient Boosting model (XGBoost) for high-precision risk assessment and a Large Language Model (Llama 3 via Groq) for explainable AI (XAI) justifications.


## Repository Structure

This repository contains the complete end-to-end pipeline, from raw data collection to the deployed inference engine.

*   **`backend_server/`**: The core Python Flask API. It hosts the pre-trained XGBoost model, the PCA transformation pipeline (`pca_model.pkl`), and the historical statistics engine (`stats_complete.json`).
*   **`extension/`**: The client-side Chrome extension source code. It injects the analysis interface directly into the OpenStack Gerrit UI.
*   **`data/`**: Contains the datasets and the extraction logic:
    *   **`scraper.py`**: Script to mine historical change data from OpenStack repositories.
    *   **`features_generator.py`**: Feature engineering logic converting raw text/metadata into mathematical vectors.
    *   `raw_data/`: Storage for extracted JSONL files.
    *   `processed_data/`: Storage for cleaned CSV datasets ready for training.
*   **`tools/`**: Auxiliary engineering utilities for model preparation:
    *   `train_save_pca.py`: Generates the PCA model for semantic analysis.
    *   `build_history.py`: Generates the historical statistics database.

***
## System Architecture

The solution operates on a three-tier architecture:

1.  **Data Layer**: Extraction of semantic features using CodeBERT (Microsoft) and dimensional reduction via PCA. Historical analysis of author reliability and file churn.
2.  **Predictive Layer**: An XGBoost classifier trained on historical backporting decisions, optimized via Time Series Split validation to respect temporal causality.
3.  **Explainability Layer**: Integration of a Large Language Model (Llama 3) to interpret the numerical confidence score and generate a human-readable justification for the maintainer.

## Key Features

*   **Real-Time Inference**: Instantaneous probability calculation upon visiting a Gerrit change page.
*   **Full Reproducibility**: The repository includes raw data, processing scripts, and training logic, allowing for complete verification of the scientific method.
*   **Hybrid Classification**: Distinguishes between critical bug fixes, refactoring, and feature additions to adjust risk parameters dynamically.
*   **Explainable AI (XAI)**: Provides context-aware natural language explanations for every prediction, detailing why a specific change was recommended or rejected.
*   **Adaptive Sensitivity**: Allows users to configure the decision threshold based on their specific tolerance for false positives versus false negatives.

## Installation and Usage

### 1. Backend Service Setup

Ensure Python 3.8 or higher is installed.

1.  Navigate to the server directory:
    ```bash
    cd backend_server
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration (Environment Variables):**
    To enable the Generative AI explanation features, you must configure your Groq API key securely.
    *   Create a file named `.env` inside the `backend_server/` directory.
    *   Add your API key to this file in the following format:
        ```env
        GROQ_API_KEY=gsk_your_api_key_here
        ```
    *   The application will automatically load this variable upon startup.

4.  Start the service:
    ```bash
    python app.py
    ```
    The API will initialize and listen on `http://127.0.0.1:5000`.

### 2. Chrome Extension Installation

1.  Open the Chrome browser and navigate to `chrome://extensions/`.
2.  Enable **Developer mode** in the top right corner.
3.  Click **Load unpacked**.
4.  Select the `extension/` directory from this repository.
5.  Navigate to an OpenStack Gerrit review page (e.g., `review.opendev.org`) to observe BackportCheck in action.

## Data Pipeline and Reproduction

To reproduce the training phase or update the models with new data:

1.  **Data Collection**: Use the scraper script located in the `scripts/` directory to fetch the latest changes from Gerrit.
2.  **Feature Engineering**: Run the processing script to generate the `openstack_complete.csv` file within `data/processed_data/`.
3.  **Model Training**: Execute `train_save_pca.py` and `build_history.py` to regenerate the `pca_model.pkl` and `stats_complete.json` files required by the backend.

## Technologies Used

*   **Core Logic**: Python, Flask, Pandas, NumPy.
*   **Machine Learning**: XGBoost, Scikit-learn (PCA).
*   **NLP & Embeddings**: Sentence-Transformers (CodeBERT), Regular Expressions.
*   **Generative AI**: Llama 3 (via Groq API).
*   **Frontend**: JavaScript, HTML/CSS (Manifest V3).
