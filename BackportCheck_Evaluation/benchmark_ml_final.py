import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

DATA_FILE = "dataset_for_ai.csv"

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "FPR": fpr,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "G-Mean": g_mean
    }

def run_benchmark():
    print("1. Loading Data...")
    df = pd.read_csv(DATA_FILE)
    drop_cols = ['change_id', 'project', 'author_name', 'target', 'file_list', 'created_date']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target']

    # Strict Temporal Split (80/20)
    split_idx = int(len(df) * 0.80)
    
    # --- FIX WAS HERE (Changed split_index to split_idx) ---
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Calculate scale_pos_weight for XGBoost (Minority class boost)
    # Check if sum is 0 to avoid division by zero (rare but possible)
    y_sum = y_train.sum()
    if y_sum > 0:
        pos_weight = (len(y_train) - y_sum) / y_sum
    else:
        pos_weight = 1.0

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # DEFINING OPTIMIZED MODELS
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, 
            max_depth=8, 
            learning_rate=0.05, 
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight, # CRITICAL FOR IMBALANCE
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, 
            max_depth=10, 
            class_weight='balanced', # CRITICAL FOR IMBALANCE
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB()
    }

    results = []
    print("2. Benchmarking...")
    
    for name, model in models.items():
        print(f"   Running {name}...")
        # XGBoost/RF don't strictly need scaling, but it's fine. LR/NB need it.
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        # Get probabilities (Handle models that might fail predict_proba slightly differently)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            probs = preds # Fallback
        
        m = calculate_metrics(y_test, preds, probs)
        m["Model"] = name
        results.append(m)

    results_df = pd.DataFrame(results)
    cols = ["Model", "Accuracy", "Precision", "Recall", "FPR", "MCC", "AUC", "G-Mean"]
    results_df = results_df[cols].sort_values("Accuracy", ascending=False)
    
    print("\n" + "="*80)
    print(" FINAL ML BENCHMARK RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv("ml_benchmark_final.csv", index=False)

if __name__ == "__main__":
    run_benchmark()