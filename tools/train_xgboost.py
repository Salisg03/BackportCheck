import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# 1. DATA LOADING & PREPROCESSING

INPUT_FILE = r"data/processed_data/openstack_complete.csv"
print(f"[INFO] Loading dataset from: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    # Fallback for local testing
    df = pd.read_csv("openstack_features_enhanced_v2.csv")

# Exclude metadata and non-numeric columns from the feature set
metadata_cols = ['project', 'change_id', 'target', 'full_text_dl']
y = df['target']
X = df.drop(columns=metadata_cols)

# One-Hot Encoding for categorical NLP analysis features
X = pd.get_dummies(X, columns=['11_nlp_change_type'], drop_first=True)
X = X.astype(float)

# Calculate class weight to address dataset imbalance
# (Backport candidates are significantly fewer than regular commits)
count_neg = (y == 0).sum()
count_pos = (y == 1).sum()
scale_weight = count_neg / count_pos
print(f"[INFO] Class Imbalance Ratio: 1:{scale_weight:.2f}")


# 2. HYPERPARAMETER SEARCH SPACE

param_grid = {
    'n_estimators': [100, 200, 300, 500, 700],        # Number of boosting rounds
    'max_depth': [3, 4, 5, 6, 8, 10],                 # Max tree depth (controls overfitting)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],          # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],                     # Row sampling ratio per tree
    'colsample_bytree': [0.6, 0.8, 1.0],              # Feature sampling ratio per tree
    'gamma': [0, 0.1, 0.5, 1],                        # Minimum loss reduction for partition
    'scale_pos_weight': [scale_weight, 1, scale_weight * 1.2] # Balancing strategies
}

# Base Classifier
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='hist', # Faster training algorithm
    random_state=42
)

# 3. HYPERPARAMETER OPTIMIZATION (RandomizedSearch)
print("\n[INFO] Starting Hyperparameter Optimization...")

# Use TimeSeriesSplit to respect temporal order and prevent data leakage
tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=50,                # Number of parameter settings sampled
    scoring='accuracy',       # Optimization metric
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X, y)

print(f"\n[SUCCESS] Best Parameters: {search.best_params_}")
best_model = search.best_estimator_

# 4. THRESHOLD MOVING

# Standard threshold (0.5) is rarely optimal for imbalanced classes.
# We iterate to find the decision boundary that maximizes Accuracy.
print("\n[INFO] Optimizing Decision Threshold...")

# Get probability estimates for the positive class
probs = best_model.predict_proba(X)[:, 1]

best_threshold = 0.5
best_acc = 0

thresholds = np.arange(0.2, 0.7, 0.01)
for thresh in thresholds:
    preds_custom = (probs >= thresh).astype(int)
    acc = accuracy_score(y, preds_custom)
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

print(f"[RESULT] Optimal Threshold: {best_threshold:.2f}")
print(f"[RESULT] Max Potential Accuracy: {best_acc:.4f}")

# 5. FINAL VALIDATION
print("\n[INFO] Running Final Time-Series Validation...")

aucs, f1s, accs = [], [], []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train on past, predict on future folds
    best_model.fit(X_train, y_train)
    
    # Inference
    probs_test = best_model.predict_proba(X_test)[:, 1]
    
    # Apply optimal threshold
    preds_test = (probs_test >= best_threshold).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, probs_test)
    f1 = f1_score(y_test, preds_test)
    acc = accuracy_score(y_test, preds_test)
    
    aucs.append(auc)
    f1s.append(f1)
    accs.append(acc)
    
    print(f"   Fold {fold+1}: Accuracy={acc:.4f} | AUC={auc:.4f}")

print(f"AVERAGE ACCURACY : {np.mean(accs):.4f}")
print(f"AVERAGE AUC      : {np.mean(aucs):.4f}")

# 6. ARTIFACT EXPORT

# Retrain on full dataset for production deployment
best_model.fit(X, y) 
booster = best_model.get_booster()

# Save model structure and weights
booster.save_model("xgboost_optimized.json")

# Save optimal threshold for the Inference Engine
with open("threshold.txt", "w") as f:
    f.write(str(best_threshold))

print(f"\n[SUCCESS] Model saved to 'xgboost_optimized.json'.")
print(f"[SUCCESS] Threshold ({best_threshold}) saved to 'threshold.txt'.")