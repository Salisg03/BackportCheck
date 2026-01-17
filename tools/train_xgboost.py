import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_optimized():
    print("1. Loading Data...")
    df = pd.read_csv(r"data\processed_data\openstack_complete.csv")
    
    # Drop Identifiers
    drop_cols = ['change_id', 'project', 'author_name', 'target', 'file_list', 'created_date']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target']

    # 2. TEMPORAL SPLIT 
    split_index = int(len(df) * 0.80)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    print(f"   > Train Set: {len(X_train)} samples (Past)")
    print(f"   > Test Set:  {len(X_test)} samples (Future)")

    # Calculate Scale Weight for Class Balance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / pos_count
    print(f"   > Class Weighting: {scale_weight:.4f}")

    # 3. GRID SEARCH
    print("2. Training Model with Interaction Features...")
    
    # We stick to the params that worked best previously, but search around them
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8], 
        'learning_rate': [0.03, 0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'scale_pos_weight': [scale_weight]
    }

    xgb = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42
    )
    
    # Use TimeSeriesSplit to prevent data leakage during Cross Validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='f1', 
        verbose=0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    
    print("\n--- RESULTS ON FUTURE DATA ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Feature Importance
    print("\n--- TOP DRIVERS OF DECISIONS ---")
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(20)
    print(feat_df)
    
    # Save
    best_model.get_booster().save_model("Xgboost_optimized.json")
    print("\nSaved model to 'backport_predictor_optimized.json'")

if __name__ == "__main__":
    train_optimized()