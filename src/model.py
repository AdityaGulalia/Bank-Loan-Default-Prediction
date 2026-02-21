from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier

def get_random_forest(n_estimators=200):
    return RandomForestClassifier(
        n_estimators=1000,        # More trees to capture the rare 'Default' cases
        max_depth=7,            # Keep it shallow to prevent overfitting the majority class
        min_samples_leaf=100,     # Ensure each leaf has enough data
        class_weight="balanced_subsample", # Harder focus on minority class
        random_state=42,
        n_jobs=-1
    )
    
from xgboost import XGBClassifier

def get_model():
    return XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        scale_pos_weight=10, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def save_artifact(model, feature_names, folder="models"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    artifact = {
        "model": model,
        "features": list(feature_names)
    }

    joblib.dump(artifact, f"{folder}/bank_loan_model.pkl")
    print(f"Model saved to {folder}/bank_loan_model.pkl")