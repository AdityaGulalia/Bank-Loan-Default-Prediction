from src.preprocess import load_and_preprocess
from src.model import get_model, train_model
from src.evaluate import perform_evaluation
import pandas as pd
from imblearn.over_sampling import SMOTE

def train_pipeline(data_path, target_column):
    # 1. Preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path, target_column)

    # 2. Initialize model
    model = get_model()

    # 3. Train
    trained_model = train_model(model, X_train, y_train)

    # 4. Evaluate
    perform_evaluation(trained_model, X_test, y_test, threshold=0.3)
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    importance = pd.Series(trained_model.feature_importances_, index=X_train.columns)
    print(importance.sort_values(ascending=False).head(10))

    return trained_model, X_train.columns

