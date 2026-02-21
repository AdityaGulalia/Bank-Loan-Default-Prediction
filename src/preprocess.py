import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_and_preprocess(path, target_column):
    # 1. Load Data
    df = pd.read_csv(path)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # 2. Drop Identifiers, Constants, and Leakage
    # IDs and 'Payment Plan' (1 unique value) provide no info. 
    # 'Recoveries' etc. are 'leakage' because they are only known AFTER a default.
    to_drop = [
        'ID', 'Payment Plan', 'Accounts Delinquent', 'Batch Enrolled', 'Loan Title',
        'Recoveries', 'Collection Recovery Fee', 'Total Collection Amount', 
        'Total Received Late Fee', 'Total Received Interest'
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')

    # 3. FEATURE ENGINEERING (Calculate these BEFORE encoding)
    # Note: In your dataset, 'Home Ownership' column actually looks like Income ($80k avg)
    if "Revolving Balance" in df.columns and "Total Revolving Credit Limit" in df.columns:
        df["utilization_ratio"] = df["Revolving Balance"] / (df["Total Revolving Credit Limit"] + 1)

    if "Loan Amount" in df.columns and "Home Ownership" in df.columns:
        # Calculating Loan to Income (using Home Ownership as income proxy)
        df["loan_to_income"] = df["Loan Amount"] / (df["Home Ownership"] + 1)

    if "Interest Rate" in df.columns and "Debit to Income" in df.columns:
        df["interest_dti_interaction"] = df["Interest Rate"] * df["Debit to Income"]

    # 4. HANDLE SKEWNESS (Log Transform)
    # Dollar amounts in banking are usually skewed. Log transform helps the model.
    skewed_cols = ['Home Ownership', 'Total Current Balance', 'Revolving Balance', 'Loan Amount']
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # 5. HANDLE MISSING VALUES
    # Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 6. ENCODING
    # Use One-Hot Encoding for remaining categories (Grade, Sub Grade, etc.)
    # We ignore the ones we dropped (Batch Enrolled, Loan Title) to reduce noise.
    df = pd.get_dummies(df, drop_first=True)

    # 7. SPLIT
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found in processed data.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logging.info(f"Final feature shape: {X.shape}")

    # Stratify=y is essential to keep the 9% default rate consistent in both sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)