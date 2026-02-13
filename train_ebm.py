import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor
)

# ---------------------------------------------------
# Encode categorical columns
# ---------------------------------------------------
def encode_categorical(df):
    encoders = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    
    print(f"  > Encoding {len(cat_cols)} categorical columns...")
    for col in tqdm(cat_cols, desc="Encoding", leave=False):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

# ---------------------------------------------------
# Clean Missing Values
# ---------------------------------------------------
def clean_missing(df):
    # Numeric → median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical → Unknown
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")
    return df

# ---------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------
def train_models():
    # ==================================================
    # 1️⃣ FINANCE MODEL — Loan Approval
    # ==================================================
    print("\n[1/2] Training Financial Model...")
    
    try:
        df_f = pd.read_csv("loan_data.csv")
        df_f.columns = df_f.columns.str.strip()
        df_f = df_f.replace("", np.nan)

        # Normalize labels
        df_f["loan_status"] = df_f["loan_status"].astype(str).str.strip().str.capitalize()
        df_f.loc[~df_f["loan_status"].isin(["Approved", "Rejected"]), "loan_status"] = "Rejected"

        # Feature Engineering
        df_f["loan_to_income_ratio"] = df_f["loan_amount"] / (df_f["income_annum"] + 1)
        df_f["total_assets"] = (df_f["residential_assets_value"] + df_f["commercial_assets_value"] + 
                               df_f["luxury_assets_value"] + df_f["bank_asset_value"])
        
        y_f = df_f["loan_status"].map({"Approved": 1, "Rejected": 0})
        X_f = df_f.drop(columns=["loan_id", "loan_status"], errors="ignore")
        
        X_f = clean_missing(X_f)
        X_f, fin_encoders = encode_categorical(X_f)

        Xf_train, Xf_test, yf_train, yf_test = train_test_split(
            X_f, y_f, test_size=0.2, random_state=42, stratify=y_f
        )

        ebm_fin = ExplainableBoostingClassifier(
            interactions=10, # Slightly reduced for speed
            n_jobs=-1,      # Use all CPU cores
            random_state=42
        )
        
        print("  > Fitting Finance EBM...")
        ebm_fin.fit(Xf_train, yf_train)
        print(f"  > Financial Accuracy: {accuracy_score(yf_test, ebm_fin.predict(Xf_test)):.4f}")

    except FileNotFoundError:
        print("  ! Error: realistic_loan_data.csv not found.")
        return

    # ==================================================
    # 2️⃣ HEALTH MODEL — Claim Prediction
    # ==================================================
    print("\n[2/2] Training Health Model...")
    
    try:
        df_h = pd.read_csv("hi.csv")
        df_h.columns = df_h.columns.str.strip()
        
        if "claim" not in df_h.columns:
            raise ValueError(f"'claim' missing. Found: {df_h.columns.tolist()}")

        y_h = df_h["claim"]
        X_h = df_h.drop(columns=["claim"])

        # FIX: Safe Numeric Conversion
        for col in X_h.columns:
            # Attempt numeric conversion, invalid becomes NaN
            converted = pd.to_numeric(X_h[col], errors='coerce')
            # If the column is mostly numbers, keep the conversion
            if converted.notna().sum() > (len(X_h) * 0.5):
                X_h[col] = converted

        X_h = clean_missing(X_h)
        X_h, health_encoders = encode_categorical(X_h)

        Xh_train, Xh_test, yh_train, yh_test = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42
        )

        # OPTIMIZED: Reduced interactions and multi-core support
        ebm_health = ExplainableBoostingRegressor(
            interactions=5,  # Prevents long hangs
            n_jobs=-1,       # High performance
            max_bins=256,
            learning_rate=0.01,
            random_state=42
        )

        print("  > Fitting Health EBM (this may take a minute)...")
        ebm_health.fit(Xh_train, yh_train)
        
        r2 = r2_score(yh_test, ebm_health.predict(Xh_test))
        print(f"  > Health R2 Score: {r2:.4f}")

    except FileNotFoundError:
        print("  ! Error: hi.csv not found.")
        return

    # ==================================================
    # SAVE MODELS
    # ==================================================
    print("\nSaving Models...")
    files = {
        "ebm_finance.pkl": ebm_fin,
        "ebm_health.pkl": ebm_health,
        "fin_encoders.pkl": fin_encoders,
        "health_encoders.pkl": health_encoders
    }
    
    for filename, obj in tqdm(files.items(), desc="Saving"):
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    print("\nSUCCESS: Training Complete!")

if __name__ == "__main__":
    train_models()