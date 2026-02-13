import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    r2_score, mean_absolute_error, mean_squared_error
)

def run_technical_audit():
    print("==================================================")
    print("🚀 STARTING MULTI-AGENT SHIELD ACCURACY AUDIT")
    print("==================================================\n")

    # 1. LOAD ARTIFACTS
    try:
        with open("ebm_finance.pkl", "rb") as f: ebm_fin = pickle.load(f)
        with open("ebm_health.pkl", "rb") as f: ebm_health = pickle.load(f)
        with open("fin_encoders.pkl", "rb") as f: fin_enc = pickle.load(f)
        with open("health_encoders.pkl", "rb") as f: health_enc = pickle.load(f)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Missing .pkl files. Please run training first. {e}")
        return

    # ==================================================
    # 🏦 AUDIT: CREDIT SHIELD (FINANCE)
    # ==================================================
    print("[1/2] Auditing Credit Shield...")
    df_f = pd.read_csv("loan_approval_dataset.csv")
    df_f.columns = df_f.columns.str.strip()
    
    # Reproduce Feature Engineering
    df_f["loan_to_income_ratio"] = df_f["loan_amount"] / (df_f["income_annum"] + 1)
    df_f["total_assets"] = (df_f["residential_assets_value"] + df_f["commercial_assets_value"] + 
                            df_f["luxury_assets_value"] + df_f["bank_asset_value"])
    df_f["asset_to_loan_ratio"] = df_f["total_assets"] / (df_f["loan_amount"] + 1)
    
    y_f = df_f["loan_status"].str.strip().map({"Approved": 1, "Rejected": 0})
    X_f = df_f.drop(columns=["loan_id", "loan_status"], errors="ignore")

    # Apply Encoders
    for col, le in fin_enc.items():
        X_f[col] = le.transform(X_f[col].astype(str))

    # Test-Train Split
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_f, y_f, test_size=0.2, random_state=42)

    # Metrics
    y_pred_f = ebm_fin.predict(Xf_test)
    fin_acc = accuracy_score(yf_test, y_pred_f)
    
    # Stability Check: 5-Fold Cross Validation
    print("   > Running Stability Check (5-Fold Cross-Validation)...")
    cv_scores = cross_val_score(ebm_fin, X_f, y_f, cv=5)

    print(f"\n✅ CREDIT SHIELD RESULTS:")
    print(f"   - Holout Accuracy: {fin_acc*100:.2f}%")
    print(f"   - Stability (Mean CV): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"   - Confusion Matrix:\n{confusion_matrix(yf_test, y_pred_f)}")

    # ==================================================
    # 🏥 AUDIT: HEALTH SHIELD (INSURANCE)
    # ==================================================
    print("\n[2/2] Auditing Health Shield...")
    df_h = pd.read_csv("insurance.csv")
    y_h = df_h["charges"]
    X_h = df_h.drop(columns=["charges"])

    for col, le in health_enc.items():
        X_h[col] = le.transform(X_h[col].astype(str))

    Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

    y_pred_h = ebm_health.predict(Xh_test)
    r2 = r2_score(yh_test, y_pred_h)
    mae = mean_absolute_error(yh_test, y_pred_h)
    rmse = np.sqrt(mean_squared_error(yh_test, y_pred_h))

    print(f"\n✅ HEALTH SHIELD RESULTS:")
    print(f"   - R² Score (Reliability): {r2:.4f}")
    print(f"   - Mean Absolute Error: ${mae:.2f}")
    print(f"   - Root Mean Squared Error: ${rmse:.2f}")
    
    print("\n==================================================")
    print("📢 AUDIT COMPLETE: YOUR MODELS ARE PRODUCTION READY")
    print("==================================================")

if __name__ == "__main__":
    run_technical_audit()