import pandas as pd
import numpy as np


# -----------------------------
# Load original data
# -----------------------------
df = pd.read_csv("synthetic_loan_data.csv")

np.random.seed(42)


# -----------------------------
# 1. Add Noise to Numeric Columns
# -----------------------------

num_cols = [
    "income_annum",
    "loan_amount",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

for col in num_cols:
    if col in df.columns:
        noise = np.random.normal(0, df[col].std() * 0.1, size=len(df))
        df[col] = df[col] + noise


# -----------------------------
# 2. Add Random Missing Values (5%)
# -----------------------------

for col in num_cols:
    if col in df.columns:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan


# -----------------------------
# 3. Add Outliers (2%)
# -----------------------------

outlier_count = int(0.02 * len(df))

for col in ["income_annum", "loan_amount"]:
    if col in df.columns:

        idx = np.random.choice(df.index, outlier_count)

        df.loc[idx, col] = df[col].mean() * np.random.randint(
            5, 10, size=outlier_count
        )


# -----------------------------
# 4. Add Decision Noise (Label Noise)
# -----------------------------

# Flip 8% of decisions randomly
flip_mask = np.random.rand(len(df)) < 0.08

df.loc[flip_mask, "loan_status"] = df.loc[
    flip_mask, "loan_status"
].map({
    "Approved": "Rejected",
    "Rejected": "Approved"
})


# -----------------------------
# 5. Make CIBIL Slightly Unreliable
# -----------------------------

if "cibil_score" in df.columns:

    df["cibil_score"] = df["cibil_score"] + np.random.normal(
        0, 25, len(df)
    )

    df["cibil_score"] = df["cibil_score"].clip(300, 900)


# -----------------------------
# Save New Dataset
# -----------------------------

df.to_csv("realistic_loan_data.csv", index=False)

print("SUCCESS: realistic_loan_data.csv created")
