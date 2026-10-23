"""Module 1 - data ingestion and preprocessing."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw_telco.csv"
CLEAN_PATH = ROOT / "data" / "cleaned_telco.csv"

# Columns where "No internet service" / "No phone service" is just a longer way of saying "No"
SERVICE_COLS = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_ORDER = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"]


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw IBM Telco CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    Steps: strip whitespace, drop duplicate IDs, convert TotalCharges to numeric
    (11 blank strings belong to brand-new customers with tenure 0), tidy category
    labels, encode Churn as 0/1 and add a TenureGroup column for EDA.

    require_target=False lets you clean a new file that has no Churn column
    (used by the dashboard's "Score new data" tab).
    """
    required = REQUIRED_COLUMNS if require_target else [c for c in REQUIRED_COLUMNS if c != "Churn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df.columns = df.columns.str.strip()
    df["customerID"] = df["customerID"].astype(str).str.strip()
    df = df.drop_duplicates(subset="customerID").reset_index(drop=True)

    total = df["TotalCharges"].astype(str).str.strip().replace("", np.nan)
    df["TotalCharges"] = pd.to_numeric(total, errors="coerce")
    new_customer = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customer, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    if "SeniorCitizen" in df.columns and pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    for col in SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1, 0: 0, 1: 1})
        if df["Churn"].isna().any():
            raise ValueError("Churn column contains values other than Yes/No")
        df["Churn"] = df["Churn"].astype(int)

    df["TenureGroup"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_ORDER).astype(str)
    return df