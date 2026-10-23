"""Modules 3 & 4 - Logistic Regression, evaluation, risk bands, explanations."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL

LOW_CUTOFF = 0.30
HIGH_CUTOFF = 0.60
RISK_LABELS = ["Low", "Medium", "High"]


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Preprocessing + Logistic Regression in one object (no train/test leakage)."""
    prep = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)]
    )
    # class_weight="balanced" because only ~27% of customers churn
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    return Pipeline([("prep", prep), ("clf", clf)])


def risk_band(p, low: float = LOW_CUTOFF, high: float = HIGH_CUTOFF):
    """Map churn probability to Low / Medium / High. Works for a scalar or an array."""
    if low >= high:
        raise ValueError("low cutoff must be smaller than high cutoff")
    arr = np.asarray(p, dtype=float)
    out = np.select([arr < low, arr < high], RISK_LABELS[:2], default=RISK_LABELS[2])
    return str(out) if out.ndim == 0 else out


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Classification metrics at a given decision threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "baseline_accuracy": 1 - y_true.mean(),   # accuracy of "nobody churns"
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def pretty_feature(name: str) -> str:
    name = name.split("__", 1)[1]
    return name if name in NUMERIC else name.replace("_", " = ", 1)


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Logistic-regression coefficients as a tidy table (positive = higher churn odds).

    Numeric features are standardised, so their coefficients are 'per 1 std-dev'.
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": [pretty_feature(n) for n in names], "coefficient": coef})
    out["odds_ratio"] = np.exp(out["coefficient"])
    out["direction"] = np.where(out["coefficient"] > 0, "raises churn risk", "lowers churn risk")
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def explain_customer(pipe: Pipeline, row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Per-customer contribution of each active feature to the log-odds of churn."""
    X = pipe.named_steps["prep"].transform(row[FEATURES].iloc[[0]])[0]
    names = pipe.named_steps["prep"].get_feature_names_out()
    contrib = X * pipe.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": [pretty_feature(n) for n in names], "contribution": contrib})
    out = out[out["contribution"].abs() > 1e-9]
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top_n)
    return out.reset_index(drop=True)


def recommend_actions(row: pd.Series) -> list:
    """Simple rule-based retention ideas keyed to the factors associated with churn."""
    actions = []
    if row["Contract"] == "Month-to-month":
        actions.append("Offer a discount for moving to a 1- or 2-year contract.")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        actions.append("Offer a free trial of tech support / online security bundle.")
    if row["PaymentMethod"] == "Electronic check":
        actions.append("Encourage auto-pay (card or bank transfer) with a small bill credit.")
    if row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] > 80:
        actions.append("Review pricing - high-bill fibre customers churn more; consider a loyalty plan.")
    if row["tenure"] <= 12:
        actions.append("Schedule an onboarding / check-in call in the first year.")
    return actions or ["No specific red flags - keep the customer on a standard engagement plan."]
