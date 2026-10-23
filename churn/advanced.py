"""Advanced analysis: feature engineering, model comparison, and retention ROI."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import model

SERVICE_FLAGS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ENG_NUMERIC = model.NUMERIC + ["NumServices", "AvgMonthlySpend", "BillVsAverage"]
ENG_CATEGORICAL = model.CATEGORICAL + ["IsAutoPay", "IsNewCustomer"]
ENG_FEATURES = ENG_NUMERIC + ENG_CATEGORICAL


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-meaningful columns derived from existing ones.

    NumServices     - how many add-on services the customer uses (stickiness)
    AvgMonthlySpend - lifetime spend / months (historical average bill)
    BillVsAverage   - current bill minus historical average (proxy for a recent price rise)
    IsAutoPay       - pays by automatic card / bank transfer
    IsNewCustomer   - tenure of 6 months or less
    """
    out = df.copy()
    flags = (out[SERVICE_FLAGS] == "Yes").sum(axis=1) + (out["InternetService"] != "No").astype(int)
    out["NumServices"] = flags
    out["AvgMonthlySpend"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["BillVsAverage"] = out["MonthlyCharges"] - out["AvgMonthlySpend"]
    out["IsAutoPay"] = np.where(out["PaymentMethod"].str.contains("automatic"), "Yes", "No")
    out["IsNewCustomer"] = np.where(out["tenure"] <= 6, "Yes", "No")
    return out


def make_pipeline(estimator, numeric, categorical) -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    return Pipeline([("prep", prep), ("clf", estimator)])


def compare_models(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """5-fold stratified cross-validation of several models on the same folds."""
    d = add_engineered_features(df)
    y = d["Churn"]
    lr = lambda **kw: LogisticRegression(class_weight="balanced", max_iter=2000, **kw)
    tuned = GridSearchCV(lr(), {"C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=3)
    candidates = {
        "Logistic Regression (baseline)": (make_pipeline(lr(), model.NUMERIC, model.CATEGORICAL), model.FEATURES),
        "Logistic Regression + engineered features": (make_pipeline(lr(), ENG_NUMERIC, ENG_CATEGORICAL), ENG_FEATURES),
        "Logistic Regression + engineered, tuned C": (make_pipeline(tuned, ENG_NUMERIC, ENG_CATEGORICAL), ENG_FEATURES),
        "Random Forest": (make_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed), ENG_NUMERIC, ENG_CATEGORICAL), ENG_FEATURES),
        "Gradient Boosting": (make_pipeline(HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=200, class_weight="balanced",
            random_state=seed), ENG_NUMERIC, ENG_CATEGORICAL), ENG_FEATURES),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scoring = ["roc_auc", "average_precision", "precision", "recall", "f1"]
    rows = []
    for name, (pipe, feats) in candidates.items():
        res = cross_validate(pipe, d[feats], y, cv=cv, scoring=scoring)
        rows.append({"model": name, **{m: res[f"test_{m}"].mean() for m in scoring},
                     "roc_auc_std": res["test_roc_auc"].std()})
    out = pd.DataFrame(rows).rename(columns={"average_precision": "pr_auc"})
    return out.round(3)


def retention_outcome(y_true, proba, monthly_charge, threshold, offer_cost,
                      success_rate, months_retained) -> dict:
    """Expected business result of offering a retention deal to everyone scored >= threshold.

    Every targeted customer costs `offer_cost`. Of the targeted customers who *would have*
    churned, `success_rate` are saved, each worth monthly_charge * months_retained.
    """
    y_true, proba, monthly = map(np.asarray, (y_true, proba, monthly_charge))
    target = proba >= threshold
    reached = target & (y_true == 1)
    cost = offer_cost * target.sum()
    saved = success_rate * (monthly[reached] * months_retained).sum()
    net = saved - cost
    return {"threshold": float(threshold), "customers_targeted": int(target.sum()),
            "churners_reached": int(reached.sum()), "campaign_cost": float(cost),
            "revenue_saved": float(saved), "net_benefit": float(net),
            "roi": float(net / cost) if cost > 0 else 0.0}


def sweep_thresholds(y_true, proba, monthly_charge, offer_cost, success_rate, months_retained,
                     thresholds=None) -> pd.DataFrame:
    thresholds = np.arange(0.05, 0.96, 0.05) if thresholds is None else thresholds
    return pd.DataFrame([retention_outcome(y_true, proba, monthly_charge, t, offer_cost,
                                           success_rate, months_retained) for t in thresholds])


def best_threshold(sweep: pd.DataFrame) -> pd.Series:
    return sweep.loc[sweep["net_benefit"].idxmax()]
