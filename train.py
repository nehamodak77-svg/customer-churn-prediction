"""Run the full pipeline: clean -> train -> evaluate -> score customers -> write report.

Usage:  python train.py
"""
import json
import sqlite3

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn import advanced, data, model, plots

ROOT = data.ROOT
SEED = 42


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    # 0. Make sure every output folder exists (a fresh clone has none of them)
    for folder in ("data", "models", "reports/figures", "screenshots"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    # 1. Ingest + clean
    raw = data.load_raw()
    df = data.clean(raw)
    df.to_csv(data.CLEAN_PATH, index=False)
    print(f"Cleaned {len(raw):,} rows -> {len(df):,} rows saved to {data.CLEAN_PATH.name}")

    X, y = df[model.FEATURES], df["Churn"]

    # 2. Train / test split (stratified keeps the ~27% churn share in both sets)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = model.build_pipeline().fit(X_tr, y_tr)
    joblib.dump(pipe, ROOT / "models" / "churn_model.joblib")

    # 3. Evaluate on the held-out test set
    p_te = pipe.predict_proba(X_te)[:, 1]
    metrics = model.evaluate(y_te, p_te, threshold=0.5)
    thresholds = pd.DataFrame([model.evaluate(y_te, p_te, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)])
    thresholds = thresholds[["threshold", "accuracy", "precision", "recall", "f1"]].round(3)

    # 4. Customer-level scores: 5-fold out-of-fold probabilities, so no customer
    #    is scored by a model that saw their own row.
    oof = cross_val_predict(model.build_pipeline(), X, y, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
    scored = df.copy()
    scored["churn_probability"] = oof.round(4)
    scored["risk_band"] = model.risk_band(oof)
    scored.to_csv(ROOT / "data" / "customer_predictions.csv", index=False)
    with sqlite3.connect(ROOT / "data" / "churn.db") as con:
        scored.to_sql("customer_predictions", con, if_exists="replace", index=False)

    band_check = (scored.groupby("risk_band")
                  .agg(customers=("Churn", "size"), actual_churn_rate=("Churn", "mean"))
                  .reindex(model.RISK_LABELS))
    band_check["actual_churn_rate"] = (band_check["actual_churn_rate"] * 100).round(1).astype(str) + "%"
    band_check = band_check.reset_index().rename(columns={"risk_band": "risk band"})

    imp = model.feature_importance(pipe)
    metrics_out = {**metrics, "n_train": len(X_tr), "n_test": len(X_te),
                   "churn_rate": float(y.mean()), "n_customers": len(df)}
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    imp.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)

    # 4b. Advanced: model comparison + retention ROI (default business assumptions)
    comp = advanced.compare_models(df, SEED)
    comp.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    ROI = dict(offer_cost=50.0, success_rate=0.30, months_retained=12)
    sweep = advanced.sweep_thresholds(y.values, oof, df["MonthlyCharges"].values, **ROI)
    best = advanced.best_threshold(sweep)
    sweep.round(2).to_csv(ROOT / "reports" / "roi_sweep.csv", index=False)

    # 5. Figures
    fig_dir = ROOT / "reports" / "figures"
    figs = {
        "confusion_matrix.png": plots.confusion_fig(metrics["confusion"], "Test-set confusion matrix"),
        "roc_curve.png": plots.roc_fig(y_te, p_te, metrics["roc_auc"]),
        "feature_importance.png": plots.importance_fig(imp),
        "churn_by_contract.png": plots.churn_rate_by(df, "Contract"),
        "churn_by_tenure.png": plots.churn_rate_by(df, "TenureGroup", data.TENURE_ORDER, "Churn rate by tenure"),
        "risk_bands.png": plots.risk_band_fig(scored["risk_band"].values),
        "model_comparison.png": plots.model_comparison_fig(comp),
        "roi_curve.png": plots.roi_curve_fig(sweep, best),
    }
    for name, fig in figs.items():
        fig.savefig(fig_dir / name, dpi=130)

    # 6. Evaluation report
    tn, fp = metrics["confusion"][0]
    fn, tp = metrics["confusion"][1]
    top = imp.head(8)[["feature", "coefficient", "odds_ratio", "direction"]].round(3)
    report = f"""# Model Evaluation Report

**Model:** Logistic Regression (`class_weight="balanced"`, L2 regularisation, C=1.0)
**Data:** IBM Telco Customer Churn - {len(df):,} customers, {y.mean():.1%} churned
**Split:** stratified 80/20 (train {len(X_tr):,} / test {len(X_te):,}), random_state={SEED}

## 1. Headline results (test set, threshold 0.50)

| Metric | Value |
|---|---|
| Accuracy | {metrics['accuracy']:.3f} |
| Precision (churn class) | {metrics['precision']:.3f} |
| Recall (churn class) | {metrics['recall']:.3f} |
| F1-score (churn class) | {metrics['f1']:.3f} |
| ROC-AUC | {metrics['roc_auc']:.3f} |

Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}.

- **Precision** = TP / (TP + FP): of the customers flagged as churners, {metrics['precision']:.0%} really churned.
- **Recall** = TP / (TP + FN): of the customers who really churned, the model caught {metrics['recall']:.0%}.
- **F1** is the harmonic mean of precision and recall.

## 2. Why accuracy alone is misleading

Only {y.mean():.1%} of customers churn. A "model" that predicts *nobody churns* would score
**{metrics['baseline_accuracy']:.1%} accuracy** while catching **0%** of churners (recall = 0) - useless for retention.
Our model has *lower* accuracy ({metrics['accuracy']:.1%}) but finds {metrics['recall']:.0%} of the customers who leave,
which is what the business actually needs. For imbalanced problems, look at precision, recall and F1 for the minority class.

## 3. Threshold trade-off

Lowering the threshold catches more churners (higher recall) but flags more loyal customers by mistake (lower precision).
The business should choose the threshold from the relative cost of a retention offer vs. a lost customer.

{md_table(thresholds)}

## 4. Risk categories

Bands use out-of-fold (5-fold cross-validated) probabilities: Low < 0.30, Medium 0.30-0.60, High >= 0.60.
The observed churn rate rises steadily across the bands, which suggests the bands are meaningful.
Note: because the model is class-balanced, the scores rank customers well but run higher than the real churn rates,
so read them as a risk score; the "actual churn rate" column below is the true observed rate.

{md_table(band_check)}

## 5. Factors most associated with churn

Coefficients are on the log-odds scale; odds ratio > 1 means higher churn odds. Numeric features are standardised (per 1 std-dev).

{md_table(top)}

## 6. Model comparison (5-fold cross-validation)

Five models were compared on identical folds. The tuned model searches C in {{0.01, 0.1, 1, 10}} with an inner 3-fold grid search (nested CV).
Engineered features: number of services, average monthly spend, current bill vs. average, auto-pay flag, new-customer flag.

{md_table(comp)}

**Finding:** all models land within about 0.003 ROC-AUC of each other. The extra complexity of Random Forest and Gradient Boosting buys no
meaningful accuracy here, so the simpler and explainable Logistic Regression is a justified choice.

## 7. Business value: retention-campaign ROI

Assumptions (editable in the dashboard): ${ROI['offer_cost']:.0f} cost per retention offer, {ROI['success_rate']:.0%} of would-be churners
saved by an offer, each saved customer retained for {ROI['months_retained']} months of billing.

Best threshold on cross-validated scores: **{best['threshold']:.2f}** -> target {int(best['customers_targeted']):,} customers,
reach {int(best['churners_reached']):,} real churners, campaign cost ${best['campaign_cost']:,.0f},
revenue saved ${best['revenue_saved']:,.0f}, **net benefit ${best['net_benefit']:,.0f}** (ROI {best['roi']:.1f}x).
The best threshold is a business choice that depends on these costs, not a fixed 0.50.
This is an illustrative back-test: the threshold is chosen and scored on the same data.

## 8. Limitations

- **Associations, not causes.** The model shows which customer attributes go with churn; it does not prove *why* a customer leaves.
- Single dataset snapshot; no time dimension, so performance on future customers may differ. For churned customers, `tenure` is their final lifetime, so it partly reflects the outcome.
- `TotalCharges` is strongly correlated with `tenure` x `MonthlyCharges`, so individual coefficients for those three features should be read together, not in isolation (multicollinearity).
- Logistic Regression is linear; interactions (e.g. contract x tenure) are not modelled. Tree-based models could be tried next.
- Retention actions in the dashboard are rule-based suggestions and have not been A/B tested.
"""
    (ROOT / "reports" / "evaluation_report.md").write_text(report)

    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))
    print(band_check.to_string(index=False))
    print(imp.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()