"""Generates notebooks/churn_analysis.ipynb (run once; the notebook is then executed)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
M("""# Customer Churn Prediction - IBM Telco dataset
**Goal:** identify customers likely to leave, understand the factors associated with churn, and categorise customers by risk.

Sections: 1) Load & clean · 2) EDA · 3) Logistic Regression · 4) Evaluation · 5) Risk bands · 6) Takeaways"""),
C("""import sys; sys.path.insert(0, "..")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from churn import data, model, plots
%matplotlib inline
pd.set_option("display.max_columns", 30)"""),
M("## 1. Data ingestion & preprocessing"),
C("""raw = data.load_raw()
print(raw.shape)
raw.head()"""),
C("""# Data-quality checks on the raw file
print("Duplicate IDs      :", raw.customerID.duplicated().sum())
print("Missing values     :", raw.isna().sum().sum())
blank = raw.TotalCharges.astype(str).str.strip() == ""
print("Blank TotalCharges :", blank.sum(), "<- stored as text, so the column is read as 'object'")
raw.loc[blank, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].head()"""),
M("The 11 blank `TotalCharges` all belong to customers with `tenure = 0` (haven't been billed yet), so they are set to 0. Other fixes: numeric `SeniorCitizen` -> Yes/No, \"No internet/phone service\" -> \"No\", `Churn` -> 0/1."),
C("""df = data.clean(raw)
df.to_csv(data.CLEAN_PATH, index=False)
print(df.dtypes.value_counts())
df.head()"""),
M("## 2. Exploratory data analysis"),
C("""print(f"Overall churn rate: {df.Churn.mean():.1%}  ({df.Churn.sum():,} of {len(df):,} customers)")
df.groupby("Churn")[["tenure","MonthlyCharges","TotalCharges"]].mean().round(1).rename(index={0:"Stayed",1:"Churned"})"""),
C("""fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (col, order) in zip(axes.ravel(), [("Contract", None), ("TenureGroup", data.TENURE_ORDER),
                                           ("InternetService", None), ("PaymentMethod", None)]):
    rate = df.groupby(col).Churn.mean().mul(100)
    if order: rate = rate.reindex(order)
    b = ax.bar(rate.index, rate.values, color=plots.CHURN); ax.bar_label(b, fmt="%.1f%%")
    ax.set_title(f"Churn rate by {col}"); ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.show()"""),
M("**Observations:** month-to-month contracts churn roughly 15x more than two-year contracts; churn falls steadily with tenure; fibre-optic and electronic-check customers churn more. These are *patterns*, not proof of cause."),
C("""plots.charges_hist(df); plt.show()"""),
M("## 3. Churn prediction with Logistic Regression\nPipeline = StandardScaler (numeric) + OneHotEncoder (categorical) + `LogisticRegression(class_weight='balanced')`. Bundling preprocessing into the pipeline prevents leakage from the test set."),
C("""X, y = df[model.FEATURES], df.Churn
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe = model.build_pipeline().fit(X_tr, y_tr)
print("train:", X_tr.shape, " test:", X_te.shape, " churn share train/test:", round(y_tr.mean(),3), round(y_te.mean(),3))"""),
M("## 4. Model evaluation"),
C("""p_te = pipe.predict_proba(X_te)[:, 1]
m = model.evaluate(y_te, p_te)
pd.Series({k: round(v, 3) for k, v in m.items() if k in ["accuracy","precision","recall","f1","roc_auc","baseline_accuracy"]})"""),
C("""from IPython.display import display
display(plots.confusion_fig(m["confusion"], "Test-set confusion matrix"), plots.roc_fig(y_te, p_te, m["roc_auc"]))"""),
M("""### Why accuracy alone is misleading
Only ~27% of customers churn, so a model that always predicts "stays" is already ~73% accurate - yet it finds **zero** churners."""),
C("""dummy = model.evaluate(y_te, np.zeros(len(y_te)))
pd.DataFrame({"Predict nobody churns": dummy, "Logistic Regression": m}).loc[["accuracy","precision","recall","f1"]].round(3)"""),
M("Precision = of those we flag, how many really churn. Recall = of those who churn, how many we catch. Choosing the threshold is a business decision:"),
C("""pd.DataFrame([model.evaluate(y_te, p_te, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)])[["threshold","accuracy","precision","recall","f1"]].round(3)"""),
M("### Which factors are associated with churn?"),
C("""imp = model.feature_importance(pipe)
display(imp.head(10).round(3))
plots.importance_fig(imp); plt.show()"""),
M("Positive coefficients (odds ratio > 1) go with higher churn; negative with lower. **Caution:** `TotalCharges`, `tenure` and `MonthlyCharges` are correlated, so read them together. Associations are not causes."),
M("## 5. Risk categorisation\nCustomer-level scores use 5-fold out-of-fold probabilities so no customer is scored by a model that saw them in training."),
C("""oof = cross_val_predict(model.build_pipeline(), X, y, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
scored = df.assign(churn_probability=oof.round(4), risk_band=model.risk_band(oof))
scored.groupby("risk_band").agg(customers=("Churn","size"), actual_churn_rate=("Churn","mean"),
                                avg_monthly_charge=("MonthlyCharges","mean")).reindex(model.RISK_LABELS).round(3)"""),
C("""scored.sort_values("churn_probability", ascending=False)[["customerID","churn_probability","risk_band","Contract","tenure","MonthlyCharges","Churn"]].head(10)"""),
M("""## 6. Going further: feature engineering & model comparison
We add five derived features (number of services, average monthly spend, current bill vs. average, auto-pay flag, new-customer flag)
and compare Logistic Regression with a tuned version, Random Forest and Gradient Boosting on identical 5-fold cross-validation folds."""),
C("""from churn import advanced
eng = advanced.add_engineered_features(df)
eng[["customerID","NumServices","AvgMonthlySpend","BillVsAverage","IsAutoPay","IsNewCustomer","Churn"]].head()"""),
C("""comp = advanced.compare_models(df)
display(comp)
plots.model_comparison_fig(comp); plt.show()"""),
M("**Finding:** every model lands within ~0.003 ROC-AUC of the others. The extra complexity buys nothing here, so Logistic Regression is a justified choice - it is as accurate *and* explainable."),
M("""## 7. Business value: retention-campaign ROI
Offers cost money, so which customers should we target? Assumptions: $50 per offer, 30% of would-be churners saved, 12 months of billing retained per saved customer."""),
C("""sweep = advanced.sweep_thresholds(y.values, oof, df.MonthlyCharges.values, offer_cost=50, success_rate=0.30, months_retained=12)
best = advanced.best_threshold(sweep)
display(sweep.round(2).iloc[::2])
print(f"Best threshold {best.threshold:.2f}: target {int(best.customers_targeted):,} customers, net benefit ${best.net_benefit:,.0f}, ROI {best.roi:.1f}x")
plots.roi_curve_fig(sweep, best); plt.show()"""),
M("The most profitable threshold depends on offer cost, success rate and customer value - it moves as those assumptions change (try a cheaper offer and watch it drop), so it should not be fixed at 0.50 by default."),
M("""## 8. Takeaways
- Model catches ~79% of churners (recall) with ~51% precision on unseen data; ROC-AUC ≈ 0.84.
- Observed churn climbs from ~5% (Low) to ~56% (High), so the bands are useful for prioritising outreach.
- Biggest associated factors: short tenure, month-to-month contract, fibre-optic internet, electronic-check payment, no tech support.
- Five models perform almost identically (ROC-AUC ≈ 0.845), so simple Logistic Regression is justified.
- A business-cost threshold (not 0.50) maximises campaign profit.
- **Limitation:** these are associations in historical data - they do not prove why a particular customer leaves."""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/churn_analysis.ipynb")
