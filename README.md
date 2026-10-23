# Customer Churn Prediction & Retention Dashboard

Predicts which telecom customers are likely to leave, explains the factors associated with churn,
and presents customer-level risk in an interactive Streamlit dashboard.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 columns, 26.5% churn).
**Stack:** Python · Pandas · Scikit-learn · Matplotlib · Streamlit · CSV + SQLite.

## Headline results (held-out 20% test set, threshold 0.50)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.740 | 0.507 | 0.786 | 0.616 | 0.842 |

A "nobody churns" model scores 73.5% accuracy but 0% recall - that is why accuracy alone is misleading.
Observed churn in the risk bands: **Low 4.8% → Medium 23.5% → High 56.2%**.
Full details: [`reports/evaluation_report.md`](reports/evaluation_report.md).

## Quick start

```bash
pip install -r requirements.txt
python train.py              # clean data, train model, write report + predictions (≈10 s)
streamlit run app.py         # open the dashboard at http://localhost:8501
pytest                       # 28 tests
```

`data/raw_telco.csv` is included. (Original source: IBM's public `telco-customer-churn-on-icp4d` repository.)

## Project structure

```
app.py                      Streamlit dashboard (Module 5)
train.py                    Pipeline: clean → train → evaluate → score → report (Modules 1, 3, 4)
churn/data.py               Ingestion & cleaning (Module 1)
churn/model.py              Logistic Regression pipeline, metrics, risk bands, explanations
churn/advanced.py           Feature engineering, model comparison, retention-ROI calculator
churn/plots.py              Matplotlib charts shared by notebook and dashboard
notebooks/churn_analysis.ipynb   Full walkthrough incl. EDA (Module 2), executed with outputs
data/raw_telco.csv          Original data
data/cleaned_telco.csv      Cleaned dataset (deliverable)
data/customer_predictions.csv  Every customer with churn probability + risk band
data/churn.db               Same predictions in SQLite (table: customer_predictions)
models/churn_model.joblib   Trained pipeline;  models/metrics.json  test-set metrics
reports/                    evaluation_report.md, feature_importance.csv, figures/
screenshots/                Dashboard previews
tests/                      28 pytest tests (cleaning, model, advanced analysis, dashboard smoke test)
```

## Modules

1. **Ingestion & preprocessing** - `TotalCharges` stored as text with 11 blanks (new customers, tenure 0 → set to 0);
   `SeniorCitizen` 0/1 → Yes/No; "No internet/phone service" → "No"; `Churn` → 0/1; tenure groups added.
2. **EDA** - churn by contract, tenure, internet service, payment method, monthly charges.
3. **Model** - `StandardScaler` + `OneHotEncoder` + `LogisticRegression(class_weight="balanced")` in a single sklearn `Pipeline`
   (stratified 80/20 split, `random_state=42`).
4. **Evaluation & risk bands** - precision, recall, F1, ROC-AUC, confusion matrix, threshold trade-off.
   Customer scores use 5-fold out-of-fold probabilities, so nobody is scored by a model that saw their own row.
   Bands: Low < 0.30 ≤ Medium < 0.60 ≤ High (adjustable in the dashboard sidebar).
5. **Dashboard** - filters, KPIs (incl. monthly revenue at risk), churn patterns, coefficient chart, customer lookup with
   per-customer explanation + rule-based retention actions + what-if scenario, downloadable at-risk list, live threshold slider.

6. **Going further (advanced)**
   - *Feature engineering:* number of services, average monthly spend, current bill vs. average (price-rise proxy), auto-pay flag, new-customer flag.
   - *Model comparison:* Logistic Regression (baseline / + engineered / + tuned C with nested CV), Random Forest and Gradient Boosting on identical 5-fold CV folds.
     **All land at ROC-AUC ≈ 0.845**, so the simple, explainable Logistic Regression is justified (`reports/model_comparison.csv`).
   - *Retention ROI:* a cost-based calculator. With $50 per offer, 30% success and 12 months retained, the most profitable threshold is 0.45
     (target 3,190 customers, net benefit ≈ $273k, ROI 1.7x). Offer cost, success rate and months are adjustable in the dashboard.

## Key findings

Churn is associated with: short tenure, month-to-month contracts (42.7% vs 2.8% on two-year), fibre-optic internet (41.9%),
electronic-check payment, and lack of tech support / online security.

## Limitations

Associations not causation · single snapshot, no time dimension · linear model (no interactions) ·
retention suggestions are rule-based and untested · ROI figures rest on assumed offer cost and success rate · the 5-fold scores and the saved model differ slightly
(saved model is trained on the 80% split it was evaluated on).
