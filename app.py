"""Module 5 - Streamlit dashboard.   Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from churn import advanced, data, model, plots

PRED_PATH = ROOT / "data" / "customer_predictions.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Churn Risk Dashboard", page_icon="📉", layout="wide")


@st.cache_data
def load_predictions():
    return pd.read_csv(PRED_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


if not (PRED_PATH.exists() and MODEL_PATH.exists() and METRICS_PATH.exists()):
    st.error("Model artefacts not found. Run `python train.py` first, then reload this page.")
    st.stop()

df_all = load_predictions()
pipe = load_model()
metrics = json.loads(METRICS_PATH.read_text())

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(df_all["Contract"].unique()),
                                   default=sorted(df_all["Contract"].unique()))
internet = st.sidebar.multiselect("Internet service", sorted(df_all["InternetService"].unique()),
                                  default=sorted(df_all["InternetService"].unique()))
t_min, t_max = int(df_all["tenure"].min()), int(df_all["tenure"].max())
tenure_range = st.sidebar.slider("Tenure (months)", t_min, t_max, (t_min, t_max))

st.sidebar.header("Risk band cut-offs")
low_cut, high_cut = st.sidebar.slider("Medium starts / High starts", 0.05, 0.95,
                                      (model.LOW_CUTOFF, model.HIGH_CUTOFF), 0.05,
                                      help="Risk score at which a customer becomes Medium / High risk.")
if low_cut >= high_cut:
    st.sidebar.error("Medium cut-off must be lower than High cut-off.")
    st.stop()

df_all = df_all.copy()
df_all["risk_band"] = model.risk_band(df_all["churn_probability"].values, low_cut, high_cut)
df = df_all[df_all["Contract"].isin(contracts) & df_all["InternetService"].isin(internet)
            & df_all["tenure"].between(*tenure_range)]

# ---------------------------------------------------------------- header + KPIs
st.title("📉 Customer Churn Prediction & Retention Dashboard")
st.caption("IBM Telco Customer Churn dataset · Logistic Regression · predictions show associations, not proof of why a customer leaves.")

if df.empty:
    st.warning("No customers match the current filters.")
    st.stop()

high = df[df["risk_band"] == "High"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Actual churn rate", f"{df['Churn'].mean():.1%}")
k3.metric("High-risk customers", f"{len(high):,}")
k4.metric("Monthly revenue of High-risk", f"${high['MonthlyCharges'].sum():,.0f}",
          help="Sum of monthly charges of customers in the High band (not all of them will churn).")
k5.metric("Avg. tenure", f"{df['tenure'].mean():.0f} months")

tab_over, tab_drv, tab_cust, tab_list, tab_model, tab_comp, tab_roi, tab_new = st.tabs(
    ["Overview", "Churn drivers", "Customer lookup", "At-risk customers", "Model & metrics",
     "Model comparison", "Retention ROI", "Score new data"])

# ---------------------------------------------------------------- overview
with tab_over:
    st.subheader("Churn patterns")
    c1, c2 = st.columns(2)
    with c1:
        show(plots.churn_rate_by(df, "Contract", title="Churn rate by contract type"))
        show(plots.churn_rate_by(df, "InternetService", title="Churn rate by internet service"))
    with c2:
        show(plots.churn_rate_by(df, "TenureGroup", data.TENURE_ORDER, "Churn rate by tenure"))
        show(plots.churn_rate_by(df, "PaymentMethod", title="Churn rate by payment method"))
    c3, c4 = st.columns(2)
    with c3:
        show(plots.charges_hist(df))
    with c4:
        show(plots.risk_band_fig(df["risk_band"].values))
    st.info("Dashed line = overall churn rate of the filtered customers. "
            "Month-to-month, fibre-optic, short-tenure and electronic-check customers churn most.")

# ---------------------------------------------------------------- drivers
with tab_drv:
    st.subheader("Factors associated with churn")
    imp = model.feature_importance(pipe)
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.importance_fig(imp, 12))
    with c2:
        st.dataframe(imp.head(12).round(3), hide_index=True, width="stretch")
    st.caption("Odds ratio > 1 = higher odds of churn, < 1 = lower. Numeric features are standardised (per 1 std-dev). "
               "These are **associations**, not proof of cause.")

# ---------------------------------------------------------------- customer lookup
with tab_cust:
    st.subheader("Customer-level risk")
    ranked = df.sort_values("churn_probability", ascending=False)
    lookup = ranked.set_index("customerID")["churn_probability"]
    cust_id = st.selectbox("Select a customer (highest risk first)", lookup.index,
                           format_func=lambda i: f"{i}  ·  risk {lookup[i]:.0%}")
    row = df_all[df_all["customerID"] == cust_id].iloc[0]
    band_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[row["risk_band"]]

    left, right = st.columns([2, 3])
    with left:
        st.metric(
    "Predicted churn probability",
    f"{row['churn_probability']:.1%}",
    help="Estimated probability that this customer will churn. Use this score to rank customers by risk; it is not a guarantee that a customer will leave."
)
        st.markdown(f"### {band_color} {row['risk_band']} risk")
        st.progress(float(row["churn_probability"]))
        st.caption("Recorded outcome in dataset: " + ("**churned**" if row["Churn"] == 1 else "**still a customer**"))
        profile = row[["tenure", "Contract", "InternetService", "TechSupport", "OnlineSecurity",
                       "PaymentMethod", "MonthlyCharges", "TotalCharges"]]
        st.dataframe(profile.astype(str).rename("value"), width="stretch")
    with right:
        contrib = model.explain_customer(pipe, row.to_frame().T)
        show(plots.contribution_fig(contrib))
        st.markdown("**Suggested retention actions**")
        for a in model.recommend_actions(row):
            st.markdown(f"- {a}")

    st.markdown("---")
    st.markdown("**What-if analysis** - change a few attributes and see how the model's score moves.")
    has_net = row["InternetService"] != "No"
    w1, w2, w3, w4 = st.columns(4)
    new = row.copy()
    new["Contract"] = w1.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                   index=["Month-to-month", "One year", "Two year"].index(row["Contract"]))
    new["TechSupport"] = w2.selectbox("Tech support", ["No", "Yes"], index=["No", "Yes"].index(row["TechSupport"]),
                                      disabled=not has_net)
    new["OnlineSecurity"] = w3.selectbox("Online security", ["No", "Yes"],
                                         index=["No", "Yes"].index(row["OnlineSecurity"]), disabled=not has_net)
    pay_opts = sorted(df_all["PaymentMethod"].unique())
    new["PaymentMethod"] = w4.selectbox("Payment method", pay_opts, index=pay_opts.index(row["PaymentMethod"]))
    if not has_net:
        st.caption("This customer has no internet service, so tech support and online security cannot be changed.")
    p_new = float(pipe.predict_proba(new.to_frame().T[model.FEATURES])[:, 1][0])
    p_old = float(pipe.predict_proba(row.to_frame().T[model.FEATURES])[:, 1][0])
    st.metric("Scenario risk score", f"{p_new:.1%}", delta=f"{(p_new - p_old) * 100:+.1f} pts vs. current profile",
              delta_color="inverse")
    st.caption("Scenario scores show how the model's output changes - they do not prove that making the change would "
               "cause the customer to stay.")

# ---------------------------------------------------------------- at-risk list
with tab_list:
    st.subheader("Customer-level results")
    band_filter = st.multiselect("Risk band", model.RISK_LABELS, default=["High"])
    cols = ["customerID", "risk_band", "churn_probability", "Contract", "InternetService", "tenure",
            "MonthlyCharges", "PaymentMethod", "Churn"]
    table = (df[df["risk_band"].isin(band_filter)].sort_values("churn_probability", ascending=False)[cols])
    st.write(f"{len(table):,} customers")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "risk score", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download as CSV", table.to_csv(index=False).encode(), "at_risk_customers.csv", "text/csv")

# ---------------------------------------------------------------- model metrics
with tab_model:
    st.subheader("How good is the model?")
    st.markdown("Scores below use **cross-validated** probabilities for all customers, so no customer was scored by "
                "a model that had seen their own record.")
    thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                    help="Customers with a score ≥ threshold are predicted to churn.")
    m = model.evaluate(df_all["Churn"].values, df_all["churn_probability"].values, thr)
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{m['accuracy']:.1%}")
    b.metric("Precision", f"{m['precision']:.1%}")
    c.metric("Recall", f"{m['recall']:.1%}")
    d.metric("F1-score", f"{m['f1']:.1%}")
    c1, c2 = st.columns(2)
    with c1:
        show(plots.confusion_fig(m["confusion"], f"Confusion matrix (threshold {thr:.2f})"))
    with c2:
        show(plots.roc_fig(df_all["Churn"].values, df_all["churn_probability"].values, m["roc_auc"]))
    st.warning(
        f"**Why accuracy alone is misleading:** only {metrics['churn_rate']:.1%} of customers churn, so predicting "
        f"'nobody churns' already scores **{m['baseline_accuracy']:.1%} accuracy** with **0% recall**. "
        "Precision tells us how many flagged customers really churn; recall tells us how many churners we catch.")
    st.caption(f"Held-out test set (20% split, threshold 0.50): accuracy {metrics['accuracy']:.3f}, "
               f"precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, "
               f"F1 {metrics['f1']:.3f}, ROC-AUC {metrics['roc_auc']:.3f}.")

# ---------------------------------------------------------------- model comparison
with tab_comp:
    st.subheader("Is Logistic Regression good enough?")
    comp_path = ROOT / "reports" / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        spread = comp["roc_auc"].max() - comp["roc_auc"].min()
        show(plots.model_comparison_fig(comp))
        st.dataframe(comp, hide_index=True, width="stretch")
        st.success(f"All {len(comp)} models score within about {spread:.3f} ROC-AUC of each other, so the extra "
                   "complexity of Random Forest / Gradient Boosting buys no real gain here. "
                   "Logistic Regression is as accurate **and** explainable.")
        st.caption("5-fold stratified cross-validation on identical folds. 'Engineered' adds: number of services, average "
                   "monthly spend, current bill vs. average, auto-pay flag, new-customer flag. 'Tuned' searches C with an "
                   "inner grid search.")
    else:
        st.info("Run `python train.py` to generate the model comparison.")

# ---------------------------------------------------------------- retention ROI
with tab_roi:
    st.subheader("Which customers are worth a retention offer?")
    st.markdown("Contacting every at-risk customer costs money. Enter your assumptions and the dashboard finds the "
                "threshold that maximises expected profit. Scores are cross-validated.")
    r1, r2, r3 = st.columns(3)
    offer_cost = r1.number_input("Cost per retention offer ($)", 1.0, 500.0, 50.0, 5.0,
                                 help="Discount, call-centre time, gift, etc.")
    success = r2.slider("Offer success rate", 0.05, 0.90, 0.30, 0.05,
                        help="Share of would-be churners who stay because of the offer.")
    months = r3.slider("Months of revenue retained", 1, 36, 12)
    sweep = advanced.sweep_thresholds(df_all["Churn"].values, df_all["churn_probability"].values,
                                      df_all["MonthlyCharges"].values, offer_cost, success, months)
    best = advanced.best_threshold(sweep)
    pick = st.slider("Threshold to inspect", 0.05, 0.95, float(round(best["threshold"], 2)), 0.05)
    out = advanced.retention_outcome(df_all["Churn"].values, df_all["churn_probability"].values,
                                     df_all["MonthlyCharges"].values, pick, offer_cost, success, months)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Customers targeted", f"{out['customers_targeted']:,}")
    q2.metric("Real churners reached", f"{out['churners_reached']:,}")
    q3.metric("Campaign cost", f"${out['campaign_cost']:,.0f}")
    q4.metric("Net benefit", f"${out['net_benefit']:,.0f}", help=f"ROI {out['roi']:.1f}x")
    show(plots.roi_curve_fig(sweep, best))
    if best["net_benefit"] <= 0:
        st.warning("With these assumptions no campaign is profitable - the offer costs more than it saves.")
    else:
        st.info(f"Best threshold with these assumptions: **{best['threshold']:.2f}** "
                f"(net benefit ${best['net_benefit']:,.0f}). The best cut-off moves as you change the costs, so it should not be fixed at 0.50 by default.")
    st.caption("Simplified model: every targeted customer costs the offer (including those who would have stayed anyway), "
               "and the success rate is assumed equal for all customers. Illustrative back-test on the same data.")

# ---------------------------------------------------------------- score new data
with tab_new:
    st.subheader("Score a new customer file")
    st.caption("Upload a CSV with the same columns as the IBM Telco data. A Churn column is optional.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            new_df = data.clean(pd.read_csv(up), require_target=False)
            missing = [c for c in model.FEATURES if c not in new_df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            nums = new_df[model.NUMERIC].apply(pd.to_numeric, errors="coerce")
            new_df[model.NUMERIC] = nums.fillna(nums.median())
            new_df["churn_probability"] = pipe.predict_proba(new_df[model.FEATURES])[:, 1].round(4)
            new_df["risk_band"] = model.risk_band(new_df["churn_probability"].values, low_cut, high_cut)
        except Exception as e:
            st.error(f"Could not score this file: {e}")
        else:
            st.success(f"Scored {len(new_df):,} customers.")
            st.bar_chart(new_df["risk_band"].value_counts().reindex(model.RISK_LABELS).fillna(0))
            out_new = new_df.sort_values("churn_probability", ascending=False)
            cols_new = ["customerID", "risk_band", "churn_probability", "Contract", "tenure", "MonthlyCharges"]
            st.dataframe(out_new[cols_new], hide_index=True, width="stretch")
            st.download_button("Download scored file", out_new.to_csv(index=False).encode(),
                               "scored_customers.csv", "text/csv")