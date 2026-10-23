"""Renders dashboard-style preview images into screenshots/ using the dashboard's own chart code.
(Replace with real browser screenshots for your submission - see README.)"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from churn import data, model, plots

ROOT = data.ROOT
df = pd.read_csv(ROOT / "data" / "customer_predictions.csv")
pipe = joblib.load(ROOT / "models" / "churn_model.joblib")
met = json.loads((ROOT / "models" / "metrics.json").read_text())
OUT = ROOT / "screenshots"


def panel(ax, fig_src):
    """Paste a rendered figure into an axes."""
    fig_src.canvas.draw()
    ax.imshow(fig_src.canvas.buffer_rgba()); ax.axis("off"); plt.close(fig_src)


def kpi_strip(fig, items, y=0.93):
    for i, (label, val) in enumerate(items):
        x = 0.04 + i * 0.19
        fig.text(x, y, val, fontsize=20, weight="bold"); fig.text(x, y - 0.035, label, fontsize=9, color="grey")


# 1. Overview
high = df[df.risk_band == "High"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9)); fig.subplots_adjust(top=0.82, hspace=.05, wspace=.02)
kpi_strip(fig, [("Customers", f"{len(df):,}"), ("Actual churn rate", f"{df.Churn.mean():.1%}"),
                ("High-risk customers", f"{len(high):,}"), ("Monthly revenue at risk", f"${high.MonthlyCharges.sum():,.0f}"),
                ("Avg. tenure", f"{df.tenure.mean():.0f} months")])
fig.suptitle("Customer Churn Dashboard - Overview (preview)", x=0.04, ha="left", fontsize=14, y=0.99)
panel(axes[0, 0], plots.churn_rate_by(df, "Contract", title="Churn rate by contract type"))
panel(axes[0, 1], plots.churn_rate_by(df, "TenureGroup", data.TENURE_ORDER, "Churn rate by tenure"))
panel(axes[1, 0], plots.churn_rate_by(df, "InternetService", title="Churn rate by internet service"))
panel(axes[1, 1], plots.risk_band_fig(df.risk_band.values))
fig.savefig(OUT / "01_overview.png", dpi=110); plt.close(fig)

# 2. Churn drivers
imp = model.feature_importance(pipe)
fig = plots.importance_fig(imp, 12); fig.suptitle("Churn drivers (preview)", x=0.02, ha="left"); fig.tight_layout()
fig.savefig(OUT / "02_churn_drivers.png", dpi=120); plt.close(fig)

# 3. Customer lookup
row = df.sort_values("churn_probability", ascending=False).iloc[40]
contrib = model.explain_customer(pipe, row.to_frame().T)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
axes[0].axis("off")
txt = (f"Customer {row.customerID}\n\nPredicted churn probability: {row.churn_probability:.1%}\nRisk band: {row.risk_band.upper()}\n\n"
       f"Contract: {row.Contract}\nTenure: {row.tenure} months\nInternet: {row.InternetService}\nTech support: {row.TechSupport}\n"
       f"Payment: {row.PaymentMethod}\nMonthly charges: ${row.MonthlyCharges:.2f}\n\nSuggested actions:\n" +
       "\n".join("  - " + a for a in model.recommend_actions(row)[:3]))
axes[0].text(0, 1, txt, va="top", fontsize=10, wrap=True)
panel(axes[1], plots.contribution_fig(contrib))
fig.suptitle("Customer lookup (preview)", x=0.04, ha="left", fontsize=14)
fig.savefig(OUT / "03_customer_lookup.png", dpi=110); plt.close(fig)

# 4. Model metrics
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
panel(axes[0], plots.confusion_fig(met["confusion"], "Test-set confusion matrix"))
y_all, p_all = df.Churn.values, df.churn_probability.values
panel(axes[1], plots.roc_fig(y_all, p_all, met["roc_auc"]))
fig.suptitle(f"Model & metrics (preview) - precision {met['precision']:.2f} · recall {met['recall']:.2f} · "
             f"F1 {met['f1']:.2f} · accuracy {met['accuracy']:.2f} (do-nothing baseline accuracy {met['baseline_accuracy']:.2f}, recall 0)",
             fontsize=10)
fig.savefig(OUT / "04_model_metrics.png", dpi=110); plt.close(fig)
print("saved", sorted(p.name for p in OUT.iterdir()))
