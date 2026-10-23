"""Matplotlib helpers shared by the notebook, training script and dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

CHURN = "#E45756"
STAY = "#4C78A8"


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def churn_rate_by(df, col, order=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rate = df.groupby(col)["Churn"].mean() * 100
    if order:
        rate = rate.reindex([o for o in order if o in rate.index])
    bars = ax.bar([str(i) for i in rate.index], rate.values, color=CHURN)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.axhline(df["Churn"].mean() * 100, ls="--", color="grey", lw=1)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title or f"Churn rate by {col}", fontsize=11)
    ax.set_ylim(0, max(rate.max() * 1.2, 5))
    plt.setp(ax.get_xticklabels(), rotation=15 if len(rate) > 3 else 0, ha="right" if len(rate) > 3 else "center", fontsize=8)
    _clean(ax)
    fig.tight_layout()
    return fig


def charges_hist(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 25)
    ax.hist(df.loc[df.Churn == 0, "MonthlyCharges"], bins=bins, alpha=.7, color=STAY, label="Stayed")
    ax.hist(df.loc[df.Churn == 1, "MonthlyCharges"], bins=bins, alpha=.7, color=CHURN, label="Churned")
    ax.set_xlabel("Monthly charges ($)"); ax.set_ylabel("Customers")
    ax.set_title("Monthly charges: churned vs stayed", fontsize=11)
    ax.legend(frameon=False); _clean(ax); fig.tight_layout()
    return fig


def risk_band_fig(bands):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    counts = [int((bands == b).sum()) for b in ["Low", "Medium", "High"]]
    bars = ax.bar(["Low", "Medium", "High"], counts, color=["#59A14F", "#F2B134", CHURN])
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_ylabel("Customers"); ax.set_title("Customers by risk band", fontsize=11)
    _clean(ax); fig.tight_layout()
    return fig


def confusion_fig(cm, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def roc_fig(y, proba, auc):
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, color=CHURN, lw=2, label=f"Model (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve", fontsize=11); ax.legend(frameon=False, loc="lower right")
    _clean(ax); fig.tight_layout()
    return fig


def importance_fig(imp, top_n=12):
    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["feature"], top["coefficient"], color=[CHURN if c > 0 else STAY for c in top["coefficient"]])
    ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("Logistic-regression coefficient  (red = higher churn risk, blue = lower)")
    ax.set_title("Top factors associated with churn", fontsize=11)
    _clean(ax); fig.tight_layout()
    return fig


def contribution_fig(contrib):
    c = contrib.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(c["feature"], c["contribution"], color=[CHURN if v > 0 else STAY for v in c["contribution"]])
    ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("Effect on churn log-odds"); ax.set_title("What drives this customer's score", fontsize=11)
    _clean(ax); fig.tight_layout()
    return fig


def model_comparison_fig(comp):
    metrics = ["roc_auc", "recall", "f1"]
    labels = ["ROC-AUC", "Recall", "F1"]
    colors = ["#4C78A8", CHURN, "#F2B134"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    n = len(comp); w = 0.25
    y = np.arange(n)
    for i, (m, lab, col) in enumerate(zip(metrics, labels, colors)):
        bars = ax.barh(y + (i - 1) * w, comp[m], height=w, color=col, label=lab)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_yticks(y, [s.replace(" + ", "\n+ ") for s in comp["model"]], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_title("Model comparison (5-fold cross-validation)", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8); _clean(ax); fig.tight_layout()
    return fig


def roi_curve_fig(sweep, best):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(sweep["threshold"], sweep["net_benefit"], color=CHURN, lw=2, marker="o", ms=3)
    ax.axhline(0, color="grey", lw=.8)
    ax.scatter([best["threshold"]], [best["net_benefit"]], s=90, color="black", zorder=5,
               label=f"Best threshold {best['threshold']:.2f}")
    ax.set_xlabel("Decision threshold (target customers with risk ≥ this)")
    ax.set_ylabel("Expected net benefit ($)")
    ax.set_title("Retention campaign: net benefit vs. threshold", fontsize=11)
    ax.legend(frameon=False); _clean(ax); fig.tight_layout()
    return fig
