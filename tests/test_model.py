import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from churn import data, model


@pytest.fixture(scope="module")
def fitted():
    df = data.clean(data.load_raw())
    X_tr, X_te, y_tr, y_te = train_test_split(df[model.FEATURES], df["Churn"], test_size=.2,
                                              stratify=df["Churn"], random_state=42)
    pipe = model.build_pipeline().fit(X_tr, y_tr)
    return pipe, df, X_te, y_te


def test_probabilities_between_0_and_1(fitted):
    pipe, _, X_te, _ = fitted
    p = pipe.predict_proba(X_te)[:, 1]
    assert p.min() >= 0 and p.max() <= 1 and len(p) == len(X_te)


def test_model_beats_do_nothing_baseline_on_recall_and_auc(fitted):
    pipe, _, X_te, y_te = fitted
    m = model.evaluate(y_te, pipe.predict_proba(X_te)[:, 1])
    assert m["recall"] > 0.6          # baseline "nobody churns" has recall 0
    assert m["roc_auc"] > 0.80
    assert m["f1"] > 0.55


def test_evaluate_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])      # preds @0.5 -> 1,0,1,0
    m = model.evaluate(y, p, 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5
    assert m["confusion"] == [[1, 1], [1, 1]]


def test_accuracy_can_mislead():
    y = np.array([0] * 90 + [1] * 10)
    m = model.evaluate(y, np.zeros(100))     # "nobody churns"
    assert m["accuracy"] == 0.9 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_risk_band_boundaries():
    assert model.risk_band(0.10) == "Low"
    assert model.risk_band(0.30) == "Medium"
    assert model.risk_band(0.599) == "Medium"
    assert model.risk_band(0.60) == "High"
    assert list(model.risk_band([0.1, 0.4, 0.9])) == ["Low", "Medium", "High"]


def test_risk_band_rejects_bad_cutoffs():
    with pytest.raises(ValueError):
        model.risk_band(0.5, low=0.7, high=0.3)


def test_higher_risk_band_has_higher_actual_churn(fitted):
    pipe, df, _, _ = fitted
    p = pipe.predict_proba(df[model.FEATURES])[:, 1]
    bands = model.risk_band(p)
    rates = [df["Churn"][bands == b].mean() for b in model.RISK_LABELS]
    assert rates[0] < rates[1] < rates[2]


def test_known_associations_have_expected_sign(fitted):
    pipe = fitted[0]
    imp = model.feature_importance(pipe).set_index("feature")["coefficient"]
    assert imp["Contract = Month-to-month"] > 0
    assert imp["Contract = Two year"] < 0
    assert imp["tenure"] < 0


def test_explain_customer_returns_top_n(fitted):
    pipe, df, _, _ = fitted
    out = model.explain_customer(pipe, df.iloc[[0]], top_n=5)
    assert len(out) == 5 and {"feature", "contribution"} <= set(out.columns)


def test_recommend_actions(fitted):
    _, df, _, _ = fitted
    risky = df[(df.Contract == "Month-to-month") & (df.PaymentMethod == "Electronic check")].iloc[0]
    assert any("contract" in a.lower() for a in model.recommend_actions(risky))
    loyal = risky.copy()
    loyal["Contract"], loyal["PaymentMethod"], loyal["tenure"] = "Two year", "Credit card (automatic)", 60
    loyal["TechSupport"] = "Yes"
    assert len(model.recommend_actions(loyal)) >= 1
