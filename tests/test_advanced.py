import numpy as np
import pytest

from churn import advanced, data


@pytest.fixture(scope="module")
def df():
    return data.clean(data.load_raw())


def test_engineered_features_valid(df):
    e = advanced.add_engineered_features(df)
    assert set(advanced.ENG_FEATURES) <= set(e.columns)
    assert e[advanced.ENG_FEATURES].isna().sum().sum() == 0
    assert np.isfinite(e["AvgMonthlySpend"]).all()
    assert e["NumServices"].between(0, 9).all()
    assert set(e["IsAutoPay"]) == {"Yes", "No"}
    assert (e.loc[e.tenure <= 6, "IsNewCustomer"] == "Yes").all()


def test_engineering_does_not_change_row_count_or_inputs(df):
    before = df.copy()
    e = advanced.add_engineered_features(df)
    assert len(e) == len(df)
    assert df.equals(before)           # original untouched


def test_retention_outcome_known_values():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.2, 0.8, 0.1]); m = np.array([100, 100, 100, 100])
    o = advanced.retention_outcome(y, p, m, threshold=0.5, offer_cost=10, success_rate=0.5, months_retained=10)
    assert o["customers_targeted"] == 2 and o["churners_reached"] == 1
    assert o["campaign_cost"] == 20
    assert o["revenue_saved"] == 0.5 * 100 * 10            # one churner reached
    assert o["net_benefit"] == 500 - 20


def test_no_one_targeted_means_zero_everything():
    o = advanced.retention_outcome([1, 0], [0.2, 0.1], [50, 50], 0.9, 10, 0.5, 12)
    assert o["customers_targeted"] == 0 and o["net_benefit"] == 0 and o["roi"] == 0


def test_best_threshold_is_the_maximum(df):
    rng = np.random.default_rng(0)
    p = np.clip(df["Churn"].values * 0.4 + rng.random(len(df)) * 0.6, 0, 1)
    sw = advanced.sweep_thresholds(df["Churn"].values, p, df["MonthlyCharges"].values, 50, 0.3, 12)
    best = advanced.best_threshold(sw)
    assert best["net_benefit"] == sw["net_benefit"].max()


def test_higher_offer_cost_never_increases_net_benefit(df):
    args = (df["Churn"].values, np.clip(df["Churn"].values * .5 + .2, 0, 1), df["MonthlyCharges"].values)
    cheap = advanced.retention_outcome(*args, 0.5, 10, 0.3, 12)["net_benefit"]
    dear = advanced.retention_outcome(*args, 0.5, 100, 0.3, 12)["net_benefit"]
    assert dear <= cheap


def test_compare_models_returns_all_models(df):
    comp = advanced.compare_models(df.sample(1500, random_state=1))
    assert len(comp) == 5
    assert (comp["roc_auc"] > 0.7).all() and (comp["roc_auc"] <= 1).all()
