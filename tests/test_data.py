import pandas as pd
import pytest

from churn import data


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return data.clean(raw)


def test_row_count_preserved(raw, cleaned):
    assert len(cleaned) == len(raw) == 7043


def test_no_missing_values(cleaned):
    assert cleaned.drop(columns=[]).isna().sum().sum() == 0


def test_total_charges_numeric_and_blank_rows_fixed(raw, cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    blanks = raw["TotalCharges"].astype(str).str.strip() == ""
    assert blanks.sum() == 11
    assert (cleaned.loc[blanks.values, "TotalCharges"] == 0).all()   # tenure 0 customers


def test_churn_is_binary_with_expected_rate(cleaned):
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert 0.25 < cleaned["Churn"].mean() < 0.28


def test_customer_ids_unique(cleaned):
    assert cleaned["customerID"].is_unique


def test_category_labels_tidied(cleaned):
    for col in data.SERVICE_COLS:
        assert not cleaned[col].isin(["No internet service", "No phone service"]).any()
    assert set(cleaned["SeniorCitizen"].unique()) == {"Yes", "No"}


def test_tenure_group_created(cleaned):
    assert set(cleaned["TenureGroup"].unique()) == set(data.TENURE_ORDER)


def test_missing_required_column_raises(raw):
    with pytest.raises(ValueError):
        data.clean(raw.drop(columns=["Contract"]))


def test_invalid_churn_value_raises(raw):
    bad = raw.copy()
    bad.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        data.clean(bad)
