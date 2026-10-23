"""Headless smoke test of the Streamlit dashboard (needs `python train.py` to have been run)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"
ARTEFACT = Path(__file__).resolve().parent.parent / "data" / "customer_predictions.csv"


@pytest.mark.skipif(not ARTEFACT.exists(), reason="run train.py first")
def test_dashboard_runs_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception
    assert len(at.tabs) == 8
    labels = [m.label for m in at.metric]
    assert "High-risk customers" in labels and "Predicted churn probability" in labels


@pytest.mark.skipif(not ARTEFACT.exists(), reason="run train.py first")
def test_empty_filter_shows_warning_not_crash():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("No customers" in w.value for w in at.warning)
