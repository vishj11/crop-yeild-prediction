"""Small Streamlit widget check; no web server is needed."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import CLEAN_FILE


pytestmark = pytest.mark.skipif(
    not CLEAN_FILE.is_file(),
    reason="UI integration check requires locally prepared ICRISAT data",
)


def test_dependent_dropdowns_prediction_and_no_history_message():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert [item.label for item in app.selectbox] == [
        "Crop", "State", "District", "Prediction year"
    ]

    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].label == "Predicted Yield"
    assert app.metric[0].value == "3,569 kg/ha"
    assert [item.value for item in app.subheader] == [
        "Prediction Parameters", "Prediction Results", "Historical Yield Trends", "Model Performance"
    ]
    assert len(app.get("vega_lite_chart")) == 1
    assert [item.label for item in app.get("expander")] == [
        "View historical values", "Historical Evaluation — Actual vs. Predicted", "About This Model"
    ]

    app.selectbox[0].select("Wheat").run()
    assert not app.exception
    assert len(app.selectbox) == 3  # default district lacks three valid prior years
    assert "No 2013–2017 prediction year" in app.info[0].value

    app.selectbox[2].select("Kurnool (code 51)").run()
    assert len(app.selectbox) == 4
    assert app.selectbox[3].value == 2013
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "1,675 kg/ha"

    app.selectbox[0].select("Maize").run()
    assert not app.exception
    assert app.selectbox[1].value == "Andhra Pradesh"
    assert "Kurnool (code 51)" in app.selectbox[2].options
    app.selectbox[2].select("Kurnool (code 51)").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "5,498 kg/ha"
