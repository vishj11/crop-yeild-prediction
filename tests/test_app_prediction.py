import json

import numpy as np
import pandas as pd
import pytest

from src.config import METADATA_FILE, MODEL_FILE
from src.predict import (
    actual_yield, available_states, available_years, build_model_input,
    district_options, historical_yields, load_model, predict_yield,
)


def history(years=(2010, 2011, 2012), yields=(1000.0, 1200.0, 1400.0)):
    return pd.DataFrame([
        {"Crop": "Rice", "State": "Example State", "District": "Same Name",
         "District_Code": "007", "Year": year, "Yield": crop_yield}
        for year, crop_yield in zip(years, yields)
    ])


def test_exact_historical_lookup_and_target_year_not_required():
    data = history()
    assert available_years(data, "Rice", "Example State", "007") == [2013]
    found = historical_yields(data, "Rice", "Example State", "007", 2013)
    assert found["Year"].tolist() == [2010, 2011, 2012]
    assert found["Historical Yield (kg/ha)"].tolist() == [1000, 1200, 1400]
    assert actual_yield(data, "Rice", "Example State", "007", 2013) is None


def test_missing_year_and_invalid_yield_reject_prediction():
    missing = history(years=(2010, 2012), yields=(1000, 1400))
    assert available_years(missing, "Rice", "Example State", "007") == []
    with pytest.raises(ValueError, match="2011"):
        historical_yields(missing, "Rice", "Example State", "007", 2013)
    bad = history(yields=(1000, float("nan"), 1400))
    assert available_years(bad, "Rice", "Example State", "007") == []
    with pytest.raises(ValueError, match="positive and finite"):
        historical_yields(bad, "Rice", "Example State", "007", 2013)
    zero = history(yields=(1000, 0, 1400))
    assert available_years(zero, "Rice", "Example State", "007") == []


def test_district_code_and_duplicate_handling():
    second = history(yields=(9000, 9000, 9000)).assign(District_Code="008")
    data = pd.concat([history(), second], ignore_index=True)
    assert available_states(data, "Rice") == ["Example State"]
    assert district_options(data, "Rice", "Example State") == {"007": "Same Name", "008": "Same Name"}
    assert historical_yields(data, "Rice", "Example State", "007", 2013)[
        "Historical Yield (kg/ha)"].tolist() == [1000, 1200, 1400]
    duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    assert available_years(duplicate, "Rice", "Example State", "007") == []
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        historical_yields(duplicate, "Rice", "Example State", "007", 2013)


def test_saved_model_input_schema_prediction_and_no_target_leakage():
    metadata = json.loads(METADATA_FILE.read_text())
    model = load_model(MODEL_FILE)
    data = history()
    # Adding an extreme target-year yield must not change the input or prediction.
    input_before, prior = build_model_input(data, "Rice", "Example State", "007", 2013, metadata, model)
    with_target = pd.concat([data, history(years=(2013,), yields=(99999,))], ignore_index=True)
    input_after, _ = build_model_input(with_target, "Rice", "Example State", "007", 2013, metadata, model)
    pd.testing.assert_frame_equal(input_before, input_after)
    assert list(input_before.columns) == metadata["input_features"]
    assert set(input_before.columns).isdisjoint({"Yield", "Production", "Area", "Split"})
    assert input_before["Previous_Year_Yield"].iloc[0] == 1400
    assert input_before["Previous_3_Year_Avg_Yield"].iloc[0] == 1200
    prediction = predict_yield(model, input_before)
    assert np.isfinite(prediction) and prediction >= 0
    assert prediction == pytest.approx(float(model.predict(input_before)[0]))
    assert prior["Year"].tolist() == [2010, 2011, 2012]


def test_bad_schema_and_prediction_values_are_reported():
    model = load_model(MODEL_FILE)
    metadata = json.loads(METADATA_FILE.read_text())
    wrong = {**metadata, "input_features": ["Yield"]}
    with pytest.raises(ValueError, match="metadata"):
        build_model_input(history(), "Rice", "Example State", "007", 2013, wrong, model)

    class BadModel:
        feature_names_in_ = np.array(["Value"])

        def __init__(self, value):
            self.value = value

        def predict(self, frame):
            return np.array([self.value])

    frame = pd.DataFrame({"Value": [1]})
    for value, message in ((float("nan"), "invalid prediction"),
                           (float("inf"), "invalid prediction"),
                           (-1, "negative yield")):
        with pytest.raises(ValueError, match=message):
            predict_yield(BadModel(value), frame)
    with pytest.raises(ValueError, match="schema"):
        predict_yield(BadModel(1), pd.DataFrame({"Wrong": [1]}))
