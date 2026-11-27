import joblib
import numpy as np
import pandas as pd
import pytest

from src.predict import HistoricalBaseline, load_model
from src.train import add_high_yield_flags, load_feature_data, make_model, model_input_columns


def example_data():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize", "Rice", "Wheat", "Maize"],
        "State": ["A", "A", "B", "A", "A", "B"],
        "District_Code": ["001", "002", "003", "001", "002", "003"],
        "Year": [2007, 2007, 2007, 2008, 2008, 2008],
        "Previous_Year_Yield": [100, 200, 300, 110, 220, 330],
        "Previous_3_Year_Avg_Yield": [95, 195, 295, 100, 200, 300],
        "Yield": [110, 220, 330, 120, 240, 360],
        "Split": ["train"] * 6,
    })


def test_inputs_exclude_target_and_future_information():
    assert model_input_columns() == ["Crop", "State", "District_Code",
                                     "Previous_Year_Yield", "Previous_3_Year_Avg_Yield"]
    assert model_input_columns(True)[-1] == "Year"
    for forbidden in ("Yield", "Production", "Area", "Split", "High_Yield_Flag"):
        assert forbidden not in model_input_columns(True)


def test_chronological_labels_are_checked(tmp_path):
    frame = example_data()
    frame.loc[0, "Split"] = "test"
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="chronological"):
        load_feature_data(path)


def test_baseline_and_ridge_prediction_shape_and_reload(tmp_path):
    frame = example_data()
    baseline = HistoricalBaseline().fit(frame[model_input_columns()], frame["Yield"])
    np.testing.assert_array_equal(baseline.predict(frame), frame["Previous_3_Year_Avg_Yield"])
    columns = model_input_columns()
    ridge = make_model("Ridge").fit(frame[columns], frame["Yield"])
    before = ridge.predict(frame[columns])
    assert before.shape == (len(frame),)
    path = tmp_path / "ridge.joblib"
    joblib.dump(ridge, path)
    np.testing.assert_allclose(load_model(path).predict(frame[columns]), before)


def test_random_forest_is_reproducible():
    frame = example_data()
    columns = model_input_columns(True)
    first = make_model("Random Forest", with_year=True, min_samples_leaf=2)
    second = make_model("Random Forest", with_year=True, min_samples_leaf=2)
    first.fit(frame[columns], frame["Yield"])
    second.fit(frame[columns], frame["Yield"])
    # Parallel tree prediction can differ at floating-point roundoff scale.
    np.testing.assert_allclose(first.predict(frame[columns]), second.predict(frame[columns]),
                               rtol=1e-12, atol=1e-10)


def test_high_yield_flags_join_by_identifier_not_position(tmp_path):
    frame = example_data()
    frame["District"] = "Same name"
    issue = pd.DataFrame([{"Issue": "suspicious_yield_outlier", "State": "A",
                           "District_Code": "002", "Crop": "Wheat", "Year": 2008}])
    path = tmp_path / "issues.csv"
    issue.to_csv(path, index=False)
    found = add_high_yield_flags(frame.sample(frac=1, random_state=1), path)
    assert found["High_Yield_Flag"].sum() == 1
    flagged = found[found["High_Yield_Flag"]].iloc[0]
    assert (flagged["Crop"], flagged["District_Code"], flagged["Year"]) == ("Wheat", "002", 2008)
