import pandas as pd
import pytest

from src.features import PREDICTOR_COLUMNS, assign_split, engineer_features, validate_split_coverage


def records(years_and_yields, code="007"):
    return pd.DataFrame([
        {"State": "Example", "District": "A", "District_Code": code,
         "Year": year, "Crop": "Rice", "Area": 1, "Production": crop_yield / 1000,
         "Yield": crop_yield}
        for year, crop_yield in years_and_yields
    ])


def test_exact_previous_year_and_three_year_average():
    data = records([(2000, 100), (2001, 200), (2002, 300), (2003, 400), (2004, 9999)])
    features, issues = engineer_features(data, data)
    assert features["Year"].tolist() == [2003, 2004]
    row = features.set_index("Year").loc[2003]
    assert row["Previous_Year_Yield"] == 300
    assert row["Previous_3_Year_Avg_Yield"] == 200
    assert "Production" not in features.columns
    assert (issues["Issue"] == "insufficient_lag_history").sum() == 3


def test_calendar_gap_and_future_values_never_fill_history():
    data = records([(2000, 100), (2001, 200), (2003, 400), (2004, 500), (2005, 600), (2006, 700)])
    features, issues = engineer_features(data, data)
    assert features["Year"].tolist() == [2006]
    assert features.loc[0, "Previous_Year_Yield"] == 600
    assert features.loc[0, "Previous_3_Year_Avg_Yield"] == 500
    missing = issues[(issues["Year"] == 2003) & (issues["Issue"] == "missing_historical_year")]
    assert len(missing) == 1


def test_district_codes_prevent_name_based_history_merging():
    first = records([(2000, 100), (2001, 200), (2002, 300), (2003, 400)], code="007")
    second = records([(2000, 900), (2001, 800), (2002, 700), (2003, 600)], code="008")
    second["District"] = "A"  # same district name, different stable code
    data = pd.concat([first, second], ignore_index=True)
    features, _ = engineer_features(data, data)
    assert features.set_index("District_Code").loc["007", "Previous_Year_Yield"] == 300
    assert features.set_index("District_Code").loc["008", "Previous_Year_Yield"] == 700


def test_current_year_target_and_future_yield_do_not_enter_inputs():
    base = records([(2000, 100), (2001, 200), (2002, 300), (2003, 400), (2004, 500)])
    changed = base.copy()
    changed.loc[changed["Year"] >= 2003, "Yield"] = [9999, 8888]
    before, _ = engineer_features(base, base)
    after, _ = engineer_features(changed, changed)
    cols = ["Previous_Year_Yield", "Previous_3_Year_Avg_Yield"]
    assert before.loc[before["Year"] == 2003, cols].reset_index(drop=True).equals(
        after.loc[after["Year"] == 2003, cols].reset_index(drop=True)
    )
    assert after.loc[after["Year"] == 2003, "Yield"].iloc[0] == 9999
    assert "Production" not in after.columns and "Area" not in after.columns
    assert "Yield" not in PREDICTOR_COLUMNS and "Split" not in PREDICTOR_COLUMNS
    assert set(PREDICTOR_COLUMNS).issubset(after.columns)


@pytest.mark.parametrize("year,expected", [(2008, "train"), (2009, "validation"),
                                           (2012, "validation"), (2013, "test"),
                                           (2017, "test")])
def test_chronological_split(year, expected):
    assert assign_split(year) == expected


def test_split_coverage_rejects_empty_periods():
    data = records([(2000, 100), (2001, 200), (2002, 300), (2003, 400)])
    features, _ = engineer_features(data, data)
    with pytest.raises(ValueError, match="Too few usable"):
        validate_split_coverage(features)
