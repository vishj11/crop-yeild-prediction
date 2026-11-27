import pandas as pd
import pytest

from src.data_loader import required_columns
from src.preprocess import clean_data, rounding_can_explain, to_long


def sample_row(year=2000):
    row = {column: "0" for column in required_columns()}
    row.update({
        "Dist Code": "007", "Year": str(year), "State Name": "Example State",
        "Dist Name": "Example District",
        "RICE AREA (1000 ha)": "2", "RICE PRODUCTION (1000 tons)": "3",
        "RICE YIELD (Kg per ha)": "1500",
        "WHEAT AREA (1000 ha)": "1", "WHEAT PRODUCTION (1000 tons)": "2",
        "WHEAT YIELD (Kg per ha)": "2000",
        "MAIZE AREA (1000 ha)": "1", "MAIZE PRODUCTION (1000 tons)": "1",
        "MAIZE YIELD (Kg per ha)": "1000",
    })
    return row


def test_wide_to_long_and_units():
    long, cleaned, issues = clean_data(pd.DataFrame([sample_row()]))
    assert len(long) == len(cleaned) == 3
    rice = cleaned.set_index("Crop").loc["Rice"]
    assert rice["District_Code"] == "007"
    assert rice["Area"] == 2  # thousand hectares, not hectares
    assert rice["Production"] == 3  # thousand tons, not tons
    assert rice["Yield"] == 1500  # kg/ha = 1000 * 3 / 2
    assert "yield_discrepancy" not in set(issues["Issue"])


def test_duplicates_and_conflicts_are_reported():
    original = sample_row()
    changed = sample_row()
    changed["RICE YIELD (Kg per ha)"] = "1700"
    _, cleaned, issues = clean_data(pd.DataFrame([original, original, changed]))
    assert (issues["Issue"] == "exact_duplicate").sum() == 5
    assert (issues["Issue"] == "conflicting_record").sum() == 2
    assert "Rice" not in set(cleaned["Crop"])
    assert len(cleaned) == 2


def test_missing_and_nonpositive_yield_are_excluded():
    blank = sample_row(2000)
    blank["RICE YIELD (Kg per ha)"] = ""
    zero = sample_row(2001)
    zero["RICE YIELD (Kg per ha)"] = "0"
    invalid = sample_row(2002)
    invalid["RICE AREA (1000 ha)"] = "bad"
    _, cleaned, issues = clean_data(pd.DataFrame([blank, zero, invalid]))
    rice = cleaned[cleaned["Crop"] == "Rice"]
    assert rice.empty
    assert "missing_yield" in set(issues["Issue"])
    assert "nonpositive_yield" in set(issues["Issue"])
    assert "invalid_area" in set(issues["Issue"])
    assert (issues["Issue"] == "excluded_record").sum() == 3


def test_invalid_area_and_production_are_excluded_and_logged():
    zero_area = sample_row(2000)
    zero_area["RICE AREA (1000 ha)"] = "0.0"
    zero_production = sample_row(2001)
    zero_production["RICE PRODUCTION (1000 tons)"] = "0.0"
    negative_area = sample_row(2002)
    negative_area["RICE AREA (1000 ha)"] = "-1"
    negative_production = sample_row(2003)
    negative_production["RICE PRODUCTION (1000 tons)"] = "-1"
    _, cleaned, issues = clean_data(pd.DataFrame([zero_area, zero_production, negative_area, negative_production]))
    assert cleaned[cleaned["Crop"] == "Rice"].empty
    excluded = issues[(issues["Issue"] == "excluded_record") & (issues["Crop"] == "Rice")]
    assert set(excluded["Year"]) == {2000, 2001, 2002, 2003}
    assert "nonpositive_area" in excluded.loc[excluded["Year"] == 2000, "Details"].iloc[0]
    assert "nonpositive_production" in excluded.loc[excluded["Year"] == 2001, "Details"].iloc[0]
    assert "nonpositive_production" in excluded.loc[excluded["Year"] == 2003, "Details"].iloc[0]


def test_missing_numeric_is_not_zero_and_missing_identifiers_are_excluded():
    blank_area = sample_row(2000)
    blank_area["RICE AREA (1000 ha)"] = pd.NA
    blank_code = sample_row(2001)
    blank_code["Dist Code"] = pd.NA
    blank_name = sample_row(2002)
    blank_name["Dist Name"] = ""
    long, cleaned, issues = clean_data(pd.DataFrame([blank_area, blank_code, blank_name]))
    assert pd.isna(long[(long["Crop"] == "Rice") & (long["Year"] == 2000)]["Area"].iloc[0])
    assert cleaned[cleaned["Crop"] == "Rice"].empty
    assert ((issues["Issue"] == "missing_area") & (issues["Year"] == 2000)).any()
    assert ((issues["Issue"] == "missing_identifier") & (issues["Year"] == 2001)).any()
    assert ((issues["Issue"] == "missing_identifier") & (issues["Year"] == 2002)).any()


def test_discrepancy_uses_source_values_and_checks_rounding():
    rounded = sample_row()
    rounded.update({"RICE AREA (1000 ha)": "0.01", "RICE PRODUCTION (1000 tons)": "0.02",
                    "RICE YIELD (Kg per ha)": "2285.71"})
    _, cleaned, issues = clean_data(pd.DataFrame([rounded]))
    detail = issues[(issues["Issue"] == "yield_discrepancy") & (issues["Crop"] == "Rice")]["Details"].iloc[0]
    assert "area=0.01" in detail and "production=0.02" in detail
    assert "yield=2285.71" in detail and "rounding_compatible=True" in detail
    assert "Rice" in set(cleaned["Crop"])
    assert rounding_can_explain("0.01", "0.02", "2285.71")
    assert not rounding_can_explain("1.0", "1.0", "5000.0")


def test_high_yield_is_flagged_and_retained():
    rows = [sample_row(year) for year in range(2000, 2010)]
    high = sample_row(2010)
    high["MAIZE PRODUCTION (1000 tons)"] = "9"
    high["MAIZE YIELD (Kg per ha)"] = "9000"
    rows.append(high)
    _, cleaned, issues = clean_data(pd.DataFrame(rows))
    assert ((cleaned["Crop"] == "Maize") & (cleaned["Year"] == 2010)).any()
    assert ((issues["Issue"] == "suspicious_yield_outlier") & (issues["Year"] == 2010)).any()


def test_required_headers_are_checked():
    with pytest.raises(ValueError, match="Missing required CSV columns"):
        to_long(pd.DataFrame([{"Year": "2000"}]))
