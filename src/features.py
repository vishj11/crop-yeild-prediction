"""Calendar-year lag features and chronological evaluation labels."""

import pandas as pd

from .config import CROPS, ISSUE_COLUMNS

PREDICTOR_COLUMNS = (
    "State", "District", "District_Code", "Year", "Crop",
    "Previous_Year_Yield", "Previous_3_Year_Avg_Yield",
)
FEATURE_COLUMNS = (
    *PREDICTOR_COLUMNS, "Yield", "Split",
)


def assign_split(year):
    """The agreed periods; years outside them are not assigned a split."""
    if year <= 2008:
        return "train"
    if 2009 <= year <= 2012:
        return "validation"
    if 2013 <= year <= 2017:
        return "test"
    raise ValueError(f"Year {year} is outside the configured evaluation periods")


def engineer_features(cleaned, all_rows=None):
    """Return (complete-history rows, history issues).

    Lag lookups use Year - 1, - 2, and - 3, never row position.
    """
    keys = ["State", "District_Code", "Crop"]
    if cleaned.duplicated(keys + ["Year"]).any():
        raise ValueError("Cannot build lag features with duplicate district/crop/year keys")
    data = cleaned.sort_values(keys + ["Year"]).copy()
    history = cleaned[keys + ["Year", "Yield"]]
    for lag in (1, 2, 3):
        lookup = history.rename(columns={"Yield": f"Yield_{lag}"}).copy()
        lookup["Year"] += lag
        data = data.merge(lookup, on=keys + ["Year"], how="left", validate="one_to_one")
    missing = data[["Yield_1", "Yield_2", "Yield_3"]].isna().any(axis=1)
    issue_rows = []
    known_years = set()
    if all_rows is not None:
        known_years = set(
            zip(all_rows["State"], all_rows["District_Code"], all_rows["Crop"], all_rows["Year"])
        )
    first_year = data.groupby(keys)["Year"].transform("min")
    for row_index in data.index[missing]:
        row = data.loc[row_index]
        absent = [int(row["Year"] - lag) for lag in (1, 2, 3) if pd.isna(row[f"Yield_{lag}"])]
        base = {
            "State": row["State"], "District": row["District"],
            "District_Code": row["District_Code"], "Year": row["Year"], "Crop": row["Crop"],
        }
        issue_rows.append({"Issue": "insufficient_lag_history", **base,
                           "Details": f"No usable positive yield for required years {absent}; row excluded"})
        gaps = [year for year in absent if year >= first_year.loc[row_index]
                and (row["State"], row["District_Code"], row["Crop"], year) not in known_years]
        if gaps and all_rows is not None:
            issue_rows.append({"Issue": "missing_historical_year", **base,
                               "Details": f"No source row for years {gaps}"})
    usable = data.loc[~missing].copy()
    usable["Previous_Year_Yield"] = usable["Yield_1"]
    usable["Previous_3_Year_Avg_Yield"] = usable[["Yield_1", "Yield_2", "Yield_3"]].mean(axis=1)
    usable["Split"] = usable["Year"].map(assign_split)
    usable = usable[list(FEATURE_COLUMNS)].sort_values(keys + ["Year"]).reset_index(drop=True)
    return usable, pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)


def validate_split_coverage(features, minimum_per_crop=30):
    """Fail if the proposed periods cannot support all three crops."""
    counts = features.groupby(["Crop", "Split"]).size()
    for crop in CROPS:
        for split in ("train", "validation", "test"):
            if counts.get((crop, split), 0) < minimum_per_crop:
                raise ValueError(f"Too few usable {crop} {split} rows: {counts.get((crop, split), 0)}")
    for year in range(2009, 2018):
        for crop in CROPS:
            if not ((features["Year"] == year) & (features["Crop"] == crop)).any():
                raise ValueError(f"No usable {crop} rows in {year}")
