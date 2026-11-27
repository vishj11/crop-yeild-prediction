"""Turn crop columns into rows, audit quality, and retain usable targets."""

from decimal import Decimal

import numpy as np
import pandas as pd

from .config import CROPS, ISSUE_COLUMNS, LONG_COLUMNS, MEASURES
from .data_loader import required_columns


def to_long(raw):
    """Return the three supported crops, with the source units unchanged."""
    missing = sorted(set(required_columns()) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    year = pd.to_numeric(raw["Year"].fillna("").astype(str).str.strip(), errors="coerce")
    if year.isna().any() or (year % 1 != 0).any():
        raise ValueError("Year must be a nonmissing integer in every source row")
    parts = []
    for crop in CROPS:
        part = pd.DataFrame({
            "State": raw["State Name"].fillna("").astype(str).str.strip(),
            "District": raw["Dist Name"].fillna("").astype(str).str.strip(),
            "District_Code": raw["Dist Code"].fillna("").astype(str).str.strip(),
            "Year": year.astype(int),
            "Crop": crop,
        })
        for name, measure in zip(("Area", "Production", "Yield"), MEASURES):
            part[name] = pd.to_numeric(
                raw[f"{crop.upper()} {measure}"].fillna("").astype(str).str.strip(), errors="coerce"
            )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[list(LONG_COLUMNS)]


def _issue_rows(data, mask, category, details):
    columns = ["State", "District", "District_Code", "Year", "Crop"]
    found = data.loc[mask, columns].copy()
    found.insert(0, "Issue", category)
    found["Details"] = details
    return found[list(ISSUE_COLUMNS)]


def rounding_can_explain(area_text, production_text, yield_text):
    """Can source rounding alone make reported yield equal production/area?"""
    area, production, crop_yield = map(Decimal, (area_text, production_text, yield_text))
    area_unit, production_unit, yield_unit = [
        Decimal(1).scaleb(value.as_tuple().exponent)
        for value in (area, production, crop_yield)
    ]
    area_low = max(Decimal(0), area - area_unit / 2)
    area_high = area + area_unit / 2
    production_low = max(Decimal(0), production - production_unit / 2)
    production_high = production + production_unit / 2
    possible_low = 1000 * production_low / area_high
    possible_high = (
        1000 * production_high / area_low if area_low > 0 else Decimal("Infinity")
    )
    reported_low = crop_yield - yield_unit / 2
    reported_high = crop_yield + yield_unit / 2
    return reported_low <= possible_high and reported_high >= possible_low


def clean_data(raw):
    """Return (all crop rows, quality-screened rows, issue rows)."""
    long = to_long(raw)
    issues = []
    source_values = {}
    for name in ("Area", "Production", "Yield"):
        source = pd.concat(
            [raw[f"{crop.upper()} {measure}"].fillna("").astype(str).str.strip()
             for crop in CROPS
             for candidate, measure in zip(("Area", "Production", "Yield"), MEASURES)
             if candidate == name], ignore_index=True
        )
        source_values[name] = source
        missing = source.eq("")
        invalid = long[name].isna() & ~missing
        issues.append(_issue_rows(long, missing, f"missing_{name.lower()}", "Blank source value"))
        issues.append(_issue_rows(long, invalid, f"invalid_{name.lower()}", "Non-numeric source value"))

    identifier_missing = long[["State", "District", "District_Code"]].eq("").any(axis=1)
    issues.append(_issue_rows(long, identifier_missing, "missing_identifier", "Blank state, district, or district code"))
    invalid_area = long["Area"].lt(0) | (long["Area"].eq(0) & long["Yield"].gt(0))
    invalid_production = long["Production"].lt(0) | (long["Production"].eq(0) & long["Yield"].gt(0))
    issues.append(_issue_rows(long, invalid_area, "invalid_area", "Negative area, or recorded zero area with positive yield; excluded if yield is positive"))
    issues.append(_issue_rows(long, invalid_production, "invalid_production", "Negative production, or recorded zero production with positive yield; excluded if yield is positive"))
    issues.append(_issue_rows(long, long["Yield"].notna() & long["Yield"].le(0),
                              "nonpositive_yield", "Excluded from modeling; zero can mean no crop"))

    # Area is 1000 ha and production is 1000 tons. Their ratio is tons/ha;
    # multiplying by 1000 gives kg/ha. Allow source rounding in small values.
    valid_ratio = long["Area"].gt(0) & long["Production"].ge(0) & long["Yield"].gt(0)
    calculated = 1000 * long["Production"] / long["Area"]
    discrepancy = valid_ratio & (calculated - long["Yield"]).abs().gt(
        (long["Yield"] * 0.10).clip(lower=50)
    )
    mismatch = _issue_rows(long, discrepancy, "yield_discrepancy", "Reported yield differs from 1000 × production / area by more than max(50 kg/ha, 10%)")
    if not mismatch.empty:
        mismatch["Details"] = [
            (f"Source area={source_values['Area'].iloc[index]} (1000 ha), "
             f"production={source_values['Production'].iloc[index]} (1000 tons), "
             f"yield={source_values['Yield'].iloc[index]} (kg/ha); "
             f"calculated={calculated.iloc[index]:.2f} kg/ha; "
             f"rounding_compatible={rounding_can_explain(source_values['Area'].iloc[index], source_values['Production'].iloc[index], source_values['Yield'].iloc[index])}")
            for index in long.index[discrepancy]
        ]
    issues.append(mismatch)

    duplicates = long.duplicated(subset=list(LONG_COLUMNS), keep="first")
    issues.append(_issue_rows(long, duplicates, "exact_duplicate", "Identical row removed"))
    unique = long.loc[~duplicates].copy()
    key = ["State", "District_Code", "Crop", "Year"]
    conflicts = unique.duplicated(subset=key, keep=False)
    issues.append(_issue_rows(unique, conflicts, "conflicting_record", "Multiple distinct rows for the same state, district code, crop, and year; all excluded"))
    excluded = pd.DataFrame({
        "conflicting_record": conflicts,
        "missing_identifier": identifier_missing.loc[unique.index],
        "missing_yield": unique["Yield"].isna(),
        "nonpositive_yield": unique["Yield"].le(0),
        "missing_or_invalid_area": unique["Yield"].gt(0) & unique["Area"].isna(),
        "nonpositive_area": unique["Yield"].gt(0) & unique["Area"].le(0),
        "missing_or_invalid_production": unique["Yield"].gt(0) & unique["Production"].isna(),
        "nonpositive_production": unique["Yield"].gt(0) & unique["Production"].le(0),
    }, index=unique.index)
    # A source row with zero yield and zero area/production is a no-crop row;
    # the yield reason alone is sufficient. All positive-yield rows need both
    # recorded measures to be positive, even if rounded zero is plausible.
    drop = excluded.any(axis=1)
    removed = _issue_rows(unique, drop, "excluded_record", "")
    removed["Details"] = [
        ", ".join(excluded.columns[excluded.loc[index]].tolist())
        for index in unique.index[drop]
    ]
    issues.append(removed)
    cleaned = unique.loc[~drop].copy()
    cleaned = cleaned.sort_values(["Crop", "State", "District_Code", "Year"]).reset_index(drop=True)
    # A conservative, crop-specific review flag. These rows remain in the data.
    for crop in CROPS:
        subset = cleaned.loc[cleaned["Crop"] == crop]
        if subset.empty:
            continue
        q1, q3 = np.percentile(subset["Yield"], [25, 75])
        cutoff = q3 + 5 * (q3 - q1)
        outliers = cleaned["Crop"].eq(crop) & cleaned["Yield"].gt(cutoff)
        issues.append(_issue_rows(cleaned, outliers, "suspicious_yield_outlier",
                                  f"Yield above {cutoff:.2f} kg/ha (Q3 + 5 × IQR for {crop}); retained for review"))
    report = pd.concat(issues, ignore_index=True)
    return long, cleaned, report
