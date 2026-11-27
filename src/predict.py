"""The historical baseline and helpers for loading a saved predictor."""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .config import CLEAN_FILE, CROPS


class HistoricalBaseline(RegressorMixin, BaseEstimator):
    """Predict each row's mean yield from its previous three calendar years."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        values = np.asarray(X["Previous_3_Year_Avg_Yield"], dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Baseline requires a finite three-year average for every row")
        return values


def load_model(path):
    """Load the complete fitted estimator or scikit-learn pipeline."""
    return joblib.load(path)


def load_history(path=CLEAN_FILE):
    """Read cleaned district history, preserving district codes as text."""
    data = pd.read_csv(path, dtype={"District_Code": str})
    required = {"State", "District", "District_Code", "Crop", "Year", "Yield"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Historical data is missing columns: {sorted(missing)}")
    data["Yield"] = pd.to_numeric(data["Yield"], errors="coerce")
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    if data["Year"].isna().any() or (data["Year"] % 1 != 0).any():
        raise ValueError("Historical data contains an invalid year")
    data["Year"] = data["Year"].astype(int)
    return data


def available_states(history, crop):
    if crop not in CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    return sorted(history.loc[history["Crop"].eq(crop), "State"].dropna().unique().tolist())


def district_options(history, crop, state):
    """Return {district_code: display_name} for the selected crop and state."""
    if state not in available_states(history, crop):
        raise ValueError("The selected state has no records for this crop")
    rows = history.loc[history["Crop"].eq(crop) & history["State"].eq(state)]
    rows = rows.sort_values("Year").drop_duplicates("District_Code", keep="last")
    return {str(row.District_Code): str(row.District)
            for row in rows.itertuples(index=False) if pd.notna(row.District_Code)}


def _district_rows(history, crop, state, district_code):
    if crop not in CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    rows = history.loc[
        history["Crop"].eq(crop) & history["State"].eq(state)
        & history["District_Code"].eq(str(district_code)),
        ["Year", "Yield"],
    ]
    if rows.empty:
        raise ValueError("No history found for this crop, state, and district code")
    return rows


def available_years(history, crop, state, district_code, first=2013, last=2017):
    """Offer a year only when Y−1, Y−2, and Y−3 are unique positive records."""
    rows = _district_rows(history, crop, state, district_code)
    counts = rows.groupby("Year").size()
    valid = rows.loc[rows["Yield"].notna() & np.isfinite(rows["Yield"]) & rows["Yield"].gt(0), "Year"]
    valid_years = set(valid)
    return [year for year in range(first, last + 1)
            if all(counts.get(year - lag, 0) == 1 and year - lag in valid_years
                   for lag in (1, 2, 3))]


def historical_yields(history, crop, state, district_code, year):
    """Return only the three exact prior calendar years, oldest first."""
    rows = _district_rows(history, crop, state, district_code)
    found = []
    for prior_year in (year - 3, year - 2, year - 1):
        match = rows.loc[rows["Year"].eq(prior_year), "Yield"]
        if match.empty:
            raise ValueError(f"Missing historical yield for {prior_year}")
        if len(match) != 1:
            raise ValueError(f"Ambiguous duplicate historical yields for {prior_year}")
        value = float(match.iloc[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Historical yield for {prior_year} must be positive and finite")
        found.append({"Year": prior_year, "Historical Yield (kg/ha)": value})
    return pd.DataFrame(found)


def build_model_input(history, crop, state, district_code, year, metadata, model):
    """Build exactly the saved model's input columns from observed prior yields."""
    history_table = historical_yields(history, crop, state, district_code, year)
    values = history_table["Historical Yield (kg/ha)"].to_numpy()
    inputs = {
        "Crop": crop,
        "State": state,
        "District_Code": str(district_code),
        "Previous_Year_Yield": float(values[-1]),
        "Previous_3_Year_Avg_Yield": float(values.mean()),
        "Year": int(year),
    }
    expected = metadata.get("input_features")
    if not isinstance(expected, list) or set(expected) != set(inputs) or len(expected) != len(inputs):
        raise ValueError("Model metadata does not match the permitted historical inputs")
    model_schema = getattr(model, "feature_names_in_", None)
    if model_schema is None or list(model_schema) != expected:
        raise ValueError("Saved model input schema does not match the metadata")
    return pd.DataFrame([inputs], columns=expected), history_table


def predict_yield(model, model_input):
    """Return one physically meaningful kg/ha prediction or a clear error."""
    model_schema = getattr(model, "feature_names_in_", None)
    if model_schema is None or list(model_schema) != list(model_input.columns):
        raise ValueError("Prediction columns do not match the saved model schema")
    result = np.asarray(model.predict(model_input))
    if result.shape != (1,) or not np.isfinite(result[0]):
        raise ValueError("The model returned an invalid prediction")
    prediction = float(result[0])
    if prediction < 0:
        raise ValueError("The model predicted a negative yield. This is outside the physically meaningful range and is not a valid agricultural estimate")
    return prediction


def actual_yield(history, crop, state, district_code, year):
    """Look up the target year only for optional retrospective evaluation."""
    rows = _district_rows(history, crop, state, district_code)
    match = rows.loc[rows["Year"].eq(year), "Yield"]
    if match.empty:
        return None
    if len(match) != 1:
        raise ValueError(f"Ambiguous duplicate actual yields for {year}")
    value = float(match.iloc[0])
    return value if np.isfinite(value) and value > 0 else None
