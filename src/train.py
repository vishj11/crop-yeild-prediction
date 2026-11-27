"""Compare small, chronological experiments and save the validation winner.

Run from the project root with: python -m src.train
"""

import json
import platform

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FEATURE_FILE, ISSUES_FILE, ROOT, MODEL_FILE, METADATA_FILE, FINAL_TEST_FILE
from .evaluate import per_crop_metrics, regression_metrics, save_test_plots
from .features import assign_split
from .predict import HistoricalBaseline

SEED = 42
CATEGORICAL = ("Crop", "State", "District_Code")
NUMERIC = ("Previous_Year_Yield", "Previous_3_Year_Avg_Yield")
COMPARISON_FILE = ROOT / "reports/model_comparison.csv"
PER_CROP_FILE = ROOT / "reports/per_crop_metrics.csv"
FINAL_FILE = FINAL_TEST_FILE
SCATTER_FILE = ROOT / "reports/actual_vs_predicted.png"
ERRORS_FILE = ROOT / "reports/prediction_errors.png"


def model_input_columns(with_year=False):
    """Only values available before the current year's yield is observed."""
    return [*CATEGORICAL, *NUMERIC, *(["Year"] if with_year else [])]


def load_feature_data(path=FEATURE_FILE):
    data = pd.read_csv(path, dtype={"District_Code": str})
    required = set(model_input_columns(True)) | {"Yield", "Split"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Features, target, and split labels must be complete")
    if data.duplicated(["State", "District_Code", "Crop", "Year"]).any():
        raise ValueError("Duplicate district/crop/year feature rows")
    expected = data["Year"].map(assign_split)
    if not expected.equals(data["Split"]):
        raise ValueError("Split labels do not match chronological boundaries")
    return data


def add_high_yield_flags(data, issues_path=ISSUES_FILE):
    """Match Phase 2 flags by stable identity and year, never by row order."""
    issues = pd.read_csv(issues_path, dtype={"District_Code": str})
    keys = ["State", "District_Code", "Crop", "Year"]
    flagged = issues.loc[issues["Issue"] == "suspicious_yield_outlier", keys].drop_duplicates()
    merged = data.merge(flagged.assign(High_Yield_Flag=True), on=keys,
                        how="left", validate="one_to_one")
    merged["High_Yield_Flag"] = merged["High_Yield_Flag"].fillna(False).astype(bool)
    return merged


def make_model(model_type, with_year=False, min_samples_leaf=None):
    if model_type == "Historical baseline":
        return HistoricalBaseline()
    numeric = list(NUMERIC) + (["Year"] if with_year else [])
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if model_type == "Ridge":
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(CATEGORICAL)),
    ])
    if model_type == "Ridge":
        estimator = Ridge(alpha=10.0)
    elif model_type == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=60, max_depth=14, min_samples_leaf=min_samples_leaf,
            random_state=SEED, n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def configurations():
    """One baseline and a small validation-only model/feature search."""
    yield {"Model": "Historical baseline", "With_Year": False,
           "Exclude_Flagged_Training": False, "Min_Samples_Leaf": None}
    for model_type in ("Ridge", "Random Forest"):
        for with_year in (False, True):
            for exclude_flagged in (False, True):
                leaves = (2, 5) if model_type == "Random Forest" else (None,)
                for leaf in leaves:
                    yield {"Model": model_type, "With_Year": with_year,
                           "Exclude_Flagged_Training": exclude_flagged,
                           "Min_Samples_Leaf": leaf}


def run():
    data = add_high_yield_flags(load_feature_data())
    train = data.loc[data["Split"] == "train"].copy()
    validation = data.loc[data["Split"] == "validation"].copy()
    # The held-out test rows are first selected and evaluated after validation
    # has chosen a configuration and that configuration has been refitted.
    comparison_rows, crop_rows = [], []
    best = None
    for number, config in enumerate(configurations(), start=1):
        config = {"Configuration": f"C{number:02d}", **config}
        fit_rows = train.loc[~train["High_Yield_Flag"]] if config["Exclude_Flagged_Training"] else train
        columns = (["Previous_3_Year_Avg_Yield"] if config["Model"] == "Historical baseline"
                   else model_input_columns(config["With_Year"]))
        model = make_model(config["Model"], config["With_Year"], config["Min_Samples_Leaf"])
        model.fit(fit_rows[columns], fit_rows["Yield"])
        for split_name, frame in (("train", train), ("validation", validation)):
            predicted = model.predict(frame[columns])
            metrics = regression_metrics(frame["Yield"], predicted)
            comparison_rows.append({**config, "Split": split_name,
                                    "Fit_Records": len(fit_rows), "Evaluation_Records": len(frame),
                                    **metrics})
            crop_rows.extend({**config, "Split": split_name, **row}
                             for row in per_crop_metrics(frame, predicted))
            if split_name == "validation" and (best is None or metrics["RMSE_kg_ha"] < best["RMSE_kg_ha"]):
                best = {**config, "RMSE_kg_ha": metrics["RMSE_kg_ha"],
                        "Validation_Metrics": metrics, "Input_Features": columns}
        print(f"{config['Configuration']} {config['Model']}: validation RMSE "
              f"{comparison_rows[-1]['RMSE_kg_ha']:.2f} kg/ha")

    # Model choice is frozen here. Validation rows become fitting rows, while
    # test observations remain held out. A selected outlier exclusion applies
    # to the combined fitting period only.
    combined = data.loc[data["Split"].isin(("train", "validation"))].copy()
    fit_rows = (combined.loc[~combined["High_Yield_Flag"]]
                if best["Exclude_Flagged_Training"] else combined)
    selected = make_model(best["Model"], best["With_Year"], best["Min_Samples_Leaf"])
    selected.fit(fit_rows[best["Input_Features"]], fit_rows["Yield"])
    test = data.loc[data["Split"] == "test"].copy()
    predictions = selected.predict(test[best["Input_Features"]])
    overall = regression_metrics(test["Yield"], predictions)
    final_rows = [{"Configuration": best["Configuration"], "Scope": "Overall",
                   "Records": len(test), **overall}]
    test_crop = per_crop_metrics(test, predictions)
    final_rows.extend({"Configuration": best["Configuration"], "Scope": row["Crop"],
                       "Records": row["Records"],
                       **{metric: row[metric] for metric in ("MAE_kg_ha", "RMSE_kg_ha", "R2")}}
                      for row in test_crop)
    crop_rows.extend({**{key: best[key] for key in ("Configuration", "Model", "With_Year",
                                                    "Exclude_Flagged_Training", "Min_Samples_Leaf")},
                      "Split": "test", **row} for row in test_crop)

    for path in (MODEL_FILE, METADATA_FILE, COMPARISON_FILE, PER_CROP_FILE,
                 FINAL_FILE, SCATTER_FILE, ERRORS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_FILE)
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_FILE, index=False)
    pd.DataFrame(crop_rows).to_csv(PER_CROP_FILE, index=False)
    pd.DataFrame(final_rows).to_csv(FINAL_FILE, index=False)
    save_test_plots(test, predictions, SCATTER_FILE, ERRORS_FILE)
    metadata = {
        "selected_configuration": {key: best[key] for key in ("Configuration", "Model", "With_Year",
                                                        "Exclude_Flagged_Training", "Min_Samples_Leaf")},
        "target": "Yield (kg/ha)", "input_features": best["Input_Features"],
        "fit_years": [int(fit_rows["Year"].min()), int(fit_rows["Year"].max())],
        "fit_records": int(len(fit_rows)),
        "flagged_records_excluded_from_final_fit": int(len(combined) - len(fit_rows)),
        "validation_metrics_before_refit": best["Validation_Metrics"],
        "final_test_metrics": overall,
        "evaluation_assumption": "Rolling one-step-ahead: prior observed yields are available for validation and test rows.",
        "seed": SEED,
        "package_versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                             "joblib": joblib.__version__, "matplotlib": matplotlib.__version__},
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {best['Configuration']} {best['Model']} by validation RMSE "
          f"{best['RMSE_kg_ha']:.2f} kg/ha; final test RMSE {overall['RMSE_kg_ha']:.2f} kg/ha")
    return metadata


if __name__ == "__main__":
    run()
