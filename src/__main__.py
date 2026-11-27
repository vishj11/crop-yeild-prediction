"""Run with: python -m src"""

import pandas as pd

from .config import CLEAN_FILE, FEATURE_FILE, ISSUES_FILE, SUMMARY_FILE, CROPS
from .data_loader import load_raw
from .features import engineer_features, validate_split_coverage
from .preprocess import clean_data


def build_summary(long, cleaned, features):
    rows = []
    for crop in CROPS:
        source = long.loc[long["Crop"] == crop]
        clean = cleaned.loc[cleaned["Crop"] == crop]
        ready = features.loc[features["Crop"] == crop]
        yield_stats = clean["Yield"].describe()
        rows.append({
            "Crop": crop,
            "Source_Records": len(source),
            "Clean_Records": len(clean),
            "Records_Excluded_For_Quality": len(source) - len(clean),
            "Records_Removed_For_History": len(clean) - len(ready),
            "Feature_Records": len(ready),
            "First_Year": int(source["Year"].min()),
            "Last_Year": int(source["Year"].max()),
            "Years_Represented": source["Year"].nunique(),
            "States": source["State"].nunique(),
            "District_Codes": source["District_Code"].nunique(),
            "Missing_Area": int(source["Area"].isna().sum()),
            "Missing_Production": int(source["Production"].isna().sum()),
            "Missing_Yield": int(source["Yield"].isna().sum()),
            "Yield_Min_kg_ha": yield_stats.get("min"),
            "Yield_Median_kg_ha": clean["Yield"].median(),
            "Yield_Mean_kg_ha": yield_stats.get("mean"),
            "Yield_Std_kg_ha": yield_stats.get("std"),
            "Yield_Max_kg_ha": yield_stats.get("max"),
            "Train_Records": int((ready["Split"] == "train").sum()),
            "Validation_Records": int((ready["Split"] == "validation").sum()),
            "Test_Records": int((ready["Split"] == "test").sum()),
        })
    return pd.DataFrame(rows)


def main():
    raw = load_raw()
    long, cleaned, quality_issues = clean_data(raw)
    features, history_issues = engineer_features(cleaned, long)
    validate_split_coverage(features)
    issues = pd.concat([quality_issues, history_issues], ignore_index=True)
    summary = build_summary(long, cleaned, features)
    for path in (CLEAN_FILE, FEATURE_FILE, ISSUES_FILE, SUMMARY_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_FILE, index=False)
    features.to_csv(FEATURE_FILE, index=False)
    issues.to_csv(ISSUES_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    print(summary.to_string(index=False))
    print("\nIssue counts:\n" + issues["Issue"].value_counts().to_string())


if __name__ == "__main__":
    main()
