"""Dataset names, units, and output locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data/raw/ICRISAT-District Level Data.csv"
CLEAN_FILE = ROOT / "data/processed/crop_yield_clean.csv"
FEATURE_FILE = ROOT / "data/processed/crop_yield_features.csv"
SUMMARY_FILE = ROOT / "reports/dataset_summary.csv"
ISSUES_FILE = ROOT / "reports/data_quality_issues.csv"
MODEL_FILE = ROOT / "models/crop_yield_model.joblib"
METADATA_FILE = ROOT / "models/model_metadata.json"
FINAL_TEST_FILE = ROOT / "reports/final_test_metrics.csv"

CROPS = ("Rice", "Wheat", "Maize")
IDENTIFIER_COLUMNS = ("Dist Code", "Year", "State Name", "Dist Name")
MEASURES = ("AREA (1000 ha)", "PRODUCTION (1000 tons)", "YIELD (Kg per ha)")
LONG_COLUMNS = ("State", "District", "District_Code", "Year", "Crop", "Area", "Production", "Yield")
ISSUE_COLUMNS = ("Issue", "State", "District", "District_Code", "Year", "Crop", "Details")
