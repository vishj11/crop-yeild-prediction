"""Read the source CSV and check its actual headers."""

import pandas as pd

from .config import CROPS, IDENTIFIER_COLUMNS, MEASURES, RAW_FILE


def required_columns():
    return list(IDENTIFIER_COLUMNS) + [
        f"{crop.upper()} {measure}" for crop in CROPS for measure in MEASURES
    ]


def load_raw(path=RAW_FILE):
    # Keep district codes as text so leading zeros, if any, survive.
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(required_columns()) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    return data
