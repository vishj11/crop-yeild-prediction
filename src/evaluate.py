"""Shared regression metrics and final test plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import CROPS


def regression_metrics(actual, predicted):
    """Return metrics in kg/ha for errors and unitless R²."""
    return {
        "MAE_kg_ha": float(mean_absolute_error(actual, predicted)),
        "RMSE_kg_ha": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def per_crop_metrics(frame, predicted):
    """Calculate metrics on the same rows as `frame`, separately by crop."""
    predicted = np.asarray(predicted)
    if len(predicted) != len(frame):
        raise ValueError("Prediction count differs from row count")
    rows = []
    for crop in CROPS:
        mask = frame["Crop"].to_numpy() == crop
        if mask.any():
            rows.append({"Crop": crop, "Records": int(mask.sum()),
                         **regression_metrics(frame.loc[mask, "Yield"], predicted[mask])})
    return rows


def save_test_plots(frame, predicted, scatter_path, errors_path):
    """Plot only the selected model's held-out test predictions."""
    predicted = np.asarray(predicted)
    actual = frame["Yield"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for crop in CROPS:
        mask = frame["Crop"].to_numpy() == crop
        ax.scatter(actual[mask], predicted[mask], s=9, alpha=0.4, label=crop)
    upper = max(float(actual.max()), float(predicted.max()))
    ax.plot([0, upper], [0, upper], color="black", linewidth=1, label="Perfect prediction")
    ax.set(xlabel="Actual yield (kg/ha)", ylabel="Predicted yield (kg/ha)",
           title="Held-out test: actual vs predicted yield")
    ax.legend()
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    errors = predicted - actual
    for crop in CROPS:
        mask = frame["Crop"].to_numpy() == crop
        ax.hist(errors[mask], bins=35, alpha=0.45, label=crop)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Prediction error: predicted − actual (kg/ha)",
           ylabel="Test records (count)", title="Held-out test: prediction errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(errors_path, dpi=160)
    plt.close(fig)
