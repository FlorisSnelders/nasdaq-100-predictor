import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score


FOLDS = [
    {"train_end": 2019, "val_year": 2020, "test_year": 2021},
    {"train_end": 2020, "val_year": 2021, "test_year": 2022},
    {"train_end": 2021, "val_year": 2022, "test_year": 2023},
    {"train_end": 2022, "val_year": 2023, "test_year": 2024},
]

EMBARGO_DAYS = 5

# Default location for walk-forward result JSONs (read by the dashboard).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"


def _split_fold(meta, fold):
    years = meta["date"].dt.year
    train_mask = years <= fold["train_end"]
    val_mask = years == fold["val_year"]
    test_mask = years == fold["test_year"]
    # apply embargo at train/val boundary and val/test boundary
    train_end = meta.loc[train_mask, "date"].max()
    val_start = train_end + pd.Timedelta(days=EMBARGO_DAYS)
    val_mask = val_mask & (meta["date"] >= val_start)
    val_end = meta.loc[val_mask, "date"].max()
    test_start = val_end + pd.Timedelta(days=EMBARGO_DAYS)
    test_mask = test_mask & (meta["date"] >= test_start)
    return train_mask.values, val_mask.values, test_mask.values


def evaluate_model(model_factory, X, y, meta):
    results = []
    for i, fold in enumerate(FOLDS, start=1):
        train_idx, val_idx, test_idx = _split_fold(meta, fold)
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue
        model = model_factory()
        t0 = time.time()
        model.fit(X[train_idx], y[train_idx])
        train_time = time.time() - t0
        y_pred = model.predict(X[test_idx])
        results.append({
            "fold": i,
            "train_end_year": fold["train_end"],
            "test_year": fold["test_year"],
            "n_train": int(train_idx.sum()),
            "n_test": int(test_idx.sum()),
            "accuracy": float(accuracy_score(y[test_idx], y_pred)),
            "precision": float(precision_score(y[test_idx], y_pred, zero_division=0)),
            "f1": float(f1_score(y[test_idx], y_pred, zero_division=0)),
            "train_time_seconds": float(train_time),
        })
    return results


def summarize_results(results):
    df = pd.DataFrame(results)
    summary = {
        "mean_accuracy": float(df["accuracy"].mean()),
        "std_accuracy": float(df["accuracy"].std()),
        "mean_precision": float(df["precision"].mean()),
        "std_precision": float(df["precision"].std()),
        "mean_f1": float(df["f1"].mean()),
        "std_f1": float(df["f1"].std()),
        "total_train_time_seconds": float(df["train_time_seconds"].sum()),
    }
    return summary


def save_results(model_name, results, output_dir=None):
    out_dir = RESULTS_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "per_fold": results,
        "summary": summarize_results(results),
    }
    out_path = out_dir / f"{model_name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path