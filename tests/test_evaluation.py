# tests/test_evaluation.py
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from src.evaluation import _split_fold, evaluate_model, summarize_results, save_results, FOLDS


def _meta(years):
    dates = []
    for y in years:
        dates += [pd.Timestamp(f"{y}-{m:02d}-15") for m in range(1, 13)]
    return pd.DataFrame({"date": pd.to_datetime(dates)})


def _factory():
    return DummyClassifier(strategy="most_frequent")


def test_split_fold_returns_bool_arrays():
    meta = _meta(range(2018, 2023))
    train, val, test = _split_fold(meta, FOLDS[0])
    assert train.dtype == bool and val.dtype == bool and test.dtype == bool
    assert train.sum() > 0 and test.sum() > 0
    # train years (<=2019) must not overlap test year (2021)
    assert not (train & test).any()


def test_evaluate_model_runs_some_folds():
    meta = _meta(range(2018, 2023))  # 2018..2022 -> folds 1 & 2 have train+test
    n = len(meta)
    rng = np.random.default_rng(0)
    X = rng.random((n, 3))
    y = rng.integers(0, 2, n)
    results = evaluate_model(_factory, X, y, meta)
    assert len(results) == 2  # folds 3 & 4 skipped (no 2023/2024 test data)
    assert {r["fold"] for r in results} == {1, 2}
    for r in results:
        for k in ["accuracy", "precision", "f1", "n_train", "n_test", "train_time_seconds"]:
            assert k in r


def test_evaluate_model_skips_when_no_test_data():
    meta = _meta([2021])  # no fold has both train and test
    n = len(meta)
    X = np.zeros((n, 3))
    y = np.zeros(n, dtype=int)
    assert evaluate_model(_factory, X, y, meta) == []


def test_summarize_results():
    results = [
        {"accuracy": 0.6, "precision": 0.5, "f1": 0.55, "train_time_seconds": 1.0},
        {"accuracy": 0.8, "precision": 0.7, "f1": 0.75, "train_time_seconds": 2.0},
    ]
    s = summarize_results(results)
    assert abs(s["mean_accuracy"] - 0.7) < 1e-9
    assert abs(s["total_train_time_seconds"] - 3.0) < 1e-9
    assert "std_f1" in s


def test_save_results_writes_json(tmp_path):
    results = [
        {"fold": 1, "accuracy": 0.6, "precision": 0.5, "f1": 0.55, "train_time_seconds": 1.0},
        {"fold": 2, "accuracy": 0.8, "precision": 0.7, "f1": 0.75, "train_time_seconds": 2.0},
    ]
    path = save_results("dummy", results, output_dir=str(tmp_path))
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["model_name"] == "dummy"
    assert len(payload["per_fold"]) == 2
    assert "summary" in payload