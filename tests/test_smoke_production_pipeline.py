# tests/test_smoke_production_pipeline.py
#
# Smoke tests for the production pipeline
#
# These tests verify the full pipeline runs end-to-end without errors.
# They use the same patching style as test_pipeline.py to avoid hitting
# real data files or databases.

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import pipeline
from src.pipeline import run_pipeline
from src.features import build_feature_matrix
from src.evaluation import evaluate_model, save_results

# ---------------------------------------------------------------------------
# Minimal synthetic data — mirrors the shape of the real dataset
# ---------------------------------------------------------------------------

# Cover 2015–2024 so all 4 walk-forward folds have enough data
DATES = pd.bdate_range("2015-01-02", "2024-12-31").strftime("%Y-%m-%d").tolist()
TICKERS = ["AAA", "BBB", "CCC"]


def _raw_ohlcv():
    """Synthetic OHLCV rows for all tickers across all dates (2015-2024)."""
    rows = []
    rng = np.random.default_rng(42)
    for ticker in TICKERS:
        price = 100.0
        for date in DATES:
            price *= 1 + rng.normal(0, 0.01)
            rows.append({
                "date":         f"{date} 12:00:00",
                "open":         price * 0.99,
                "high":         price * 1.01,
                "low":          price * 0.98,
                "close":        price,
                "volume":       int(rng.uniform(500_000, 2_000_000)),
                "transactions": int(rng.uniform(1_000, 5_000)),
                "ticker":       ticker,
            })
    return pd.DataFrame(rows)


def _macro_df():
    """Synthetic macro data aligned to DATES."""
    return pd.DataFrame({
        "date":               [pd.Timestamp(d) for d in DATES],
        "vix":                np.linspace(15, 25, len(DATES)),
        "treasury_yield_10y": np.linspace(4.0, 4.5, len(DATES)),
    })


@pytest.fixture
def patched(monkeypatch):
    """Patch all I/O so tests run without real files or databases."""
    monkeypatch.setattr(pipeline, "load_snapshot",        lambda source: _raw_ohlcv())
    monkeypatch.setattr(pipeline, "load_and_clean_macro", lambda **k: _macro_df())
    monkeypatch.setattr(pipeline, "validate_stage",       lambda *a, **k: None)
    return monkeypatch


# ---------------------------------------------------------------------------
# 1. Pipeline smoke tests
# ---------------------------------------------------------------------------

def test_pipeline_returns_dataframe(patched):
    """run_pipeline must return a non-empty DataFrame."""
    df = run_pipeline(source="dummy/path")
    assert isinstance(df, pd.DataFrame), "run_pipeline must return a DataFrame"
    assert len(df) > 0, "DataFrame must not be empty"


def test_pipeline_has_required_columns(patched):
    """run_pipeline output must contain the macro and feature columns."""
    df = run_pipeline(source="dummy/path")
    required = ["vix", "treasury_yield_10y", "daily_return", "label_valid", "ticker", "date"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_pipeline_no_temp_columns_leaked(patched):
    """Merge helper columns must be cleaned up."""
    df = run_pipeline(source="dummy/path")
    assert "date_key"   not in df.columns, "date_key should be cleaned up after merge"
    assert "date_macro" not in df.columns, "date_macro should be cleaned up after merge"


# ---------------------------------------------------------------------------
# 2. Feature matrix smoke tests
# ---------------------------------------------------------------------------

def test_build_feature_matrix_shape(patched):
    """build_feature_matrix must return (X, y, meta) with 17 locked features."""
    df = run_pipeline(source="dummy/path")
    X, y, meta = build_feature_matrix(df)

    assert X.shape[1] == 17, f"Expected 17 features, got {X.shape[1]}"
    assert len(X) == len(y), "X and y must have the same number of rows"
    assert len(X) == len(meta), "X and meta must have the same number of rows"


def test_build_feature_matrix_no_nulls(patched):
    """build_feature_matrix must return a clean matrix with no NaN values."""
    df = run_pipeline(source="dummy/path")
    X, y, meta = build_feature_matrix(df)

    assert X.isnull().sum().sum() == 0, "X must not contain NaN values"
    assert pd.isnull(y).sum() == 0,     "y must not contain NaN values"


def test_build_feature_matrix_binary_labels(patched):
    """Labels must be binary (0 or 1) only."""
    df = run_pipeline(source="dummy/path")
    _, y, _ = build_feature_matrix(df)

    unique_labels = set(np.unique(y))
    assert unique_labels <= {0, 1}, f"Labels must be 0 or 1, got: {unique_labels}"


def test_build_feature_matrix_locked_feature_names(patched):
    """build_feature_matrix must return exactly the 17 locked feature names."""
    expected_features = {
        "daily_return", "rsi_14", "sma_20", "sma_50", "sma_ratio",
        "macd", "macd_signal", "macd_hist",
        "bb_upper_20", "bb_lower_20", "bb_width_20",
        "return_lag_1", "return_lag_3", "return_lag_5", "return_lag_10",
        "vix", "treasury_yield_10y",
    }
    df = run_pipeline(source="dummy/path")
    X, _, _ = build_feature_matrix(df)

    actual_features = set(X.columns)
    assert actual_features == expected_features, (
        f"Feature mismatch.\nMissing: {expected_features - actual_features}\n"
        f"Extra:   {actual_features - expected_features}"
    )


# ---------------------------------------------------------------------------
# 3. Model smoke tests
# ---------------------------------------------------------------------------

class _SimpleWrapper:
    """Minimal sklearn-style wrapper used only for smoke testing."""

    def __init__(self):
        self.scaler_ = None
        self.model_  = None

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled     = self.scaler_.fit_transform(X)
        self.model_  = LogisticRegression(max_iter=200, random_state=42)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X))


def test_model_fit_predict_runs(patched):
    """A model wrapper must be able to fit and predict without errors."""
    df = run_pipeline(source="dummy/path")
    X, y, _ = build_feature_matrix(df)

    model = _SimpleWrapper()
    model.fit(X.values, y.values)
    preds = model.predict(X.values)

    assert len(preds) == len(y), "predict must return one label per row"
    assert set(np.unique(preds)) <= {0, 1}, "predictions must be binary"


def test_evaluate_model_returns_four_folds(patched):
    """evaluate_model must return results for all 4 walk-forward folds."""
    df = run_pipeline(source="dummy/path")
    X, y, meta = build_feature_matrix(df)

    results = evaluate_model(lambda: _SimpleWrapper(), X.values, y.values, meta)

    assert isinstance(results, list), "evaluate_model must return a list"
    assert len(results) == 4, f"Expected 4 folds, got {len(results)}"


def test_evaluate_model_fold_keys(patched):
    """Each fold result must contain precision, f1, and accuracy keys."""
    df = run_pipeline(source="dummy/path")
    X, y, meta = build_feature_matrix(df)

    results = evaluate_model(lambda: _SimpleWrapper(), X.values, y.values, meta)

    for i, fold in enumerate(results):
        for key in ("precision", "f1", "accuracy"):
            assert key in fold, f"Fold {i+1} missing key: '{key}'"
        assert 0.0 <= fold["precision"] <= 1.0, f"Fold {i+1} precision out of range"
        assert 0.0 <= fold["accuracy"]  <= 1.0, f"Fold {i+1} accuracy out of range"


# ---------------------------------------------------------------------------
# 4. save_results smoke test
# ---------------------------------------------------------------------------

def test_save_results_writes_file(patched, tmp_path, monkeypatch):
    """save_results must write a JSON file to the results directory."""
    import json
    from pathlib import Path
    import src.evaluation as ev

    # Find where save_results actually writes — try common attribute names
    original_dir = None
    for attr in ("REPORTS_DIR", "BAKEOFF_DIR", "OUTPUT_DIR", "RESULTS_DIR"):
        if hasattr(ev, attr):
            original_dir = getattr(ev, attr)
            reports_dir = tmp_path / "reports" / "bakeoff"
            reports_dir.mkdir(parents=True)
            monkeypatch.setattr(ev, attr, reports_dir)
            break

    df = run_pipeline(source="dummy/path")
    X, y, meta = build_feature_matrix(df)
    results = evaluate_model(lambda: _SimpleWrapper(), X.values, y.values, meta)

    if original_dir is not None:
        # Patching worked — check tmp_path
        save_results("smoke_test_model", results)
        output_file = reports_dir / "smoke_test_model.json"
        assert output_file.exists(), f"Expected file at {output_file}"
    else:
        # Can't patch the dir — just verify save_results runs without error
        # and that it writes *somewhere* in the project reports folder
        real_reports = Path("reports") / "bakeoff"
        real_reports.mkdir(parents=True, exist_ok=True)
        save_results("smoke_test_model", results)
        output_file = real_reports / "smoke_test_model.json"
        assert output_file.exists(), f"Expected file at {output_file}"
        # Clean up
        output_file.unlink(missing_ok=True)
