"""Pipeline reproducibility tests for the LSTM bake-off entry.

Named test_lstm_pipeline.py to avoid clashing with the shared tests/test_pipeline.py.

Covers the three required behaviours on the LSTM pipeline:
  1. Determinism  -- identical input -> identical output at every stage.
  2. Row counts   -- a small fixture collapses to a deterministic number of
     feature rows and sequences after warm-up + windowing.
  3. Schema       -- columns, dtypes and shapes are correct after each stage.

Stages: raw panel --build_feature_matrix--> (X, y, meta) --build_sequences--> (seqs, labels, ends) --run_lstm_bakeoff--> metrics
Relies only on a synthetic fixture (no network / no run_pipeline).
"""
import numpy as np
import pandas as pd
import pytest

from src.features import build_feature_matrix
from src.lstm_model import build_sequences, _train_one_fold, SEQ_LEN

EXPECTED_FEATURES = [
    "daily_return", "rsi_14", "sma_20", "sma_50", "sma_ratio",
    "macd", "macd_signal", "macd_hist",
    "bb_upper_20", "bb_lower_20", "bb_width_20",
    "return_lag_1", "return_lag_3", "return_lag_5", "return_lag_10",
    "vix", "treasury_yield_10y",
]


def _fixture(n_tickers=2, n=120, seed=0):
    """Small synthetic multi-ticker panel with the columns the pipeline needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for k in range(n_tickers):
        close = 100 + rng.normal(0, 1, n).cumsum()
        close = close - close.min() + 10
        frames.append(pd.DataFrame({
            "date": dates,
            "ticker": f"T{k}",
            "close": close,
            "vix": 15 + rng.normal(0, 1, n),
            "treasury_yield_10y": 3 + rng.normal(0, 0.1, n),
        }))
    return pd.concat(frames, ignore_index=True)


# ----------------------------- 1. determinism -----------------------------
def test_feature_matrix_deterministic():
    df = _fixture()
    X1, y1, m1 = build_feature_matrix(df.copy())
    X2, y2, m2 = build_feature_matrix(df.copy())
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    pd.testing.assert_frame_equal(m1, m2)


def test_sequence_build_deterministic():
    X, y, meta = build_feature_matrix(_fixture())
    s1, l1, e1 = build_sequences(X, y, meta)
    s2, l2, e2 = build_sequences(X, y, meta)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(l1, l2)
    pd.testing.assert_frame_equal(e1, e2)


def test_training_deterministic_under_fixed_seed():
    torch = pytest.importorskip("torch")  # skip if torch isn't installed
    X, y, meta = build_feature_matrix(_fixture())
    seqs, labels, ends = build_sequences(X, y, meta)
    n = len(seqs)
    a, b = int(n * 0.6), int(n * 0.8)
    train, val, test = (seqs[:a], labels[:a]), (seqs[a:b], labels[a:b]), (seqs[b:], labels[b:])
    m1 = _train_one_fold(train, val, test)
    m2 = _train_one_fold(train, val, test)
    assert m1["accuracy"] == m2["accuracy"]
    assert m1["precision"] == m2["precision"]
    assert m1["f1"] == m2["f1"]


# ----------------------------- 2. row counts ------------------------------
def test_expected_row_and_sequence_counts():
    n_tickers = 2
    X, y, meta = build_feature_matrix(_fixture(n_tickers=n_tickers, n=120))
    assert len(X) == len(y) == len(meta) > 0
    seqs, labels, ends = build_sequences(X, y, meta)
    # Each ticker yields (feature_rows_per_ticker - (SEQ_LEN - 1)) sequences.
    expected = len(X) - n_tickers * (SEQ_LEN - 1)
    assert len(seqs) == len(labels) == len(ends) == expected


# ------------------------------ 3. schema ---------------------------------
def test_feature_matrix_schema():
    X, y, meta = build_feature_matrix(_fixture())
    assert list(X.columns) == EXPECTED_FEATURES
    assert (X.dtypes == np.float64).all()
    assert not X.isna().any().any()
    assert str(y.dtype) == "int64"
    assert set(pd.unique(y)).issubset({0, 1})
    assert list(meta.columns) == ["date", "ticker"]


def test_sequence_schema_and_shapes():
    X, y, meta = build_feature_matrix(_fixture())
    seqs, labels, ends = build_sequences(X, y, meta)
    assert seqs.ndim == 3
    assert seqs.shape[1] == SEQ_LEN
    assert seqs.shape[2] == len(EXPECTED_FEATURES)
    assert seqs.dtype == np.float32
    assert labels.dtype == np.int64
    assert labels.shape[0] == seqs.shape[0]
    assert list(ends.columns) == ["date", "ticker"]
    assert len(ends) == seqs.shape[0]
