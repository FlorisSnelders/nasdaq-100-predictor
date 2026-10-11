# tests/test_storage.py
#
# Unit tests for src/storage.py
# Uses tmp_path + monkeypatching DB_PATH to avoid touching real data.

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import src.storage as storage
from src.storage import (
    create_tables,
    save_features,
    save_macro_indicators,
    save_predictions,
    load_features,
    load_macro_indicators,
    load_predictions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_db(tmp_path, monkeypatch):
    """Redirect all DB operations to a temp file for every test."""
    db = tmp_path / "test_nasdaq100.db"
    monkeypatch.setattr(storage, "DB_PATH", db)
    yield db


def _make_prices(n_tickers=2, n_days=5):
    """Minimal OHLCV DataFrame."""
    rows = []
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    for ticker in tickers:
        for date in dates:
            rows.append({
                "ticker": ticker,
                "date": date,
                "open": 100.0, "high": 102.0,
                "low": 99.0, "close": 101.0,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def _seed_cleaned_prices(n_tickers=2, n_days=5):
    """Insert rows into cleaned_prices to satisfy FOREIGN KEY constraints."""
    import sqlite3 as _sq
    create_tables()
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = [
        (t, str(d.date()), 100.0, 102.0, 99.0, 101.0, 1_000_000, None)
        for t in tickers for d in dates
    ]
    conn = _sq.connect(storage.DB_PATH)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executemany(
        "INSERT OR IGNORE INTO cleaned_prices "
        "(ticker, date, open, high, low, close, volume, label) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _make_features(n_tickers=2, n_days=5):
    rows = []
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    for ticker in tickers:
        for date in dates:
            rows.append({"ticker": ticker, "date": date, "return_1d": 0.01})
    return pd.DataFrame(rows)


def _make_macro(n_days=5):
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    return pd.DataFrame({
        "date": dates,
        "vix_close": [15.0] * n_days,
        "vix_change_1d": [0.1] * n_days,
    })


def _make_predictions(n_tickers=2, n_days=5):
    rows = []
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    for ticker in tickers:
        for date in dates:
            rows.append({
                "ticker": ticker, "date": date,
                "prediction": 1, "probability": 0.6,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# create_tables
# ---------------------------------------------------------------------------

def test_create_tables_is_idempotent(patch_db):
    """create_tables must not raise when called multiple times."""
    create_tables()
    create_tables()


def test_create_tables_creates_all_tables(patch_db):
    """All four tables must exist after create_tables."""
    create_tables()
    conn = sqlite3.connect(patch_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"cleaned_prices", "macro_indicators", "features", "predictions"} <= tables


# ---------------------------------------------------------------------------
# save_features / load_features
# ---------------------------------------------------------------------------

def test_save_features_returns_count(patch_db):
    _seed_cleaned_prices()
    df = _make_features()
    n = save_features(df)
    assert n == len(df)


def test_save_features_missing_ticker_raises(patch_db):
    df = _make_features().drop(columns=["ticker"])
    with pytest.raises(ValueError, match="ticker"):
        save_features(df)


def test_save_features_missing_date_raises(patch_db):
    df = _make_features().drop(columns=["date"])
    with pytest.raises(ValueError, match="date"):
        save_features(df)


def test_load_features_roundtrip(patch_db):
    _seed_cleaned_prices()
    df = _make_features()
    save_features(df)
    loaded = load_features()
    assert len(loaded) == len(df)
    assert "return_1d" in loaded.columns


def test_load_features_ticker_filter(patch_db):
    _seed_cleaned_prices(n_tickers=3)
    df = _make_features(n_tickers=3)
    save_features(df)
    loaded = load_features(tickers=["T00"])
    assert set(loaded["ticker"].unique()) == {"T00"}


def test_load_features_date_filter(patch_db):
    _seed_cleaned_prices(n_days=10)
    df = _make_features(n_days=10)
    save_features(df)
    loaded = load_features(start_date="2024-01-04", end_date="2024-01-05")
    assert len(loaded) > 0
    assert len(loaded) < len(df)


def test_save_features_idempotent(patch_db):
    """Saving the same features twice must not duplicate rows."""
    _seed_cleaned_prices()
    df = _make_features()
    save_features(df)
    n = save_features(df)
    assert n == len(df)


# ---------------------------------------------------------------------------
# save_macro_indicators / load_macro_indicators
# ---------------------------------------------------------------------------

def test_save_macro_returns_count(patch_db):
    df = _make_macro()
    n = save_macro_indicators(df)
    assert n == len(df)


def test_save_macro_missing_date_raises(patch_db):
    df = _make_macro().drop(columns=["date"])
    with pytest.raises(ValueError, match="date"):
        save_macro_indicators(df)


def test_load_macro_roundtrip(patch_db):
    df = _make_macro()
    save_macro_indicators(df)
    loaded = load_macro_indicators()
    assert len(loaded) == len(df)
    assert "vix_close" in loaded.columns


def test_load_macro_date_filter(patch_db):
    df = _make_macro(n_days=10)
    save_macro_indicators(df)
    loaded = load_macro_indicators(start_date="2024-01-04", end_date="2024-01-05")
    assert len(loaded) > 0
    assert len(loaded) < len(df)


def test_save_macro_idempotent(patch_db):
    df = _make_macro()
    save_macro_indicators(df)
    n = save_macro_indicators(df)
    assert n == len(df)


# ---------------------------------------------------------------------------
# save_predictions / load_predictions
# ---------------------------------------------------------------------------

def test_save_predictions_returns_count(patch_db):
    _seed_cleaned_prices()
    df = _make_predictions()
    n = save_predictions(df)
    assert n == len(df)


def test_save_predictions_missing_columns_raises(patch_db):
    df = _make_predictions().drop(columns=["probability"])
    with pytest.raises(ValueError, match="probability"):
        save_predictions(df)


def test_load_predictions_roundtrip(patch_db):
    _seed_cleaned_prices()
    df = _make_predictions()
    save_predictions(df)
    loaded = load_predictions()
    assert len(loaded) == len(df)
    assert "prediction" in loaded.columns
    assert "probability" in loaded.columns


def test_load_predictions_ticker_filter(patch_db):
    _seed_cleaned_prices(n_tickers=3)
    df = _make_predictions(n_tickers=3)
    save_predictions(df)
    loaded = load_predictions(tickers=["T00"])
    assert set(loaded["ticker"].unique()) == {"T00"}


def test_save_predictions_idempotent(patch_db):
    _seed_cleaned_prices()
    df = _make_predictions()
    save_predictions(df)
    n = save_predictions(df)
    assert n == len(df)


# ---------------------------------------------------------------------------
# Legacy column warning
# ---------------------------------------------------------------------------

def test_legacy_columns_warning(patch_db, monkeypatch):
    """save_to_db must warn when legacy columns are present."""
    import warnings

    # Patch EXPECTED_TICKERS to match our small test df
    monkeypatch.setattr(storage, "EXPECTED_TICKERS", 2)
    # Patch _compute_label to avoid needing pandas_market_calendars
    monkeypatch.setattr(storage, "_compute_label", lambda df: pd.Series(1, index=df.index, dtype="Int64"))

    df = _make_prices(n_tickers=2)
    df["label_valid"] = True  # legacy column

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        storage.save_to_db(df)

    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) > 0
