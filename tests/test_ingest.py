# tests/test_ingest.py
import pandas as pd
import pytest
from src import ingest
from src.ingest import load_snapshot, load_yfinance


def _snapshot_csvs(directory, n):
    for i in range(n):
        pd.DataFrame({
            "date": ["2024-01-02"],
            "open": [10], "high": [12], "low": [9], "close": [11],
            "volume": [1000], "transactions": [5],
        }).to_csv(directory / f"T{i:03d}.csv", index=False)


def test_load_snapshot_happy(tmp_path):
    _snapshot_csvs(tmp_path, 100)
    df = load_snapshot(tmp_path)
    assert df["ticker"].nunique() == 100


def test_load_snapshot_wrong_count_raises(tmp_path):
    _snapshot_csvs(tmp_path, 2)
    with pytest.raises(AssertionError):
        load_snapshot(tmp_path)


def _fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=2), name="Date")
    if ticker == "EMPTY":
        return pd.DataFrame()
    if ticker == "MULTI":
        cols = pd.MultiIndex.from_tuples([
            ("Open", "MULTI"), ("High", "MULTI"), ("Low", "MULTI"),
            ("Close", "MULTI"), ("Adj Close", "MULTI"), ("Volume", "MULTI"),
        ])
        return pd.DataFrame(
            [[1.0, 2.0, 0.5, 1.5, 1.5, 100], [2.0, 3.0, 1.5, 2.5, 2.5, 200]],
            index=idx, columns=cols,
        )
    return pd.DataFrame({
        "Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5],
        "Close": [1.5, 2.5], "Adj Close": [1.5, 2.5], "Volume": [100, 200],
    }, index=idx)


@pytest.fixture
def mock_yf(monkeypatch):
    monkeypatch.setattr(ingest.yf, "download", _fake_download)


EXPECTED = ["date", "open", "high", "low", "close", "volume", "transactions", "ticker"]


def test_load_yfinance_single_string(mock_yf):
    df = load_yfinance("AAA", "2024-01-01", "2024-01-05")
    assert list(df.columns) == EXPECTED
    assert len(df) == 2
    assert (df["ticker"] == "AAA").all()
    assert df["transactions"].isna().all()


def test_load_yfinance_skips_empty(mock_yf):
    df = load_yfinance(["AAA", "EMPTY"], "2024-01-01", "2024-01-05")
    assert (df["ticker"] == "AAA").all()
    assert len(df) == 2


def test_load_yfinance_multiindex_columns(mock_yf):
    df = load_yfinance(["MULTI"], "2024-01-01", "2024-01-05")
    assert list(df.columns) == EXPECTED
    assert len(df) == 2
    assert (df["ticker"] == "MULTI").all()