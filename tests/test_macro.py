# tests/test_macro.py
import pandas as pd
import pytest
from src import macro
from src.macro import fetch_macro, clean_macro, load_and_clean_macro


def _fake_download(ticker, start=None, end=None, progress=False, auto_adjust=True):
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=3), name="Date")
    base = 15.0 if ticker == "^VIX" else 4.2
    return pd.DataFrame({
        "Open": [base] * 3, "High": [base] * 3, "Low": [base] * 3,
        "Close": [base, base + 1, base + 2], "Volume": [0] * 3,
    }, index=idx)


@pytest.fixture
def mock_yf(monkeypatch):
    monkeypatch.setattr(macro.yf, "download", _fake_download)


def test_fetch_macro_columns_and_values(mock_yf):
    df = fetch_macro("2024-01-02", "2024-01-05")
    assert list(df.columns) == ["date", "vix", "treasury_yield_10y"]
    assert len(df) == 3
    assert df["vix"].iloc[0] == 15.0
    assert df["treasury_yield_10y"].iloc[0] == 4.2


def test_clean_macro_filters_and_ffills():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"),
                 pd.Timestamp("2024-01-06")],  # Saturday -> dropped
        "vix": [15.0, 17.0, 99.0],
        "treasury_yield_10y": [4.0, 5.0, 9.0],
    })
    out = clean_macro(df, start="2024-01-02", end="2024-01-05")
    assert len(out) == 4  # 02,03,04,05 trading days
    # 01-03 was missing -> forward-filled from 01-02
    val_03 = out.loc[out["date"] == pd.Timestamp("2024-01-03"), "vix"].iloc[0]
    assert val_03 == 15.0
    assert 99.0 not in out["vix"].values  # weekend row removed


def test_clean_macro_drops_leading_nans():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-04")],
        "vix": [17.0],
        "treasury_yield_10y": [5.0],
    })
    out = clean_macro(df, start="2024-01-02", end="2024-01-05")
    assert len(out) == 2  # 01-02 and 01-03 have no prior value -> dropped
    assert out["date"].min() == pd.Timestamp("2024-01-04")


def test_load_and_clean_macro_end_to_end(mock_yf):
    out = load_and_clean_macro(start="2024-01-02", end="2024-01-05")
    assert list(out.columns) == ["date", "vix", "treasury_yield_10y"]
    assert len(out) == 4