import pandas as pd
import pytest

from src.jobs.daily_ingest import (
    read_tickers,
    compute_window,
    assert_no_nulls,
    assert_no_future_dates,
)


def test_read_tickers_strips_and_skips_blanks(tmp_path):
    p = tmp_path / "tickers.txt"
    p.write_text("AAPL\n  MSFT  \n\nGOOG\n")
    assert read_tickers(p) == ["AAPL", "MSFT", "GOOG"]


def test_compute_window():
    today = pd.Timestamp("2026-06-10", tz="America/New_York")
    start, end = compute_window(today, lookback_days=10)
    assert start == "2026-05-31"
    assert end == "2026-06-11"


def test_assert_no_nulls_passes():
    df = pd.DataFrame({"ticker": ["AAPL"], "close": [1.0]})
    assert_no_nulls(df, ["ticker", "close"])


def test_assert_no_nulls_raises():
    df = pd.DataFrame({"ticker": ["AAPL"], "close": [None]})
    with pytest.raises(ValueError):
        assert_no_nulls(df, ["ticker", "close"])


def test_assert_no_future_dates_passes():
    df = pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-01", tz="America/New_York")]}
    )
    assert_no_future_dates(df)


def test_assert_no_future_dates_raises():
    df = pd.DataFrame(
        {"ticker": ["AAPL"], "date": [pd.Timestamp("2099-01-01", tz="America/New_York")]}
    )
    with pytest.raises(ValueError):
        assert_no_future_dates(df)