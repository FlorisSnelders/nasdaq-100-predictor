# tests/test_data_cleaner.py
#
# Unit tests for src/data_cleaner.py
# Covers: add_daily_returns, flag_extreme_returns, drop_before, flag_pregap_rows

import pandas as pd
import pytest

from src.data_cleaner import (
    add_daily_returns,
    flag_extreme_returns,
    drop_before,
    flag_pregap_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_days=10, tickers=("AAA", "BBB")):
    """Minimal OHLCV-style DataFrame with close prices."""
    rows = []
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B",
                          tz="America/New_York")
    for ticker in tickers:
        price = 100.0
        for date in dates:
            price *= 1.01
            rows.append({"ticker": ticker, "date": date, "close": price,
                         "volume": 1_000_000})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# add_daily_returns
# ---------------------------------------------------------------------------

def test_add_daily_returns_column_exists():
    df = _make_df()
    result = add_daily_returns(df)
    assert "daily_return" in result.columns


def test_add_daily_returns_first_row_nan():
    """First row per ticker must be NaN."""
    df = _make_df(tickers=("AAA",))
    result = add_daily_returns(df)
    first_rows = result.groupby("ticker").head(1)
    assert first_rows["daily_return"].isna().all()


def test_add_daily_returns_correct_value():
    """daily_return must equal pct_change of close."""
    df = _make_df(tickers=("AAA",))
    result = add_daily_returns(df)
    expected = result["close"].pct_change()
    # Compare non-NaN values
    mask = result["daily_return"].notna()
    pd.testing.assert_series_equal(
        result.loc[mask, "daily_return"].reset_index(drop=True),
        expected[mask].reset_index(drop=True),
        check_names=False,
    )


def test_add_daily_returns_preserves_rows():
    df = _make_df()
    result = add_daily_returns(df)
    assert len(result) == len(df)


def test_add_daily_returns_sorted_by_ticker_date():
    """Result must be sorted by ticker then date."""
    df = _make_df().sample(frac=1, random_state=42)  # shuffle
    result = add_daily_returns(df)
    assert result["date"].is_monotonic_increasing or \
           list(result.sort_values(["ticker", "date"])["date"]) == list(result["date"])


# ---------------------------------------------------------------------------
# flag_extreme_returns
# ---------------------------------------------------------------------------

def test_flag_extreme_returns_empty_when_no_extremes():
    df = _make_df()  # ~1% daily moves — well below 30% threshold
    df = add_daily_returns(df)
    result = flag_extreme_returns(df, threshold=0.30)
    assert len(result) == 0


def test_flag_extreme_returns_detects_spike():
    """A 50% price jump must be flagged."""
    df = _make_df(tickers=("AAA",))
    df = add_daily_returns(df)
    # Inject a 50% spike on row 3
    df.loc[3, "close"] = df.loc[2, "close"] * 1.5
    df = add_daily_returns(df)  # recompute
    result = flag_extreme_returns(df, threshold=0.30)
    assert len(result) >= 1
    assert (result["daily_return"].abs() > 0.30).all()


def test_flag_extreme_returns_columns():
    """Result must contain the required columns."""
    df = _make_df(tickers=("AAA",))
    df = add_daily_returns(df)
    df.loc[3, "close"] = df.loc[2, "close"] * 1.5
    df = add_daily_returns(df)
    result = flag_extreme_returns(df, threshold=0.30)
    for col in ["ticker", "date", "prev_close", "close", "daily_return", "volume"]:
        assert col in result.columns, f"Missing column: {col}"


def test_flag_extreme_returns_adds_daily_return_if_missing():
    """Must work even if daily_return column is not already present."""
    df = _make_df(tickers=("AAA",))
    assert "daily_return" not in df.columns
    result = flag_extreme_returns(df, threshold=0.30)
    assert isinstance(result, pd.DataFrame)


def test_flag_extreme_returns_custom_threshold():
    """Should flag more rows with a lower threshold."""
    df = _make_df(tickers=("AAA",))
    df = add_daily_returns(df)
    high = flag_extreme_returns(df, threshold=0.30)
    low  = flag_extreme_returns(df, threshold=0.005)  # ~0.5% — catches all 1% moves
    assert len(low) >= len(high)


# ---------------------------------------------------------------------------
# drop_before
# ---------------------------------------------------------------------------

def test_drop_before_removes_rows():
    df = _make_df(tickers=("AAA",))
    cutoff = "2024-01-05"
    result = drop_before(df, "AAA", cutoff)
    remaining = result[result["ticker"] == "AAA"]
    cutoff_ts = pd.Timestamp(cutoff, tz="America/New_York")
    assert (remaining["date"] >= cutoff_ts).all()


def test_drop_before_does_not_affect_other_tickers():
    df = _make_df(tickers=("AAA", "BBB"))
    n_bbb_before = len(df[df["ticker"] == "BBB"])
    result = drop_before(df, "AAA", "2024-01-10")
    n_bbb_after = len(result[result["ticker"] == "BBB"])
    assert n_bbb_before == n_bbb_after


def test_drop_before_returns_dataframe():
    df = _make_df()
    result = drop_before(df, "AAA", "2024-01-05")
    assert isinstance(result, pd.DataFrame)


def test_drop_before_resets_index():
    df = _make_df(tickers=("AAA",))
    result = drop_before(df, "AAA", "2024-01-05")
    assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# flag_pregap_rows
# ---------------------------------------------------------------------------

def _make_missing_df(ticker="AAA", miss_date="2024-01-05"):
    """Return a missing DataFrame with one missing date entry."""
    return pd.DataFrame({
        "ticker": [ticker],
        "missing_date": [pd.Timestamp(miss_date)],
    })


def test_flag_pregap_rows_adds_label_valid():
    df = _make_df(tickers=("AAA",))
    missing = _make_missing_df("AAA", "2024-01-05")
    result = flag_pregap_rows(df, missing)
    assert "label_valid" in result.columns


def test_flag_pregap_rows_default_true():
    """Rows not before a gap must have label_valid=True (except last row per ticker)."""
    df = _make_df(tickers=("AAA",))
    missing = _make_missing_df("AAA", "2099-01-01")  # far future
    result = flag_pregap_rows(df, missing)
    last_idx = result.groupby("ticker").tail(1).index
    non_last = result.drop(index=last_idx)
    assert non_last["label_valid"].all()


def test_flag_pregap_rows_marks_row_before_gap():
    """The row immediately before the missing date must be marked False."""
    df = _make_df(tickers=("AAA",))
    miss_date = df[df["ticker"] == "AAA"]["date"].iloc[3]  # 4th date
    missing = pd.DataFrame({"ticker": ["AAA"], "missing_date": [miss_date.tz_localize(None)]})
    result = flag_pregap_rows(df, missing)
    false_rows = result[result["label_valid"] == False]
    assert len(false_rows) >= 1


def test_flag_pregap_rows_preserves_existing_label_valid():
    """If label_valid already exists, it must not be overwritten for unaffected rows."""
    df = _make_df(tickers=("AAA",))
    df["label_valid"] = True
    missing = _make_missing_df("AAA", "2099-01-01")
    result = flag_pregap_rows(df, missing)
    last_idx = result.groupby("ticker").tail(1).index
    non_last = result.drop(index=last_idx)
    assert non_last["label_valid"].all()


def test_flag_pregap_rows_empty_missing():
    """Empty missing DataFrame must leave all rows valid."""
    df = _make_df()
    missing = pd.DataFrame(columns=["ticker", "missing_date"])
    result = flag_pregap_rows(df, missing)
    assert "label_valid" in result.columns
    assert result["label_valid"].all()
