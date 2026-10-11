import pandas as pd
import pytest
from src.features import add_daily_return, add_rsi, add_sma, add_sma_ratio, add_all_indicators


def _sample_df():
    dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="America/New_York")
    return pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["AAA"] * 60 + ["BBB"] * 60,
        "close": list(range(100, 160)) + list(range(200, 260)),
    })


def test_daily_return_first_row_nan():
    df = add_daily_return(_sample_df())
    first_rows = df.groupby("ticker").head(1)
    assert first_rows["daily_return"].isna().all()


def test_daily_return_values():
    df = add_daily_return(_sample_df())
    aaa = df[df["ticker"] == "AAA"].reset_index(drop=True)
    assert abs(aaa.loc[1, "daily_return"] - (101 / 100 - 1)) < 1e-9


def test_rsi_warmup_nan():
    df = add_rsi(_sample_df(), period=14)
    aaa = df[df["ticker"] == "AAA"].reset_index(drop=True)
    assert aaa.loc[:12, "rsi_14"].isna().all()
    assert not pd.isna(aaa.loc[14, "rsi_14"])


def test_rsi_range():
    df = add_rsi(_sample_df(), period=14)
    rsi = df["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_sma_warmup_nan():
    df = add_sma(_sample_df(), period=20)
    aaa = df[df["ticker"] == "AAA"].reset_index(drop=True)
    assert aaa.loc[:18, "sma_20"].isna().all()
    assert not pd.isna(aaa.loc[19, "sma_20"])


def test_sma_ratio_columns():
    df = add_sma_ratio(_sample_df(), short=20, long=50)
    assert "sma_20" in df.columns
    assert "sma_50" in df.columns
    assert "sma_ratio" in df.columns


def test_no_leakage():
    df = _sample_df()
    df_full = add_all_indicators(df.copy())
    df_partial = add_all_indicators(df.iloc[:50].copy())
    common = df_full.iloc[:50][["daily_return", "rsi_14", "sma_20", "sma_50", "sma_ratio"]]
    assert common.equals(df_partial[["daily_return", "rsi_14", "sma_20", "sma_50", "sma_ratio"]])


def test_per_ticker_isolation():
    df = add_all_indicators(_sample_df())
    aaa = df[df["ticker"] == "AAA"].reset_index(drop=True)
    bbb = df[df["ticker"] == "BBB"].reset_index(drop=True)
    assert not aaa["daily_return"].equals(bbb["daily_return"])

def test_macd_columns():
    from src.features import add_macd
    df = add_macd(_sample_df())
    assert {"macd", "macd_signal", "macd_hist"}.issubset(df.columns)


def test_bollinger_warmup():
    from src.features import add_bollinger
    df = add_bollinger(_sample_df(), period=20)
    aaa = df[df["ticker"] == "AAA"].reset_index(drop=True)
    assert aaa.loc[:18, "bb_upper_20"].isna().all()
    assert not pd.isna(aaa.loc[19, "bb_upper_20"])


def test_lagged_returns_columns():
    from src.features import add_lagged_returns
    df = add_lagged_returns(_sample_df(), lags=(1, 3, 5))
    assert {"return_lag_1", "return_lag_3", "return_lag_5"}.issubset(df.columns)


def test_lagged_returns_shift():
    from src.features import add_lagged_returns
    df = add_lagged_returns(_sample_df(), lags=(1,))
    aaa = df[df["ticker"] == "AAA"].reset_index(drop=True)
    assert pd.isna(aaa.loc[0, "return_lag_1"])
    assert pd.isna(aaa.loc[1, "return_lag_1"])