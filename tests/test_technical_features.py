# tests/test_technical_features.py
#
# Unit tests for src/technical_features.py
# Covers the indicator helpers (sma, ema, rsi, macd_histogram,
# stochastic_k, atr, bollinger_position, obv, rolling_slope), the
# per-ticker feature builder _features_for_one, the multi-ticker
# build_features wrapper, and the FEATURE_COLS constant.
#
# All functions are pure (Series/DataFrame in, Series/DataFrame out), so
# nothing is mocked. Expected values are derived from the indicator
# definitions and hand-checked where practical.

import numpy as np
import pandas as pd
import pytest

from src.technical_features import (
    FEATURE_COLS,
    _features_for_one,
    atr,
    bollinger_position,
    build_features,
    ema,
    macd_histogram,
    obv,
    rolling_slope,
    rsi,
    sma,
    stochastic_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(values):
    return pd.Series(values, dtype="float64")


def _ohlc(n, ticker="AAA", *, start="2024-01-02", base_close=100.0):
    """Per-ticker OHLCV frame with tz-aware NY dates and a +1/step close.
    open = close - 0.5, high = close + 1.0, low = close - 1.0, so
    derived ratios (intraday_return, hl_range) have known closed forms."""
    dates = pd.date_range(start, periods=n, freq="B", tz="America/New_York")
    closes = [base_close + i for i in range(n)]
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "date": dates,
        "open": [c - 0.5 for c in closes],
        "high": [c + 1.0 for c in closes],
        "low":  [c - 1.0 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


# ---------------------------------------------------------------------------
# sma
# ---------------------------------------------------------------------------

def test_sma_warmup_is_window_minus_one_nan():
    s = _series([1, 2, 3, 4, 5])
    out = sma(s, window=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].notna().all()


def test_sma_constant_series_equals_constant_after_warmup():
    s = _series([5.0] * 10)
    out = sma(s, window=3)
    assert (out.iloc[2:] == 5.0).all()


def test_sma_known_value():
    # Last window of [1,2,3,4,5] is (3,4,5) -> mean = 4.0.
    s = _series([1, 2, 3, 4, 5])
    out = sma(s, window=3)
    assert out.iloc[-1] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# ema
# ---------------------------------------------------------------------------

def test_ema_warmup_is_span_minus_one_nan():
    s = _series([1, 2, 3])
    out = ema(s, span=2)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1:].notna().all()


def test_ema_recursive_value_matches_hand_calc():
    # span=2, alpha = 2/(span+1) = 2/3. adjust=False (recursive form):
    #   internal y_0 = x_0 = 1
    #   y_1 = (2/3)*2 + (1/3)*1     = 5/3
    #   y_2 = (2/3)*3 + (1/3)*(5/3) = 2 + 5/9 = 23/9
    # min_periods=span masks y_0 to NaN; y_1 and y_2 are returned.
    s = _series([1, 2, 3])
    out = ema(s, span=2)
    assert out.iloc[1] == pytest.approx(5.0 / 3.0)
    assert out.iloc[2] == pytest.approx(23.0 / 9.0)


def test_ema_constant_series_equals_constant_after_warmup():
    s = _series([7.0] * 10)
    out = ema(s, span=3)
    assert (out.iloc[2:] == 7.0).all()


# ---------------------------------------------------------------------------
# rsi
# ---------------------------------------------------------------------------

def test_rsi_warmup_is_period_nan():
    # delta drops index 0 to NaN, then ewm(min_periods=14) needs 14
    # non-null observations -> first non-null RSI sits at index 14.
    s = _series(list(range(1, 30)))
    out = rsi(s, period=14)
    assert out.iloc[:14].isna().all()


def test_rsi_monotonic_down_is_zero_after_warmup():
    # gains all zero, losses all positive -> rs = 0 -> rsi = 0.
    s = _series(list(range(40, 10, -1)))
    out = rsi(s, period=14)
    assert (out.iloc[14:] == 0.0).all()


def test_rsi_monotonic_up_is_nan_after_warmup():
    # losses all zero -> avg_loss = 0 -> .replace(0, NaN) -> rs = NaN
    # -> rsi = NaN. Regression for the divide-by-zero guard.
    s = _series(list(range(1, 30)))
    out = rsi(s, period=14)
    assert out.iloc[14:].isna().all()


def test_rsi_constant_series_is_nan_after_warmup():
    # delta = 0 -> both gain and loss = 0 -> avg_loss = 0 -> NaN.
    s = _series([10.0] * 30)
    out = rsi(s, period=14)
    assert out.iloc[14:].isna().all()


# ---------------------------------------------------------------------------
# macd_histogram
# ---------------------------------------------------------------------------

def test_macd_histogram_first_valid_at_index_33():
    # ema26 is valid from index 25; ema(macd_line, 9) needs 9 valid
    # macd_line values, so the histogram is first valid at 25 + 8 = 33.
    s = _series([float(i) for i in range(50)])
    out = macd_histogram(s)
    assert out.iloc[:33].isna().all()
    assert pd.notna(out.iloc[33])


def test_macd_histogram_constant_series_is_zero_after_warmup():
    # All EMAs of a constant equal the constant -> macd_line = 0 -> hist = 0.
    s = _series([12.0] * 50)
    out = macd_histogram(s)
    assert (out.iloc[33:] == 0.0).all()


# ---------------------------------------------------------------------------
# stochastic_k
# ---------------------------------------------------------------------------

def test_stochastic_k_warmup_is_period_minus_one_nan():
    n = 20
    high  = _series([2.0 + i for i in range(n)])
    low   = _series([0.0 + i for i in range(n)])
    close = _series([1.0 + i for i in range(n)])
    out = stochastic_k(high, low, close, period=14)
    assert out.iloc[:13].isna().all()
    assert out.iloc[13:].notna().all()


def test_stochastic_k_close_pinned_to_high_is_100():
    # close == rolling high -> numerator equals denominator -> %K = 100.
    n = 20
    high  = _series([10.0] * n)
    low   = _series([0.0] * n)
    close = _series([10.0] * n)
    out = stochastic_k(high, low, close, period=14)
    assert (out.iloc[13:] == 100.0).all()


def test_stochastic_k_close_pinned_to_low_is_0():
    n = 20
    high  = _series([10.0] * n)
    low   = _series([0.0] * n)
    close = _series([0.0] * n)
    out = stochastic_k(high, low, close, period=14)
    assert (out.iloc[13:] == 0.0).all()


def test_stochastic_k_flat_range_is_nan():
    # high == low across the whole window -> rng = 0 -> .replace(0, NaN).
    n = 20
    high  = _series([5.0] * n)
    low   = _series([5.0] * n)
    close = _series([5.0] * n)
    out = stochastic_k(high, low, close, period=14)
    assert out.iloc[13:].isna().all()


# ---------------------------------------------------------------------------
# atr
# ---------------------------------------------------------------------------

def test_atr_warmup_is_period_minus_one_nan():
    # ewm with min_periods=14: indices 0..12 NaN; index 13 first valid,
    # because TR[0] survives via skipna fallback (see next test).
    n = 20
    high  = _series([2.0 + i for i in range(n)])
    low   = _series([0.0 + i for i in range(n)])
    close = _series([1.0 + i for i in range(n)])
    out = atr(high, low, close, period=14)
    assert out.iloc[:13].isna().all()
    assert pd.notna(out.iloc[13])


def test_atr_skipna_fallback_keeps_tr_at_row_0_finite():
    # Constant prices high=2, low=1, close=1.5. On row 0 prev_close is
    # NaN, so |high - prev_close| and |low - prev_close| are NaN; max(
    # axis=1, skipna=True) returns (high - low) = 1. All TR = 1 -> ATR
    # after warmup is exactly 1.0. If the skipna fallback ever breaks,
    # TR[0] would be NaN, ewm(min_periods=14) would not be satisfied
    # until index 14, and ATR[13] would be NaN -- this assertion fails.
    n = 20
    high  = _series([2.0] * n)
    low   = _series([1.0] * n)
    close = _series([1.5] * n)
    out = atr(high, low, close, period=14)
    assert (out.iloc[13:] == 1.0).all()


def test_atr_constant_prices_are_zero_after_warmup():
    n = 20
    high  = _series([5.0] * n)
    low   = _series([5.0] * n)
    close = _series([5.0] * n)
    out = atr(high, low, close, period=14)
    assert (out.iloc[13:] == 0.0).all()


# ---------------------------------------------------------------------------
# bollinger_position
# ---------------------------------------------------------------------------

def test_bollinger_position_warmup_is_window_minus_one_nan():
    s = _series([float(i) for i in range(25)])
    out = bollinger_position(s, window=20)
    assert out.iloc[:19].isna().all()
    assert pd.notna(out.iloc[19])


def test_bollinger_position_constant_series_is_nan():
    # std = 0 -> rng = 0 -> .replace(0, NaN). Regression for the
    # divide-by-zero guard.
    s = _series([10.0] * 25)
    out = bollinger_position(s, window=20)
    assert out.iloc[19:].isna().all()


def test_bollinger_position_known_value():
    # window=3, series=[1,2,3].  m=2, s=std(ddof=0)=sqrt(2/3).
    # upper = 2 + 2s, lower = 2 - 2s, rng = 4s.
    # close=3 -> (3 - lower)/rng = (1 + 2s)/(4s) = 1/(4s) + 1/2
    #                            = sqrt(6)/8 + 0.5.
    s = _series([1.0, 2.0, 3.0])
    out = bollinger_position(s, window=3)
    expected = np.sqrt(6) / 8 + 0.5
    assert out.iloc[2] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# obv
# ---------------------------------------------------------------------------

def test_obv_first_row_is_zero():
    # sign(diff[0]) = sign(NaN -> fillna(0)) = 0 -> obv[0] = 0.
    close  = _series([10.0, 11.0])
    volume = _series([100.0, 200.0])
    out = obv(close, volume)
    assert out.iloc[0] == 0.0


def test_obv_hand_calc_for_mixed_directions():
    # close  = [10, 11, 10, 12]
    # diff   = [NaN, +1, -1, +2]   .fillna(0) -> [0, 1, -1, 2]
    # sign   = [0, 1, -1, 1]
    # volume = [100, 200, 300, 400]
    # signed = [0, 200, -300, 400]
    # cumsum = [0, 200, -100, 300]
    close  = _series([10.0, 11.0, 10.0, 12.0])
    volume = _series([100.0, 200.0, 300.0, 400.0])
    assert obv(close, volume).tolist() == [0.0, 200.0, -100.0, 300.0]


# ---------------------------------------------------------------------------
# rolling_slope
# ---------------------------------------------------------------------------

def test_rolling_slope_warmup_is_window_minus_one_nan():
    s = _series([float(i) for i in range(5)])
    out = rolling_slope(s, window=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].notna().all()


def test_rolling_slope_on_linear_ramp_equals_one():
    # series = [0,1,2,3,4], window=3. Every window is a perfect ramp:
    # x_dev = [-1, 0, 1], y_dev = [-1, 0, 1], slope = 2/2 = 1.0.
    s = _series([0.0, 1.0, 2.0, 3.0, 4.0])
    out = rolling_slope(s, window=3)
    assert (out.iloc[2:] == 1.0).all()


def test_rolling_slope_constant_series_equals_zero():
    s = _series([5.0] * 10)
    out = rolling_slope(s, window=3)
    assert (out.iloc[2:] == 0.0).all()


# ---------------------------------------------------------------------------
# FEATURE_COLS
# ---------------------------------------------------------------------------

def test_feature_cols_has_18_entries():
    # 14 technical (set by _features_for_one) + 4 macro = 18.
    assert len(FEATURE_COLS) == 18


def test_feature_cols_contains_anchor_names():
    # Renames of these three would silently break the bundle wiring.
    for name in ("return_1d", "vix_close", "treasury_10y"):
        assert name in FEATURE_COLS


def test_feature_cols_has_no_duplicates():
    assert len(set(FEATURE_COLS)) == len(FEATURE_COLS)


# ---------------------------------------------------------------------------
# _features_for_one
# ---------------------------------------------------------------------------

# The 14 columns _features_for_one must add. A missing entry here
# indicates the per-ticker builder dropped an indicator without
# updating FEATURE_COLS.
TECH_COLS = [
    "return_1d", "return_5d", "intraday_return", "hl_range",
    "sma_ratio_10", "sma_ratio_50", "ema_ratio_20",
    "macd_hist", "rsi_14", "stoch_k_14",
    "atr_14_norm", "bb_position_20",
    "volume_ratio_20", "obv_slope_10",
]


def test_features_for_one_adds_all_14_technical_columns():
    df = _ohlc(60)
    out = _features_for_one(df)
    for col in TECH_COLS:
        assert col in out.columns, f"missing technical column: {col}"


def test_features_for_one_return_1d_first_row_is_nan():
    df = _ohlc(60)
    out = _features_for_one(df).sort_values("date").reset_index(drop=True)
    assert pd.isna(out.loc[0, "return_1d"])


def test_features_for_one_return_1d_hand_calc():
    # close = [100, 101, 102, 103, 104]. return_1d[1] = 101/100 - 1 = 0.01.
    df = _ohlc(5, base_close=100.0)
    out = _features_for_one(df).sort_values("date").reset_index(drop=True)
    assert out.loc[1, "return_1d"] == pytest.approx(0.01)
    assert out.loc[2, "return_1d"] == pytest.approx(102.0 / 101.0 - 1.0)


def test_features_for_one_intraday_return_hand_calc():
    # Factory: open = close - 0.5. With close=100, open=99.5 ->
    # intraday_return = 100/99.5 - 1.
    df = _ohlc(5, base_close=100.0)
    out = _features_for_one(df).sort_values("date").reset_index(drop=True)
    assert out.loc[0, "intraday_return"] == pytest.approx(100.0 / 99.5 - 1.0)


def test_features_for_one_hl_range_hand_calc():
    # Factory: high = close + 1, low = close - 1 -> hl_range = 2 / close.
    df = _ohlc(5, base_close=100.0)
    out = _features_for_one(df).sort_values("date").reset_index(drop=True)
    assert out.loc[0, "hl_range"] == pytest.approx(2.0 / 100.0)


def test_features_for_one_sorts_by_date_on_shuffled_input():
    df = _ohlc(60).sample(frac=1, random_state=7)
    out = _features_for_one(df)
    assert out["date"].is_monotonic_increasing


def test_features_for_one_preserves_tz_aware_ny_date():
    df = _ohlc(60)
    out = _features_for_one(df)
    assert str(out["date"].dt.tz) == "America/New_York"


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

def test_build_features_runs_across_two_tickers():
    df = pd.concat([_ohlc(60, "AAA"), _ohlc(60, "BBB")], ignore_index=True)
    out = build_features(df)
    assert set(out["ticker"].unique()) == {"AAA", "BBB"}
    for col in TECH_COLS:
        assert col in out.columns


def test_build_features_no_cross_ticker_leakage_in_return_1d():
    # First row per ticker must have return_1d == NaN. If the per-ticker
    # groupby boundary is ever lost, BBB's first close would be diffed
    # against AAA's last close and produce a numeric value here.
    df = pd.concat([_ohlc(60, "AAA"), _ohlc(60, "BBB")], ignore_index=True)
    out = build_features(df).sort_values(["ticker", "date"]).reset_index(drop=True)
    bbb_first = out[out["ticker"] == "BBB"].iloc[0]
    assert pd.isna(bbb_first["return_1d"])


def test_build_features_preserves_row_count():
    df = pd.concat([_ohlc(60, "AAA"), _ohlc(60, "BBB")], ignore_index=True)
    out = build_features(df)
    assert len(out) == len(df)


def test_build_features_preserves_tz_aware_ny_date():
    df = pd.concat([_ohlc(60, "AAA"), _ohlc(60, "BBB")], ignore_index=True)
    out = build_features(df)
    assert str(out["date"].dt.tz) == "America/New_York"
