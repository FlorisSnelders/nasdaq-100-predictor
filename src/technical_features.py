"""
Technical indicator and feature functions for the NASDAQ-100 pipeline.

Only the indicator calculations and the per-ticker feature builder live here.
All other steps (loading macro data, joining, walk-forward, feature selection,
tuning, evaluation) are in notebooks/03_xgboost_model.ipynb.

Every indicator uses only past or current data (rolling/ewm with
min_periods, and shift(1) where needed), so no leakage occurs. The
warm-up length is documented in each function's docstring.
"""

import numpy as np
import pandas as pd


# The 18 model columns (14 technical indicators + 4 macro columns).
FEATURE_COLS = [
    "return_1d", "return_5d", "intraday_return", "hl_range",
    "sma_ratio_10", "sma_ratio_50", "ema_ratio_20",
    "macd_hist", "rsi_14", "stoch_k_14",
    "atr_14_norm", "bb_position_20",
    "volume_ratio_20", "obv_slope_10",
    "vix_close", "vix_change_1d",
    "treasury_10y", "treasury_10y_change_1d",
]


# ---------------------------------------------------------------------------
# Indicator functions
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. Warm-up: first ``window - 1`` values are NaN."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentially-weighted moving average with ``adjust=False`` (standard
    recursive form). Warm-up: first ``span - 1`` values are NaN
    (enforced via ``min_periods=span``)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index. ``period=14`` is the default.

    Warm-up: first ``period`` values are NaN (one is lost to ``diff``, then
    Wilder smoothing needs ``period`` non-null observations).
    Leakage-safe: only uses current and past closes.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing is equivalent to EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_histogram(close: pd.Series) -> pd.Series:
    """MACD histogram = (EMA12 - EMA26) - EMA9 of that difference.

    Warm-up: ~33 rows (EMA26 needs 25 prior values; EMA9 of the
    difference adds 8 more).
    """
    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    macd_line = ema_fast - ema_slow
    signal = ema(macd_line, 9)
    return macd_line - signal


def stochastic_k(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Fast stochastic %K: 100 * (close - lowest_low) / (highest_high - lowest_low).

    Warm-up: ``period - 1`` NaN values.
    """
    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()
    rng = (highest_high - lowest_low).replace(0, np.nan)
    return 100.0 * (close - lowest_low) / rng


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Wilder's Average True Range. Warm-up: ``period - 1`` NaN values.

    True Range = max(high-low, ``|high-prev_close|``, ``|low-prev_close|``). On
    the very first row prev_close is NaN, but skipna-max falls back to
    (high-low), so TR is non-null from index 0 (standard Wilder
    convention). Wilder smoothing (alpha = 1/period) with
    min_periods=period then becomes valid at index ``period - 1``.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """%B position inside +/-2-stddev Bollinger bands.

    Returns 0 at the lower band, 1 at the upper band; can go outside [0, 1].
    Warm-up: ``window - 1`` NaN values.
    """
    m = close.rolling(window=window, min_periods=window).mean()
    s = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = m + 2.0 * s
    lower = m - 2.0 * s
    rng = (upper - lower).replace(0, np.nan)
    return (close - lower) / rng


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative signed volume. No warm-up (first row 0)."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope of ``series`` against a 0..window-1 index, over a rolling
    window. Warm-up: ``window - 1`` NaN values."""
    x = np.arange(window, dtype=float)
    x_dev = x - x.mean()
    denom = float((x_dev ** 2).sum())

    def _slope(arr: np.ndarray) -> float:
        y_dev = arr - arr.mean()
        return float((x_dev * y_dev).sum() / denom)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Per-ticker feature builder
# ---------------------------------------------------------------------------

def _features_for_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    open_ = g["open"]
    vol = g["volume"].astype(float)

    g["return_1d"] = close / close.shift(1) - 1.0
    g["return_5d"] = close / close.shift(5) - 1.0
    g["intraday_return"] = close / open_ - 1.0
    g["hl_range"] = (high - low) / close
    g["sma_ratio_10"] = close / sma(close, 10)
    g["sma_ratio_50"] = close / sma(close, 50)
    g["ema_ratio_20"] = close / ema(close, 20)
    g["macd_hist"] = macd_histogram(close)
    g["rsi_14"] = rsi(close, 14)
    g["stoch_k_14"] = stochastic_k(high, low, close, 14)
    g["atr_14_norm"] = atr(high, low, close, 14) / close
    g["bb_position_20"] = bollinger_position(close, 20)
    g["volume_ratio_20"] = vol / sma(vol, 20)
    g["obv_slope_10"] = rolling_slope(obv(close, vol), 10)
    return g


def build_features(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute the 14 technical indicators per ticker (groupby) on top of
    the frame that has already been joined with the macro data."""
    pieces = [
        _features_for_one(g)
        for _, g in joined.groupby("ticker", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True)
