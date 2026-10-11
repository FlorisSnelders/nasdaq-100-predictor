import pandas as pd


def add_daily_return(df):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    return df


def add_rsi(df, period=14):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    delta = df.groupby("ticker")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda s: s.rolling(period, min_periods=period).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda s: s.rolling(period, min_periods=period).mean())
    rs = avg_gain / avg_loss
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_sma(df, period):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df[f"sma_{period}"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(period, min_periods=period).mean())
    return df


def add_sma_ratio(df, short=20, long=50):
    df = add_sma(df, short)
    df = add_sma(df, long)
    df["sma_ratio"] = df[f"sma_{short}"] / df[f"sma_{long}"]
    return df


def add_all_indicators(df):
    df = add_daily_return(df)
    df = add_rsi(df, 14)
    df = add_sma_ratio(df, 20, 50)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_lagged_returns(df)
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    ema_fast = df.groupby("ticker")["close"].transform(lambda s: s.ewm(span=fast, adjust=False).mean())
    ema_slow = df.groupby("ticker")["close"].transform(lambda s: s.ewm(span=slow, adjust=False).mean())
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda s: s.ewm(span=signal, adjust=False).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df, period=20, n_std=2):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    rolling_mean = df.groupby("ticker")["close"].transform(lambda s: s.rolling(period, min_periods=period).mean())
    rolling_std = df.groupby("ticker")["close"].transform(lambda s: s.rolling(period, min_periods=period).std())
    df[f"bb_upper_{period}"] = rolling_mean + n_std * rolling_std
    df[f"bb_lower_{period}"] = rolling_mean - n_std * rolling_std
    df[f"bb_width_{period}"] = (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]) / rolling_mean
    return df


def add_lagged_returns(df, lags=(1, 3, 5, 10)):
    df = add_daily_return(df)
    for lag in lags:
        df[f"return_lag_{lag}"] = df.groupby("ticker")["daily_return"].shift(lag)
    return df

def build_feature_matrix(df):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = add_all_indicators(df)
    df["label"] = (df.groupby("ticker")["close"].shift(-1) > df["close"]).astype("Int64")
    df.loc[df.groupby("ticker").tail(1).index, "label"] = pd.NA
    if "label_valid" in df.columns:
        df.loc[~df["label_valid"], "label"] = pd.NA
    feature_cols = [
        "daily_return", "rsi_14", "sma_20", "sma_50", "sma_ratio",
        "macd", "macd_signal", "macd_hist",
        "bb_upper_20", "bb_lower_20", "bb_width_20",
        "return_lag_1", "return_lag_3", "return_lag_5", "return_lag_10",
        "vix", "treasury_yield_10y",
    ]
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    X = df[feature_cols].astype("float64")
    y = df["label"].astype("int64")
    meta = df[["date", "ticker"]].reset_index(drop=True)
    return X, y, meta