import pandas as pd


def add_daily_returns(df):
    # Add per-ticker daily return column (close.pct_change).
    # First row per ticker is NaN by construction.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    return df


def flag_extreme_returns(df, threshold=0.30):
    # Return rows where daily_return > threshold.
    # Each row is a candidate for manual review: real market event vs. data issue
    if "daily_return" not in df.columns:
        df = add_daily_returns(df)
    df = df.copy()
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    flagged = df[df["daily_return"].abs() > threshold]
    cols = ["ticker", "date", "prev_close", "close", "daily_return", "volume"]
    return flagged[cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def drop_before(df, ticker, cutoff_date):
    # Drop rows for `ticker` strictly before `cutoff_date`.
    # Use for confirmed ticker reassignments: e.g. META was FB before 2022-06-09
    cutoff = pd.Timestamp(cutoff_date, tz="America/New_York")
    mask = (df["ticker"] == ticker) & (df["date"] < cutoff)
    return df[~mask].reset_index(drop=True)


def find_missing_trading_days(df):
    # For each ticker, find official NASDAQ trading days between its first
    # and last row that are missing from df. Returns DataFrame with columns
    # ['ticker', 'missing_date']. Run BEFORE drop_before so our own drops
    # aren't reported as missing.
    import pandas_market_calendars as mcal
    nasdaq = mcal.get_calendar("NASDAQ")

    out = []
    for ticker, g in df.groupby("ticker"):
        dates = pd.DatetimeIndex(g["date"].dt.tz_localize(None).dt.normalize().unique())
        cal = nasdaq.schedule(start_date=dates.min(), end_date=dates.max()).index
        if cal.tz is not None:
            cal = cal.tz_localize(None)
        cal = cal.normalize()
        for d in cal.difference(dates):
            out.append((ticker, d))
    return pd.DataFrame(out, columns=["ticker", "missing_date"])


def flag_pregap_rows(df, missing):
    # For each (ticker, missing_date), mark the immediately preceding row's
    # label as invalid. Adds 'label_valid' boolean (default True). The row's
    # next-day forward label is unreliable because the next row in data is
    # not the next trading day.
    df = df.copy()
    if "label_valid" not in df.columns:
        df["label_valid"] = True
    for ticker, miss_grp in missing.groupby("ticker"):
        ticker_subset = df[df["ticker"] == ticker]
        for miss_date in miss_grp["missing_date"]:
            miss_dt = pd.Timestamp(miss_date, tz="America/New_York")
            before_rows = ticker_subset[ticker_subset["date"] < miss_dt]
            if not before_rows.empty:
                idx = before_rows["date"].idxmax()
                df.loc[idx, "label_valid"] = False
    return df
