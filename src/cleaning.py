import pandas as pd
import pandas_market_calendars as mcal

_TRADING_DAYS_END = (
    pd.Timestamp.today() + pd.Timedelta(days=7)
).strftime("%Y-%m-%d")
_trading_days = (
    mcal.get_calendar("XNYS")
    .valid_days(start_date="2015-01-01", end_date=_TRADING_DAYS_END)
)
_trading_days = pd.DatetimeIndex(_trading_days).normalize().tz_localize(None)


def remove_weekends_and_holidays(df):
    df = df.copy()
    before = len(df)
    dates = df["date"].dt.tz_localize(None).dt.normalize()
    df = df[dates.isin(_trading_days)].reset_index(drop=True)
    print(f"[calendar] Removed: {before - len(df):,} | Remaining: {len(df):,}")
    return df


def remove_duplicate_rows(df):
    df = df.copy()
    before = len(df)
    df = (df.sort_values("volume", ascending=False)
            .drop_duplicates(subset=["ticker", "date"], keep="first")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True))
    print(f"[dedupe] Removed: {before - len(df):,} | Remaining: {len(df):,}")
    return df


def validate_price_consistency(df):
    df = df.copy()
    before = len(df)
    rule1 = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
    rule2 = df["high"] >= df["low"]
    rule3 = (df["high"] >= df["open"]) & (df["high"] >= df["close"])
    rule4 = (df["low"] <= df["open"]) & (df["low"] <= df["close"])
    df = df[rule1 & rule2 & rule3 & rule4].reset_index(drop=True)
    print(f"[ohlc] Removed: {before - len(df):,} | Remaining: {len(df):,}")
    return df