from pathlib import Path
import pandas as pd
import yfinance as yf


def load_snapshot(path):
    path = Path(path)
    frames = []
    for csv in sorted(path.glob("*.csv")):
        df = pd.read_csv(csv)
        df["ticker"] = csv.stem
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    assert combined["ticker"].nunique() == 100, f"Expected 100 tickers, got {combined['ticker'].nunique()}"
    return combined


def load_yfinance(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            continue
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})
        df["ticker"] = t
        df["transactions"] = pd.NA
        # yfinance returns Date as a tz-naive datetime64[s] sitting at
        # UTC midnight of the day after the trading day (verified by
        # reverse-engineering from cleaned_prices: a 2026-06-03 trading
        # day landed as "2026-06-03 20:00:00-04:00", which only happens
        # if the input was tz-naive "2026-06-04 00:00:00"). standardize
        # subsequently does pd.to_datetime(..., utc=True).tz_convert(NY),
        # which yields the 20:00 wallclock and skews the format away
        # from the training CSV's "00:00:00-EE:00" midnight. Normalize
        # here -- after the same UTC->NY conversion that standardize
        # uses, so the calendar day stays correct across DST -- so
        # downstream save_*-functions receive the exact training format.
        df["date"] = (
            pd.to_datetime(df["date"])
            .dt.tz_localize("America/New_York")
            .dt.normalize()
        )
        df = df[["date", "open", "high", "low", "close", "volume", "transactions", "ticker"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)