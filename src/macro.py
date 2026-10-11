"""
src/macro.py
============
Fetch and clean macroeconomic indicators.

Pulls VIX and 10-Year Treasury Yield from Yahoo Finance via yfinance,
aligns to NYSE trading calendar, and forward-fills any gaps. This path is
used by ``run_pipeline`` (the shared bake-off harness).

The production model's VIX features are built by ``clean_vix_frame``
instead, from the FRED ``VIXCLS`` series (see scripts/build_database.py
and src/jobs/daily_macro.py).

Provides:
    fetch_macro(start, end)          -> downloads from Yahoo Finance
    clean_macro(df)                  -> aligns to trading calendar + ffill
    load_and_clean_macro(start, end) -> fetch + clean in one call
    clean_vix_frame(raw, ...)        -> vix_close / vix_change_1d for storage
"""

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal


# ── NYSE trading calendar ─────────────────────────────────────────────────────
_cal = mcal.get_calendar("XNYS")


def _get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    days = _cal.valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days).normalize().tz_localize(None)


def fetch_macro(start: str = "2015-01-01",
                end: str   = "2025-02-28") -> pd.DataFrame:
    """
    Download VIX and 10-Year Treasury Yield from Yahoo Finance.

    Parameters
    ----------
    start : str  Start date  (YYYY-MM-DD)
    end   : str  End date    (YYYY-MM-DD)

    Returns
    -------
    DataFrame with columns: date, vix, treasury_yield_10y
    """
    print("[fetch_macro] Downloading VIX (^VIX) ...")
    vix_raw = yf.download("^VIX", start=start, end=end,
                           progress=False, auto_adjust=True)

    print("[fetch_macro] Downloading 10Y Treasury Yield (^TNX) ...")
    tnx_raw = yf.download("^TNX", start=start, end=end,
                           progress=False, auto_adjust=True)

    # Keep closing price only (handle multi-level columns from yfinance)
    vix = vix_raw["Close"].squeeze().rename("vix")
    tnx = tnx_raw["Close"].squeeze().rename("treasury_yield_10y")

    # Merge on date index
    df = pd.concat([vix, tnx], axis=1)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = "date"
    df = df.reset_index()

    print(f"[fetch_macro] Downloaded {len(df)} rows")
    return df


def clean_macro(df: pd.DataFrame,
                start: str = "2015-01-01",
                end: str   = "2025-02-28") -> pd.DataFrame:
    """
    Align macro data to the NYSE trading calendar and forward-fill gaps.

    Parameters
    ----------
    df    : DataFrame from fetch_macro()
    start : Start date for trading calendar
    end   : End date for trading calendar

    Returns
    -------
    Clean DataFrame with columns: date, vix, treasury_yield_10y
    """
    trading_days = _get_trading_days(start, end)

    # Keep only trading days
    before = len(df)
    df = df[df["date"].isin(trading_days)].copy()
    print(f"[clean_macro] Non-trading rows removed : {before - len(df)}")

    # Reindex to full trading calendar and forward-fill
    df = (df.set_index("date")
            .reindex(trading_days)
            .ffill()
            .reset_index()
            .rename(columns={"index": "date"}))

    # Drop rows before first valid observation
    before = len(df)
    df = df.dropna(subset=["vix", "treasury_yield_10y"]).reset_index(drop=True)
    print(f"[clean_macro] Rows dropped (no prior value) : {before - len(df)}")
    print(f"[clean_macro] Final rows  : {len(df):,}")
    print(f"[clean_macro] Date range  : {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def load_and_clean_macro(start: str = "2015-01-01",
                          end: str   = "2025-02-28") -> pd.DataFrame:
    """Convenience wrapper: fetch + clean in one call."""
    df = fetch_macro(start, end)
    return clean_macro(df, start, end)


def clean_vix_frame(
    raw: pd.DataFrame,
    *,
    date_col: str,
    vix_col: str,
) -> pd.DataFrame:
    """Single source of truth for VIX macro cleaning.

    Takes a raw DataFrame containing a date column and a VIX column (may
    be unsorted, may contain NaN) and reproduces the canonical cleaning
    chain: coerce -> dropna -> sort -> reset -> pct_change -> drop first
    NaN -> tz_localize America/New_York + normalize.

    Returns a DataFrame with columns [date, vix_close, vix_change_1d] in
    that order, `date` tz-aware America/New_York, ready to hand to
    `save_macro_indicators`. Output column names are normalised to the
    storage contract regardless of how `vix_col` was named in `raw`.

    IMPORTANT: `raw` must contain the FULL series, not just new rows.
    pct_change references the previous row in the series; with only the
    newest day it returns NaN and the row gets dropped.

    Both the batch build (scripts/build_database.py) and the live FRED
    job (src/jobs/daily_macro.py) call this helper. Re-implementing the
    chain elsewhere risks train/serve skew.
    """
    vix = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]),
        "vix_close": pd.to_numeric(raw[vix_col], errors="coerce"),
    })
    vix = vix.dropna(subset=["vix_close"]).sort_values("date").reset_index(drop=True)

    vix["vix_change_1d"] = vix["vix_close"].pct_change()
    vix = vix.dropna(subset=["vix_change_1d"]).reset_index(drop=True)

    vix["date"] = vix["date"].dt.tz_localize("America/New_York").dt.normalize()
    return vix[["date", "vix_close", "vix_change_1d"]]
