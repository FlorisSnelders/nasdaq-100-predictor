"""
Automated pipeline validation checks for the NASDAQ-100 data pipeline.

After each pipeline stage, call `validate_stage()` (or one of the
stage-specific helpers) to assert data quality invariants. Any failure
raises PipelineValidationError with a clear message so the pipeline
halts immediately rather than producing silently-broken downstream data.

Typical use inside run_pipeline():
    df = load_all_tickers("data/")
    df = standardize(df)
    validate_stage(df, stage="after_load")

    df = clean_pipeline(df)
    validate_stage(df, stage="after_clean", expected_min_rows=200_000)

    features = build_features(df)
    validate_stage(features, stage="after_features", require_label=True)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Iterable, Optional
import logging

logger = logging.getLogger(__name__)


class PipelineValidationError(Exception):
    """Raised when a pipeline-stage validation check fails."""
    pass


# -------------------------- individual checks -----------------------------
# Each check returns None on success and raises PipelineValidationError on
# failure. They're small and composable so you can call them individually
# in unit tests or chain them via validate_stage().

def check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Every column in `required` must exist in df."""
    missing = set(required) - set(df.columns)
    if missing:
        raise PipelineValidationError(
            f"Missing required columns: {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )


def check_no_nulls(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """No null values allowed in the given columns."""
    offenders = {}
    for col in columns:
        if col not in df.columns:
            continue  # column-existence check is a separate concern
        n_null = df[col].isna().sum()
        if n_null > 0:
            offenders[col] = int(n_null)
    if offenders:
        raise PipelineValidationError(
            f"Null values found in columns that must be non-null: {offenders}"
        )


def check_no_duplicates(df: pd.DataFrame, subset: Iterable[str]) -> None:
    """No duplicate rows allowed on the (ticker, date) compound key."""
    subset = list(subset)
    dup_mask = df.duplicated(subset=subset, keep=False)
    if dup_mask.any():
        sample = df.loc[dup_mask, subset].head(5).to_dict(orient="records")
        raise PipelineValidationError(
            f"Found {int(dup_mask.sum())} duplicate rows on {subset}. "
            f"First few: {sample}"
        )


def check_no_future_dates(df: pd.DataFrame, date_col: str = "date") -> None:
    """No row may be dated in the future."""
    if date_col not in df.columns:
        raise PipelineValidationError(f"Date column '{date_col}' not in df")
    # Normalize: drop tz so comparison works no matter what tz the data uses
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    future = dates > now
    if future.any():
        n = int(future.sum())
        bad = df.loc[future, date_col].head(3).tolist()
        raise PipelineValidationError(
            f"{n} rows have dates in the future (after {now}). Examples: {bad}"
        )


def check_no_negative_prices(
    df: pd.DataFrame,
    price_cols: Iterable[str] = ("open", "high", "low", "close"),
) -> None:
    """OHLC prices must be strictly positive."""
    offenders = {}
    for col in price_cols:
        if col not in df.columns:
            continue
        bad = df[col] <= 0
        if bad.any():
            offenders[col] = int(bad.sum())
    if offenders:
        raise PipelineValidationError(
            f"Non-positive prices found: {offenders}. "
            f"Prices must be > 0 in all OHLC columns."
        )


def check_row_count_in_range(
    df: pd.DataFrame,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> None:
    """Row count must lie inside [minimum, maximum]."""
    n = len(df)
    if minimum is not None and n < minimum:
        raise PipelineValidationError(
            f"Too few rows: got {n:,}, expected at least {minimum:,}"
        )
    if maximum is not None and n > maximum:
        raise PipelineValidationError(
            f"Too many rows: got {n:,}, expected at most {maximum:,}"
        )


def check_ohlc_consistency(df: pd.DataFrame) -> None:
    """high >= max(open, close) and low <= min(open, close)."""
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return  # silently skip if OHLC isn't all present
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)
    n_high = int(bad_high.sum())
    n_low = int(bad_low.sum())
    if n_high or n_low:
        raise PipelineValidationError(
            f"OHLC inconsistency: {n_high} rows where high < max(open, close), "
            f"{n_low} rows where low > min(open, close)"
        )


def check_no_negative_volume(df: pd.DataFrame, col: str = "volume") -> None:
    """Volume must be >= 0."""
    if col not in df.columns:
        return
    bad = df[col] < 0
    if bad.any():
        raise PipelineValidationError(
            f"{int(bad.sum())} rows with negative {col}. Volume must be >= 0."
        )


# ------------------------- stage orchestrator -----------------------------

# Per-stage profiles: what to check, with what tolerances.
STAGE_PROFILES = {
    "after_load": dict(
        required_cols=("date", "open", "high", "low", "close",
                       "volume", "ticker"),
        no_null_cols=("date", "ticker"),
        dup_subset=("ticker", "date"),
        check_dates=True,
        check_prices=True,
        check_ohlc=False,  # raw data may have odd rows; we clean those later
        min_rows=100_000,
        max_rows=500_000,
    ),
    "after_clean": dict(
        required_cols=("date", "open", "high", "low", "close",
                       "volume", "ticker"),
        no_null_cols=("date", "ticker", "open", "high", "low", "close"),
        dup_subset=("ticker", "date"),
        check_dates=True,
        check_prices=True,
        check_ohlc=True,
        min_rows=200_000,
        max_rows=240_000,
    ),
    "after_features": dict(
        required_cols=("date", "ticker"),
        no_null_cols=("date", "ticker"),
        dup_subset=("ticker", "date"),
        check_dates=True,
        check_prices=False,  # features table doesn't carry OHLC
        check_ohlc=False,
        min_rows=200_000,
        max_rows=240_000,
    ),
}


def validate_stage(
    df: pd.DataFrame,
    stage: str,
    *,
    expected_min_rows: Optional[int] = None,
    expected_max_rows: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Run all validation checks for a given pipeline stage.

    Parameters
    ----------
    df : the dataframe to validate
    stage : one of the keys in STAGE_PROFILES ("after_load", "after_clean",
            "after_features"). Defines which checks run.
    expected_min_rows / expected_max_rows : optional overrides for row count
    verbose : print PASS/FAIL summaries

    Raises
    ------
    PipelineValidationError on the first failed check.
    """
    if stage not in STAGE_PROFILES:
        raise ValueError(
            f"Unknown stage '{stage}'. "
            f"Pick one of: {sorted(STAGE_PROFILES)}"
        )
    p = STAGE_PROFILES[stage]
    logger.info("--- Validating stage: %s ---", stage)

    check_required_columns(df, p["required_cols"])
    logger.info("[OK] required columns present")

    check_no_nulls(df, p["no_null_cols"])
    logger.info("[OK] no nulls in %s", list(p["no_null_cols"]))

    check_no_duplicates(df, p["dup_subset"])
    logger.info("[OK] no duplicates on %s", list(p["dup_subset"]))

    if p["check_dates"]:
        check_no_future_dates(df)
        logger.info("[OK] no future-dated rows")

    if p["check_prices"]:
        check_no_negative_prices(df)
        check_no_negative_volume(df)
        logger.info("[OK] no negative prices or volume")

    if p["check_ohlc"]:
        check_ohlc_consistency(df)
        logger.info("[OK] OHLC consistency holds")

    min_rows = expected_min_rows if expected_min_rows is not None else p["min_rows"]
    max_rows = expected_max_rows if expected_max_rows is not None else p["max_rows"]
    check_row_count_in_range(df, minimum=min_rows, maximum=max_rows)
    if verbose:
        print(f"  [OK] row count {len(df):,} in [{min_rows:,}, {max_rows:,}]")

    if verbose:
        print(f"--- {stage}: ALL CHECKS PASSED ---\n")
