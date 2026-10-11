# tests/test_datasets.py
#
# Unit tests for the raw-file loaders (src/datasets.py), the shared VIX
# cleaner (src.macro.clean_vix_frame) and the offline bake-off frame
# (src.pipeline.build_bakeoff_frame). Everything runs on tiny synthetic
# files in tmp_path; no network.

import numpy as np
import pandas as pd
import pytest

from src.datasets import load_macro, load_prices
from src.macro import clean_vix_frame
from src.pipeline import build_bakeoff_frame

DAYS = pd.bdate_range("2024-01-02", "2024-01-31")


def _write_prices(folder, tickers=("AAA", "BBB"), date_fmt="%Y-%m-%d 00:00:00-05:00"):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for t in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(DAYS)))
        pd.DataFrame({
            "Date": DAYS.strftime(date_fmt),
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1_000, 5_000, len(DAYS)),
        }).to_csv(folder / f"{t}.csv", index=False)


def _write_macro(path):
    dates = pd.date_range("2023-12-20", "2024-02-05")
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "vix_close": np.linspace(12, 20, len(dates)),
        "treasury_10y": np.linspace(3.8, 4.2, len(dates)),
    }).to_csv(path, index=False)


def test_load_prices_lowercases_and_adds_ticker(tmp_path):
    _write_prices(tmp_path / "prices")
    df = load_prices(tmp_path / "prices")
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert {"date", "open", "high", "low", "close", "volume", "transactions"} <= set(df.columns)
    assert df["transactions"].isna().all()
    assert len(df) == 2 * len(DAYS)


def test_load_prices_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_prices(tmp_path / "empty")


def test_load_macro_columns_and_dtypes(tmp_path):
    _write_macro(tmp_path / "macro.csv")
    df = load_macro(tmp_path / "macro.csv")
    assert list(df.columns) == ["date", "vix_close", "treasury_10y"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_macro_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_macro(tmp_path / "nope.csv")


def test_load_macro_missing_column_raises(tmp_path):
    pd.DataFrame({"date": ["2024-01-02"], "vix_close": [13.0]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(ValueError, match="treasury_10y"):
        load_macro(tmp_path / "m.csv")


def test_clean_vix_frame_drops_gaps_before_pct_change():
    raw = pd.DataFrame({
        "d": ["2024-01-04", "2024-01-02", "2024-01-03", "2024-01-05"],
        "v": [11.0, 10.0, None, "bad"],
    })
    out = clean_vix_frame(raw, date_col="d", vix_col="v")
    # 01-03 (NaN) and 01-05 (non-numeric) dropped; first valid row has no change -> dropped
    assert list(out.columns) == ["date", "vix_close", "vix_change_1d"]
    assert len(out) == 1
    assert out["vix_change_1d"].iloc[0] == pytest.approx(0.10)  # 10 -> 11, gap skipped
    assert str(out["date"].dt.tz) == "America/New_York"


def test_build_bakeoff_frame_merges_macro_on_calendar_day(tmp_path):
    _write_prices(tmp_path / "prices")
    _write_macro(tmp_path / "macro.csv")
    df = build_bakeoff_frame(tmp_path / "prices", tmp_path / "macro.csv")
    assert {"vix", "treasury_yield_10y", "label_valid"} <= set(df.columns)
    assert df["date"].dt.tz is None
    assert df["vix"].notna().all()  # every trading day found its macro row
    assert len(df) == 2 * len(DAYS) - 2  # 2024-01-15 (MLK day) removed per ticker
