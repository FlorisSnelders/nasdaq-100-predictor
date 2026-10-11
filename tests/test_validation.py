import pandas as pd
import pytest
from src import validation as V
from src.validation import PipelineValidationError, validate_stage
 
 
def _valid(dates=("2024-01-02", "2024-01-03")):
    n = len(dates)
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)).tz_localize("America/New_York"),
        "ticker": ["AAA"] * n,
        "open": [10.0] * n,
        "high": [12.0] * n,
        "low": [9.0] * n,
        "close": [11.0] * n,
        "volume": [100] * n,
    })
 
 
# ---- check_required_columns ----
def test_required_columns_ok():
    assert V.check_required_columns(_valid(), ["date", "ticker"]) is None
 
 
def test_required_columns_missing():
    with pytest.raises(PipelineValidationError):
        V.check_required_columns(_valid(), ["date", "nope"])
 
 
# ---- check_no_nulls ----
def test_no_nulls_ok():
    assert V.check_no_nulls(_valid(), ["date", "ticker"]) is None
 
 
def test_no_nulls_skips_absent_column():
    assert V.check_no_nulls(_valid(), ["does_not_exist"]) is None
 
 
def test_no_nulls_detects_null():
    df = _valid()
    df.loc[0, "close"] = None
    with pytest.raises(PipelineValidationError):
        V.check_no_nulls(df, ["close"])
 
 
# ---- check_no_duplicates ----
def test_no_duplicates_ok():
    assert V.check_no_duplicates(_valid(), ["ticker", "date"]) is None
 
 
def test_no_duplicates_detects_dupes():
    df = pd.concat([_valid(["2024-01-02"]), _valid(["2024-01-02"])], ignore_index=True)
    with pytest.raises(PipelineValidationError):
        V.check_no_duplicates(df, ["ticker", "date"])
 
 
# ---- check_no_future_dates ----
def test_no_future_dates_ok():
    assert V.check_no_future_dates(_valid()) is None
 
 
def test_no_future_dates_missing_column():
    with pytest.raises(PipelineValidationError):
        V.check_no_future_dates(_valid().drop(columns=["date"]))
 
 
def test_no_future_dates_detects_future():
    df = _valid()
    df.loc[0, "date"] = pd.Timestamp("2999-01-01", tz="America/New_York")
    with pytest.raises(PipelineValidationError):
        V.check_no_future_dates(df)
 
 
# ---- check_no_negative_prices ----
def test_no_negative_prices_ok():
    assert V.check_no_negative_prices(_valid()) is None
 
 
def test_no_negative_prices_skips_absent():
    assert V.check_no_negative_prices(_valid(), price_cols=("missing",)) is None
 
 
def test_no_negative_prices_detects():
    df = _valid()
    df.loc[0, "open"] = 0.0
    with pytest.raises(PipelineValidationError):
        V.check_no_negative_prices(df)
 
 
# ---- check_row_count_in_range ----
def test_row_count_ok():
    assert V.check_row_count_in_range(_valid(), minimum=1, maximum=10) is None
 
 
def test_row_count_too_few():
    with pytest.raises(PipelineValidationError):
        V.check_row_count_in_range(_valid(), minimum=100)
 
 
def test_row_count_too_many():
    with pytest.raises(PipelineValidationError):
        V.check_row_count_in_range(_valid(), maximum=1)
 
 
# ---- check_ohlc_consistency ----
def test_ohlc_ok():
    assert V.check_ohlc_consistency(_valid()) is None
 
 
def test_ohlc_skips_when_incomplete():
    assert V.check_ohlc_consistency(_valid().drop(columns=["high"])) is None
 
 
def test_ohlc_bad_high():
    df = _valid()
    df.loc[0, "high"] = 5.0  # below max(open, close)
    with pytest.raises(PipelineValidationError):
        V.check_ohlc_consistency(df)
 
 
def test_ohlc_bad_low():
    df = _valid()
    df.loc[0, "low"] = 50.0  # above min(open, close)
    with pytest.raises(PipelineValidationError):
        V.check_ohlc_consistency(df)
 
 
# ---- check_no_negative_volume ----
def test_no_negative_volume_ok():
    assert V.check_no_negative_volume(_valid()) is None
 
 
def test_no_negative_volume_skips_absent():
    assert V.check_no_negative_volume(_valid().drop(columns=["volume"])) is None
 
 
def test_no_negative_volume_detects():
    df = _valid()
    df.loc[0, "volume"] = -1
    with pytest.raises(PipelineValidationError):
        V.check_no_negative_volume(df)
 
 
# ---- validate_stage ----
def test_validate_stage_unknown_raises():
    with pytest.raises(ValueError):
        validate_stage(_valid(), stage="nonsense")
 
 
def test_validate_stage_after_load_verbose():
    validate_stage(_valid(), stage="after_load",
                   expected_min_rows=1, expected_max_rows=10, verbose=True)
 
 
def test_validate_stage_after_clean_quiet():
    validate_stage(_valid(), stage="after_clean",
                   expected_min_rows=1, expected_max_rows=10, verbose=False)
 
 
def test_validate_stage_after_features():
    df = _valid()[["date", "ticker"]]
    validate_stage(df, stage="after_features",
                   expected_min_rows=1, expected_max_rows=10, verbose=True)
 
 
def test_validate_stage_propagates_failure():
    df = _valid()
    df.loc[0, "high"] = 1.0  # OHLC violation caught at after_clean
    with pytest.raises(PipelineValidationError):
        validate_stage(df, stage="after_clean",
                       expected_min_rows=1, expected_max_rows=10, verbose=False)