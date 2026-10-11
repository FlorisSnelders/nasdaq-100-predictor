import pandas as pd
import pytest
from src.data_utils import load_all_tickers, standardize, check_schema, EXPECTED_COLUMNS


def _raw_df(ticker="AAA", upper=False):
    cols = {
        "date": ["2024-01-02", "2024-01-03"],
        "open": [10, 11],
        "high": [12, 13],
        "low": [9, 10],
        "close": [11, 12],
        "volume": [1000, 2000],
        "transactions": [10, 20],
        "ticker": [ticker, ticker],
    }

    df = pd.DataFrame(cols)

    if upper:
        df.columns = [
            c.capitalize() if c != "ticker" else c
            for c in df.columns
        ]

    return df


def _write_csvs(directory, n):
    for i in range(n):
        _raw_df(ticker=f"T{i:03d}")[
            ["date", "open", "high", "low", "close", "volume", "transactions"]
        ].to_csv(
            directory / f"T{i:03d}.csv",
            index=False,
        )


def test_load_all_tickers_happy(tmp_path):
    _write_csvs(tmp_path, 100)

    df = load_all_tickers(tmp_path)

    assert df["ticker"].nunique() == 100
    assert "ticker" in df.columns
    assert len(df) == 200


def test_load_all_tickers_wrong_count_raises(tmp_path):
    _write_csvs(tmp_path, 3)

    with pytest.raises(AssertionError):
        load_all_tickers(tmp_path)


def test_standardize_types_and_columns():
    out = standardize(_raw_df(upper=True))

    assert list(out.columns) == EXPECTED_COLUMNS
    assert out["date"].dt.tz is not None
    assert str(out["date"].dt.tz) == "America/New_York"

    for c in ["open", "high", "low", "close"]:
        assert out[c].dtype == "float64"

    assert out["volume"].dtype == "int64"
    assert str(out["transactions"].dtype) == "Int64"
    assert str(out["ticker"].dtype) == "string"


def test_check_schema_passes_on_standardized():
    assert check_schema(standardize(_raw_df())) is True


def test_check_schema_missing_column():
    df = standardize(_raw_df()).drop(columns=["volume"])

    with pytest.raises(AssertionError):
        check_schema(df)


def test_check_schema_date_not_datetime():
    df = standardize(_raw_df())
    df["date"] = df["date"].astype(str)

    with pytest.raises(AssertionError):
        check_schema(df)


def test_check_schema_wrong_price_dtype():
    df = standardize(_raw_df())
    df["open"] = df["open"].astype("int64")

    with pytest.raises(AssertionError):
        check_schema(df)


def test_check_schema_wrong_volume_dtype():
    df = standardize(_raw_df())
    df["volume"] = df["volume"].astype("float64")

    with pytest.raises(AssertionError):
        check_schema(df)


def test_check_schema_naive_date_rejected():
    df = standardize(_raw_df())
    df["date"] = df["date"].dt.tz_localize(None)

    with pytest.raises(AssertionError):
        check_schema(df)
