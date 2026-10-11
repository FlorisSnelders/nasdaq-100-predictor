from pathlib import Path
import pandas as pd


def load_all_tickers(path):
    path = Path(path)
    frames = []
    for csv in sorted(path.glob("*.csv")):
        df = pd.read_csv(csv)
        df["ticker"] = csv.stem
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    assert combined["ticker"].nunique() == 100, f"Expected 100 tickers, got {combined['ticker'].nunique()}"
    return combined


EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "transactions", "ticker"]


def standardize(df):
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    df["transactions"] = df["transactions"].astype("Int64")
    df["ticker"] = df["ticker"].astype("string")
    return df[EXPECTED_COLUMNS]


def check_schema(df):
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date must be datetime"
    for col in ["open", "high", "low", "close"]:
        assert df[col].dtype == "float64", f"{col} must be float64"
    assert df["volume"].dtype == "int64", "volume must be int64"
    assert str(df["transactions"].dtype) in ("int64", "Int64"), "transactions must be int64 or Int64"
    assert df["date"].dt.tz is not None, "date must be timezone-aware"
    return True