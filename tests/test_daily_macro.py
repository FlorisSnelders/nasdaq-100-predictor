import pandas as pd
import pytest

from src.jobs.daily_macro import (
    series_to_raw_frame,
    fetch_vix_series,
    _get_fred_client,
)


def test_series_to_raw_frame():
    s = pd.Series([12.0, 13.0], index=pd.to_datetime(["2026-06-08", "2026-06-09"]))
    df = series_to_raw_frame(s)
    assert list(df.columns) == ["date", "VIXCLS"]
    assert df["VIXCLS"].tolist() == [12.0, 13.0]


def test_fetch_vix_series_delegates():
    class FakeFred:
        def get_series(self, series_id, observation_start, observation_end):
            return pd.Series([1.0])

    out = fetch_vix_series(FakeFred(), "2015-01-01", "2026-06-10")
    assert out.tolist() == [1.0]


def test_get_fred_client_missing_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _get_fred_client()