# tests/test_pipeline.py
import pandas as pd
import pytest
from src import pipeline
from src.pipeline import run_pipeline

DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _raw():
    rows = []
    for t in ["AAA", "BBB"]:
        for d in DATES:
            rows.append({
                # noon UTC stays on the same NY calendar day after tz_convert
                "date": f"{d} 12:00:00",
                "open": 10, "high": 12, "low": 9, "close": 11,
                "volume": 1000, "transactions": 5, "ticker": t,
            })
    return pd.DataFrame(rows)


def _macro_df():
    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d in DATES],
        "vix": [15.0, 16.0, 17.0, 18.0],
        "treasury_yield_10y": [4.0, 4.1, 4.2, 4.3],
    })


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(pipeline, "load_and_clean_macro", lambda **k: _macro_df())
    monkeypatch.setattr(pipeline, "validate_stage", lambda *a, **k: None)
    return monkeypatch


def test_run_pipeline_snapshot(patched):
    patched.setattr(pipeline, "load_snapshot", lambda source: _raw())
    df = run_pipeline(source="dummy/path")
    assert len(df) == 8
    assert "vix" in df.columns and "treasury_yield_10y" in df.columns
    assert "daily_return" in df.columns and "label_valid" in df.columns
    # merge helper columns are cleaned up
    assert "date_key" not in df.columns and "date_macro" not in df.columns


def test_run_pipeline_yfinance_branch(patched):
    called = {}
    def fake_yf(tickers, start, end):
        called["yes"] = (tuple(tickers), start, end)
        return _raw()
    patched.setattr(pipeline, "load_yfinance", fake_yf)
    patched.setattr(pipeline, "load_snapshot",
                    lambda *a, **k: pytest.fail("snapshot should not be called"))
    df = run_pipeline(tickers=["AAA", "BBB"], start="2024-01-01", end="2024-01-06")
    assert called["yes"][0] == ("AAA", "BBB")
    assert len(df) == 8


def test_run_pipeline_save_and_report(patched):
    calls = {"save": 0, "report": 0}
    patched.setattr(pipeline, "load_snapshot", lambda source: _raw())
    patched.setattr(pipeline, "save_to_db",
                    lambda df: calls.__setitem__("save", len(df)))
    patched.setattr(pipeline, "generate_quality_report",
                    lambda **k: calls.__setitem__("report", 1))
    run_pipeline(source="x", save_db=True, write_report=True)
    assert calls["save"] == 8
    assert calls["report"] == 1