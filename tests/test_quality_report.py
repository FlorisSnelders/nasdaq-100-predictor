# tests/test_quality_report.py
import pandas as pd
import pytest
from src import quality_report as QR
from src.quality_report import (
    StageStats, ValidationResult, generate_quality_report,
    generate_report_from_storage,
)


# ---- dataclasses ----
def test_stagestats_normal():
    s = StageStats("clean", rows_in=100, rows_out=80)
    assert s.rows_dropped == 20
    assert abs(s.pct_dropped - 20.0) < 1e-9


def test_stagestats_zero_in():
    s = StageStats("load", rows_in=0, rows_out=0)
    assert s.pct_dropped == 0.0


def test_stagestats_negative_clamped():
    s = StageStats("grow", rows_in=5, rows_out=10)
    assert s.rows_dropped == 0


# ---- _format_stage_table ----
def test_format_stage_table():
    out = QR._format_stage_table([StageStats("load", 0, 10)])
    assert "Stage" in out and "load" in out


# ---- _format_validation_table ----
def test_format_validation_table_empty():
    assert "No validation checks" in QR._format_validation_table([])


def test_format_validation_table_pass_and_fail():
    out = QR._format_validation_table([
        ValidationResult("after_load", "no_nulls", True, ""),
        ValidationResult("after_clean", "ohlc", False, "bad | pipe"),
    ])
    assert "PASS" in out and "FAIL" in out


# ---- _format_problem_tickers ----
def test_problem_tickers_none():
    assert "No cleaning report" in QR._format_problem_tickers(None)


def test_problem_tickers_all_zero():
    df = pd.DataFrame({"ticker": ["AAA", "BBB"]})  # no issue columns
    assert "No tickers had cleaning issues" in QR._format_problem_tickers(df)


def test_problem_tickers_with_issues():
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "rows_dropped_dup": [3, 0],
        "rows_dropped_price": [1, 0],
        "rows_flagged_extreme": [0, 0],
    })
    out = QR._format_problem_tickers(df)
    assert "AAA" in out and "total_issues" in out


# ---- _format_final_stats ----
def test_final_stats_none():
    assert "No final dataframe" in QR._format_final_stats(None)


def test_final_stats_with_nulls():
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "close": [1.0, None],
    })
    out = QR._format_final_stats(df)
    assert "Total rows" in out and "Unique tickers" in out and "`close`" in out


def test_final_stats_no_nulls():
    df = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2024-01-02"])})
    out = QR._format_final_stats(df)
    assert "none" in out


# ---- generate_quality_report ----
def _stages():
    return [StageStats("load", 0, 100), StageStats("clean", 100, 80)]


def _final():
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
    })


def test_generate_report_pass_with_metadata(tmp_path):
    path = generate_quality_report(
        stage_stats=_stages(),
        validation_results=[ValidationResult("after_load", "all", True, "")],
        cleaning_report=pd.DataFrame({"ticker": ["AAA"], "rows_dropped_dup": [2]}),
        final_df=_final(),
        output_dir=str(tmp_path),
        extra_metadata={"git_sha": "abc123"},
    )
    assert path.exists()
    text = path.read_text()
    assert "PASS" in text and "Run metadata" in text and "abc123" in text


def test_generate_report_fail(tmp_path):
    path = generate_quality_report(
        stage_stats=_stages(),
        validation_results=[ValidationResult("after_clean", "ohlc", False, "broke")],
        output_dir=str(tmp_path),
    )
    assert "FAIL" in path.read_text()


def test_generate_report_no_validations(tmp_path):
    path = generate_quality_report(
        stage_stats=_stages(),
        validation_results=[],
        output_dir=str(tmp_path),
    )
    assert "no validation checks recorded" in path.read_text()


def test_generate_report_default_output_dir():
    # output_dir=None branch -> writes under <project>/reports
    path = generate_quality_report(
        stage_stats=_stages(),
        validation_results=[ValidationResult("x", "y", True, "")],
    )
    assert path.exists() and path.name.startswith("quality_")
    path.unlink()  # clean up


# ---- generate_report_from_storage ----
def test_generate_report_from_storage(tmp_path, monkeypatch):
    from src import storage
    monkeypatch.setattr(
        storage, "load_from_db",
        lambda *a, **k: pd.DataFrame({
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2024-01-02"]),
        }),
    )
    path = generate_report_from_storage(output_dir=str(tmp_path))
    assert path.exists()
    assert "loaded_from_db" in path.read_text()