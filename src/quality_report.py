"""
Automated data quality report generator.

After a successful pipeline run, call `generate_quality_report()` to write
a timestamped markdown report to /reports/quality_<timestamp>.md. The
report summarizes:
  - rows in/out per pipeline stage
  - validation check results (which checks passed/failed)
  - tickers with the most data quality issues
  - timestamp and pipeline metadata

Typical use at the end of run_pipeline():

    from src.quality_report import (
        StageStats, generate_quality_report
    )

    stage_stats = []
    df_raw = load_all_tickers("data/")
    stage_stats.append(StageStats("load", rows_in=0, rows_out=len(df_raw)))

    df_std = standardize(df_raw)
    stage_stats.append(StageStats("standardize",
                                   rows_in=len(df_raw),
                                   rows_out=len(df_std)))

    df_clean, cleaning_report = clean_pipeline(df_std)
    stage_stats.append(StageStats("clean",
                                   rows_in=len(df_std),
                                   rows_out=len(df_clean)))

    features = build_features(df_clean)
    stage_stats.append(StageStats("features",
                                   rows_in=len(df_clean),
                                   rows_out=len(features)))

    generate_quality_report(
        stage_stats=stage_stats,
        validation_results=collected_validation_results,
        cleaning_report=cleaning_report,
        final_df=df_clean,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd


# ----------------------------- data classes -------------------------------

@dataclass
class StageStats:
    """How many rows came in and went out of a single pipeline stage."""
    name: str
    rows_in: int
    rows_out: int
    notes: str = ""

    @property
    def rows_dropped(self) -> int:
        return max(0, self.rows_in - self.rows_out)

    @property
    def pct_dropped(self) -> float:
        if self.rows_in == 0:
            return 0.0
        return 100.0 * self.rows_dropped / self.rows_in


@dataclass
class ValidationResult:
    """Outcome of one validation check during the pipeline run."""
    stage: str           # e.g. "after_clean"
    check: str           # e.g. "no_nulls", "no_duplicates"
    passed: bool
    message: str = ""    # error message if failed; brief detail if passed


# --------------------------- report generation ----------------------------

def _format_stage_table(stage_stats: List[StageStats]) -> str:
    """Markdown table of per-stage row counts."""
    lines = [
        "| Stage | Rows in | Rows out | Dropped | % dropped | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in stage_stats:
        lines.append(
            f"| {s.name} | {s.rows_in:,} | {s.rows_out:,} | "
            f"{s.rows_dropped:,} | {s.pct_dropped:.2f}% | {s.notes} |"
        )
    return "\n".join(lines)


def _format_validation_table(results: List[ValidationResult]) -> str:
    """Markdown table of validation outcomes."""
    if not results:
        return "_No validation checks were recorded._"

    lines = [
        "| Stage | Check | Result | Detail |",
        "| --- | --- | :---: | --- |",
    ]
    for r in results:
        mark = "PASS" if r.passed else "**FAIL**"
        detail = r.message.replace("|", "\\|") if r.message else ""
        lines.append(f"| {r.stage} | {r.check} | {mark} | {detail} |")
    return "\n".join(lines)


def _format_problem_tickers(
    cleaning_report: Optional[pd.DataFrame],
    top_n: int = 10,
) -> str:
    """Markdown table of the tickers that lost the most rows during cleaning."""
    if cleaning_report is None or len(cleaning_report) == 0:
        return "_No cleaning report available._"

    df = cleaning_report.copy()

    # Compute a "total issues" score that combines all problem columns we
    # know about. Missing columns are treated as 0 so this works on any
    # cleaning_report shape.
    issue_cols = [
        "rows_dropped_dup",
        "rows_dropped_price",
        "rows_flagged_extreme",
    ]
    for c in issue_cols:
        if c not in df.columns:
            df[c] = 0
    df["total_issues"] = df[issue_cols].sum(axis=1)

    worst = df.sort_values("total_issues", ascending=False).head(top_n)
    if worst["total_issues"].sum() == 0:
        return "_No tickers had cleaning issues. All raw rows passed through._"

    cols = ["ticker"] + issue_cols + ["total_issues"]
    cols = [c for c in cols if c in worst.columns]

    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in worst.iterrows():
        cells = [str(row[c]) if c == "ticker" else f"{int(row[c]):,}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_final_stats(final_df: Optional[pd.DataFrame]) -> str:
    """Quick summary stats on the final dataframe (date range, ticker count)."""
    if final_df is None or len(final_df) == 0:
        return "_No final dataframe provided._"

    lines = []
    lines.append(f"- **Total rows:** {len(final_df):,}")
    if "ticker" in final_df.columns:
        lines.append(f"- **Unique tickers:** {final_df['ticker'].nunique()}")
    if "date" in final_df.columns:
        dates = pd.to_datetime(final_df["date"], utc=True, errors="coerce")
        lines.append(f"- **Date range:** {dates.min()} to {dates.max()}")

    # Per-column null counts (only columns with at least one null)
    nulls = final_df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        lines.append("- **Columns with nulls (final df):**")
        for col, n in nulls.items():
            lines.append(f"  - `{col}`: {int(n):,}")
    else:
        lines.append("- **Columns with nulls (final df):** none")
    return "\n".join(lines)


def generate_quality_report(
    *,
    stage_stats: List[StageStats],
    validation_results: List[ValidationResult],
    cleaning_report: Optional[pd.DataFrame] = None,
    final_df: Optional[pd.DataFrame] = None,
    output_dir: str = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a markdown quality report and return its file path.

    Parameters
    ----------
    stage_stats
        Ordered list of StageStats, one per pipeline stage.
    validation_results
        List of ValidationResult objects collected during the run.
    cleaning_report
        Optional `cleaning_report` dataframe (per-ticker drop counts).
    final_df
        Optional final dataframe (used for date range, null counts, etc.).
    output_dir
        Where to write the report. Default `reports/`.
    extra_metadata
        Any extra fields to put in the header (git sha, environment, etc.).

    Returns
    -------
    pathlib.Path
        Full path to the report that was written.
    """
    timestamp = datetime.now()
    if output_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "reports"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"quality_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    out_path = out_dir / fname

    # Build report body
    parts: List[str] = []
    parts.append(f"# Data Quality Report")
    parts.append("")
    parts.append(f"_Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}_")
    parts.append("")

    # Overall pass/fail at the top so a human can see status in 2 seconds
    n_total = len(validation_results)
    n_pass = sum(1 for r in validation_results if r.passed)
    if n_total == 0:
        overall = "_no validation checks recorded_"
    elif n_pass == n_total:
        overall = f"**PASS** — all {n_total} validation checks passed"
    else:
        overall = f"**FAIL** — {n_total - n_pass}/{n_total} checks failed"
    parts.append(f"**Status:** {overall}")
    parts.append("")

    if extra_metadata:
        parts.append("## Run metadata")
        for k, v in extra_metadata.items():
            parts.append(f"- **{k}:** {v}")
        parts.append("")

    parts.append("## Stage-by-stage row counts")
    parts.append("")
    parts.append(_format_stage_table(stage_stats))
    parts.append("")

    parts.append("## Validation check results")
    parts.append("")
    parts.append(_format_validation_table(validation_results))
    parts.append("")

    parts.append("## Final dataset summary")
    parts.append("")
    parts.append(_format_final_stats(final_df))
    parts.append("")

    parts.append("## Tickers with the most cleaning issues (top 10)")
    parts.append("")
    parts.append(_format_problem_tickers(cleaning_report))
    parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(f"_Report path: `{out_path}`_")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


# ---------------------- convenience: run from storage ---------------------

def generate_report_from_storage(
    output_dir: str = "reports",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Convenience wrapper: build the report directly from the SQLite store
    set up in src/storage.py. Useful when you want a report after-the-fact
    (e.g. running this notebook against a DB that already exists) instead
    of wiring it into run_pipeline.

    The SQLite layer only persists the cleaned dataframe, so cleaning_report
    and the features table aren't available here. We construct minimal
    stage stats from the cleaned dataframe and a synthetic post-hoc
    validation record.
    """
    from src.storage import load_from_db

    final = load_from_db()

    stages = [
        StageStats(
            "loaded_from_db",
            rows_in=0,
            rows_out=len(final),
            notes="loaded via src.storage.load_from_db() — upstream counts not persisted",
        ),
    ]

    # No live validation results when reading post-hoc, but record a single
    # synthetic 'reconstructed' line so the section isn't empty.
    validations = [
        ValidationResult(
            stage="post-hoc",
            check="report-from-storage",
            passed=True,
            message="reconstructed from the SQLite store, no live checks ran",
        ),
    ]

    return generate_quality_report(
        stage_stats=stages,
        validation_results=validations,
        cleaning_report=None,  # not persisted in the SQLite store
        final_df=final,
        output_dir=output_dir,
        extra_metadata=extra_metadata,
    )
