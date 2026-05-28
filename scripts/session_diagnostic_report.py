#!/usr/bin/env python3
"""Session diagnostic report generator for bass-trainer.

Reads a .session.json file and writes two output files side-by-side:

  <stem>.timeline.png  — two-lane timing visualization with error bars
  <stem>.summary.txt   — plain-text numeric evaluation summary

All evaluation, matching, and plotting logic is delegated to existing
helpers in plot_session_timeline and core.realtime_evaluator.  Nothing
is duplicated here.

Usage
-----
Default output directory (same folder as the session file):

    python scripts/session_diagnostic_report.py diagnostics/example.session.json

Custom output directory:

    python scripts/session_diagnostic_report.py diagnostics/example.session.json \\
        --out-dir reports/

Adjust evaluation window:

    python scripts/session_diagnostic_report.py session.json \\
        --tolerance-ms 100 --on-time-ms 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root and scripts/ are importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# plot_session_timeline sets matplotlib.use("Agg") at import time.
from plot_session_timeline import (
    EvaluationSummary,
    TimingScenario,
    _DEFAULT_ON_TIME_S,
    _DEFAULT_TOLERANCE_S,
    build_figure,
    save_figure,
    scenario_from_session_file,
    summarize_evaluations,
)

import matplotlib.pyplot as plt  # backend already configured by the import above

from core.realtime_evaluator import evaluate_targets

_DEFAULT_TOLERANCE_MS = _DEFAULT_TOLERANCE_S * 1000   # 80 ms
_DEFAULT_ON_TIME_MS   = _DEFAULT_ON_TIME_S   * 1000   # 30 ms


# ── Output path derivation ────────────────────────────────────────────────────

def derive_output_paths(
    session_path: str | Path,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return (png_path, txt_path) derived from the session file path.

    The stem is the filename with ``.session.json`` (or ``.json``) stripped.
    When *out_dir* is None the output files land in the session file's own
    parent directory; otherwise they go into *out_dir*.

    >>> derive_output_paths("diagnostics/example.session.json")
    (PosixPath('diagnostics/example.timeline.png'),
     PosixPath('diagnostics/example.summary.txt'))

    >>> derive_output_paths("example.session.json", out_dir="reports/")
    (PosixPath('reports/example.timeline.png'),
     PosixPath('reports/example.summary.txt'))
    """
    p    = Path(session_path)
    name = p.name
    if name.endswith(".session.json"):
        stem = name[: -len(".session.json")]
    else:
        stem = p.stem
    base = Path(out_dir) if out_dir is not None else p.parent
    return base / f"{stem}.timeline.png", base / f"{stem}.summary.txt"


# ── Text report formatter ─────────────────────────────────────────────────────

def format_summary_text(
    session_path: str | Path,
    scenario: TimingScenario,
    summary: EvaluationSummary,
    tolerance_ms: float,
    on_time_ms: float,
) -> str:
    """Return a deterministic plain-text report as a single string.

    Error stats are shown as ``—`` when all targets are misses (no matched
    onsets to compute from).
    """
    def _ms_signed(v: float | None) -> str:
        return f"{v:+.1f} ms" if v is not None else "—"

    def _ms_abs(v: float | None) -> str:
        return f"{v:.1f} ms" if v is not None else "—"

    def _pct(n: int) -> str:
        return f"({100 * n / summary.n_targets:.1f}%)" if summary.n_targets else ""

    n            = summary.n_targets
    hit_rate_pct = 100 * (n - summary.n_miss) / n if n else 0.0

    lines = [
        "Bass Trainer — Session Diagnostic Report",
        "=" * 41,
        f"Session:            {Path(session_path).resolve()}",
        f"Title:              {scenario.title}",
        f"Description:        {scenario.description}",
        "",
        "Tolerance settings",
        "-" * 18,
        f"Match tolerance:    {tolerance_ms:.1f} ms",
        f"On-time window:     {on_time_ms:.1f} ms",
        "",
        "Evaluation summary",
        "-" * 18,
        f"Total targets:      {n}",
        f"  On time:          {summary.n_on_time:<4}  {_pct(summary.n_on_time)}",
        f"  Early:            {summary.n_early:<4}  {_pct(summary.n_early)}",
        f"  Late:             {summary.n_late:<4}  {_pct(summary.n_late)}",
        f"  Miss:             {summary.n_miss:<4}  {_pct(summary.n_miss)}",
        f"Extra onsets:       {summary.n_unmatched_onsets}",
        f"Hit rate:           {hit_rate_pct:.1f}%",
        "",
        "Timing errors (matched targets only)",
        "-" * 36,
        f"Mean signed error:  {_ms_signed(summary.mean_signed_error_ms)}",
        f"Mean abs error:     {_ms_abs(summary.mean_abs_error_ms)}",
        f"Max abs error:      {_ms_abs(summary.max_abs_error_ms)}",
    ]
    return "\n".join(lines) + "\n"


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    session_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    tolerance_ms: float = _DEFAULT_TOLERANCE_MS,
    on_time_ms: float = _DEFAULT_ON_TIME_MS,
) -> tuple[Path, Path]:
    """Generate the timeline PNG and text summary for one session log.

    Parameters
    ----------
    session_path
        Input ``.session.json`` file.
    out_dir
        Output directory.  Defaults to the session file's parent.
    tolerance_ms, on_time_ms
        Evaluation window parameters in milliseconds.

    Returns
    -------
    (png_path, txt_path)
        Resolved absolute paths of the two written files.
    """
    tolerance_s = tolerance_ms / 1000.0
    on_time_s   = on_time_ms   / 1000.0

    scenario = scenario_from_session_file(session_path)
    evals    = evaluate_targets(
        scenario.target_times_s,
        scenario.onset_times_s,
        tolerance_s         = tolerance_s,
        on_time_threshold_s = on_time_s,
    )
    summary = summarize_evaluations(evals, scenario.onset_times_s)

    png_path, txt_path = derive_output_paths(session_path, out_dir)

    # Timeline PNG — delegates all plotting to build_figure / save_figure
    fig     = build_figure([scenario], tolerance_s=tolerance_s, on_time_s=on_time_s)
    png_out = save_figure(fig, png_path)
    plt.close(fig)

    # Plain-text summary
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    text = format_summary_text(
        session_path, scenario, summary, tolerance_ms, on_time_ms
    )
    txt_path.write_text(text, encoding="utf-8")
    txt_out = txt_path.resolve()

    return png_out, txt_out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a timeline PNG and plain-text summary for a session log."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "session",
        metavar="FILE.session.json",
        help="Input session log file.",
    )
    p.add_argument(
        "--out-dir", default=None, dest="out_dir", metavar="DIR",
        help=(
            "Output directory.  Defaults to the session file's own directory."
        ),
    )
    p.add_argument(
        "--tolerance-ms", type=float, default=_DEFAULT_TOLERANCE_MS,
        dest="tolerance_ms", help="Match tolerance in ms.",
    )
    p.add_argument(
        "--on-time-ms", type=float, default=_DEFAULT_ON_TIME_MS,
        dest="on_time_ms", help="On-time classification threshold in ms.",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    path = Path(args.session)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    png_out, txt_out = generate_report(
        path,
        out_dir      = args.out_dir,
        tolerance_ms = args.tolerance_ms,
        on_time_ms   = args.on_time_ms,
    )
    print(f"Timeline: {png_out}")
    print(f"Summary:  {txt_out}")


if __name__ == "__main__":
    main()
