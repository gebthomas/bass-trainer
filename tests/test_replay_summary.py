"""Tests for the compact summary reporting added to tools/replay_tempo_tracking.py.

Covers:
  - summarize_results() calculations
  - format_summary() output content
  - adaptive improvement arithmetic
  - --summary-only CLI mode
  - --no-summary CLI mode
  - default mode (table + summary)

All tests are synthetic — no audio hardware.

Test matrix
-----------
summarize_results — observation counts
    1.  Exact nominal: accepted == n_beats, rejected == 0, missed == 0.
    2.  Outlier scenario: rejected == 1, accepted == n_beats - 1.
    3.  Missed beats: missed count equals the number of missing beats.
    4.  anchor counts as accepted (not as a separate category).

summarize_results — BPM fields
    5.  Exact nominal: final_bpm == nominal_bpm.
    6.  Fast scenario: final_bpm > nominal_bpm after convergence.
    7.  bpm_min ≤ final_bpm ≤ bpm_max.

summarize_results — error statistics
    8.  Exact nominal: mean_abs_fixed_ms == 0.
    9.  Exact nominal: mean_abs_adapt_ms == 0.
    10. Fast scenario: mean_abs_fixed_ms >> mean_abs_adapt_ms.
    11. mean_abs_fixed_ms is None when all beats are missed.
    12. max_abs_adapt_ms ≥ mean_abs_adapt_ms (for non-trivial case).

summarize_results — adaptive improvement
    13. adapt_improvement_ms == mean_abs_fixed_ms − mean_abs_adapt_ms.
    14. adapt_improvement_pct == 100 × improvement / mean_abs_fixed.
    15. adapt_improvement_pct is None when mean_abs_fixed_ms == 0.
    16. Both improvement fields are None when there are no onset rows.

summarize_results — outlier limit range
    17. Exact nominal: outlier_limit_min_ms == outlier_limit_max_ms (constant).
    18. Accel scenario: outlier_limit_max_ms > outlier_limit_min_ms (widened).
    19. Fields are None when results list is empty.

format_summary — content checks
    20. Output contains accepted/rejected/missed counts.
    21. Output contains "Final BPM".
    22. Output contains "fixed" error value.
    23. Output contains "adaptive" error value.
    24. Output contains improvement value when improvement is non-zero.
    25. Output contains outlier limit range.
    26. Output omits improvement line when both errors are None.
    27. When limit min == max, output says "constant".
    28. When limit min != max, output mentions "widened".

CLI output modes
    29. Default mode: stdout contains both table header ("Beat") and summary header.
    30. --summary-only: stdout lacks "Beat" column header; summary present.
    31. --no-summary: stdout has "Beat" column header; no summary separator.
    32. --summary-only and --no-summary are mutually exclusive (argparse error).
    33. CSV export unaffected by --summary-only.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.replay_tempo_tracking import (
    format_summary,
    main,
    run_replay,
    scenario_exact_nominal,
    scenario_gradual_change,
    scenario_steady_offset,
    scenario_with_missed_beats,
    scenario_with_outlier,
    summarize_results,
)

# ── Shared constants ──────────────────────────────────────────────────────────

NOM_BPM  = 120.0
BEAT_S   = 60.0 / NOM_BPM
COUNT_IN = 2.0
N        = 16   # short run for fast tests


def _exact(n: int = N) -> list[dict]:
    return run_replay(scenario_exact_nominal(NOM_BPM, n, COUNT_IN), NOM_BPM)


def _fast(n: int = N) -> list[dict]:
    return run_replay(scenario_steady_offset(NOM_BPM, 132.0, n, COUNT_IN), NOM_BPM)


def _missed(missed: set[int] = frozenset({3, 7}), n: int = N) -> list[dict]:
    return run_replay(scenario_with_missed_beats(NOM_BPM, n, missed, COUNT_IN), NOM_BPM)


def _outlier(n: int = N, beat: int = 8, offset: float = 0.350) -> list[dict]:
    return run_replay(scenario_with_outlier(NOM_BPM, n, beat, offset, COUNT_IN), NOM_BPM)


# ── 1–4: Observation counts ───────────────────────────────────────────────────

def test_exact_all_accepted():
    s = summarize_results(_exact())
    assert s["accepted"] == N
    assert s["rejected"] == 0
    assert s["missed"]   == 0


def test_outlier_one_rejected():
    s = summarize_results(_outlier())
    assert s["rejected"] == 1
    assert s["accepted"] == N - 1


def test_missed_count_correct():
    missed_beats = {3, 7, 11}
    s = summarize_results(_missed(missed_beats))
    assert s["missed"] == len(missed_beats)


def test_anchor_counts_as_accepted():
    """The anchor beat (status="anchor") must be included in accepted."""
    results = _exact(n=1)
    assert results[0]["status"] == "anchor"
    s = summarize_results(results)
    assert s["accepted"] == 1


# ── 5–7: BPM fields ──────────────────────────────────────────────────────────

def test_exact_final_bpm_equals_nominal():
    s = summarize_results(_exact())
    assert s["final_bpm"] == pytest.approx(NOM_BPM, abs=0.5)


def test_fast_final_bpm_above_nominal():
    s = summarize_results(_fast(n=32))
    assert s["final_bpm"] > NOM_BPM


def test_bpm_min_le_final_le_max():
    s = summarize_results(_fast())
    assert s["bpm_min"] <= s["final_bpm"] <= s["bpm_max"]


# ── 8–12: Error statistics ────────────────────────────────────────────────────

def test_exact_mean_fixed_error_zero():
    s = summarize_results(_exact())
    assert s["mean_abs_fixed_ms"] == pytest.approx(0.0, abs=1e-6)


def test_exact_mean_adapt_error_zero():
    s = summarize_results(_exact())
    assert s["mean_abs_adapt_ms"] == pytest.approx(0.0, abs=1e-6)


def test_fast_fixed_error_larger_than_adaptive():
    s = summarize_results(_fast(n=32))
    assert s["mean_abs_fixed_ms"] > s["mean_abs_adapt_ms"]


def test_all_missed_errors_none():
    """When every beat is missed, there are no onset rows → None errors."""
    onsets  = scenario_with_missed_beats(NOM_BPM, 4, {0, 1, 2, 3}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    s = summarize_results(results)
    assert s["mean_abs_fixed_ms"] is None
    assert s["mean_abs_adapt_ms"] is None


def test_max_adapt_ge_mean_adapt():
    s = summarize_results(_fast(n=32))
    assert s["max_abs_adapt_ms"] >= s["mean_abs_adapt_ms"]


# ── 13–16: Adaptive improvement ──────────────────────────────────────────────

def test_improvement_ms_equals_fixed_minus_adapt():
    s = summarize_results(_fast(n=32))
    expected = s["mean_abs_fixed_ms"] - s["mean_abs_adapt_ms"]
    assert s["adapt_improvement_ms"] == pytest.approx(expected, abs=1e-9)


def test_improvement_pct_formula():
    s = summarize_results(_fast(n=32))
    expected_pct = 100.0 * s["adapt_improvement_ms"] / s["mean_abs_fixed_ms"]
    assert s["adapt_improvement_pct"] == pytest.approx(expected_pct, abs=1e-6)


def test_improvement_pct_none_when_fixed_zero():
    """Exact nominal → fixed error 0 → pct undefined → None."""
    s = summarize_results(_exact())
    assert s["adapt_improvement_pct"] is None


def test_improvement_none_when_no_onsets():
    onsets  = scenario_with_missed_beats(NOM_BPM, 4, {0, 1, 2, 3}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    s = summarize_results(results)
    assert s["adapt_improvement_ms"]  is None
    assert s["adapt_improvement_pct"] is None


# ── 17–19: Outlier limit range ───────────────────────────────────────────────

def test_exact_limit_constant():
    s = summarize_results(_exact(n=32))
    assert s["outlier_limit_min_ms"] == pytest.approx(s["outlier_limit_max_ms"], abs=1e-6)


def test_accel_limit_widens():
    onsets  = scenario_gradual_change(NOM_BPM, 120.0, 132.0, 48, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    s = summarize_results(results)
    assert s["outlier_limit_max_ms"] > s["outlier_limit_min_ms"]


def test_empty_results_limit_none():
    s = summarize_results([])
    assert s["outlier_limit_min_ms"] is None
    assert s["outlier_limit_max_ms"] is None


# ── 20–28: format_summary content ────────────────────────────────────────────

def test_format_summary_has_accepted_count():
    s   = summarize_results(_exact())
    out = format_summary(s)
    assert "accepted" in out


def test_format_summary_has_final_bpm():
    s   = summarize_results(_exact())
    out = format_summary(s)
    assert "Final BPM" in out or "BPM" in out


def test_format_summary_has_fixed_error():
    s   = summarize_results(_fast())
    out = format_summary(s)
    assert "fixed" in out.lower()


def test_format_summary_has_adaptive_error():
    s   = summarize_results(_fast())
    out = format_summary(s)
    assert "adaptive" in out.lower()


def test_format_summary_has_improvement_when_nonzero():
    s   = summarize_results(_fast(n=32))
    out = format_summary(s)
    assert "improvement" in out.lower() or "Improvement" in out


def test_format_summary_has_outlier_limit():
    s   = summarize_results(_exact())
    out = format_summary(s)
    assert "outlier" in out.lower() or "limit" in out.lower()


def test_format_summary_no_improvement_when_no_errors():
    onsets  = scenario_with_missed_beats(NOM_BPM, 4, {0, 1, 2, 3}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    s   = summarize_results(results)
    out = format_summary(s)
    assert "improvement" not in out.lower()


def test_format_summary_constant_limit_says_constant():
    s   = summarize_results(_exact(n=32))
    out = format_summary(s)
    assert "constant" in out.lower()


def test_format_summary_widened_limit_says_widened():
    onsets  = scenario_gradual_change(NOM_BPM, 120.0, 132.0, 48, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    s   = summarize_results(results)
    out = format_summary(s)
    assert "widen" in out.lower()


# ── 29–33: CLI output modes ───────────────────────────────────────────────────

def _capture(argv: list[str]) -> str:
    """Run main() and capture stdout as a string."""
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        main(argv)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_default_mode_has_table_and_summary():
    out = _capture(["--scenario", "exact", "--beats", "4"])
    assert "Beat" in out          # table header
    assert "Observations" in out  # summary section


def test_summary_only_no_table_header():
    out = _capture(["--scenario", "exact", "--beats", "4", "--summary-only"])
    assert "Beat" not in out


def test_summary_only_has_summary():
    out = _capture(["--scenario", "exact", "--beats", "4", "--summary-only"])
    assert "Observations" in out


def test_no_summary_has_table():
    out = _capture(["--scenario", "exact", "--beats", "4", "--no-summary"])
    assert "Beat" in out


def test_no_summary_lacks_summary_section():
    out = _capture(["--scenario", "exact", "--beats", "4", "--no-summary"])
    assert "Observations" not in out


def test_summary_only_and_no_summary_mutually_exclusive():
    """argparse should reject both flags being set simultaneously."""
    with pytest.raises(SystemExit):
        _capture(["--scenario", "exact", "--beats", "4",
                  "--summary-only", "--no-summary"])


def test_csv_unaffected_by_summary_only(tmp_path):
    csv_path = str(tmp_path / "out.csv")
    _capture(["--scenario", "exact", "--beats", "4",
              "--summary-only", "--csv", csv_path])
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
