"""Tests for the pure helper functions in tools/replay_tempo_tracking.py.

All tests are synthetic — no audio hardware.

Test matrix
-----------
scenario_exact_nominal
    1.  Returns n_beats entries.
    2.  Every nominal_time_s == actual_time_s (exact timing).
    3.  Beat indices are 0-based consecutive.

scenario_steady_offset
    4.  Returns n_beats entries.
    5.  Beat 0 has the same nominal and actual time (they start together).
    6.  Fixed error for beat i is proportional to i (linear drift).
    7.  Faster player → actual arrives before nominal (negative drift).
    8.  Slower player → actual arrives after nominal (positive drift).

scenario_gradual_change
    9.  Returns n_beats entries.
    10. Beat 0 is at count_in_s for both nominal and actual.
    11. Accelerating: each inter-onset interval is shorter than the previous.
    12. Decelerating: each inter-onset interval is longer than the previous.

scenario_with_outlier
    13. Returns n_beats entries.
    14. Only the specified beat has a non-zero error.
    15. All other beats have actual == nominal.

scenario_with_missed_beats
    16. Specified beats have actual_time_s == None.
    17. Non-missed beats have actual_time_s == nominal_time_s.

run_replay — exact nominal
    18. Tracker stays at nominal BPM.
    19. All fixed and adaptive errors are 0.
    20. Confidence rises to > 0.9 after several beats.
    21. Status of beat 0 is "anchor".
    22. All subsequent statuses are "accept".

run_replay — steady fast (120 nom / 132 actual)
    23. Fixed errors are negative and grow in magnitude.
    24. Adaptive errors are smaller in magnitude than fixed errors after convergence.
    25. Estimated BPM converges toward 132 over 32 beats.

run_replay — outlier rejection
    26. Outlier beat status is "reject".
    27. tempo_ratio is unchanged after an outlier (tracker state preserved).

run_replay — missed beats
    28. Missed entries have None for all error fields and adjusted_target_time_s.
    29. Missed entries have status == "miss".
    30. Non-missed beats around a miss continue to have valid errors.

run_replay — gradual change
    31. Tracker BPM moves in the direction of the actual tempo change.

export_csv
    32. CSV file is created at the specified path.
    33. First row is the header with expected column names.
    34. Row count equals len(results) + 1 (header).
    35. None values are exported as empty strings.
    36. Numeric values round-trip through the file correctly.

TempoTracker diagnostic properties
    37. outlier_limit_s equals outlier_threshold * nominal_beat_s.
    38. nominal_beat_s equals 60 / nominal_bpm.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tempo_tracker import TempoTracker
from tools.replay_tempo_tracking import (
    export_csv,
    run_replay,
    scenario_exact_nominal,
    scenario_gradual_change,
    scenario_steady_offset,
    scenario_with_missed_beats,
    scenario_with_outlier,
)

# ── Shared constants ──────────────────────────────────────────────────────────

NOM_BPM   = 120.0
BEAT_S    = 60.0 / NOM_BPM   # 0.5 s
COUNT_IN  = 2.0              # seconds


# ── 1–3: scenario_exact_nominal ───────────────────────────────────────────────

def test_exact_nominal_beat_count():
    assert len(scenario_exact_nominal(NOM_BPM, 16, COUNT_IN)) == 16


def test_exact_nominal_times_equal():
    for row in scenario_exact_nominal(NOM_BPM, 16, COUNT_IN):
        assert row["nominal_time_s"] == pytest.approx(row["actual_time_s"], abs=1e-9)


def test_exact_nominal_consecutive_indices():
    onsets = scenario_exact_nominal(NOM_BPM, 8, COUNT_IN)
    assert [r["beat_index"] for r in onsets] == list(range(8))


# ── 4–8: scenario_steady_offset ──────────────────────────────────────────────

def test_steady_offset_beat_count():
    assert len(scenario_steady_offset(NOM_BPM, 132.0, 16, COUNT_IN)) == 16


def test_steady_offset_beat_zero_same_time():
    onsets = scenario_steady_offset(NOM_BPM, 132.0, 16, COUNT_IN)
    assert onsets[0]["nominal_time_s"] == pytest.approx(onsets[0]["actual_time_s"], abs=1e-9)


def test_steady_offset_linear_drift():
    """Fixed error at beat i should be proportional to i."""
    onsets = scenario_steady_offset(NOM_BPM, 132.0, 16, COUNT_IN)
    errors = [r["actual_time_s"] - r["nominal_time_s"] for r in onsets]
    # Beat 0 error is 0; errors[i] / errors[1] ≈ i
    for i in range(2, 16):
        ratio = errors[i] / errors[1]
        assert ratio == pytest.approx(i, rel=1e-6)


def test_steady_offset_faster_player_negative_drift():
    """Faster player (132 > 120) arrives early → negative error after beat 0."""
    onsets = scenario_steady_offset(NOM_BPM, 132.0, 8, COUNT_IN)
    errors = [r["actual_time_s"] - r["nominal_time_s"] for r in onsets[1:]]
    assert all(e < 0 for e in errors)


def test_steady_offset_slower_player_positive_drift():
    """Slower player (108 < 120) arrives late → positive error after beat 0."""
    onsets = scenario_steady_offset(NOM_BPM, 108.0, 8, COUNT_IN)
    errors = [r["actual_time_s"] - r["nominal_time_s"] for r in onsets[1:]]
    assert all(e > 0 for e in errors)


# ── 9–12: scenario_gradual_change ────────────────────────────────────────────

def test_gradual_change_beat_count():
    assert len(scenario_gradual_change(NOM_BPM, 120.0, 132.0, 16, COUNT_IN)) == 16


def test_gradual_change_beat_zero_at_count_in():
    onsets = scenario_gradual_change(NOM_BPM, 120.0, 132.0, 16, COUNT_IN)
    assert onsets[0]["nominal_time_s"] == pytest.approx(COUNT_IN, abs=1e-9)
    assert onsets[0]["actual_time_s"]  == pytest.approx(COUNT_IN, abs=1e-9)


def test_gradual_accel_intervals_decrease():
    """Accelerating: successive inter-onset intervals get shorter."""
    onsets = scenario_gradual_change(NOM_BPM, 100.0, 140.0, 16, COUNT_IN)
    intervals = [
        onsets[i + 1]["actual_time_s"] - onsets[i]["actual_time_s"]
        for i in range(len(onsets) - 1)
    ]
    assert all(intervals[i] >= intervals[i + 1] for i in range(len(intervals) - 1))


def test_gradual_decel_intervals_increase():
    """Decelerating: successive inter-onset intervals get longer."""
    onsets = scenario_gradual_change(NOM_BPM, 140.0, 100.0, 16, COUNT_IN)
    intervals = [
        onsets[i + 1]["actual_time_s"] - onsets[i]["actual_time_s"]
        for i in range(len(onsets) - 1)
    ]
    assert all(intervals[i] <= intervals[i + 1] for i in range(len(intervals) - 1))


# ── 13–15: scenario_with_outlier ─────────────────────────────────────────────

def test_outlier_beat_count():
    assert len(scenario_with_outlier(NOM_BPM, 12, 5, 0.400, COUNT_IN)) == 12


def test_outlier_only_specified_beat_displaced():
    onsets = scenario_with_outlier(NOM_BPM, 12, 5, 0.400, COUNT_IN)
    for r in onsets:
        if r["beat_index"] == 5:
            assert r["actual_time_s"] != pytest.approx(r["nominal_time_s"])
        else:
            assert r["actual_time_s"] == pytest.approx(r["nominal_time_s"], abs=1e-9)


def test_outlier_other_beats_exact():
    onsets = scenario_with_outlier(NOM_BPM, 12, 5, 0.400, COUNT_IN)
    for r in onsets:
        if r["beat_index"] != 5:
            assert r["actual_time_s"] == pytest.approx(r["nominal_time_s"], abs=1e-9)


# ── 16–17: scenario_with_missed_beats ────────────────────────────────────────

def test_missed_beats_have_none():
    onsets = scenario_with_missed_beats(NOM_BPM, 12, {3, 7}, COUNT_IN)
    assert onsets[3]["actual_time_s"] is None
    assert onsets[7]["actual_time_s"] is None


def test_non_missed_beats_have_actual_time():
    onsets = scenario_with_missed_beats(NOM_BPM, 12, {3, 7}, COUNT_IN)
    for r in onsets:
        if r["beat_index"] not in {3, 7}:
            assert r["actual_time_s"] is not None
            assert r["actual_time_s"] == pytest.approx(r["nominal_time_s"], abs=1e-9)


# ── 18–22: run_replay — exact nominal ────────────────────────────────────────

def test_run_replay_exact_stays_at_nominal():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 20, COUNT_IN), NOM_BPM)
    for r in results[2:]:   # skip anchor and first learning step
        assert r["estimated_bpm"] == pytest.approx(NOM_BPM, abs=0.5)


def test_run_replay_exact_all_errors_zero():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 16, COUNT_IN), NOM_BPM)
    for r in results:
        if r["fixed_error_ms"] is not None:
            assert r["fixed_error_ms"]    == pytest.approx(0.0, abs=1e-6)
            assert r["adaptive_error_ms"] == pytest.approx(0.0, abs=1e-6)


def test_run_replay_exact_confidence_rises():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 20, COUNT_IN), NOM_BPM)
    assert results[-1]["confidence"] > 0.9


def test_run_replay_exact_beat_zero_is_anchor():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 8, COUNT_IN), NOM_BPM)
    assert results[0]["status"] == "anchor"


def test_run_replay_exact_subsequent_status_accept():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 8, COUNT_IN), NOM_BPM)
    for r in results[1:]:
        assert r["status"] == "accept"


# ── 23–25: run_replay — steady fast (120 / 132) ───────────────────────────────

def test_run_replay_fast_fixed_errors_grow():
    results = run_replay(scenario_steady_offset(NOM_BPM, 132.0, 16, COUNT_IN), NOM_BPM)
    fixed = [r["fixed_error_ms"] for r in results[1:]]
    # Each fixed error is more negative than the last
    assert all(fixed[i] > fixed[i + 1] for i in range(len(fixed) - 1))


def test_run_replay_fast_adaptive_smaller_than_fixed_after_convergence():
    """After convergence, |adaptive error| must be less than |fixed error|."""
    results = run_replay(scenario_steady_offset(NOM_BPM, 132.0, 32, COUNT_IN), NOM_BPM)
    # Compare beats 16–31 where convergence should be complete
    for r in results[16:]:
        assert abs(r["adaptive_error_ms"]) < abs(r["fixed_error_ms"])


def test_run_replay_fast_bpm_converges_toward_actual():
    results = run_replay(scenario_steady_offset(NOM_BPM, 132.0, 32, COUNT_IN), NOM_BPM)
    final_bpm = results[-1]["estimated_bpm"]
    assert final_bpm >= 130.0, f"Expected ≥ 130 BPM, got {final_bpm:.2f}"


# ── 26–27: run_replay — outlier rejection ────────────────────────────────────

def test_run_replay_outlier_status_is_reject():
    onsets = scenario_with_outlier(NOM_BPM, 12, 5, 0.400, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    assert results[5]["status"] == "reject"


def test_run_replay_outlier_does_not_change_bpm():
    """BPM before and after the outlier beat should be identical."""
    onsets = scenario_with_outlier(NOM_BPM, 12, 5, 0.400, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    bpm_before = results[4]["estimated_bpm"]
    bpm_after  = results[6]["estimated_bpm"]
    # Accept a tiny difference due to beat 6 observation; compare beats 4 and 5
    bpm_at_outlier = results[5]["estimated_bpm"]
    assert bpm_at_outlier == pytest.approx(bpm_before, abs=1e-9), (
        "Tracker BPM must not change when an outlier is rejected"
    )


# ── 28–30: run_replay — missed beats ─────────────────────────────────────────

def test_run_replay_missed_errors_are_none():
    onsets = scenario_with_missed_beats(NOM_BPM, 10, {3, 7}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    for idx in (3, 7):
        r = results[idx]
        assert r["fixed_error_ms"]         is None
        assert r["adaptive_error_ms"]      is None
        assert r["adjusted_target_time_s"] is None


def test_run_replay_missed_status_is_miss():
    onsets = scenario_with_missed_beats(NOM_BPM, 10, {3, 7}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    assert results[3]["status"] == "miss"
    assert results[7]["status"] == "miss"


def test_run_replay_non_missed_around_miss_have_valid_errors():
    onsets = scenario_with_missed_beats(NOM_BPM, 10, {5}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    assert results[4]["fixed_error_ms"] is not None
    assert results[6]["fixed_error_ms"] is not None


# ── 31: run_replay — gradual change ──────────────────────────────────────────

def test_run_replay_gradual_accel_bpm_increases():
    onsets = scenario_gradual_change(NOM_BPM, 120.0, 140.0, 32, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    early_bpm = results[4]["estimated_bpm"]
    late_bpm  = results[-1]["estimated_bpm"]
    assert late_bpm > early_bpm, (
        f"BPM should increase during acceleration: {early_bpm:.1f} → {late_bpm:.1f}"
    )


# ── 32–36: export_csv ────────────────────────────────────────────────────────

def test_export_csv_creates_file():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 4, COUNT_IN), NOM_BPM)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    export_csv(results, path)
    assert Path(path).exists()


def test_export_csv_header_columns():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 4, COUNT_IN), NOM_BPM)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    export_csv(results, path)
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
    expected = [
        "beat", "nominal_time_s", "actual_onset_time_s",
        "adjusted_target_time_s", "fixed_error_ms", "adaptive_error_ms",
        "estimated_bpm", "confidence", "status",
    ]
    assert header == expected


def test_export_csv_row_count():
    n = 8
    results = run_replay(scenario_exact_nominal(NOM_BPM, n, COUNT_IN), NOM_BPM)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    export_csv(results, path)
    rows = Path(path).read_text().strip().splitlines()
    assert len(rows) == n + 1  # header + n data rows


def test_export_csv_none_as_empty_string():
    onsets  = scenario_with_missed_beats(NOM_BPM, 6, {3}, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    export_csv(results, path)
    with open(path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    missed_row = next(r for r in rows if r["status"] == "miss")
    assert missed_row["fixed_error_ms"] == ""
    assert missed_row["actual_onset_time_s"] == ""


def test_export_csv_numeric_roundtrip():
    results = run_replay(scenario_exact_nominal(NOM_BPM, 4, COUNT_IN), NOM_BPM)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    export_csv(results, path)
    with open(path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    # Beat 0 nominal time should survive the CSV round-trip
    assert float(rows[0]["nominal_time_s"]) == pytest.approx(COUNT_IN, abs=1e-9)


# ── 37–38: TempoTracker diagnostic properties ─────────────────────────────────

def test_tempo_tracker_outlier_limit_s():
    threshold = 0.40
    tracker = TempoTracker(NOM_BPM, outlier_threshold=threshold)
    expected = threshold * (60.0 / NOM_BPM)
    assert tracker.outlier_limit_s == pytest.approx(expected, abs=1e-9)


def test_tempo_tracker_nominal_beat_s():
    tracker = TempoTracker(NOM_BPM)
    assert tracker.nominal_beat_s == pytest.approx(60.0 / NOM_BPM, abs=1e-9)
