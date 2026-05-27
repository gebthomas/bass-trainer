"""Tests for core/realtime_evaluator — deterministic timing evaluation.

All tests use synthetic timestamp data; no audio hardware required.

Conventions
-----------
- Times in seconds; errors in milliseconds.
- Positive signed_error_ms → onset arrived late.
- Negative signed_error_ms → onset arrived early.
- Default tolerance_s      = 0.08 s (80 ms).
- Default on_time threshold = 0.03 s (30 ms).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime_evaluator import TargetEvaluation, evaluate_targets


# ── Helpers ───────────────────────────────────────────────────────────────────

_TOL = 0.08   # default tolerance (seconds)
_OT  = 0.03   # default on_time threshold (seconds)


def _eval(targets, onsets, tolerance_s=_TOL, on_time_threshold_s=_OT):
    return evaluate_targets(
        targets, onsets,
        tolerance_s=tolerance_s,
        on_time_threshold_s=on_time_threshold_s,
    )


def _single(targets, onsets, **kw):
    """Evaluate and assert exactly one result is returned."""
    results = _eval(targets, onsets, **kw)
    assert len(results) == 1
    return results[0]


# ── TargetEvaluation dataclass ────────────────────────────────────────────────

class TestTargetEvaluationDataclass:
    def test_all_fields_present(self):
        te = TargetEvaluation(
            target_index=0,
            expected_time_s=1.0,
            actual_time_s=1.01,
            signed_error_ms=10.0,
            classification="on_time",
            matched_onset_index=0,
        )
        assert te.target_index        == 0
        assert te.expected_time_s     == 1.0
        assert te.actual_time_s       == 1.01
        assert te.signed_error_ms     == 10.0
        assert te.classification      == "on_time"
        assert te.matched_onset_index == 0

    def test_miss_fields_are_none(self):
        te = TargetEvaluation(
            target_index=0,
            expected_time_s=1.0,
            actual_time_s=None,
            signed_error_ms=None,
            classification="miss",
            matched_onset_index=None,
        )
        assert te.actual_time_s       is None
        assert te.signed_error_ms     is None
        assert te.matched_onset_index is None

    def test_is_immutable(self):
        te = TargetEvaluation(0, 1.0, 1.0, 0.0, "on_time", 0)
        with pytest.raises(Exception):
            te.target_index = 99   # frozen dataclass raises


# ── Empty inputs ──────────────────────────────────────────────────────────────

class TestEmptyInputs:
    def test_empty_targets_returns_empty_list(self):
        assert _eval([], [0.5, 1.0]) == []

    def test_empty_onsets_all_misses(self):
        results = _eval([1.0, 2.0, 3.0], [])
        assert len(results) == 3
        assert all(r.classification == "miss" for r in results)

    def test_both_empty_returns_empty(self):
        assert _eval([], []) == []

    def test_empty_onsets_miss_fields_are_none(self):
        r = _single([1.0], [])
        assert r.actual_time_s       is None
        assert r.signed_error_ms     is None
        assert r.matched_onset_index is None


# ── Return structure ──────────────────────────────────────────────────────────

class TestReturnStructure:
    def test_returns_list_of_target_evaluations(self):
        results = _eval([1.0], [1.0])
        assert isinstance(results, list)
        assert all(isinstance(r, TargetEvaluation) for r in results)

    def test_result_count_matches_target_count(self):
        assert len(_eval([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])) == 4

    def test_target_index_matches_position(self):
        results = _eval([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        for i, r in enumerate(results):
            assert r.target_index == i

    def test_expected_time_s_matches_input(self):
        targets = [0.5, 1.5, 2.5]
        results = _eval(targets, [0.5, 1.5, 2.5])
        for r in results:
            assert r.expected_time_s == targets[r.target_index]

    def test_classification_is_valid_string(self):
        valid = {"on_time", "early", "late", "miss"}
        results = _eval([1.0, 2.0], [1.0, 99.0])
        for r in results:
            assert r.classification in valid


# ── Perfect timing ────────────────────────────────────────────────────────────

class TestPerfectTiming:
    def test_exact_match_is_on_time(self):
        assert _single([1.0], [1.0]).classification == "on_time"

    def test_exact_match_zero_error(self):
        assert _single([1.0], [1.0]).signed_error_ms == pytest.approx(0.0)

    def test_exact_match_actual_time_set(self):
        assert _single([1.0], [1.0]).actual_time_s == pytest.approx(1.0)

    def test_exact_match_onset_index_set(self):
        assert _single([1.0], [1.0]).matched_onset_index == 0

    def test_within_on_time_threshold_is_on_time(self):
        # 20 ms late < 30 ms threshold → on_time
        assert _single([1.0], [1.020]).classification == "on_time"

    def test_multiple_exact_matches_all_on_time(self):
        results = _eval([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert all(r.classification == "on_time" for r in results)


# ── Early hits ────────────────────────────────────────────────────────────────

class TestEarlyHits:
    def test_early_within_threshold_is_on_time(self):
        # 20 ms early < 30 ms threshold → on_time
        assert _single([1.0], [0.980]).classification == "on_time"

    def test_early_beyond_threshold_is_early(self):
        # 50 ms early > 30 ms threshold → early
        assert _single([1.0], [0.950]).classification == "early"

    def test_early_signed_error_is_negative(self):
        r = _single([1.0], [0.950])
        assert r.signed_error_ms == pytest.approx(-50.0)

    def test_early_within_tolerance_matched(self):
        # 70 ms early, tolerance 80 ms → matched as early
        r = _single([1.0], [0.930], tolerance_s=0.08)
        assert r.classification == "early"
        assert r.actual_time_s  == pytest.approx(0.930)

    def test_early_outside_tolerance_is_miss(self):
        # 90 ms early, tolerance 80 ms → miss
        assert _single([1.0], [0.910], tolerance_s=0.08).classification == "miss"


# ── Late hits ─────────────────────────────────────────────────────────────────

class TestLateHits:
    def test_late_within_threshold_is_on_time(self):
        # 20 ms late < 30 ms threshold → on_time
        assert _single([1.0], [1.020]).classification == "on_time"

    def test_late_beyond_threshold_is_late(self):
        # 50 ms late > 30 ms threshold → late
        assert _single([1.0], [1.050]).classification == "late"

    def test_late_signed_error_is_positive(self):
        r = _single([1.0], [1.050])
        assert r.signed_error_ms == pytest.approx(50.0)

    def test_late_within_tolerance_matched(self):
        r = _single([1.0], [1.070], tolerance_s=0.08)
        assert r.classification == "late"
        assert r.actual_time_s  == pytest.approx(1.070)

    def test_late_outside_tolerance_is_miss(self):
        # 90 ms late, tolerance 80 ms → miss
        assert _single([1.0], [1.090], tolerance_s=0.08).classification == "miss"


# ── Misses ────────────────────────────────────────────────────────────────────

class TestMisses:
    def test_no_onset_is_miss(self):
        assert _single([1.0], []).classification == "miss"

    def test_onset_outside_window_is_miss(self):
        assert _single([1.0], [2.0]).classification == "miss"

    def test_miss_actual_time_is_none(self):
        assert _single([1.0], []).actual_time_s is None

    def test_miss_signed_error_is_none(self):
        assert _single([1.0], []).signed_error_ms is None

    def test_miss_onset_index_is_none(self):
        assert _single([1.0], []).matched_onset_index is None

    def test_all_misses_without_onsets(self):
        results = _eval([1.0, 2.0, 3.0], [])
        assert all(r.classification == "miss" for r in results)

    def test_first_hit_second_miss(self):
        results = _eval([1.0, 2.0], [1.0])
        assert results[0].classification == "on_time"
        assert results[1].classification == "miss"


# ── Tolerance boundaries ──────────────────────────────────────────────────────

class TestToleranceBoundaries:
    def test_onset_at_right_boundary_is_matched(self):
        # tolerance_s = 0.25 = 1/4, exact in float64.
        # 1.0 + 0.25 = 1.25, also exact.  abs(1.25 - 1.0) = 0.25 = tolerance_s → matched.
        r = _single([1.0], [1.25], tolerance_s=0.25)
        assert r.classification != "miss"

    def test_onset_at_left_boundary_is_matched(self):
        r = _single([1.0], [0.75], tolerance_s=0.25)
        assert r.classification != "miss"

    def test_onset_just_outside_right_boundary_is_miss(self):
        # 90 ms > 80 ms tolerance → miss
        r = _single([1.0], [1.09], tolerance_s=0.08)
        assert r.classification == "miss"

    def test_onset_just_outside_left_boundary_is_miss(self):
        r = _single([1.0], [0.91], tolerance_s=0.08)
        assert r.classification == "miss"


# ── On-time threshold boundaries ──────────────────────────────────────────────

class TestOnTimeThresholdBoundaries:
    def test_error_exactly_at_threshold_is_on_time(self):
        # on_time_threshold_s = 0.0625 = 1/16, exact in float64.
        # onset at 1.0625 → signed_error_ms = 62.5 exactly = threshold → on_time.
        r = _single([1.0], [1.0625], on_time_threshold_s=0.0625)
        assert r.classification == "on_time"

    def test_error_just_above_threshold_is_late(self):
        # 70 ms late > 30 ms default threshold → late
        r = _single([1.0], [1.070], on_time_threshold_s=0.03)
        assert r.classification == "late"

    def test_negative_error_exactly_at_threshold_is_on_time(self):
        r = _single([1.0], [0.9375], on_time_threshold_s=0.0625)
        assert r.classification == "on_time"

    def test_negative_error_just_below_threshold_is_early(self):
        # 70 ms early > 30 ms default threshold → early
        r = _single([1.0], [0.930], on_time_threshold_s=0.03)
        assert r.classification == "early"


# ── Duplicate onset prevention ────────────────────────────────────────────────

class TestDuplicateOnsetPrevention:
    def test_one_onset_matches_only_one_target(self):
        # Single onset near two targets; first target (left to right) claims it.
        results = _eval([1.0, 1.05], [1.0])
        assert results[0].classification == "on_time"
        assert results[1].classification == "miss"

    def test_each_onset_used_at_most_once(self):
        # 3 targets, 2 onsets → third target misses
        results = _eval([1.0, 2.0, 3.0], [1.0, 2.0])
        assert results[0].classification == "on_time"
        assert results[1].classification == "on_time"
        assert results[2].classification == "miss"

    def test_double_hit_nearest_wins(self):
        # Two onsets both inside tolerance; target at 1.0.
        # Onset at 0.970 (30 ms early) vs onset at 1.020 (20 ms late) → 1.020 is closer.
        r = _single([1.0], [0.970, 1.020])
        assert r.actual_time_s == pytest.approx(1.020)

    def test_consumed_onset_not_reused_by_later_target(self):
        # Both onsets are near target 0; target 1 at 1.5 s gets nothing.
        results = _eval([1.0, 1.5], [0.970, 1.020])
        assert results[0].matched_onset_index is not None
        assert results[1].classification == "miss"


# ── Nearest-onset selection ───────────────────────────────────────────────────

class TestNearestOnsetSelection:
    def test_nearest_onset_selected(self):
        # Target 1.0; onsets at 0.960 (40 ms) and 1.010 (10 ms) → 1.010 wins.
        r = _single([1.0], [0.960, 1.010])
        assert r.actual_time_s == pytest.approx(1.010)

    def test_nearest_onset_index_reported(self):
        # Onset 0 at 0.96, onset 1 at 1.01 → nearest is index 1.
        r = _single([1.0], [0.960, 1.010])
        assert r.matched_onset_index == 1

    def test_three_candidates_nearest_chosen(self):
        # Onsets at 0.930 (70 ms), 0.970 (30 ms), 1.040 (40 ms) → 0.970 wins.
        r = _single([1.0], [0.930, 0.970, 1.040])
        assert r.actual_time_s == pytest.approx(0.970)

    def test_equidistant_onsets_earlier_wins(self):
        # Onset A at 0.970 (−30 ms) and onset B at 1.030 (+30 ms) — equal distance.
        # Sorted order puts A first → A wins the tie.
        r = _single([1.0], [0.970, 1.030])
        assert r.actual_time_s == pytest.approx(0.970)


# ── Out-of-order onset input ──────────────────────────────────────────────────

class TestOutOfOrderInput:
    def test_unsorted_onsets_same_classifications_as_sorted(self):
        targets          = [1.0, 2.0, 3.0]
        onsets_sorted    = [1.01, 1.98, 3.02]
        onsets_unsorted  = [3.02, 1.01, 1.98]
        r_sorted    = _eval(targets, onsets_sorted)
        r_unsorted  = _eval(targets, onsets_unsorted)
        for s, u in zip(r_sorted, r_unsorted):
            assert s.classification == u.classification

    def test_reversed_onsets_both_targets_matched(self):
        results = _eval([1.0, 2.0], [2.0, 1.0])
        assert results[0].classification != "miss"
        assert results[1].classification != "miss"

    def test_unsorted_onset_times_match_correctly(self):
        # Onset 0 at 2.01 (near target 1), onset 1 at 1.01 (near target 0).
        # After sorting: [(1, 1.01), (0, 2.01)].
        results = _eval([1.0, 2.0], [2.01, 1.01])
        assert results[0].actual_time_s == pytest.approx(1.01)
        assert results[1].actual_time_s == pytest.approx(2.01)


# ── Deterministic behavior ────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_inputs_produce_identical_results(self):
        targets = [1.0, 2.0, 3.0]
        onsets  = [0.99, 2.01, 2.95]
        r1 = _eval(targets, onsets)
        r2 = _eval(targets, onsets)
        assert r1 == r2

    def test_classification_stable_across_repeated_calls(self):
        for _ in range(5):
            r = _single([1.0], [1.050])
            assert r.classification == "late"


# ── signed_error_ms convention ────────────────────────────────────────────────

class TestSignedErrorConvention:
    def test_late_onset_gives_positive_error(self):
        assert _single([1.0], [1.050]).signed_error_ms > 0

    def test_early_onset_gives_negative_error(self):
        assert _single([1.0], [0.950]).signed_error_ms < 0

    def test_error_magnitude_correct(self):
        r = _single([1.0], [1.045])
        assert r.signed_error_ms == pytest.approx(45.0)

    def test_error_zero_for_exact_match(self):
        assert _single([2.5], [2.5]).signed_error_ms == pytest.approx(0.0)


# ── matched_onset_index ───────────────────────────────────────────────────────

class TestMatchedOnsetIndex:
    def test_index_0_for_single_onset(self):
        assert _single([1.0], [1.0]).matched_onset_index == 0

    def test_index_points_to_correct_onset_time(self):
        onsets = [0.5, 1.0, 1.5]
        r = _single([1.0], onsets)
        assert onsets[r.matched_onset_index] == pytest.approx(r.actual_time_s)

    def test_index_for_second_onset(self):
        # Onset 0 at 0.5 (far), onset 1 at 1.0 (exact) → index 1
        r = _single([1.0], [0.5, 1.0])
        assert r.matched_onset_index == 1

    def test_original_index_preserved_when_input_unsorted(self):
        # onsets[0]=1.5 (far), onsets[1]=1.0 (exact) → original index 1 returned
        r = _single([1.0], [1.5, 1.0])
        assert r.matched_onset_index == 1


# ── Multiple targets, multiple onsets ────────────────────────────────────────

class TestMultipleTargetsOnsets:
    def test_three_targets_three_onsets_all_on_time(self):
        results = _eval([1.0, 2.0, 3.0], [1.005, 2.010, 2.995])
        assert all(r.classification == "on_time" for r in results)

    def test_extra_onsets_ignored(self):
        # More onsets than targets; surplus onsets produce no output entries.
        results = _eval([1.0], [0.5, 1.0, 1.5])
        assert len(results) == 1
        assert results[0].classification  == "on_time"
        assert results[0].matched_onset_index == 1

    def test_earlier_target_has_first_claim_on_shared_onset(self):
        # Onset at 1.02 is within tolerance of both target 1.0 and target 1.05.
        results = _eval([1.0, 1.05], [1.02], tolerance_s=0.08)
        assert results[0].classification != "miss"
        assert results[1].classification == "miss"

    def test_mixed_classifications_correct(self):
        # target 0 → on_time (10 ms); target 1 → late (50 ms);
        # target 2 → miss (onset at 5.0 s is 2 s away);
        # target 3 → miss (no onset remaining)
        targets = [1.0, 2.0, 3.0, 4.0]
        onsets  = [1.01, 2.05, 5.00]
        results = _eval(targets, onsets)
        assert results[0].classification == "on_time"
        assert results[1].classification == "late"
        assert results[2].classification == "miss"
        assert results[3].classification == "miss"


# ── Custom tolerance and threshold ────────────────────────────────────────────

class TestCustomParameters:
    def test_tight_tolerance_causes_miss(self):
        # 40 ms late; tolerance only 30 ms → miss
        r = _single([1.0], [1.040], tolerance_s=0.03)
        assert r.classification == "miss"

    def test_wider_on_time_threshold_reclassifies_as_on_time(self):
        # 50 ms late; threshold raised to 60 ms → on_time
        r = _single([1.0], [1.050], on_time_threshold_s=0.06)
        assert r.classification == "on_time"

    def test_narrower_on_time_threshold_reclassifies_as_late(self):
        # 20 ms late; threshold narrowed to 10 ms → late
        r = _single([1.0], [1.020], on_time_threshold_s=0.01)
        assert r.classification == "late"
