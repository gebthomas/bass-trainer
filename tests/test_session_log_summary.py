"""Tests for pure summary helpers in scripts/session_log_summary.py.

Test matrix
-----------
hit_errors_ordered
    1.  Returns empty list when no TARGET_HIT events.
    2.  Returns errors in target_index order, not insertion order.
    3.  Skips TARGET_HIT events with value=None.
    4.  Skips TARGET_HIT events with target_index=None.
    5.  Includes events with negative errors.
    6.  Ignores TARGET_MISS and EXTRA_ONSET events.

split_half_means
    7.  Empty list → (None, None).
    8.  Single element → (None, first_value) — first half empty.
    9.  Two elements → (first, second).
    10. Even count: both halves equal length.
    11. Odd count: first half has n//2, second has the rest.
    12. All positive errors: both means positive.
    13. Mixed signs: means reflect the split.

longest_run
    14. Empty list → 0.
    15. No element satisfies predicate → 0.
    16. All elements satisfy predicate → len(errors).
    17. Single-element match → 1.
    18. Run at start of list.
    19. Run at end of list.
    20. Run in middle; flanked by non-matching.
    21. Multiple runs of different lengths: longest is returned.
    22. Predicate for late (> 0) works correctly.
    23. Predicate for early (< 0) works correctly.
    24. Zero breaks a late run.
    25. Zero breaks an early run.

missed_target_indices
    26. Empty log → empty list.
    27. No TARGET_MISS events → empty list.
    28. Single miss → [target_index].
    29. Multiple misses returned in sorted order.
    30. TARGET_MISS with target_index=None is excluded.
    31. TARGET_HIT events are not included.

severity_counts
    32. Empty log → all zero.
    33. One good hit → good=1, others 0.
    34. One warn hit → warn=1, others 0.
    35. One miss-tier hit → miss=1, others 0.
    36. TARGET_MISS events are not counted.
    37. Mixed hits: counts correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_log import (
    EXTRA_ONSET,
    SCHEMA_VERSION,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
)
from scripts.session_log_summary import (
    hit_errors_ordered,
    longest_run,
    missed_target_indices,
    severity_counts,
    split_half_means,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _log(*events: SessionEvent) -> SessionLog:
    return SessionLog(
        schema_version = SCHEMA_VERSION,
        started_at     = "2026-05-25T12:00:00+00:00",
        events         = list(events),
    )


def _hit(target_index: int, error_s: float, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec     = time_sec,
        event_type   = TARGET_HIT,
        target_index = target_index,
        value        = error_s,
    )


def _miss(target_index: int, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec     = time_sec,
        event_type   = TARGET_MISS,
        target_index = target_index,
    )


def _extra(time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec   = time_sec,
        event_type = EXTRA_ONSET,
    )


# ── hit_errors_ordered ────────────────────────────────────────────────────────

class TestHitErrorsOrdered:

    def test_empty_when_no_hits(self):
        assert hit_errors_ordered(_log()) == []

    def test_sorted_by_target_index_not_insertion(self):
        # Inserted as beat 2, then beat 0 — result should be [beat-0-err, beat-2-err].
        log = _log(_hit(2, 0.030), _hit(0, -0.020))
        assert hit_errors_ordered(log) == pytest.approx([-0.020, 0.030])

    def test_skips_hit_with_none_value(self):
        ev = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, target_index=0, value=None)
        assert hit_errors_ordered(_log(ev)) == []

    def test_skips_hit_with_none_target_index(self):
        ev = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, target_index=None, value=0.01)
        assert hit_errors_ordered(_log(ev)) == []

    def test_includes_negative_errors(self):
        log = _log(_hit(0, -0.025))
        assert hit_errors_ordered(log) == pytest.approx([-0.025])

    def test_ignores_miss_and_extra(self):
        log = _log(_miss(0), _extra(), _hit(1, 0.010))
        assert hit_errors_ordered(log) == pytest.approx([0.010])


# ── split_half_means ──────────────────────────────────────────────────────────

class TestSplitHalfMeans:

    def test_empty_returns_none_none(self):
        assert split_half_means([]) == (None, None)

    def test_single_element_first_half_empty(self):
        # n=1 → mid=0 → first=[], second=[v]
        first, second = split_half_means([0.010])
        assert first  is None
        assert second == pytest.approx(0.010)

    def test_two_elements(self):
        first, second = split_half_means([0.020, -0.040])
        assert first  == pytest.approx(0.020)
        assert second == pytest.approx(-0.040)

    def test_even_count_equal_halves(self):
        errors = [0.01, 0.02, 0.03, 0.04]
        first, second = split_half_means(errors)
        assert first  == pytest.approx(0.015)  # mean(0.01, 0.02)
        assert second == pytest.approx(0.035)  # mean(0.03, 0.04)

    def test_odd_count_split_at_floor(self):
        # n=5, mid=2: first=[a,b], second=[c,d,e]
        errors = [0.01, 0.02, 0.03, 0.04, 0.05]
        first, second = split_half_means(errors)
        assert first  == pytest.approx(0.015)           # mean(0.01, 0.02)
        assert second == pytest.approx((0.03+0.04+0.05)/3)

    def test_all_positive_both_positive(self):
        errors = [0.01, 0.02, 0.03, 0.04]
        first, second = split_half_means(errors)
        assert first  > 0
        assert second > 0

    def test_mixed_signs_means_reflect_split(self):
        # First half all negative, second half all positive.
        errors = [-0.03, -0.02, 0.02, 0.03]
        first, second = split_half_means(errors)
        assert first  < 0
        assert second > 0


# ── longest_run ───────────────────────────────────────────────────────────────

class TestLongestRun:

    _late  = lambda self, e: e > 0
    _early = lambda self, e: e < 0

    def test_empty_list_zero(self):
        assert longest_run([], self._late) == 0

    def test_no_match_zero(self):
        assert longest_run([-0.01, -0.02], self._late) == 0

    def test_all_match_full_length(self):
        assert longest_run([0.01, 0.02, 0.03], self._late) == 3

    def test_single_match(self):
        assert longest_run([0.01], self._late) == 1

    def test_run_at_start(self):
        assert longest_run([0.01, 0.02, -0.01, -0.02], self._late) == 2

    def test_run_at_end(self):
        assert longest_run([-0.01, 0.01, 0.02], self._late) == 2

    def test_run_in_middle(self):
        assert longest_run([-0.01, 0.01, 0.02, 0.03, -0.01], self._late) == 3

    def test_longest_of_multiple_runs(self):
        # [late, early, late, late, late] → longest = 3
        errors = [0.01, -0.01, 0.02, 0.03, 0.04]
        assert longest_run(errors, self._late) == 3

    def test_early_predicate(self):
        errors = [-0.01, -0.02, 0.01, -0.03]
        assert longest_run(errors, self._early) == 2

    def test_zero_breaks_late_run(self):
        # 0.01, 0.0, 0.02 → runs of 1 and 1, not 2
        assert longest_run([0.01, 0.0, 0.02], self._late) == 1

    def test_zero_breaks_early_run(self):
        assert longest_run([-0.01, 0.0, -0.02], self._early) == 1


# ── missed_target_indices ─────────────────────────────────────────────────────

class TestMissedTargetIndices:

    def test_empty_log_empty_list(self):
        assert missed_target_indices(_log()) == []

    def test_no_misses_empty_list(self):
        assert missed_target_indices(_log(_hit(0, 0.01))) == []

    def test_single_miss(self):
        assert missed_target_indices(_log(_miss(3))) == [3]

    def test_multiple_misses_sorted(self):
        # Inserted out of order → returned sorted.
        log = _log(_miss(5), _miss(1), _miss(3))
        assert missed_target_indices(log) == [1, 3, 5]

    def test_miss_with_none_target_index_excluded(self):
        ev = SessionEvent(time_sec=1.0, event_type=TARGET_MISS, target_index=None)
        assert missed_target_indices(_log(ev)) == []

    def test_hits_not_included(self):
        log = _log(_hit(0, 0.01), _miss(1))
        assert missed_target_indices(log) == [1]


# ── severity_counts ───────────────────────────────────────────────────────────

class TestSeverityCounts:

    def test_empty_log_all_zero(self):
        counts = severity_counts(_log())
        assert counts == {"good": 0, "warn": 0, "miss": 0}

    def test_one_good_hit(self):
        # DEFAULT_GOOD_THRESHOLD_S = 0.05 s → 0.01 s is good
        counts = severity_counts(_log(_hit(0, 0.010)))
        assert counts["good"] == 1
        assert counts["warn"] == 0
        assert counts["miss"] == 0

    def test_one_warn_hit(self):
        # Between 0.05 s and 0.12 s → warn
        counts = severity_counts(_log(_hit(0, 0.080)))
        assert counts["warn"] == 1
        assert counts["good"] == 0
        assert counts["miss"] == 0

    def test_one_miss_tier_hit(self):
        # Beyond 0.12 s → miss-tier
        counts = severity_counts(_log(_hit(0, 0.200)))
        assert counts["miss"] == 1
        assert counts["good"] == 0
        assert counts["warn"] == 0

    def test_target_miss_events_not_counted(self):
        counts = severity_counts(_log(_miss(0), _miss(1)))
        assert counts == {"good": 0, "warn": 0, "miss": 0}

    def test_mixed_hits_correct_counts(self):
        log = _log(
            _hit(0,  0.010),   # good
            _hit(1,  0.080),   # warn
            _hit(2,  0.200),   # miss-tier
            _hit(3, -0.010),   # good (early but within threshold)
            _hit(4, -0.090),   # warn
            _miss(5),          # not counted
        )
        counts = severity_counts(log)
        assert counts["good"] == 2
        assert counts["warn"] == 2
        assert counts["miss"] == 1
