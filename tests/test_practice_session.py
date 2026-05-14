"""Tests for core/practice_session.py — no audio hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.practice_session import PracticeSession


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Quarter notes at beats 0, 1, 2, 3.
_TARGETS = [
    {"time": 0, "note": "D2"},
    {"time": 1, "note": "F2"},
    {"time": 2, "note": "A2"},
    {"time": 3, "note": "C2"},
]

# BPM=120, count_in=4, SR=48000, margin=0.15 (default)
#   beat_s=0.5, count_in_s=2.0, gap=0.5, win_end_offset=0.30
#   threshold_i = 2.45 + i*0.5 seconds
#
#   idx=0 → 2.45 s → sample 117600
#   idx=1 → 2.95 s → sample 141600
#   idx=2 → 3.45 s → sample 165600
#   idx=3 → 3.95 s → sample 189600

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000
_T        = [117600, 141600, 165600, 189600]


def _session(targets=None):
    return PracticeSession(
        targets if targets is not None else _TARGETS,
        _BPM, _COUNT_IN, _SR,
    )


# ── No targets ready at start ─────────────────────────────────────────────────

def test_no_targets_ready_at_sample_zero():
    assert _session().update(0) == []


def test_no_targets_ready_just_before_first_threshold():
    assert _session().update(_T[0] - 1) == []


def test_initial_evaluated_indices_empty():
    assert _session().evaluated_indices == set()


# ── Target becomes ready after its window passes ──────────────────────────────

def test_first_target_ready_at_threshold():
    result = _session().update(_T[0])
    assert result == [0]


def test_second_target_ready_at_its_threshold():
    s = _session()
    s.update(_T[0])   # consume idx 0
    result = s.update(_T[1])
    assert result == [1]


def test_target_not_ready_one_sample_before_threshold():
    result = _session().update(_T[1] - 1)
    assert 1 not in result


# ── Same target not returned twice ────────────────────────────────────────────

def test_same_target_not_returned_on_second_call():
    s = _session()
    s.update(_T[0])
    assert s.update(_T[0]) == []


def test_same_target_not_returned_at_later_sample():
    s = _session()
    s.update(_T[0])
    assert 0 not in s.update(_T[3])


def test_each_target_returned_at_most_once_sequential():
    s = _session()
    seen = []
    for sample in _T:
        seen.extend(s.update(sample))
    assert seen == list(set(seen))   # no duplicates
    assert sorted(seen) == [0, 1, 2, 3]


# ── Multiple targets ready when current_sample jumps ahead ────────────────────

def test_all_targets_ready_when_sample_past_last_threshold():
    result = _session().update(_T[3])
    assert result == [0, 1, 2, 3]


def test_first_two_ready_when_sample_at_second_threshold():
    result = _session().update(_T[1])
    assert result == [0, 1]


def test_multiple_ready_targets_in_ascending_order():
    result = _session().update(_T[3])
    assert result == sorted(result)


# ── evaluated_indices tracked correctly ───────────────────────────────────────

def test_evaluated_indices_updated_after_update():
    s = _session()
    s.update(_T[0])
    assert 0 in s.evaluated_indices


def test_evaluated_indices_accumulate_across_calls():
    s = _session()
    s.update(_T[0])
    s.update(_T[1])
    assert {0, 1} <= s.evaluated_indices


def test_evaluated_indices_match_all_returned():
    s = _session()
    s.update(_T[3])
    assert s.evaluated_indices == {0, 1, 2, 3}


def test_unevaluated_indices_not_in_set():
    s = _session()
    s.update(_T[0])
    assert 1 not in s.evaluated_indices
    assert 2 not in s.evaluated_indices
    assert 3 not in s.evaluated_indices


def test_evaluated_indices_not_returned_again():
    s = _session()
    first  = s.update(_T[3])   # all four
    second = s.update(_T[3])   # nothing new
    assert first  == [0, 1, 2, 3]
    assert second == []


# ── Empty target list ─────────────────────────────────────────────────────────

def test_empty_targets_returns_empty_list():
    s = _session(targets=[])
    assert s.update(1_000_000) == []


def test_empty_targets_evaluated_indices_stays_empty():
    s = _session(targets=[])
    s.update(1_000_000)
    assert s.evaluated_indices == set()


# ── Pre-populated evaluated_indices ───────────────────────────────────────────

def test_pre_populated_evaluated_indices_respected():
    s = PracticeSession(_TARGETS, _BPM, _COUNT_IN, _SR, evaluated_indices={0, 1})
    result = s.update(_T[3])
    assert 0 not in result
    assert 1 not in result
    assert result == [2, 3]


def test_pre_populated_indices_not_re_evaluated():
    s = PracticeSession(_TARGETS, _BPM, _COUNT_IN, _SR, evaluated_indices={0})
    s.update(_T[3])
    assert 0 in s.evaluated_indices   # still there, not doubled
