"""Tests for core/live_feedback_controller.py — no audio hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_feedback_controller import process_ready_feedback


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Quarter notes at beats 0, 1, 2, 3.
_TARGETS = [
    {"time": 0, "note": "D2"},
    {"time": 1, "note": "F2"},
    {"time": 2, "note": "A2"},
    {"time": 3, "note": "C2"},
]

# BPM=120, count_in=4, SR=48000, margin=0.15
#   beat_s=0.5, count_in_s=2.0, gap=0.5 → win_end_offset=0.30
#   threshold_i = 2.0 + i*0.5 + 0.30 + 0.15 = 2.45 + i*0.5
#
#   i=0 → 2.45 s → sample 117600
#   i=1 → 2.95 s → sample 141600
#   i=2 → 3.45 s → sample 165600
#   i=3 → 3.95 s → sample 189600

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000
_T        = [117600, 141600, 165600, 189600]   # ready samples per target

_GOOD = {"detected_note": "D2", "timing_error_s": 0.02, "pitch_error_cents": 5.0}


def _call(sample, evaluated=None, results=None):
    return process_ready_feedback(
        _TARGETS, _BPM, _COUNT_IN, _SR,
        sample,
        evaluated if evaluated is not None else set(),
        results  if results  is not None else {},
    )


# ── Return structure ──────────────────────────────────────────────────────────

def test_return_keys_always_present():
    out = _call(0)
    assert "events"                  in out
    assert "newly_evaluated_indices" in out
    assert "all_evaluated_indices"   in out


def test_events_is_list():
    assert isinstance(_call(0)["events"], list)


def test_newly_evaluated_is_set():
    assert isinstance(_call(0)["newly_evaluated_indices"], set)


def test_all_evaluated_is_set():
    assert isinstance(_call(0)["all_evaluated_indices"], set)


# ── No ready targets ──────────────────────────────────────────────────────────

def test_no_ready_targets_at_sample_zero():
    out = _call(0, results={0: _GOOD})
    assert out["events"]                  == []
    assert out["newly_evaluated_indices"] == set()
    assert out["all_evaluated_indices"]   == set()


def test_no_ready_targets_just_before_first_threshold():
    out = _call(_T[0] - 1, results={0: _GOOD})
    assert out["events"] == []
    assert out["newly_evaluated_indices"] == set()


def test_no_ready_no_results_empty_output():
    out = _call(0)
    assert out["events"] == []
    assert out["newly_evaluated_indices"] == set()
    assert out["all_evaluated_indices"]   == set()


# ── Ready target with result ──────────────────────────────────────────────────

def test_ready_with_result_produces_one_event():
    out = _call(_T[0], results={0: _GOOD})
    assert len(out["events"]) == 1


def test_event_target_idx_matches():
    out = _call(_T[0], results={0: _GOOD})
    assert out["events"][0]["target_idx"] == 0


def test_event_expected_note_from_target():
    out = _call(_T[0], results={0: _GOOD})
    assert out["events"][0]["expected_note"] == "D2"


def test_ready_with_result_newly_evaluated():
    out = _call(_T[0], results={0: _GOOD})
    assert out["newly_evaluated_indices"] == {0}


def test_ready_with_result_all_evaluated():
    out = _call(_T[0], results={0: _GOOD})
    assert out["all_evaluated_indices"] == {0}


# ── Ready target without result is skipped ────────────────────────────────────

def test_ready_without_result_no_event():
    out = _call(_T[0], results={})
    assert out["events"] == []


def test_ready_without_result_not_marked_evaluated():
    out = _call(_T[0], results={})
    assert 0 not in out["newly_evaluated_indices"]
    assert 0 not in out["all_evaluated_indices"]


def test_mixed_ready_only_those_with_results_emitted():
    # Targets 0 and 1 are both ready; only 0 has a result.
    out = _call(_T[1], results={0: _GOOD})
    assert len(out["events"]) == 1
    assert out["events"][0]["target_idx"] == 0
    assert out["newly_evaluated_indices"] == {0}
    assert 1 not in out["newly_evaluated_indices"]


# ── Multiple ready targets preserve order ─────────────────────────────────────

def test_multiple_ready_with_results_all_emitted():
    results = {0: _GOOD, 1: _GOOD, 2: _GOOD}
    out = _call(_T[2], results=results)
    assert len(out["events"]) == 3


def test_multiple_ready_events_in_target_index_order():
    results = {0: _GOOD, 1: _GOOD, 2: _GOOD}
    out = _call(_T[2], results=results)
    indices = [e["target_idx"] for e in out["events"]]
    assert indices == [0, 1, 2]


def test_multiple_ready_partial_results_preserves_order():
    # Targets 0,1,2 ready; results for 0 and 2 only.
    results = {0: _GOOD, 2: _GOOD}
    out = _call(_T[2], results=results)
    indices = [e["target_idx"] for e in out["events"]]
    assert indices == [0, 2]


def test_all_targets_ready():
    results = {0: _GOOD, 1: _GOOD, 2: _GOOD, 3: _GOOD}
    out = _call(_T[3], results=results)
    assert [e["target_idx"] for e in out["events"]] == [0, 1, 2, 3]
    assert out["newly_evaluated_indices"] == {0, 1, 2, 3}


# ── Already evaluated not re-emitted ─────────────────────────────────────────

def test_already_evaluated_not_in_events():
    results = {0: _GOOD, 1: _GOOD}
    out = _call(_T[1], evaluated={0}, results=results)
    indices = [e["target_idx"] for e in out["events"]]
    assert 0 not in indices


def test_already_evaluated_only_new_ones_emitted():
    results = {0: _GOOD, 1: _GOOD, 2: _GOOD}
    out = _call(_T[2], evaluated={0}, results=results)
    indices = [e["target_idx"] for e in out["events"]]
    assert indices == [1, 2]


def test_already_evaluated_not_in_newly():
    results = {0: _GOOD, 1: _GOOD}
    out = _call(_T[1], evaluated={0}, results=results)
    assert 0 not in out["newly_evaluated_indices"]


# ── evaluated_indices is not mutated ─────────────────────────────────────────

def test_evaluated_set_not_mutated():
    original = {0}
    snapshot = set(original)
    _call(_T[1], evaluated=original, results={0: _GOOD, 1: _GOOD})
    assert original == snapshot


def test_evaluated_list_accepted_and_not_mutated():
    original = [0]
    _call(_T[1], evaluated=original, results={1: _GOOD})
    assert original == [0]


# ── Result for future (not-yet-ready) target is ignored ──────────────────────

def test_future_result_not_emitted():
    # At _T[0] only target 0 is ready; result for target 3 should be ignored.
    results = {3: _GOOD}
    out = _call(_T[0], results=results)
    assert out["events"] == []
    assert 3 not in out["newly_evaluated_indices"]


def test_future_result_does_not_affect_ready_target():
    results = {0: _GOOD, 3: _GOOD}
    out = _call(_T[0], results=results)
    assert len(out["events"]) == 1
    assert out["events"][0]["target_idx"] == 0


# ── Result for invalid index is ignored ──────────────────────────────────────

def test_invalid_index_in_results_no_error():
    results = {999: _GOOD}
    out = _call(_T[3], results=results)   # all 4 targets ready, invalid key present
    assert 999 not in out["newly_evaluated_indices"]


def test_invalid_index_does_not_block_valid_targets():
    results = {999: _GOOD, 0: _GOOD}
    out = _call(_T[0], results=results)
    assert len(out["events"]) == 1
    assert out["events"][0]["target_idx"] == 0


def test_negative_index_in_results_ignored():
    results = {-1: _GOOD, 0: _GOOD}
    out = _call(_T[0], results=results)
    assert [e["target_idx"] for e in out["events"]] == [0]


# ── all_evaluated_indices includes prior and new ──────────────────────────────

def test_all_evaluated_union_of_prior_and_new():
    out = _call(_T[1], evaluated={0}, results={1: _GOOD})
    assert out["all_evaluated_indices"] == {0, 1}


def test_all_evaluated_when_nothing_new():
    # Target 0 is ready but has no result — prior stays unchanged.
    out = _call(_T[0], evaluated={2}, results={})
    assert out["all_evaluated_indices"] == {2}


def test_all_evaluated_empty_prior_and_empty_new():
    out = _call(0)
    assert out["all_evaluated_indices"] == set()


def test_all_evaluated_large_prior_plus_new():
    results = {3: _GOOD}
    out = _call(_T[3], evaluated={0, 1, 2}, results=results)
    assert out["all_evaluated_indices"] == {0, 1, 2, 3}
    assert out["newly_evaluated_indices"] == {3}


# ── Idempotency ───────────────────────────────────────────────────────────────

def test_second_call_with_all_evaluated_produces_no_events():
    results = {0: _GOOD, 1: _GOOD}
    first  = _call(_T[1], results=results)
    second = _call(_T[1],
                   evaluated=first["all_evaluated_indices"],
                   results=results)
    assert second["events"]                  == []
    assert second["newly_evaluated_indices"] == set()


def test_feeding_all_evaluated_back_is_stable():
    results = {i: _GOOD for i in range(4)}
    evaluated: set[int] = set()
    for sample in _T:
        out       = _call(sample, evaluated=evaluated, results=results)
        evaluated = out["all_evaluated_indices"]
    # After processing all, a final call produces nothing new.
    final = _call(_T[3] + 100_000, evaluated=evaluated, results=results)
    assert final["events"] == []
    assert final["newly_evaluated_indices"] == set()


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_targets_returns_empty():
    out = process_ready_feedback([], _BPM, _COUNT_IN, _SR, 1_000_000, set(), {0: _GOOD})
    assert out["events"]                  == []
    assert out["newly_evaluated_indices"] == set()
    assert out["all_evaluated_indices"]   == set()


def test_empty_results_dict_never_emits():
    out = _call(_T[3], results={})
    assert out["events"] == []
    assert out["newly_evaluated_indices"] == set()
