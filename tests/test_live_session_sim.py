"""Tests for core/live_session_sim.py — no audio hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_session_sim import simulate_live_session, simulate_until_complete


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Quarter notes at beats 0, 1, 2, 3.
_TARGETS = [
    {"time": 0, "note": "D2"},
    {"time": 1, "note": "F2"},
    {"time": 2, "note": "A2"},
    {"time": 3, "note": "C2"},
]

# BPM=120, count_in=4, SR=48000, margin=0.15, tick_s=0.5 (exact binary fraction)
#   beat_s=0.5, count_in_s=2.0, gap=0.5 → win_end_offset=0.30
#   threshold_i = 2.45 + i*0.5
#
#   i=0: 2.45 s  → first caught at t=2.5  (tick 6)
#   i=1: 2.95 s  → first caught at t=3.0  (tick 7)
#   i=2: 3.45 s  → first caught at t=3.5  (tick 8)
#   i=3: 3.95 s  → first caught at t=4.0  (tick 9)
#
# tick_count formula for this tick_s=0.5: floor(duration_s/0.5) + 1

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000
_TICK     = 0.5   # 0.5 is exactly representable; no float drift

_GOOD = {"detected_note": "D2", "timing_error_s": 0.02, "pitch_error_cents": 5.0}
_ALL  = {0: _GOOD, 1: _GOOD, 2: _GOOD, 3: _GOOD}


def _sim(duration_s, tick_s=_TICK, schedule=None, count_in=_COUNT_IN):
    return simulate_live_session(
        _TARGETS, _BPM, count_in, _SR,
        duration_s, tick_s, schedule if schedule is not None else {},
    )


def _until(schedule=None, tick_s=_TICK, max_dur=30.0, count_in=_COUNT_IN, targets=None):
    return simulate_until_complete(
        targets if targets is not None else _TARGETS,
        _BPM, count_in, _SR,
        schedule if schedule is not None else {},
        tick_s=tick_s, max_duration_s=max_dur,
    )


# ── Return structure ──────────────────────────────────────────────────────────

def test_sim_return_keys():
    out = _sim(1.0)
    for k in ("events", "evaluated_indices", "tick_count", "duration_s"):
        assert k in out, f"missing key: {k!r}"


def test_until_return_keys():
    out = _until()
    for k in ("events", "evaluated_indices", "tick_count", "duration_s", "completed"):
        assert k in out, f"missing key: {k!r}"


def test_sim_events_is_list():
    assert isinstance(_sim(1.0)["events"], list)


def test_sim_evaluated_is_set():
    assert isinstance(_sim(1.0)["evaluated_indices"], set)


# ── Empty targets ─────────────────────────────────────────────────────────────

def test_sim_empty_targets():
    out = simulate_live_session([], _BPM, _COUNT_IN, _SR, 5.0, _TICK, _ALL)
    assert out["events"]            == []
    assert out["evaluated_indices"] == set()
    assert out["tick_count"]        >= 1
    assert out["duration_s"]        == 5.0


def test_until_empty_targets():
    out = simulate_until_complete([], _BPM, _COUNT_IN, _SR, _ALL,
                                  tick_s=_TICK, max_duration_s=30.0)
    assert out["events"]    == []
    assert out["completed"] is True


def test_until_empty_schedule():
    out = _until(schedule={})
    assert out["events"]            == []
    assert out["evaluated_indices"] == set()
    assert out["completed"]         is True


# ── Single target emitted once ────────────────────────────────────────────────

def test_sim_single_target_emitted():
    out = _sim(5.0, schedule={0: _GOOD})
    assert len(out["events"]) == 1
    assert out["events"][0]["target_idx"] == 0


def test_sim_single_target_only_once():
    out = _sim(10.0, schedule={0: _GOOD})
    assert len([e for e in out["events"] if e["target_idx"] == 0]) == 1


def test_until_single_target_completes():
    out = _until(schedule={0: _GOOD})
    assert out["completed"] is True
    assert len(out["events"]) == 1
    assert out["events"][0]["target_idx"] == 0


# ── Multiple targets emitted in order ─────────────────────────────────────────

def test_sim_all_targets_emitted():
    out = _sim(5.0, schedule=_ALL)
    assert len(out["events"]) == 4


def test_sim_targets_in_chronological_index_order():
    out = _sim(5.0, schedule=_ALL)
    indices = [e["target_idx"] for e in out["events"]]
    assert indices == [0, 1, 2, 3]


def test_sim_targets_strictly_increasing():
    out = _sim(5.0, schedule=_ALL)
    indices = [e["target_idx"] for e in out["events"]]
    assert all(a < b for a, b in zip(indices, indices[1:]))


def test_until_all_targets_in_order():
    out = _until(schedule=_ALL)
    assert [e["target_idx"] for e in out["events"]] == [0, 1, 2, 3]
    assert out["completed"] is True


# ── Missing evaluation results remain pending ─────────────────────────────────

def test_sim_target_without_result_not_emitted():
    schedule = {0: _GOOD, 2: _GOOD}   # no result for 1 and 3
    out = _sim(5.0, schedule=schedule)
    idxs = {e["target_idx"] for e in out["events"]}
    assert 1 not in idxs
    assert 3 not in idxs


def test_sim_target_without_result_not_evaluated():
    schedule = {0: _GOOD}
    out = _sim(5.0, schedule=schedule)
    assert 1 not in out["evaluated_indices"]
    assert 2 not in out["evaluated_indices"]
    assert 3 not in out["evaluated_indices"]


def test_until_unscheduled_target_ignored_for_completion():
    # Only target 0 is scheduled; targets 1–3 have no results.
    out = _until(schedule={0: _GOOD})
    assert out["completed"] is True
    assert len(out["events"]) == 1


def test_until_unscheduled_targets_not_in_evaluated():
    out = _until(schedule={0: _GOOD})
    assert 1 not in out["evaluated_indices"]
    assert 2 not in out["evaluated_indices"]
    assert 3 not in out["evaluated_indices"]


# ── No duplicate events ───────────────────────────────────────────────────────

def test_sim_no_duplicate_events():
    out = _sim(20.0, schedule=_ALL)   # many ticks after all thresholds
    assert len(out["events"]) == 4


def test_sim_each_target_emitted_at_most_once():
    out = _sim(20.0, schedule=_ALL)
    seen = [e["target_idx"] for e in out["events"]]
    assert len(seen) == len(set(seen))


def test_until_no_duplicate_events():
    out = _until(schedule=_ALL, max_dur=30.0)
    seen = [e["target_idx"] for e in out["events"]]
    assert len(seen) == len(set(seen))


# ── Completion detection ──────────────────────────────────────────────────────

def test_until_completes_when_all_scheduled_evaluated():
    out = _until(schedule=_ALL)
    assert out["completed"] is True
    assert out["evaluated_indices"] >= {0, 1, 2, 3}


def test_until_completed_true_sets_boolean():
    assert _until(schedule=_ALL)["completed"] is True


def test_until_all_scheduled_in_evaluated_when_complete():
    out = _until(schedule=_ALL)
    assert {0, 1, 2, 3} <= out["evaluated_indices"]


# ── max_duration_s timeout ────────────────────────────────────────────────────

def test_until_timeout_returns_completed_false():
    # max_duration_s=1.0 < first threshold (2.45 s) → no targets evaluated.
    out = _until(schedule=_ALL, max_dur=1.0)
    assert out["completed"] is False


def test_until_timeout_no_events_when_too_short():
    out = _until(schedule=_ALL, max_dur=1.0)
    assert out["events"] == []


def test_until_timeout_partial_evaluation():
    # max_duration_s=3.0 → targets 0 and 1 evaluated (thresholds 2.45, 2.95),
    # targets 2 and 3 (thresholds 3.45, 3.95) not yet reached.
    out = _until(schedule=_ALL, max_dur=3.0)
    assert out["completed"] is False
    evaluated = out["evaluated_indices"]
    assert 0 in evaluated
    assert 1 in evaluated
    assert 2 not in evaluated
    assert 3 not in evaluated


def test_until_timeout_duration_at_most_max():
    out = _until(schedule=_ALL, max_dur=1.0)
    assert out["duration_s"] <= 1.0 + 1e-9


def test_sim_short_duration_no_events():
    # duration_s=2.0 < threshold for target 0 (2.45 s)
    out = _sim(2.0, schedule=_ALL)
    assert out["events"] == []


# ── Tick count ────────────────────────────────────────────────────────────────

def test_sim_tick_count_formula():
    # tick_s=0.5, duration_s=1.0: ticks at 0.0, 0.5, 1.0 → count=3
    out = _sim(1.0, tick_s=0.5, schedule={})
    assert out["tick_count"] == 3


def test_sim_tick_count_at_least_one():
    # Even duration_s=0 must run one tick (at t=0).
    out = _sim(0.0, tick_s=0.5, schedule={})
    assert out["tick_count"] == 1


def test_sim_duration_s_in_return_equals_parameter():
    out = _sim(7.3, schedule={})
    assert out["duration_s"] == 7.3


def test_until_tick_count_positive():
    assert _until(schedule={})["tick_count"] >= 1


# ── Coarse vs fine tick sizes ────────────────────────────────────────────────

def test_coarse_and_fine_ticks_produce_same_events():
    fine   = _until(schedule=_ALL, tick_s=0.05)
    coarse = _until(schedule=_ALL, tick_s=0.5)
    fine_idxs   = {e["target_idx"] for e in fine["events"]}
    coarse_idxs = {e["target_idx"] for e in coarse["events"]}
    assert fine_idxs == coarse_idxs == {0, 1, 2, 3}


def test_coarse_and_fine_ticks_same_evaluated():
    fine   = _until(schedule=_ALL, tick_s=0.05)
    coarse = _until(schedule=_ALL, tick_s=0.5)
    assert fine["evaluated_indices"] == coarse["evaluated_indices"]


def test_very_coarse_tick_catches_all_same_tick():
    # tick_s=2.0: ticks at 0.0 and 2.0 and 4.0.
    # At t=4.0 all four targets (thresholds ≤ 3.95) become ready at once.
    out = _until(schedule=_ALL, tick_s=2.0)
    assert out["completed"] is True
    assert {e["target_idx"] for e in out["events"]} == {0, 1, 2, 3}


def test_same_tick_events_in_target_index_order():
    # Same coarse tick catches multiple targets simultaneously.
    out = _until(schedule=_ALL, tick_s=2.0)
    indices = [e["target_idx"] for e in out["events"]]
    assert indices == sorted(indices)


# ── Dense targets close together ─────────────────────────────────────────────

def test_dense_targets_all_emitted():
    # Targets at beats 0, 0.25, 0.5, 0.75 — gap = 0.25 beats = 0.125 s.
    # win_end_offset = min(0.35, 0.6×0.125) = 0.075; thresholds ≈ 2.225..2.6 s.
    dense = [
        {"time": 0.00, "note": "D2"},
        {"time": 0.25, "note": "F2"},
        {"time": 0.50, "note": "A2"},
        {"time": 0.75, "note": "C2"},
    ]
    schedule = {i: _GOOD for i in range(4)}
    out = simulate_until_complete(
        dense, _BPM, _COUNT_IN, _SR, schedule,
        tick_s=0.5, max_duration_s=10.0,
    )
    assert out["completed"] is True
    assert len(out["events"]) == 4
    assert [e["target_idx"] for e in out["events"]] == [0, 1, 2, 3]


def test_dense_targets_no_duplicates():
    dense = [{"time": i * 0.25, "note": "D2"} for i in range(4)]
    schedule = {i: _GOOD for i in range(4)}
    out = simulate_until_complete(
        dense, _BPM, _COUNT_IN, _SR, schedule,
        tick_s=0.1, max_duration_s=10.0,
    )
    seen = [e["target_idx"] for e in out["events"]]
    assert len(seen) == len(set(seen))


# ── Large count-in values ─────────────────────────────────────────────────────

def test_large_count_in_all_emitted():
    # count_in=16 → count_in_s=8.0; target[0] threshold ≈ 8.45 s.
    out = simulate_until_complete(
        _TARGETS, _BPM, 16, _SR, _ALL,
        tick_s=0.5, max_duration_s=15.0,
    )
    assert out["completed"] is True
    assert len(out["events"]) == 4


def test_large_count_in_order_preserved():
    out = simulate_until_complete(
        _TARGETS, _BPM, 16, _SR, _ALL,
        tick_s=0.5, max_duration_s=15.0,
    )
    assert [e["target_idx"] for e in out["events"]] == [0, 1, 2, 3]


def test_large_count_in_too_short_timeout():
    # max_duration_s=5.0 < first threshold at 8.45 s → timeout.
    out = simulate_until_complete(
        _TARGETS, _BPM, 16, _SR, _ALL,
        tick_s=0.5, max_duration_s=5.0,
    )
    assert out["completed"] is False
    assert out["events"] == []


# ── Simulation stops at duration_s / max_duration_s ──────────────────────────

def test_sim_does_not_emit_events_after_duration():
    # duration_s=2.5: target 0 (threshold 2.45) is caught at t=2.5.
    # target 1 (threshold 2.95) is NOT caught.
    out = _sim(2.5, schedule=_ALL)
    idxs = {e["target_idx"] for e in out["events"]}
    assert 0 in idxs
    assert 1 not in idxs


def test_sim_all_evaluated_in_evaluated_indices():
    out = _sim(5.0, schedule=_ALL)
    assert out["evaluated_indices"] == {0, 1, 2, 3}


def test_until_duration_s_reflects_actual_run_time():
    # On completion, duration_s is the time of the last processed tick.
    out = _until(schedule=_ALL, tick_s=0.5)
    # Last target (3) threshold 3.95 → caught at t=4.0.
    assert abs(out["duration_s"] - 4.0) < 1e-9
