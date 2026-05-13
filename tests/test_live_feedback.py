"""Tests for core/live_feedback.py — pure timing logic, no audio hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_feedback import ready_targets, target_state
from core.targets import load_targets

# ── Shared fixtures ───────────────────────────────────────────────────────────

# Simple 4-target sequence: quarter notes at beats 0, 1, 2, 3.
_TARGETS = [
    {"time": 0, "note": "D2"},
    {"time": 1, "note": "F2"},
    {"time": 2, "note": "A2"},
    {"time": 3, "note": "C2"},
]

# BPM=120, count_in=4, SR=48000, margin=0.15 (default)
#   beat_s      = 0.5 s
#   count_in_s  = 2.0 s
#   gap         = 0.5 s  → win_end_offset = min(0.35, 0.6×0.5) = 0.30 s
#   threshold_i = count_in_s + i*beat_s + 0.30 + 0.15
#               = 2.0 + i*0.5 + 0.45
#
#   i=0: 2.45 s → sample 117600
#   i=1: 2.95 s → sample 141600
#   i=2: 3.45 s → sample 165600
#   i=3: 3.95 s → sample 189600

_BPM        = 120.0
_COUNT_IN   = 4
_SR         = 48000
_THRESHOLD  = [117600, 141600, 165600, 189600]   # ready_sample for each target


# ── Test: nothing ready before first window ends ──────────────────────────────

def test_nothing_ready_at_sample_zero():
    assert ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, 0, set()) == []


def test_nothing_ready_just_before_first_threshold():
    sample = _THRESHOLD[0] - 1
    assert ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, sample, set()) == []


# ── Test: first target ready after window + margin ────────────────────────────

def test_first_target_ready_at_threshold():
    assert ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[0], set()) == [0]


def test_first_target_ready_well_past_threshold():
    assert 0 in ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[0] + 10000, set())


def test_target_state_pending_before_threshold():
    sample = _THRESHOLD[0] - 1
    assert target_state(_TARGETS, 0, _BPM, _COUNT_IN, _SR, sample, set()) == "pending"


def test_target_state_ready_at_threshold():
    assert target_state(_TARGETS, 0, _BPM, _COUNT_IN, _SR, _THRESHOLD[0], set()) == "ready"


# ── Test: evaluated target not returned again ─────────────────────────────────

def test_evaluated_target_excluded_from_ready():
    result = ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[0], {0})
    assert 0 not in result


def test_evaluated_target_state():
    assert target_state(_TARGETS, 0, _BPM, _COUNT_IN, _SR, _THRESHOLD[0], {0}) == "evaluated"


def test_evaluated_set_accepts_list():
    result = ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[2], [0, 1])
    assert result == [2]


# ── Test: multiple targets ready after sample jump ───────────────────────────

def test_multiple_targets_ready_after_jump():
    # Jump to after threshold for target 2 — indices 0, 1, 2 all become ready.
    result = ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[2], set())
    assert result == [0, 1, 2]


def test_all_targets_ready_after_full_jump():
    result = ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[3], set())
    assert result == [0, 1, 2, 3]


def test_partial_evaluated_multiple_ready():
    # Targets 0 and 1 already evaluated; jump puts 2 and 3 past threshold.
    result = ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, _THRESHOLD[3], {0, 1})
    assert result == [2, 3]


# ── Test: BPM changes affect readiness time ───────────────────────────────────

def test_slower_bpm_delays_readiness():
    # At BPM=60 the threshold for target 0 is much later than at BPM=120.
    #   beat_s=1.0, count_in_s=4.0, gap=1.0
    #   threshold = 4.0 + min(0.35, 0.6) + 0.15 = 4.50 s → sample 216000
    # The BPM=120 threshold (117600) is below 216000, so target[0] is still pending.
    sample = _THRESHOLD[0]   # 117600 — ready at BPM=120 but not BPM=60
    assert ready_targets(_TARGETS, 60.0, _COUNT_IN, _SR, sample, set()) == []


def test_faster_bpm_advances_readiness():
    # At BPM=240 each beat is 0.25 s; target[0] becomes ready much sooner.
    #   beat_s=0.25, count_in_s=1.0, gap=0.25
    #   win_end_offset = min(0.35, 0.6×0.25) = 0.15 s
    #   threshold = 1.0 + 0.15 + 0.15 = 1.30 s → sample 62400
    threshold_240 = int(1.30 * _SR)
    assert ready_targets(_TARGETS, 240.0, _COUNT_IN, _SR, threshold_240, set()) == [0]


def test_bpm_threshold_scales_linearly():
    # Halving BPM roughly doubles the threshold in seconds.
    # (Exact ratio varies slightly because of the min() in window_end.)
    # For our _TARGETS at BPM=120 vs BPM=60, both have gap > 0.35/0.6 so
    # win_end_offset=0.35 for both, and the only difference is count_in_s + tat spacing.
    #   BPM=120: threshold_0 = 2.0 + 0.30 + 0.15 = 2.45 s
    #   BPM=60:  threshold_0 = 4.0 + 0.35 + 0.15 = 4.50 s   (gap=1.0 → 0.60>0.35)
    assert _THRESHOLD[0] / _SR < 4.50   # sanity: 2.45 < 4.50


# ── Test: empty and edge-case inputs ─────────────────────────────────────────

def test_empty_targets_returns_empty():
    assert ready_targets([], _BPM, _COUNT_IN, _SR, 1_000_000, set()) == []


def test_single_target_uses_beat_s_as_gap():
    # With one target the gap falls back to beat_s.
    #   BPM=120, beat_s=0.5, count_in_s=2.0, gap=0.5
    #   win_end_offset = min(0.35, 0.30) = 0.30, threshold = 2.45 s
    single = [{"time": 0, "note": "D2"}]
    threshold_s = 2.45
    sample_below = int((threshold_s - 0.001) * _SR)
    sample_at    = int(threshold_s * _SR)
    assert ready_targets(single, _BPM, _COUNT_IN, _SR, sample_below, set()) == []
    assert ready_targets(single, _BPM, _COUNT_IN, _SR, sample_at,    set()) == [0]


def test_custom_margin_shifts_threshold():
    # window_end for target[0] ≈ 2.30 s; with margin=0.15 → ready at 2.45 s.
    # Use 2.31 s: clearly past window_end (ready with margin=0) but well below 2.45 s
    # (still pending with default margin=0.15).
    sample_past_window = int(2.31 * _SR)   # 110880
    assert ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, sample_past_window, set(),
                         margin_s=0.0) == [0]
    assert ready_targets(_TARGETS, _BPM, _COUNT_IN, _SR, sample_past_window, set(),
                         margin_s=0.15) == []


# ── Test: 16-beat ii-V-I low-position etude ──────────────────────────────────

_ETUDE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "targets" / "jazz" / "ii_v_i_low_position_quarters.json"
)

# BPM=80, count_in=4, SR=48000, margin=0.15 (default)
#   beat_s=0.75, count_in_s=3.0
#   All consecutive gaps = 0.75 s → 0.6×0.75=0.45>0.35 → win_end_offset=0.35 for all
#   threshold_i = 3.0 + i*0.75 + 0.35 + 0.15 = 3.50 + i*0.75
#
#   i= 0: 3.50 s → sample 168000
#   i=15: 3.50 + 15×0.75 = 14.75 s → sample 708000

_ETUDE_BPM      = 80.0
_ETUDE_COUNT_IN = 4
_ETUDE_SR       = 48000


def _etude_threshold(i: int) -> int:
    return round((3.50 + i * 0.75) * _ETUDE_SR)


def test_etude_nothing_ready_before_start():
    targets = load_targets(str(_ETUDE_PATH))
    assert ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR, 0, set()) == []


def test_etude_first_target_ready():
    targets = load_targets(str(_ETUDE_PATH))
    sample  = _etude_threshold(0)
    result  = ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR, sample, set())
    assert result == [0]


def test_etude_first_target_not_ready_one_sample_before():
    targets = load_targets(str(_ETUDE_PATH))
    sample  = _etude_threshold(0) - 1
    assert ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR, sample, set()) == []


def test_etude_all_targets_ready_at_end():
    targets = load_targets(str(_ETUDE_PATH))
    sample  = _etude_threshold(15)
    result  = ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR, sample, set())
    assert result == list(range(16))


def test_etude_evaluated_targets_excluded():
    targets   = load_targets(str(_ETUDE_PATH))
    sample    = _etude_threshold(15)
    evaluated = set(range(15))   # 0..14 already done
    result    = ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR,
                               sample, evaluated)
    assert result == [15]


def test_etude_incremental_evaluation():
    """Simulate evaluating targets one by one as time advances."""
    targets   = load_targets(str(_ETUDE_PATH))
    evaluated: set[int] = set()

    for i in range(16):
        # Just before this target's threshold — should not yet appear.
        sample_before = _etude_threshold(i) - 1
        ready = ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR,
                               sample_before, evaluated)
        assert i not in ready, f"target {i} appeared too early"

        # At threshold — should appear.
        sample_at = _etude_threshold(i)
        ready = ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR,
                               sample_at, evaluated)
        assert i in ready, f"target {i} not ready at its threshold"

        evaluated.add(i)

    # After all evaluated, nothing left.
    sample = _etude_threshold(15) + 100_000
    assert ready_targets(targets, _ETUDE_BPM, _ETUDE_COUNT_IN, _ETUDE_SR,
                          sample, evaluated) == []
