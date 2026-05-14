"""Tests for core/live_pipeline.py — no audio hardware."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_pipeline import process_realtime_audio
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
#   threshold_i = 2.45 + i×0.5 seconds
#   idx=0 → 117600 samples
#   idx=1 → 141600 samples
#   idx=2 → 165600 samples
#   idx=3 → 189600 samples

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000
_T        = [117600, 141600, 165600, 189600]

# Silent audio long enough to cover all target windows.
# Target[3] window end = round((3.5 + 0.10) × 48000) = 172800; add margin.
_AUDIO = np.zeros(200_000)

_EVAL_KEYS = ("detected", "rms", "peak", "onset_found", "onset_sample", "onset_time_s")


def _session():
    return PracticeSession(_TARGETS, _BPM, _COUNT_IN, _SR)


def _run(current_sample, audio=None, session=None):
    s = session if session is not None else _session()
    return process_realtime_audio(
        _AUDIO if audio is None else audio,
        current_sample,
        s,
    )


# ── No ready targets ──────────────────────────────────────────────────────────

def test_no_ready_targets_returns_empty_list():
    assert _run(0) == []


def test_no_ready_targets_just_before_threshold():
    assert _run(_T[0] - 1) == []


def test_no_ready_targets_returns_list_not_none():
    assert isinstance(_run(0), list)


# ── Single ready target ───────────────────────────────────────────────────────

def test_single_ready_target_one_event():
    events = _run(_T[0])
    assert len(events) == 1


def test_single_ready_target_index():
    events = _run(_T[0])
    assert events[0]["target_index"] == 0


def test_single_ready_target_has_evaluation_key():
    events = _run(_T[0])
    assert "evaluation" in events[0]


def test_single_ready_target_evaluation_keys():
    evaluation = _run(_T[0])[0]["evaluation"]
    for k in _EVAL_KEYS:
        assert k in evaluation, f"missing evaluation key: {k!r}"


def test_single_ready_target_evaluation_detected_is_bool():
    evaluation = _run(_T[0])[0]["evaluation"]
    assert isinstance(evaluation["detected"], bool)


def test_single_ready_target_evaluation_rms_is_float():
    evaluation = _run(_T[0])[0]["evaluation"]
    assert isinstance(evaluation["rms"], float)


# ── Multiple ready targets ────────────────────────────────────────────────────

def test_multiple_ready_targets_count():
    events = _run(_T[3])
    assert len(events) == 4


def test_multiple_ready_targets_indices_in_order():
    events = _run(_T[3])
    assert [e["target_index"] for e in events] == [0, 1, 2, 3]


def test_two_ready_targets():
    events = _run(_T[1])
    assert len(events) == 2
    assert [e["target_index"] for e in events] == [0, 1]


def test_each_event_has_evaluation():
    events = _run(_T[3])
    for e in events:
        assert "evaluation" in e
        for k in _EVAL_KEYS:
            assert k in e["evaluation"]


# ── Target returned only once ─────────────────────────────────────────────────

def test_target_not_returned_on_second_call():
    s = _session()
    _run(_T[0], session=s)
    second = process_realtime_audio(_AUDIO, _T[0], s)
    assert second == []


def test_target_not_returned_at_later_sample():
    s = _session()
    _run(_T[0], session=s)
    later = process_realtime_audio(_AUDIO, _T[3], s)
    assert all(e["target_index"] != 0 for e in later)


def test_sequential_calls_each_target_once():
    s = _session()
    all_events = []
    for sample in _T:
        all_events.extend(process_realtime_audio(_AUDIO, sample, s))
    indices = [e["target_index"] for e in all_events]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_session_evaluated_indices_updated():
    s = _session()
    _run(_T[0], session=s)
    assert 0 in s.evaluated_indices


# ── Evaluation attached correctly ────────────────────────────────────────────

def test_evaluation_target_index_matches_target():
    # Use audio with a spike at target[0]'s beat position.
    # Target[0] beat = 2.0 s → sample 96000; window pre_roll=0.03 → starts at 94560.
    audio = np.zeros(200_000)
    audio[96000] = 0.5
    events = process_realtime_audio(audio, _T[0], _session())
    assert events[0]["target_index"] == 0
    assert events[0]["evaluation"]["onset_found"] is True


def test_evaluation_rms_reflects_audio_content():
    # Silent audio → rms = 0, detected = False.
    events = _run(_T[0])
    assert events[0]["evaluation"]["rms"] == 0.0
    assert events[0]["evaluation"]["detected"] is False


def test_evaluation_peak_reflects_audio_content():
    # Silent audio → peak = 0.
    events = _run(_T[0])
    assert events[0]["evaluation"]["peak"] == 0.0


def test_multiple_evaluations_independent():
    # Each target gets its own evaluation dict (not the same object).
    events = _run(_T[1])   # targets 0 and 1 ready
    assert events[0]["evaluation"] is not events[1]["evaluation"]


# ── Empty audio handled ───────────────────────────────────────────────────────

def test_empty_audio_no_crash():
    events = _run(_T[0], audio=np.array([]))
    assert len(events) == 1


def test_empty_audio_evaluation_detected_false():
    events = _run(_T[0], audio=np.array([]))
    assert events[0]["evaluation"]["detected"] is False


def test_empty_audio_evaluation_rms_zero():
    events = _run(_T[0], audio=np.array([]))
    assert events[0]["evaluation"]["rms"] == 0.0


def test_empty_audio_still_returns_target_index():
    events = _run(_T[0], audio=np.array([]))
    assert events[0]["target_index"] == 0


def test_empty_audio_all_eval_keys_present():
    events = _run(_T[0], audio=np.array([]))
    for k in _EVAL_KEYS:
        assert k in events[0]["evaluation"]


# ── Empty target list ─────────────────────────────────────────────────────────

def test_empty_target_list_returns_empty():
    s = PracticeSession([], _BPM, _COUNT_IN, _SR)
    assert process_realtime_audio(_AUDIO, 1_000_000, s) == []
