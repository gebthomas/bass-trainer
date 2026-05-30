"""Tests for click-generation helpers in scripts/practice_session_demo.py.

All tests are pure — no sounddevice, no audio hardware.

Test matrix
-----------
click_beat_times_s
    1.  Empty inputs (no count-in, no targets) → empty list.
    2.  Count-in only, no targets.
    3.  Targets only, no count-in.
    4.  Count-in and targets together — correct absolute times.
    5.  Result is in ascending order.
    6.  Times scale inversely with BPM (half speed → double durations).
    7.  Single count-in beat appears at t=0.
    8.  Target at beat position 0 aligns with end of count-in.
    9.  Count list length matches count_in_beats.
    10. Total list length = count_in_beats + len(target_beat_positions).

make_click_waveform
    11. Returns a float32 ndarray.
    12. Length equals int(sample_rate * duration).
    13. Peak amplitude is at most the requested volume.
    14. Waveform is audible (not all zeros).
    15. End of waveform decays to near silence (exponential envelope).
    16. Different frequencies produce different waveforms.
    17. volume=0.0 produces a silent waveform.
    18. Different durations produce correspondingly different lengths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT    = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from practice_session_demo import click_beat_times_s, make_click_waveform


# ── click_beat_times_s ────────────────────────────────────────────────────────

def test_empty_inputs_returns_empty():
    assert click_beat_times_s(120.0, 0, []) == []


def test_count_in_only():
    # 2 count-in beats at 120 BPM (beat_s=0.5s) → [0.0, 0.5]
    times = click_beat_times_s(120.0, 2, [])
    assert times == pytest.approx([0.0, 0.5])


def test_targets_only_no_count_in():
    # No count-in; targets at beat positions 0.0 and 1.0 → 0.0s and 0.5s at 120 BPM
    times = click_beat_times_s(120.0, 0, [0.0, 1.0])
    assert times == pytest.approx([0.0, 0.5])


def test_count_in_and_targets_together():
    # 2 count-in + 4 targets at 120 BPM
    # count-in:  0.0, 0.5
    # targets (count_in_s=1.0): 1.0, 1.5, 2.0, 2.5
    times = click_beat_times_s(120.0, 2, [0.0, 1.0, 2.0, 3.0])
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])


def test_times_are_ascending():
    times = click_beat_times_s(80.0, 3, [0.0, 1.0, 2.0, 3.0])
    assert times == sorted(times)


def test_times_scale_inversely_with_bpm():
    # 60 BPM is half the speed of 120 BPM → all times are doubled
    times_120 = click_beat_times_s(120.0, 2, [0.0, 1.0])
    times_60  = click_beat_times_s(60.0,  2, [0.0, 1.0])
    assert times_60 == pytest.approx([t * 2 for t in times_120])


def test_single_count_in_beat_at_zero():
    times = click_beat_times_s(120.0, 1, [])
    assert times == pytest.approx([0.0])


def test_target_at_beat_zero_aligns_with_end_of_count_in():
    # Target at beat position 0 with 2 count-in beats at 120 BPM:
    # count_in_s = 1.0s; click at 1.0 + 0.0*0.5 = 1.0s
    times = click_beat_times_s(120.0, 2, [0.0])
    assert times == pytest.approx([0.0, 0.5, 1.0])


def test_count_in_list_length_matches_count_in_beats():
    for n in (0, 1, 2, 4):
        times = click_beat_times_s(120.0, n, [])
        assert len(times) == n


def test_total_length_equals_count_in_plus_targets():
    targets = [0.0, 1.0, 2.0, 3.0]
    times   = click_beat_times_s(80.0, 3, targets)
    assert len(times) == 3 + len(targets)


# ── make_click_waveform ───────────────────────────────────────────────────────

def test_returns_float32():
    assert make_click_waveform().dtype == np.float32


def test_length_equals_int_sample_rate_times_duration():
    sr       = 44100
    duration = 0.04
    click    = make_click_waveform(freq=1000.0, duration=duration, sample_rate=sr)
    assert len(click) == int(sr * duration)


def test_peak_amplitude_within_volume():
    volume = 0.5
    click  = make_click_waveform(volume=volume)
    assert float(np.max(np.abs(click))) <= volume + 1e-5


def test_waveform_is_audible():
    click = make_click_waveform(freq=1000.0, volume=0.4)
    assert float(np.max(np.abs(click))) > 0.1


def test_end_decays_to_near_silence():
    # Envelope is exp(-t / (dur*0.25)); at t=dur it's exp(-4) ≈ 0.018.
    # The last 10% of the click should be much quieter than the peak.
    click    = make_click_waveform(freq=1000.0, duration=0.04, sample_rate=44100, volume=0.5)
    tail_rms = float(np.sqrt(np.mean(click[int(len(click) * 0.9):] ** 2)))
    assert tail_rms < 0.05


def test_different_frequencies_differ():
    c_low  = make_click_waveform(freq=800.0)
    c_high = make_click_waveform(freq=1200.0)
    assert not np.allclose(c_low, c_high)


def test_zero_volume_is_silent():
    click = make_click_waveform(volume=0.0)
    assert float(np.max(np.abs(click))) == pytest.approx(0.0, abs=1e-9)


def test_duration_controls_length():
    sr      = 44100
    c_short = make_click_waveform(duration=0.02, sample_rate=sr)
    c_long  = make_click_waveform(duration=0.05, sample_rate=sr)
    assert len(c_short) == int(sr * 0.02)
    assert len(c_long)  == int(sr * 0.05)
    assert len(c_short) < len(c_long)
