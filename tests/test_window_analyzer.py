"""Tests for core/window_analyzer.py — synthetic signals, no audio hardware."""

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.window_analyzer import evaluate_window


# ── Helpers ───────────────────────────────────────────────────────────────────

_SR = 1000   # 1 kHz — keeps timing math simple (onset_sample / 1000 = seconds)


def _eval(audio, sr=_SR, **kw):
    return evaluate_window(audio, sr, **kw)


# ── Return structure ──────────────────────────────────────────────────────────

def test_return_keys_present():
    out = _eval(np.zeros(100))
    for k in ("detected", "rms", "peak", "onset_found", "onset_sample", "onset_time_s"):
        assert k in out, f"missing key: {k!r}"


def test_detected_is_bool():
    assert type(_eval(np.zeros(100))["detected"]) is bool


def test_rms_is_float():
    assert type(_eval(np.zeros(100))["rms"]) is float


def test_peak_is_float():
    assert type(_eval(np.zeros(100))["peak"]) is float


def test_onset_found_is_bool():
    assert type(_eval(np.zeros(100))["onset_found"]) is bool


def test_onset_sample_is_int_or_none():
    out = _eval(np.zeros(100))
    assert out["onset_sample"] is None or type(out["onset_sample"]) is int


def test_onset_sample_is_int_when_found():
    audio = np.zeros(100)
    audio[10] = 1.0
    assert type(_eval(audio)["onset_sample"]) is int


def test_onset_time_s_is_float_when_found():
    audio = np.zeros(100)
    audio[10] = 1.0
    assert type(_eval(audio)["onset_time_s"]) is float


# ── Silence ───────────────────────────────────────────────────────────────────

def test_silence_detected_false():
    assert _eval(np.zeros(100))["detected"] is False


def test_silence_rms_zero():
    assert _eval(np.zeros(100))["rms"] == 0.0


def test_silence_peak_zero():
    assert _eval(np.zeros(100))["peak"] == 0.0


def test_silence_onset_not_found():
    assert _eval(np.zeros(100))["onset_found"] is False


def test_silence_onset_sample_none():
    assert _eval(np.zeros(100))["onset_sample"] is None


def test_silence_onset_time_none():
    assert _eval(np.zeros(100))["onset_time_s"] is None


# ── RMS computation ───────────────────────────────────────────────────────────

def test_rms_constant_signal():
    # RMS of a constant value c is c.
    audio = np.full(200, 0.3)
    assert abs(_eval(audio)["rms"] - 0.3) < 1e-9


def test_rms_known_mixed_signal():
    # [0.0, 0.0, 1.0, 0.0]: mean(x^2) = 1/4 → rms = 0.5
    audio = np.array([0.0, 0.0, 1.0, 0.0])
    assert abs(_eval(audio)["rms"] - 0.5) < 1e-9


def test_rms_with_negative_values():
    # [-0.4, 0.4]: mean(x^2) = 0.16 → rms = 0.4
    audio = np.array([-0.4, 0.4])
    assert abs(_eval(audio)["rms"] - 0.4) < 1e-9


# ── Peak amplitude ────────────────────────────────────────────────────────────

def test_peak_positive_signal():
    assert abs(_eval(np.full(50, 0.7))["peak"] - 0.7) < 1e-9


def test_peak_uses_absolute_value():
    # Negative amplitude counts toward peak.
    audio = np.array([0.2, -0.9, 0.1])
    assert abs(_eval(audio)["peak"] - 0.9) < 1e-9


def test_peak_mixed_sign():
    audio = np.array([0.6, -0.3, 0.4])
    assert abs(_eval(audio)["peak"] - 0.6) < 1e-9


# ── Detection (rms vs min_rms) ────────────────────────────────────────────────

def test_detected_true_when_rms_above_min():
    audio = np.full(100, 0.1)   # rms=0.1 > default min_rms=0.005
    assert _eval(audio)["detected"] is True


def test_detected_false_when_rms_below_min():
    audio = np.full(100, 0.001)   # rms=0.001 < 0.005
    assert _eval(audio)["detected"] is False


def test_detected_true_at_exact_min_rms():
    audio = np.full(100, 0.005)   # rms == min_rms → True (>=)
    assert _eval(audio, min_rms=0.005)["detected"] is True


def test_detected_false_just_below_min_rms():
    audio = np.full(100, 0.00499)
    assert _eval(audio, min_rms=0.005)["detected"] is False


def test_detected_independent_of_onset():
    # High peak but very low RMS (single spike in long silence) → detected=False, onset=True
    audio = np.zeros(1000)
    audio[10] = 0.5   # rms = 0.5/sqrt(1000) ≈ 0.0158 ... actually let's check
    # rms = sqrt(0.25/1000) = sqrt(0.00025) ≈ 0.01581 > 0.005 → detected=True
    # Use a longer array to keep rms below min_rms:
    audio = np.zeros(10_000)
    audio[10] = 0.5   # rms = sqrt(0.25/10000) = sqrt(0.000025) = 0.005 — exactly min_rms
    audio = np.zeros(10_001)
    audio[10] = 0.5   # rms < 0.005
    out = _eval(audio)
    assert out["detected"] is False
    assert out["onset_found"] is True


# ── Impulse onset ─────────────────────────────────────────────────────────────

def test_impulse_at_sample_zero_onset_found():
    audio = np.zeros(100)
    audio[0] = 1.0
    assert _eval(audio)["onset_found"] is True


def test_impulse_at_sample_zero_onset_sample():
    audio = np.zeros(100)
    audio[0] = 1.0
    assert _eval(audio)["onset_sample"] == 0


def test_impulse_at_sample_zero_onset_time():
    audio = np.zeros(100)
    audio[0] = 1.0
    assert _eval(audio)["onset_time_s"] == 0.0


def test_impulse_onset_sample_value():
    audio = np.zeros(100)
    audio[42] = 1.0
    assert _eval(audio)["onset_sample"] == 42


def test_impulse_onset_time_s():
    audio = np.zeros(100)
    audio[42] = 1.0
    # SR=1000: 42 / 1000 = 0.042
    assert abs(_eval(audio)["onset_time_s"] - 0.042) < 1e-9


# ── Delayed onset ─────────────────────────────────────────────────────────────

def test_delayed_onset_found():
    audio = np.zeros(500)
    audio[200] = 0.5
    assert _eval(audio)["onset_found"] is True


def test_delayed_onset_sample():
    audio = np.zeros(500)
    audio[200] = 0.5
    assert _eval(audio)["onset_sample"] == 200


def test_delayed_onset_time_s():
    audio = np.zeros(500)
    audio[200] = 0.5
    assert abs(_eval(audio)["onset_time_s"] - 0.2) < 1e-9


def test_first_crossing_used_not_last():
    # Two crossings: first at 30, second at 70 — onset_sample must be 30.
    audio = np.zeros(100)
    audio[30] = 0.5
    audio[70] = 0.8
    assert _eval(audio)["onset_sample"] == 30


# ── Onset threshold ───────────────────────────────────────────────────────────

def test_signal_below_threshold_no_onset():
    audio = np.full(100, 0.01)   # 0.01 < default 0.02
    assert _eval(audio)["onset_found"] is False


def test_signal_at_exactly_threshold_sustained_not_onset():
    # A constant signal at the absolute threshold is sustained resonance, not a new
    # onset.  The dynamic threshold is max(0.02, 0.02 × 2.5) = 0.05; the constant
    # signal never exceeds 0.05, so onset_found is False.
    audio = np.full(100, 0.02)
    assert _eval(audio, onset_threshold=0.02)["onset_found"] is False


def test_signal_just_below_threshold_no_onset():
    audio = np.full(100, 0.0199)
    assert _eval(audio, onset_threshold=0.02)["onset_found"] is False


def test_custom_threshold_respected():
    audio = np.full(100, 0.1)
    # With high threshold, no onset found.
    assert _eval(audio, onset_threshold=0.5)["onset_found"] is False


# ── Stereo input ──────────────────────────────────────────────────────────────

def test_stereo_detected_consistent_with_mono_mean():
    audio = np.zeros((100, 2))
    audio[10, 0] = 0.6
    audio[10, 1] = 0.4
    # Mono mean at sample 10 = 0.5 → onset found
    out = _eval(audio)
    assert out["onset_found"] is True
    assert out["onset_sample"] == 10


def test_stereo_onset_sample_matches_mono_average():
    stereo = np.zeros((200, 2))
    stereo[50, 0] = 0.8
    stereo[50, 1] = 0.6
    mono_avg = stereo.mean(axis=1)
    out_stereo = _eval(stereo)
    out_mono   = _eval(mono_avg)
    assert out_stereo["onset_sample"] == out_mono["onset_sample"]
    assert abs(out_stereo["rms"]  - out_mono["rms"])  < 1e-9
    assert abs(out_stereo["peak"] - out_mono["peak"]) < 1e-9


def test_stereo_channel_average_for_rms():
    # Channel 0: [1.0, 1.0], Channel 1: [0.0, 0.0] → mono = [0.5, 0.5] → rms = 0.5
    audio = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert abs(_eval(audio)["rms"] - 0.5) < 1e-9


def test_stereo_equal_channels_same_as_mono():
    n = 200
    signal = np.random.default_rng(42).uniform(-0.3, 0.3, n)
    mono   = signal
    stereo = np.column_stack([signal, signal])
    out_m  = _eval(mono)
    out_s  = _eval(stereo)
    assert abs(out_m["rms"]  - out_s["rms"])  < 1e-9
    assert abs(out_m["peak"] - out_s["peak"]) < 1e-9
    assert out_m["onset_sample"] == out_s["onset_sample"]


def test_stereo_onset_below_threshold_when_averaged():
    # Each channel alone would exceed threshold, but average does not.
    audio = np.zeros((100, 2))
    audio[10, 0] =  0.03
    audio[10, 1] = -0.03
    # mean at sample 10 = 0.0 → no onset
    out = _eval(audio, onset_threshold=0.02)
    assert out["onset_found"] is False


# ── Low-energy signal ─────────────────────────────────────────────────────────

def test_low_energy_not_detected():
    audio = np.full(100, 0.001)   # rms = 0.001 < 0.005
    assert _eval(audio)["detected"] is False


def test_low_energy_rms_returned():
    audio = np.full(100, 0.001)
    assert abs(_eval(audio)["rms"] - 0.001) < 1e-9


def test_low_energy_no_onset_below_threshold():
    audio = np.full(100, 0.001)   # 0.001 < 0.02 onset_threshold
    assert _eval(audio)["onset_found"] is False


# ── Empty input ───────────────────────────────────────────────────────────────

def test_empty_detected_false():
    assert _eval(np.array([]))["detected"] is False


def test_empty_rms_zero():
    assert _eval(np.array([]))["rms"] == 0.0


def test_empty_peak_zero():
    assert _eval(np.array([]))["peak"] == 0.0


def test_empty_onset_not_found():
    assert _eval(np.array([]))["onset_found"] is False


def test_empty_onset_sample_none():
    assert _eval(np.array([]))["onset_sample"] is None


def test_empty_onset_time_none():
    assert _eval(np.array([]))["onset_time_s"] is None


def test_empty_stereo_works():
    out = _eval(np.zeros((0, 2)))
    assert out["detected"]    is False
    assert out["onset_found"] is False
    assert out["rms"]         == 0.0


# ── Rise-based onset detection ────────────────────────────────────────────────
#
# These tests verify the key scenarios motivating the change from absolute-
# threshold detection to baseline-relative detection.
#
# Test SR is 1000 Hz throughout so that timing in samples == timing in ms and
# the baseline_window_s=0.015 s default covers exactly 15 samples.

class TestRiseBasedOnset:
    """Amplitude-rise detection: onset = rise above local baseline, not first
    sample above an absolute floor."""

    def test_silence_no_onset(self):
        assert _eval(np.zeros(200))["onset_found"] is False

    def test_clean_attack_from_silence_detected(self):
        # Pre-roll of silence, single impulse at sample 50.
        # baseline from samples 0–14 = 0; dyn_threshold = 0.02.
        # abs(audio[50]) = 0.5 >= 0.02 → onset found at sample 50.
        audio = np.zeros(200)
        audio[50] = 0.5
        result = _eval(audio)
        assert result["onset_found"] is True
        assert result["onset_sample"] == 50

    def test_clean_step_attack_detected_at_rise(self):
        # 30 samples of silence, then a step to 0.3 at sample 30.
        # baseline = 0; dyn_threshold = 0.02; onset at sample 30.
        audio = np.zeros(200)
        audio[30:] = 0.3
        result = _eval(audio)
        assert result["onset_found"] is True
        assert result["onset_sample"] == 30

    def test_sustained_signal_above_floor_is_not_onset(self):
        # A constant signal at 0.05 (above the 0.02 absolute floor) fills the
        # entire window.  Old code would report onset_sample=0.  New code
        # computes baseline ≈ 0.05, dyn_threshold = 0.125; signal never
        # exceeds that → onset_found = False.
        audio = np.full(200, 0.05)
        assert _eval(audio)["onset_found"] is False

    def test_sustained_louder_signal_is_not_onset(self):
        # Even a loud constant signal (0.3) is not an onset.
        audio = np.full(200, 0.3)
        assert _eval(audio)["onset_found"] is False

    def test_sustained_plus_new_attack_onset_at_attack_not_zero(self):
        # First 60 samples: sustained resonance at 0.04 (above 0.02 floor).
        # Samples 60 onward: loud new attack at 0.4.
        # baseline ≈ 0.04; dyn_threshold = max(0.02, 0.10) = 0.10.
        # onset at sample 60 (0.4 >= 0.10), NOT at sample 0.
        audio = np.zeros(200)
        audio[:60] = 0.04
        audio[60:] = 0.4
        result = _eval(audio)
        assert result["onset_found"] is True
        assert result["onset_sample"] >= 55   # never reports sample 0

    def test_sustained_plus_attack_onset_near_attack_start(self):
        # Same setup; onset should land at or very near the attack boundary.
        audio = np.zeros(200)
        audio[:60] = 0.04
        audio[60:] = 0.4
        result = _eval(audio)
        assert result["onset_sample"] <= 65   # within a few samples of sample 60

    def test_modest_rise_above_quiet_baseline_is_detected(self):
        # Low-level background (0.005, below absolute floor) for first 20 samples,
        # then silence until a modest attack at sample 60.
        # baseline ≈ 0.005; dyn_threshold = max(0.02, 0.0125) = 0.02.
        # attack = 0.06 >= 0.02 → detected.
        audio = np.zeros(200)
        audio[:20] = 0.005
        audio[60:] = 0.06
        result = _eval(audio)
        assert result["onset_found"] is True
        assert result["onset_sample"] >= 55

    def test_onset_sample_is_none_when_no_rise(self):
        audio = np.full(200, 0.04)
        assert _eval(audio)["onset_sample"] is None

    def test_onset_time_s_is_none_when_no_rise(self):
        audio = np.full(200, 0.04)
        assert _eval(audio)["onset_time_s"] is None

    def test_rms_and_peak_unaffected_by_rise_logic(self):
        # rms/peak are computed from the raw signal, independent of rise detection.
        audio = np.full(200, 0.04)
        result = _eval(audio)
        assert abs(result["rms"]  - 0.04) < 1e-9
        assert abs(result["peak"] - 0.04) < 1e-9
        assert result["onset_found"] is False   # sustained, but rms/peak still valid

    def test_custom_rise_ratio_tighter(self):
        # With rise_ratio=1.5, the attack needs to be only 1.5× baseline.
        # Sustained 0.04, attack at 0.07 (1.75× baseline) → detected.
        audio = np.zeros(200)
        audio[:30] = 0.04
        audio[60:] = 0.07
        result = _eval(audio, rise_ratio=1.5)
        # dyn_threshold = max(0.02, 0.04×1.5) = 0.06; 0.07 >= 0.06 → onset found
        assert result["onset_found"] is True
        assert result["onset_sample"] >= 55

    def test_custom_rise_ratio_looser_misses_weak_attack(self):
        # With rise_ratio=10, only a very large rise is detected.
        # Sustained 0.04, attack at 0.07 → NOT detected (0.07 < 0.04×10 = 0.40).
        audio = np.zeros(200)
        audio[:30] = 0.04
        audio[60:] = 0.07
        result = _eval(audio, rise_ratio=10.0)
        assert result["onset_found"] is False
