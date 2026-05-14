"""Tests for core/audio_windows.py — no audio hardware."""

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio_windows import extract_target_window


# ── Shared fixtures ───────────────────────────────────────────────────────────

_BPM      = 120.0   # beat_s = 0.5
_COUNT_IN = 4       # count_in_s = 2.0
_SR       = 48000

# target["time"] = 1 beat → audio_time = 2.5 s
#   target_sample = round(2.5  × 48000) = 120000
#   start_sample  = round(2.47 × 48000) = 118560   (pre_roll=0.03)
#   end_sample    = round(2.60 × 48000) = 124800   (post_roll=0.10)
#   window length = 6240

_TARGET   = {"time": 1, "note": "F2"}
_N        = 200_000   # audio long enough to avoid clamping

_MONO     = np.arange(_N, dtype=np.float64)
_STEREO   = np.column_stack([
    np.arange(_N, dtype=np.float64),
    np.arange(_N, dtype=np.float64) + 1_000_000.0,
])

_START    = 118560
_END      = 124800
_TARGET_S = 120000
_LEN      = _END - _START   # 6240


def _extract(audio=None, target=None, bpm=_BPM, count_in=_COUNT_IN, sr=_SR, **kw):
    return extract_target_window(
        _MONO if audio is None else audio,
        _TARGET if target is None else target,
        bpm, count_in, sr, **kw,
    )


# ── Return structure ──────────────────────────────────────────────────────────

def test_return_keys_present():
    out = _extract()
    for k in ("audio", "start_sample", "end_sample", "target_sample"):
        assert k in out, f"missing key: {k!r}"


def test_start_sample_is_int():
    assert type(_extract()["start_sample"]) is int


def test_end_sample_is_int():
    assert type(_extract()["end_sample"]) is int


def test_target_sample_is_int():
    assert type(_extract()["target_sample"]) is int


# ── Exact sample calculations ─────────────────────────────────────────────────

def test_target_sample_exact():
    assert _extract()["target_sample"] == _TARGET_S


def test_start_sample_exact():
    assert _extract()["start_sample"] == _START


def test_end_sample_exact():
    assert _extract()["end_sample"] == _END


def test_window_length():
    out = _extract()
    assert len(out["audio"]) == _LEN


def test_start_less_than_end():
    out = _extract()
    assert out["start_sample"] < out["end_sample"]


# ── Mono extraction ───────────────────────────────────────────────────────────

def test_mono_audio_is_1d():
    assert _extract(audio=_MONO)["audio"].ndim == 1


def test_mono_extracted_values():
    # Ramp array: audio[i] == i, so window[j] == start_sample + j.
    out = _extract(audio=_MONO)
    npt.assert_array_equal(out["audio"], np.arange(_START, _END, dtype=np.float64))


def test_mono_window_length():
    assert len(_extract(audio=_MONO)["audio"]) == _LEN


def test_mono_first_sample_value():
    out = _extract(audio=_MONO)
    assert out["audio"][0] == _START


def test_mono_last_sample_value():
    out = _extract(audio=_MONO)
    assert out["audio"][-1] == _END - 1


# ── Stereo extraction ─────────────────────────────────────────────────────────

def test_stereo_audio_is_2d():
    assert _extract(audio=_STEREO)["audio"].ndim == 2


def test_stereo_channel_count_preserved():
    assert _extract(audio=_STEREO)["audio"].shape[1] == 2


def test_stereo_frame_count():
    assert _extract(audio=_STEREO)["audio"].shape[0] == _LEN


def test_stereo_channel_0_values():
    out = _extract(audio=_STEREO)
    npt.assert_array_equal(out["audio"][:, 0], np.arange(_START, _END, dtype=np.float64))


def test_stereo_channel_1_values():
    out = _extract(audio=_STEREO)
    npt.assert_array_equal(out["audio"][:, 1],
                           np.arange(_START + 1_000_000, _END + 1_000_000, dtype=np.float64))


def test_stereo_start_end_samples_same_as_mono():
    m = _extract(audio=_MONO)
    s = _extract(audio=_STEREO)
    assert m["start_sample"]  == s["start_sample"]
    assert m["end_sample"]    == s["end_sample"]
    assert m["target_sample"] == s["target_sample"]


# ── Clamp at beginning ────────────────────────────────────────────────────────

def test_clamp_start_at_zero():
    # count_in=0, target time=0: audio_time=0, computed start = round(-0.03×48000) = -1440 → 0
    out = extract_target_window(_MONO, {"time": 0}, _BPM, count_in_beats=0,
                                sample_rate=_SR)
    assert out["start_sample"] == 0


def test_clamp_start_end_sample_unclamped():
    # end = round(0.10 × 48000) = 4800 (within _N, no clamping)
    out = extract_target_window(_MONO, {"time": 0}, _BPM, count_in_beats=0,
                                sample_rate=_SR)
    assert out["end_sample"] == 4800


def test_clamp_start_target_sample_unclamped():
    # target_sample is never clamped — reflects true beat position
    out = extract_target_window(_MONO, {"time": 0}, _BPM, count_in_beats=0,
                                sample_rate=_SR)
    assert out["target_sample"] == 0


def test_clamp_start_audio_begins_at_sample_zero():
    out = extract_target_window(_MONO, {"time": 0}, _BPM, count_in_beats=0,
                                sample_rate=_SR)
    assert out["audio"][0] == 0.0


def test_clamp_start_window_shorter_than_full_pre_roll():
    out = extract_target_window(_MONO, {"time": 0}, _BPM, count_in_beats=0,
                                sample_rate=_SR)
    # pre-roll was clipped; full window = 0.13 s × 48000 = 6240 samples,
    # but clipped start means we get fewer than that.
    full_len = round(0.10 * _SR) + round(0.03 * _SR)   # 4800 + 1440
    assert len(out["audio"]) < full_len


# ── Clamp at end ──────────────────────────────────────────────────────────────

def test_clamp_end_at_audio_length():
    # Audio ends at 122000; computed end = 124800 → clamped
    short = _MONO[:122_000]
    out = _extract(audio=short)
    assert out["end_sample"] == 122_000


def test_clamp_end_start_sample_unchanged():
    short = _MONO[:122_000]
    out = _extract(audio=short)
    assert out["start_sample"] == _START


def test_clamp_end_audio_length_reduced():
    short = _MONO[:122_000]
    out = _extract(audio=short)
    assert len(out["audio"]) == 122_000 - _START


def test_clamp_end_target_sample_unclamped():
    short = _MONO[:100_000]   # ends before target_sample=120000
    out = _extract(audio=short)
    assert out["target_sample"] == _TARGET_S   # still 120000


def test_clamp_end_audio_truncated_correctly():
    short = _MONO[:122_000]
    out = _extract(audio=short)
    npt.assert_array_equal(out["audio"], np.arange(_START, 122_000, dtype=np.float64))


# ── Empty audio ───────────────────────────────────────────────────────────────

def test_empty_audio_mono_returns_empty():
    out = extract_target_window(np.array([]), _TARGET, _BPM, _COUNT_IN, _SR)
    assert len(out["audio"]) == 0


def test_empty_audio_start_sample_zero():
    out = extract_target_window(np.array([]), _TARGET, _BPM, _COUNT_IN, _SR)
    assert out["start_sample"] == 0


def test_empty_audio_end_sample_zero():
    out = extract_target_window(np.array([]), _TARGET, _BPM, _COUNT_IN, _SR)
    assert out["end_sample"] == 0


def test_empty_audio_target_sample_correct():
    # target_sample reflects the beat position even when audio is empty
    out = extract_target_window(np.array([]), _TARGET, _BPM, _COUNT_IN, _SR)
    assert out["target_sample"] == _TARGET_S


# ── Count-in and BPM affect timing ───────────────────────────────────────────

def test_no_count_in_earlier_window():
    out_0 = _extract(count_in=0)
    out_4 = _extract(count_in=4)
    assert out_0["start_sample"] < out_4["start_sample"]


def test_slower_bpm_later_window():
    # BPM=60: beat_s=1.0; audio_time = 4.0 + 1.0 = 5.0 s
    out_slow = _extract(bpm=60.0)
    out_fast = _extract(bpm=120.0)
    assert out_slow["target_sample"] > out_fast["target_sample"]


def test_bpm_60_target_sample():
    # beat_s=1.0, count_in=4 → count_in_s=4.0; target time=1 → audio_time=5.0
    out = _extract(bpm=60.0)
    assert out["target_sample"] == round(5.0 * _SR)


def test_window_duration_independent_of_bpm():
    # pre_roll and post_roll are in seconds, so window length is BPM-independent.
    # BPM=60: audio_time=5.0 s → need at least 5.1 s of audio (244801 samples).
    long_audio = np.arange(300_000, dtype=np.float64)
    out_slow = extract_target_window(long_audio, _TARGET, 60.0, _COUNT_IN, _SR)
    out_fast = _extract(bpm=120.0)
    assert len(out_slow["audio"]) == len(out_fast["audio"])
