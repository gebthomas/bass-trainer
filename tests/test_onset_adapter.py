"""Tests for core/onset_adapter.py — pure, no sounddevice, no threads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.onset_adapter import OnsetAdapter
from core.session_engine import SessionEngine

# ── Fixtures / helpers ─────────────────────────────────────────────────────────

SR = 48_000  # samples per second
BLOCK = 1_024  # typical block size

def silence(n: int = BLOCK) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)

def loud_block(n: int = BLOCK, amplitude: float = 0.5) -> np.ndarray:
    """Flat loud block — RMS == amplitude."""
    return np.full(n, amplitude, dtype=np.float32)

def spike_block(n: int = BLOCK, spike_pos: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """Silence with one loud sample at *spike_pos*.

    Default amplitude 1.0 gives RMS = 1/sqrt(1024) ≈ 0.031, which clears the
    default min_rms=0.018 threshold regardless of block size up to ~3100 samples.
    """
    buf = np.zeros(n, dtype=np.float32)
    buf[spike_pos] = amplitude
    return buf


# ── 1. Silence → no onset ──────────────────────────────────────────────────────

def test_silence_produces_no_onset():
    adapter = OnsetAdapter(sample_rate=SR)
    result = adapter.process_block(0, silence())
    assert result == []


# ── 2. Loud block → onset triggered ───────────────────────────────────────────

def test_loud_block_produces_onset():
    adapter = OnsetAdapter(sample_rate=SR)
    result = adapter.process_block(0, loud_block())
    assert len(result) == 1


# ── 3. Time formula correctness ───────────────────────────────────────────────

def test_onset_time_formula():
    adapter = OnsetAdapter(sample_rate=SR)
    block_start = 5 * SR  # 5 seconds in
    # spike at position 100 inside the block
    block = spike_block(spike_pos=100)
    result = adapter.process_block(block_start, block)
    assert len(result) == 1
    expected = (block_start + 100) / SR
    assert result[0] == pytest.approx(expected, abs=1e-9)


# ── 4. Peak location within block ─────────────────────────────────────────────

def test_onset_localized_to_peak_sample():
    adapter = OnsetAdapter(sample_rate=SR)
    spike_pos = 700
    block = spike_block(spike_pos=spike_pos)
    result = adapter.process_block(0, block)
    assert len(result) == 1
    assert result[0] == pytest.approx(spike_pos / SR, abs=1e-9)


# ── 5. Refractory prevents double-trigger ─────────────────────────────────────

def test_refractory_suppresses_second_onset():
    adapter = OnsetAdapter(sample_rate=SR, refractory_s=0.150)
    # First block at t=0
    r1 = adapter.process_block(0, loud_block())
    assert len(r1) == 1

    # Second block starts 50 ms later — inside the 150 ms refractory window
    gap_samples = int(0.050 * SR)
    r2 = adapter.process_block(gap_samples, loud_block())
    assert r2 == []


# ── 6. Refractory expires → onset allowed again ───────────────────────────────

def test_refractory_expires_after_window():
    adapter = OnsetAdapter(sample_rate=SR, refractory_s=0.150)
    r1 = adapter.process_block(0, loud_block())
    assert len(r1) == 1

    # Second block starts 200 ms later — outside the 150 ms window
    gap_samples = int(0.200 * SR)
    r2 = adapter.process_block(gap_samples, loud_block())
    assert len(r2) == 1


# ── 7. Count-in onsets are ignored by SessionEngine ──────────────────────────

def test_count_in_onsets_ignored_by_engine():
    """Onsets during the count-in period fall before all target nominal times
    and should not match any target."""
    targets = [{"time": 0, "note": "A2"}, {"time": 1, "note": "A2"}]
    bpm = 120.0
    count_in_beats = 2
    # nominal time of first target = 2 beats * (60/120) = 1.0 s
    engine = SessionEngine(targets, bpm=bpm, count_in_beats=count_in_beats)

    adapter = OnsetAdapter(sample_rate=SR)

    # Simulate onset at t=0.3 s (deep in count-in)
    count_in_onset_s = 0.3
    block_start = int(count_in_onset_s * SR)
    onsets = adapter.process_block(block_start, loud_block())
    assert len(onsets) == 1

    events = []
    for t in onsets:
        events.extend(engine.on_onset(t))

    # No target should be matched — count-in onset is too early
    assert events == []
    assert len(engine.evaluated_indices) == 0


# ── 8. Full synthetic block stream → correct events ───────────────────────────

def test_full_synthetic_block_stream():
    """Drive OnsetAdapter + SessionEngine through a synthetic stream.

    Setup: 2 targets at beat 0 and beat 2 (quarter notes), 120 BPM,
    2-beat count-in.  Player hits both targets on time (within ±30 ms).
    """
    bpm = 120.0
    count_in_beats = 2
    beat_s = 60.0 / bpm  # 0.5 s

    targets = [
        {"time": 0, "note": "A2"},
        {"time": 2, "note": "D2"},
    ]
    # nominal times: beat 0 → 1.0 s, beat 2 → 2.0 s
    nom_0 = count_in_beats * beat_s + targets[0]["time"] * beat_s  # 1.0
    nom_1 = count_in_beats * beat_s + targets[1]["time"] * beat_s  # 2.0

    engine = SessionEngine(targets, bpm=bpm, count_in_beats=count_in_beats)
    adapter = OnsetAdapter(sample_rate=SR)

    hit_offsets = [+0.020, -0.015]  # slightly early / slightly late (seconds)
    onset_times_s = [nom_0 + hit_offsets[0], nom_1 + hit_offsets[1]]

    # Build block stream: 3 seconds, BLOCK-sized blocks
    session_end_sample = 3 * SR
    all_events: list[dict] = []

    block_start = 0
    while block_start <= session_end_sample:
        # Build block: mostly silence, spike at onset sample if one falls here
        block = silence(BLOCK)
        for ot in onset_times_s:
            onset_sample = int(ot * SR)
            if block_start <= onset_sample < block_start + BLOCK:
                block[onset_sample - block_start] = 1.0

        onsets = adapter.process_block(block_start, block)
        for t in onsets:
            all_events.extend(engine.on_onset(t))

        t_now = block_start / SR
        all_events.extend(engine.update_time(t_now))

        block_start += BLOCK

    hit_events = [e for e in all_events if e["detected_note"] is not None]
    miss_events = [e for e in all_events if e["detected_note"] is None]

    assert len(hit_events) == 2, f"Expected 2 hits, got {len(hit_events)}"
    assert len(miss_events) == 0, f"Expected 0 misses, got {len(miss_events)}"

    errors = sorted(e["timing_error_s"] for e in hit_events)
    assert errors[0] == pytest.approx(hit_offsets[1], abs=1 / SR)  # -0.015
    assert errors[1] == pytest.approx(hit_offsets[0], abs=1 / SR)  # +0.020


# ── 9. High-peak / low-RMS transient triggers via peak gate ───────────────────

def test_high_peak_low_rms_triggers():
    """A single sample at 0.5 in a 1024-frame block has RMS ≈ 0.0156 (below
    min_rms=0.018) but peak=0.5 (above min_peak=0.15) — onset must fire."""
    adapter = OnsetAdapter(sample_rate=SR)
    block = spike_block(amplitude=0.5)  # RMS ≈ 0.0156, peak = 0.5
    # Verify the RMS-only threshold would have suppressed this:
    rms = float(np.sqrt(np.mean(block ** 2)))
    assert rms < adapter.min_rms, "precondition: RMS alone would suppress"
    # But the peak gate should save it:
    result = adapter.process_block(0, block)
    assert len(result) == 1


# ── 10. Low-level noise does not trigger ──────────────────────────────────────

def test_low_level_noise_no_trigger():
    """White noise at max amplitude 0.01 fails both gates — no onset."""
    rng = np.random.default_rng(seed=42)
    noise = (rng.random(BLOCK).astype(np.float32) * 2 - 1) * 0.01  # peak ≤ 0.01
    adapter = OnsetAdapter(sample_rate=SR)
    # Confirm both gates fail (documents the expected signal levels):
    rms  = float(np.sqrt(np.mean(noise ** 2)))
    peak = float(np.max(np.abs(noise)))
    assert rms  < adapter.min_rms,  f"RMS={rms:.4f} should be below min_rms"
    assert peak < adapter.min_peak, f"peak={peak:.4f} should be below min_peak"
    result = adapter.process_block(0, noise)
    assert result == []
