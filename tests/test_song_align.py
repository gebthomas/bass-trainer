"""Tests for song-channel cross-correlation alignment."""

import numpy as np
import pytest

from pocket_lab.song_align import align_song_channels


class TestAlignSongChannels:
    def test_identical_signals_zero_offset(self):
        rng = np.random.default_rng(42)
        sr = 44100
        signal = rng.standard_normal(sr * 2).astype(np.float32)
        offset, conf = align_song_channels(signal, signal, sr)
        assert abs(offset) < 0.002
        assert conf > 0.9

    def test_known_positive_offset(self):
        rng = np.random.default_rng(42)
        sr = 44100
        signal = rng.standard_normal(sr * 4).astype(np.float32)
        delay_samples = int(0.5 * sr)
        song_a = signal[: sr * 3]
        song_b = signal[delay_samples: delay_samples + sr * 3]
        offset, conf = align_song_channels(song_a, song_b, sr)
        assert abs(offset - 0.5) < 0.01
        assert conf > 0.7

    def test_known_negative_offset(self):
        rng = np.random.default_rng(42)
        sr = 44100
        signal = rng.standard_normal(sr * 4).astype(np.float32)
        delay_samples = int(0.5 * sr)
        song_a = signal[delay_samples: delay_samples + sr * 3]
        song_b = signal[: sr * 3]
        offset, conf = align_song_channels(song_a, song_b, sr)
        assert abs(offset - (-0.5)) < 0.01
        assert conf > 0.7

    def test_noise_robustness(self):
        rng = np.random.default_rng(42)
        sr = 44100
        signal = rng.standard_normal(sr * 3).astype(np.float32)
        delay_samples = int(0.2 * sr)
        song_a = signal[: sr * 2]
        song_b = signal[delay_samples: delay_samples + sr * 2]
        noise = rng.standard_normal(len(song_b)).astype(np.float32) * 0.3
        song_b_noisy = song_b + noise
        offset, conf = align_song_channels(song_a, song_b_noisy, sr)
        assert abs(offset - 0.2) < 0.02

    def test_gain_difference(self):
        rng = np.random.default_rng(42)
        sr = 44100
        signal = rng.standard_normal(sr * 3).astype(np.float32)
        delay_samples = int(0.3 * sr)
        song_a = signal[: sr * 2]
        song_b = signal[delay_samples: delay_samples + sr * 2] * 0.5
        offset, conf = align_song_channels(song_a, song_b, sr)
        assert abs(offset - 0.3) < 0.02

    def test_empty_inputs(self):
        offset, conf = align_song_channels(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            44100,
        )
        assert offset == 0.0
        assert conf == 0.0

    def test_max_offset_clamp(self):
        rng = np.random.default_rng(42)
        sr = 44100
        signal = rng.standard_normal(sr * 10).astype(np.float32)
        delay_samples = int(3.0 * sr)
        song_a = signal[: sr * 5]
        song_b = signal[delay_samples: delay_samples + sr * 5]
        offset, conf = align_song_channels(song_a, song_b, sr, max_offset_s=1.0)
        assert abs(offset) <= 1.1

    def test_low_sample_rate(self):
        rng = np.random.default_rng(42)
        sr = 8000
        signal = rng.standard_normal(sr * 2).astype(np.float32)
        offset, conf = align_song_channels(signal, signal, sr)
        assert abs(offset) < 0.002
        assert conf > 0.9
