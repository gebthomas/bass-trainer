"""Tests for onset detection diagnostics."""

import numpy as np
import pytest

from pocket_lab.onset_diagnostics import (
    close_pair_counts,
    frame_quantization_info,
    full_onset_diagnostic,
    onset_spacing_stats,
    spacing_histogram,
    strength_distribution,
)


class TestOnsetSpacingStats:
    def test_basic(self):
        t = np.array([0.0, 0.1, 0.25, 0.5])
        s = onset_spacing_stats(t)
        assert s["count"] == 4
        assert s["min_spacing_ms"] == pytest.approx(100.0)
        assert s["median_spacing_ms"] == pytest.approx(150.0)

    def test_single_onset(self):
        s = onset_spacing_stats(np.array([1.0]))
        assert s["count"] == 1
        assert len(s["spacings"]) == 0

    def test_empty(self):
        s = onset_spacing_stats(np.array([]))
        assert s["count"] == 0


class TestClosePairCounts:
    def test_close_onsets(self):
        t = np.array([0.0, 0.005, 0.1, 0.115, 0.5])
        counts = close_pair_counts(t, [10.0, 20.0, 50.0])
        within_10 = counts[0]
        assert within_10["count"] == 2
        within_20 = counts[1]
        assert within_20["count"] >= 2

    def test_no_close_pairs(self):
        t = np.array([0.0, 1.0, 2.0])
        counts = close_pair_counts(t, [10.0])
        assert counts[0]["count"] == 0

    def test_empty(self):
        counts = close_pair_counts(np.array([]), [10.0])
        assert counts[0]["count"] == 0


class TestSpacingHistogram:
    def test_bins(self):
        t = np.array([0.0, 0.010, 0.040, 0.200, 0.800])
        h = spacing_histogram(t)
        assert len(h) > 0
        total = sum(b["count"] for b in h)
        assert total == 4


class TestStrengthDistribution:
    def test_basic(self):
        s = np.array([0.0, 0.005, 0.1, 0.5, 1.0])
        d = strength_distribution(s)
        assert d["count"] == 5
        assert d["zero_count"] == 1
        assert d["below_001"] == 2
        assert d["below_01"] == 2
        assert d["below_05"] == 3
        assert d["min"] == 0.0
        assert d["max"] == 1.0

    def test_empty(self):
        d = strength_distribution(np.array([]))
        assert d["count"] == 0


class TestFrameQuantization:
    def test_44100(self):
        q = frame_quantization_info(44100, 512)
        assert q["frame_period_ms"] == pytest.approx(11.61, abs=0.01)
        assert q["frames_per_second"] == pytest.approx(86.13, abs=0.1)


class TestFullDiagnostic:
    def test_runs(self):
        t = np.array([0.0, 0.05, 0.1, 0.5, 1.0])
        s = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
        d = full_onset_diagnostic(t, s, "Test", 44100)
        assert d["label"] == "Test"
        assert "spacing" in d
        assert "close_pairs" in d
        assert "histogram" in d
        assert "strength" in d
        assert "quantization" in d
