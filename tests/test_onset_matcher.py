"""Tests for onset matching algorithm."""

import pytest

from pocket_lab.match_record import MatchCategory, OnsetRecord
from pocket_lab.onset_matcher import match_onsets


def _onset(time_s, strength=0.5, amp=-12.0, take="A", idx=0):
    return OnsetRecord(
        time_s=time_s, strength=strength, amplitude_db=amp,
        raw_time_s=time_s, take_label=take, onset_index=idx,
    )


class TestMatchOnsets:
    def test_perfect_match(self):
        a = [_onset(1.0, idx=0), _onset(2.0, idx=1), _onset(3.0, idx=2)]
        b = [_onset(1.0, take="B", idx=0), _onset(2.0, take="B", idx=1),
             _onset(3.0, take="B", idx=2)]
        results = match_onsets(a, b)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        assert len(matched) == 3
        for r in matched:
            assert r.timing_diff_ms == pytest.approx(0.0)

    def test_uniform_shift(self):
        a = [_onset(1.0, idx=0), _onset(2.0, idx=1)]
        b = [_onset(1.01, take="B", idx=0), _onset(2.01, take="B", idx=1)]
        results = match_onsets(a, b)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        assert len(matched) == 2
        for r in matched:
            assert r.timing_diff_ms == pytest.approx(-10.0, abs=0.1)

    def test_missing_from_b(self):
        a = [_onset(1.0, idx=0), _onset(2.0, idx=1), _onset(3.0, idx=2)]
        b = [_onset(1.0, take="B", idx=0), _onset(3.0, take="B", idx=1)]
        results = match_onsets(a, b)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        a_only = [r for r in results if r.category == MatchCategory.A_ONLY]
        assert len(matched) == 2
        assert len(a_only) == 1
        assert a_only[0].onset_a.time_s == 2.0

    def test_extra_in_b(self):
        a = [_onset(1.0, idx=0), _onset(3.0, idx=1)]
        b = [_onset(1.0, take="B", idx=0), _onset(2.0, take="B", idx=1),
             _onset(3.0, take="B", idx=2)]
        results = match_onsets(a, b)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        b_only = [r for r in results if r.category == MatchCategory.B_ONLY]
        assert len(matched) == 2
        assert len(b_only) == 1
        assert b_only[0].onset_b.time_s == 2.0

    def test_noise_filtering(self):
        a = [_onset(1.0, strength=0.02, idx=0), _onset(2.0, strength=0.5, idx=1)]
        b = [_onset(2.0, take="B", strength=0.5, idx=0)]
        results = match_onsets(a, b, noise_strength_threshold=0.08)
        noise = [r for r in results if r.category == MatchCategory.NOISE]
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        assert len(noise) == 1
        assert noise[0].onset_a.time_s == 1.0
        assert len(matched) == 1

    def test_conflict_resolution(self):
        a = [_onset(1.00, idx=0), _onset(1.03, idx=1)]
        b = [_onset(1.02, take="B", idx=0)]
        results = match_onsets(a, b, max_match_window_s=0.050)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        a_only = [r for r in results if r.category == MatchCategory.A_ONLY]
        assert len(matched) == 1
        assert matched[0].onset_a.time_s == 1.03
        assert len(a_only) == 1
        assert a_only[0].onset_a.time_s == 1.00

    def test_empty_a(self):
        b = [_onset(1.0, take="B", idx=0), _onset(2.0, take="B", idx=1)]
        results = match_onsets([], b)
        b_only = [r for r in results if r.category == MatchCategory.B_ONLY]
        assert len(b_only) == 2

    def test_empty_b(self):
        a = [_onset(1.0, idx=0), _onset(2.0, idx=1)]
        results = match_onsets(a, [])
        a_only = [r for r in results if r.category == MatchCategory.A_ONLY]
        assert len(a_only) == 2

    def test_both_empty(self):
        results = match_onsets([], [])
        assert len(results) == 0

    def test_no_cross_matching_dense_onsets(self):
        """16th notes at 120 BPM = 125ms apart. No cross-matching."""
        interval = 0.125
        a = [_onset(i * interval, idx=i) for i in range(8)]
        b = [_onset(i * interval + 0.005, take="B", idx=i) for i in range(8)]
        results = match_onsets(a, b, max_match_window_s=0.050)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        assert len(matched) == 8
        for r in matched:
            assert abs(r.timing_diff_ms) < 10

    def test_amplitude_diff_populated(self):
        a = [_onset(1.0, amp=-10.0, idx=0)]
        b = [_onset(1.0, take="B", amp=-14.0, idx=0)]
        results = match_onsets(a, b)
        matched = [r for r in results if r.category == MatchCategory.MATCHED]
        assert len(matched) == 1
        assert matched[0].amplitude_diff_db == pytest.approx(4.0)

    def test_results_sorted_by_time(self):
        a = [_onset(3.0, idx=0), _onset(1.0, idx=1)]
        b = [_onset(2.0, take="B", idx=0)]
        results = match_onsets(a, b)
        times = [r.time_s for r in results]
        assert times == sorted(times)

    def test_ambiguous_detection(self):
        a = [_onset(1.0, idx=0)]
        b = [_onset(1.02, take="B", idx=0), _onset(0.98, take="B", idx=1)]
        results = match_onsets(a, b, max_match_window_s=0.050, ambiguity_ratio=2.0)
        ambiguous = [r for r in results if r.category == MatchCategory.AMBIGUOUS]
        assert len(ambiguous) == 1
        assert len(ambiguous[0].candidates_b) == 2
