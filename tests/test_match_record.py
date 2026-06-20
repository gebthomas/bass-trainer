"""Tests for match_record data structures."""

import pytest

from pocket_lab.match_record import (
    ComparisonResult,
    MatchCategory,
    MatchRecord,
    OnsetRecord,
)


def _onset(time_s=0.0, strength=0.5, amp=-12.0, take="A", idx=0, **kw):
    return OnsetRecord(
        time_s=time_s, strength=strength, amplitude_db=amp,
        raw_time_s=time_s, take_label=take, onset_index=idx, **kw,
    )


class TestOnsetRecord:
    def test_pitch_fields_default_none(self):
        o = _onset()
        assert o.pitch_hz is None
        assert o.pitch_note is None
        assert o.pitch_confidence is None

    def test_pitch_fields_settable(self):
        o = _onset(pitch_hz=82.4, pitch_note="E2", pitch_confidence=0.95)
        assert o.pitch_hz == 82.4
        assert o.pitch_note == "E2"
        assert o.pitch_confidence == 0.95


class TestMatchRecord:
    def test_matched_time_s(self):
        r = MatchRecord(
            category=MatchCategory.MATCHED,
            onset_a=_onset(time_s=1.0),
            onset_b=_onset(time_s=1.01, take="B"),
        )
        assert r.time_s == 1.0

    def test_b_only_time_s(self):
        r = MatchRecord(
            category=MatchCategory.B_ONLY,
            onset_b=_onset(time_s=2.5, take="B"),
        )
        assert r.time_s == 2.5

    def test_empty_time_s(self):
        r = MatchRecord(category=MatchCategory.NOISE)
        assert r.time_s == 0.0

    def test_candidates_default_empty(self):
        r = MatchRecord(category=MatchCategory.A_ONLY, onset_a=_onset())
        assert r.candidates_b == []


class TestComparisonResult:
    @pytest.fixture()
    def result(self):
        return ComparisonResult(
            take_a_path="a.wav",
            take_b_path="b.wav",
            alignment_offset_s=0.1,
            alignment_confidence=0.95,
            sample_rate=44100,
            matches=[
                MatchRecord(
                    category=MatchCategory.MATCHED,
                    onset_a=_onset(time_s=1.0, amp=-10.0),
                    onset_b=_onset(time_s=1.005, take="B", amp=-12.0),
                    timing_diff_ms=-5.0,
                    amplitude_diff_db=2.0,
                ),
                MatchRecord(
                    category=MatchCategory.MATCHED,
                    onset_a=_onset(time_s=2.0, amp=-11.0),
                    onset_b=_onset(time_s=2.01, take="B", amp=-11.5),
                    timing_diff_ms=-10.0,
                    amplitude_diff_db=0.5,
                ),
                MatchRecord(category=MatchCategory.A_ONLY, onset_a=_onset(time_s=3.0)),
                MatchRecord(category=MatchCategory.B_ONLY, onset_b=_onset(time_s=4.0, take="B")),
                MatchRecord(category=MatchCategory.AMBIGUOUS, onset_a=_onset(time_s=5.0)),
                MatchRecord(category=MatchCategory.NOISE, onset_a=_onset(time_s=6.0)),
            ],
        )

    def test_matched_count(self, result):
        assert result.matched_count == 2

    def test_a_only_count(self, result):
        assert result.a_only_count == 1

    def test_b_only_count(self, result):
        assert result.b_only_count == 1

    def test_ambiguous_count(self, result):
        assert result.ambiguous_count == 1

    def test_noise_count(self, result):
        assert result.noise_count == 1

    def test_timing_diffs(self, result):
        assert result.timing_diffs_ms == [-5.0, -10.0]

    def test_amplitude_diffs(self, result):
        assert result.amplitude_diffs_db == [2.0, 0.5]
