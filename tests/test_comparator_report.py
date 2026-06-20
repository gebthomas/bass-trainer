"""Tests for comparator report HTML structure."""

import numpy as np
import pytest

from pocket_lab.comparator_report import render_comparator_report
from pocket_lab.match_record import (
    ComparisonResult,
    MatchCategory,
    MatchRecord,
    OnsetRecord,
)


def _onset(time_s=0.0, strength=0.5, amp=-12.0, take="A", idx=0):
    return OnsetRecord(
        time_s=time_s, strength=strength, amplitude_db=amp,
        raw_time_s=time_s, take_label=take, onset_index=idx,
    )


@pytest.fixture()
def comparator_html():
    sr = 44100
    audio_a = np.zeros(sr * 2, dtype=np.float32)
    audio_b = np.zeros(sr * 2, dtype=np.float32)
    result = ComparisonResult(
        take_a_path="take_a.wav",
        take_b_path="take_b.wav",
        alignment_offset_s=0.1,
        alignment_confidence=0.95,
        sample_rate=sr,
        matches=[
            MatchRecord(
                category=MatchCategory.MATCHED,
                onset_a=_onset(time_s=0.5, amp=-10.0, idx=0),
                onset_b=_onset(time_s=0.51, take="B", amp=-12.0, idx=0),
                timing_diff_ms=-10.0,
                amplitude_diff_db=2.0,
            ),
            MatchRecord(
                category=MatchCategory.A_ONLY,
                onset_a=_onset(time_s=1.0, idx=1),
            ),
            MatchRecord(
                category=MatchCategory.B_ONLY,
                onset_b=_onset(time_s=1.2, take="B", idx=1),
            ),
            MatchRecord(
                category=MatchCategory.AMBIGUOUS,
                onset_a=_onset(time_s=1.5, idx=2),
                candidates_b=[
                    _onset(time_s=1.48, take="B", idx=2),
                    _onset(time_s=1.52, take="B", idx=3),
                ],
            ),
            MatchRecord(
                category=MatchCategory.NOISE,
                onset_a=_onset(time_s=0.1, strength=0.02, idx=3),
            ),
        ],
    )
    sync_info = {
        "alignment_offset_s": 0.1,
        "alignment_confidence": 0.95,
        "window_start_a": 0.0,
        "window_start_b": -0.1,
        "bass_a_samples": sr * 2,
        "bass_b_samples": sr * 2,
        "bass_a_duration": 2.0,
        "bass_b_duration": 2.0,
        "song_duration": 2.0,
    }
    return render_comparator_report(
        result=result,
        bass_audio_a=audio_a,
        bass_audio_b=audio_b,
        sr=sr,
        duration=2.0,
        audio_src_a="take_a_bass.wav",
        audio_src_b="take_b_bass.wav",
        audio_src_song="song.wav",
        audio_src_song_b="song_b.wav",
        audio_src_stereo_a="stereo_a.wav",
        audio_src_stereo_b="stereo_b.wav",
        sync_info=sync_info,
    )


class TestComparatorHTML:
    def test_title(self, comparator_html):
        assert "<title>Take Comparator</title>" in comparator_html

    def test_audio_elements(self, comparator_html):
        assert 'id="audio-bass-a"' in comparator_html
        assert 'id="audio-bass-b"' in comparator_html
        assert 'id="audio-song"' in comparator_html
        assert 'id="audio-song-b"' in comparator_html
        assert 'id="audio-stereo-a"' in comparator_html
        assert 'id="audio-stereo-b"' in comparator_html

    def test_channel_toggles(self, comparator_html):
        assert 'data-ch="bass_a"' in comparator_html
        assert 'data-ch="bass_b"' in comparator_html
        assert 'data-ch="song"' in comparator_html
        assert 'data-ch="stereo_a"' in comparator_html
        assert 'data-ch="stereo_b"' in comparator_html

    def test_preset_buttons(self, comparator_html):
        assert 'id="preset-stereo-a"' in comparator_html
        assert 'id="preset-stereo-b"' in comparator_html
        assert 'id="preset-ab"' in comparator_html
        assert 'id="preset-abs"' in comparator_html
        assert 'id="preset-sync-check"' in comparator_html

    def test_speed_controls(self, comparator_html):
        assert 'id="speed-label"' in comparator_html
        for rate in ["1.0", "0.75", "0.5"]:
            assert f'data-speed="{rate}"' in comparator_html

    def test_loop_controls(self, comparator_html):
        for eid in ["loop-toggle", "loop-set-start", "loop-set-end",
                     "loop-reset", "loop-restart"]:
            assert f'id="{eid}"' in comparator_html

    def test_sync_diagnostic(self, comparator_html):
        assert "Sync Diagnostic" in comparator_html
        assert "+0.1000s" in comparator_html
        assert "Window start in A" in comparator_html
        assert "Window start in B" in comparator_html

    def test_summary_stats(self, comparator_html):
        assert "matched" in comparator_html
        assert "A-only" in comparator_html
        assert "B-only" in comparator_html

    def test_waveform_svgs(self, comparator_html):
        assert 'id="svg-waveform-a"' in comparator_html
        assert 'id="svg-waveform-b"' in comparator_html

    def test_comparison_timeline(self, comparator_html):
        assert 'id="svg-comparison"' in comparator_html

    def test_disagreement_cards(self, comparator_html):
        assert "disagreement-card" in comparator_html
        assert "category-badge" in comparator_html

    def test_category_badges_present(self, comparator_html):
        for cat in ["matched", "a-only", "b-only", "ambiguous", "noise"]:
            assert f"category-badge {cat}" in comparator_html

    def test_filter_buttons(self, comparator_html):
        assert "filter-btn" in comparator_html
        assert 'data-category="matched"' in comparator_html
        assert 'data-category="noise"' in comparator_html

    def test_quick_filter_presets(self, comparator_html):
        assert 'id="qf-all"' in comparator_html
        assert 'id="qf-disagreements"' in comparator_html
        assert 'id="qf-matched"' in comparator_html

    def test_zoom_button(self, comparator_html):
        assert 'id="zoom-event-btn"' in comparator_html
        assert "Zoom to event" in comparator_html

    def test_matched_detail_table(self, comparator_html):
        assert "Matched Notes (1)" in comparator_html

    def test_keyboard_shortcuts(self, comparator_html):
        assert "Keyboard Shortcuts" in comparator_html
        assert "Toggle A bass" in comparator_html
        assert "Zoom/loop" in comparator_html

    def test_tooltip_div(self, comparator_html):
        assert 'id="tooltip"' in comparator_html

    def test_js_sync(self, comparator_html):
        assert "syncAll" in comparator_html
        assert "applyChannels" in comparator_html
        assert "playActive" in comparator_html

    def test_no_align_offset_in_js(self, comparator_html):
        assert "alignOffset" not in comparator_html

    def test_loop_region_rects(self, comparator_html):
        assert comparator_html.count('class="loop-region"') >= 3

    def test_cursor_lines(self, comparator_html):
        assert comparator_html.count('class="cursor"') >= 3

    def test_noise_hidden_by_default(self, comparator_html):
        noise_filter = 'data-category="noise"'
        idx = comparator_html.index(noise_filter)
        preceding = comparator_html[max(0, idx - 100):idx]
        assert "active" not in preceding or 'filter-btn"' in preceding
