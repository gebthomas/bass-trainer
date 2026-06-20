"""Tests for pocket_lab pure helpers."""

import numpy as np
import pytest

from pocket_lab.audio import (
    audio_diagnostics,
    compute_overview,
    segment_audio,
    window_tag,
)
from pocket_lab.grid import (
    GridLine,
    OnsetClassification,
    classify_onset_against_grid,
    make_grid,
)
from pocket_lab.grid_phase import estimate_beat_zero, filter_onsets_for_phase
from pocket_lab.grid_settings import GridSource, load_grid_settings, save_grid_settings
from pocket_lab.report import render_report


# ── make_grid tests ──────────────────────────────────────────────────────────


class TestMakeGrid:
    def test_4_4_at_146_bpm_line_count(self):
        grid = make_grid(bpm=146, beats_per_measure=4, n_measures=2)
        # 2 measures × 4 beats × 2 (beat + subdivision) = 16 lines
        assert len(grid) == 16

    def test_4_4_at_146_bpm_first_beat(self):
        grid = make_grid(bpm=146, beats_per_measure=4, n_measures=1)
        assert grid[0].time == pytest.approx(0.0)
        assert grid[0].kind == "measure"
        assert grid[0].measure == 1
        assert grid[0].beat == 1

    def test_4_4_at_146_bpm_beat_spacing(self):
        grid = make_grid(bpm=146, beats_per_measure=4, n_measures=1)
        beat_s = 60.0 / 146
        on_beats = [g for g in grid if g.subdivision == 0]
        for i, g in enumerate(on_beats):
            assert g.time == pytest.approx(i * beat_s, abs=1e-9)

    def test_4_4_at_146_bpm_shuffle_subdivisions(self):
        grid = make_grid(bpm=146, beats_per_measure=4, n_measures=1,
                         shuffle_fraction=2 / 3)
        beat_s = 60.0 / 146
        subs = [g for g in grid if g.subdivision == 1]
        assert len(subs) == 4
        for i, s in enumerate(subs):
            expected = i * beat_s + (2 / 3) * beat_s
            assert s.time == pytest.approx(expected, abs=1e-9)

    def test_4_4_at_146_bpm_measure_lines(self):
        grid = make_grid(bpm=146, beats_per_measure=4, n_measures=3)
        measures = [g for g in grid if g.kind == "measure"]
        assert len(measures) == 3
        beat_s = 60.0 / 146
        for i, m in enumerate(measures):
            assert m.time == pytest.approx(i * 4 * beat_s, abs=1e-9)
            assert m.measure == i + 1

    def test_3_4_at_120_bpm_line_count(self):
        grid = make_grid(bpm=120, beats_per_measure=3, n_measures=2)
        # 2 measures × 3 beats × 2 = 12 lines
        assert len(grid) == 12

    def test_3_4_at_120_bpm_beat_spacing(self):
        grid = make_grid(bpm=120, beats_per_measure=3, n_measures=2)
        beat_s = 60.0 / 120  # 0.5s
        on_beats = [g for g in grid if g.subdivision == 0]
        assert len(on_beats) == 6
        for i, g in enumerate(on_beats):
            assert g.time == pytest.approx(i * beat_s, abs=1e-9)

    def test_3_4_at_120_bpm_measure_boundaries(self):
        grid = make_grid(bpm=120, beats_per_measure=3, n_measures=2)
        measures = [g for g in grid if g.kind == "measure"]
        assert len(measures) == 2
        assert measures[0].time == pytest.approx(0.0)
        assert measures[1].time == pytest.approx(1.5)  # 3 beats × 0.5s

    def test_3_4_at_120_bpm_beats_per_measure(self):
        grid = make_grid(bpm=120, beats_per_measure=3, n_measures=1)
        on_beats = [g for g in grid if g.subdivision == 0]
        assert [g.beat for g in on_beats] == [1, 2, 3]

    def test_shuffle_fraction_two_thirds(self):
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1,
                         shuffle_fraction=2 / 3)
        beat_s = 0.5
        subs = [g for g in grid if g.subdivision == 1]
        for i, s in enumerate(subs):
            expected = i * beat_s + (2 / 3) * beat_s
            assert s.time == pytest.approx(expected, abs=1e-9)

    def test_straight_eighth_shuffle_fraction(self):
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1,
                         shuffle_fraction=0.5)
        beat_s = 0.5
        subs = [g for g in grid if g.subdivision == 1]
        for i, s in enumerate(subs):
            expected = i * beat_s + 0.5 * beat_s
            assert s.time == pytest.approx(expected, abs=1e-9)

    def test_offset_shifts_all_times(self):
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1, offset=2.0)
        assert grid[0].time == pytest.approx(2.0)
        on_beats = [g for g in grid if g.subdivision == 0]
        for i, g in enumerate(on_beats):
            assert g.time == pytest.approx(2.0 + i * 0.5, abs=1e-9)

    def test_sorted_by_time(self):
        grid = make_grid(bpm=146, beats_per_measure=4, n_measures=4,
                         shuffle_fraction=2 / 3)
        times = [g.time for g in grid]
        assert times == sorted(times)

    def test_all_kinds_present(self):
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=2)
        kinds = {g.kind for g in grid}
        assert kinds == {"measure", "beat", "subdivision"}


# ── classify_onset_against_grid tests ────────────────────────────────────────


class TestClassifyOnsetAgainstGrid:
    def test_exact_on_beat(self):
        beat_s = 60.0 / 146
        c = classify_onset_against_grid(beat_s * 4, bpm=146, beats_per_measure=4)
        assert c.nearest_measure == 2
        assert c.nearest_beat == 1
        assert c.offset_ms == pytest.approx(0.0, abs=0.01)
        assert c.label == "on-beat"

    def test_exact_on_beat_146_beat_2(self):
        beat_s = 60.0 / 146
        c = classify_onset_against_grid(beat_s * 1, bpm=146, beats_per_measure=4)
        assert c.nearest_measure == 1
        assert c.nearest_beat == 2
        assert c.label == "on-beat"

    def test_shuffle_hit_146(self):
        beat_s = 60.0 / 146
        onset = beat_s * 2 + (2 / 3) * beat_s
        c = classify_onset_against_grid(onset, bpm=146, beats_per_measure=4,
                                         shuffle_fraction=2 / 3)
        assert c.label == "shuffle"
        assert c.beat_fraction == pytest.approx(2 / 3, abs=0.01)

    def test_early_onset(self):
        beat_s = 60.0 / 146
        onset = beat_s * 3 - 0.04
        c = classify_onset_against_grid(onset, bpm=146, beats_per_measure=4)
        assert c.offset_ms < 0
        assert c.label == "early"

    def test_3_4_at_120_on_beat(self):
        beat_s = 60.0 / 120
        c = classify_onset_against_grid(beat_s * 3, bpm=120, beats_per_measure=3)
        assert c.nearest_measure == 2
        assert c.nearest_beat == 1
        assert c.label == "on-beat"

    def test_3_4_at_120_beat_3(self):
        beat_s = 0.5
        c = classify_onset_against_grid(beat_s * 2, bpm=120, beats_per_measure=3)
        assert c.nearest_measure == 1
        assert c.nearest_beat == 3

    def test_shuffle_fraction_two_thirds_classification(self):
        beat_s = 60.0 / 120
        onset = (2 / 3) * beat_s
        c = classify_onset_against_grid(onset, bpm=120, beats_per_measure=4,
                                         shuffle_fraction=2 / 3)
        assert c.label == "shuffle"
        assert c.beat_fraction == pytest.approx(2 / 3, abs=0.01)

    def test_between_beat_and_shuffle(self):
        beat_s = 60.0 / 120
        onset = 0.4 * beat_s
        c = classify_onset_against_grid(onset, bpm=120, beats_per_measure=4,
                                         shuffle_fraction=2 / 3)
        assert c.label == "between"

    def test_offset_parameter(self):
        beat_s = 60.0 / 146
        c = classify_onset_against_grid(
            2.0 + beat_s * 4, bpm=146, beats_per_measure=4, offset=2.0,
        )
        assert c.nearest_measure == 2
        assert c.nearest_beat == 1
        assert c.offset_ms == pytest.approx(0.0, abs=0.01)

    def test_positive_offset_ms_means_late(self):
        beat_s = 60.0 / 120
        onset = beat_s + 0.010
        c = classify_onset_against_grid(onset, bpm=120, beats_per_measure=4)
        assert c.offset_ms == pytest.approx(10.0, abs=0.5)

    def test_negative_offset_ms_means_early(self):
        beat_s = 60.0 / 120
        onset = beat_s - 0.010
        c = classify_onset_against_grid(onset, bpm=120, beats_per_measure=4)
        assert c.offset_ms == pytest.approx(-10.0, abs=0.5)


# ── estimate_beat_zero tests ─────────────────────────────────────────────────


class TestEstimateBeatZero:
    def test_4_4_at_146_anchors_1_3(self):
        beat_s = 60.0 / 146
        phase = 0.1
        onsets = np.array([
            phase + 0 * beat_s,
            phase + 2 * beat_s,
            phase + 4 * beat_s,
            phase + 6 * beat_s,
            phase + 8 * beat_s,
            phase + 10 * beat_s,
        ])
        result = estimate_beat_zero(onsets, bpm=146, beats_per_measure=4,
                                     anchor_beats=[1, 3])
        assert result == pytest.approx(phase, abs=0.01)

    def test_3_4_at_120_anchor_1(self):
        beat_s = 0.5
        measure_s = 1.5
        phase = 0.2
        onsets = np.array([phase, phase + measure_s, phase + 2 * measure_s])
        result = estimate_beat_zero(onsets, bpm=120, beats_per_measure=3,
                                     anchor_beats=[1])
        assert result == pytest.approx(phase, abs=0.01)

    def test_strong_onsets_dominate(self):
        beat_s = 60.0 / 146
        phase = 0.15
        strong_times = [phase + k * 2 * beat_s for k in range(4)]
        noise_times = [0.5, 1.1, 1.8]
        all_times = sorted(strong_times + noise_times)
        onsets = np.array(all_times)
        strengths = np.array([
            1.0 if any(abs(t - s) < 0.001 for s in strong_times) else 0.1
            for t in all_times
        ])
        result = estimate_beat_zero(onsets, bpm=146, beats_per_measure=4,
                                     anchor_beats=[1, 3],
                                     onset_strengths=strengths)
        assert result == pytest.approx(phase, abs=0.02)

    def test_empty_onsets_returns_zero(self):
        result = estimate_beat_zero(np.array([]), bpm=120,
                                     beats_per_measure=4, anchor_beats=[1])
        assert result == 0.0

    def test_empty_anchors_returns_zero(self):
        result = estimate_beat_zero(np.array([0.1, 0.5]), bpm=120,
                                     beats_per_measure=4, anchor_beats=[])
        assert result == 0.0

    def test_zero_phase(self):
        beat_s = 60.0 / 120
        onsets = np.array([0.0, 2 * beat_s, 4 * beat_s, 6 * beat_s])
        result = estimate_beat_zero(onsets, bpm=120, beats_per_measure=4,
                                     anchor_beats=[1, 3])
        assert result == pytest.approx(0.0, abs=0.01)

    def test_3_4_anchor_beat_3(self):
        beat_s = 0.5
        measure_s = 1.5
        phase = 0.3
        onsets = np.array([
            phase + 2 * beat_s,
            phase + 2 * beat_s + measure_s,
            phase + 2 * beat_s + 2 * measure_s,
        ])
        result = estimate_beat_zero(onsets, bpm=120, beats_per_measure=3,
                                     anchor_beats=[3])
        assert result == pytest.approx(phase, abs=0.01)


# ── audio_diagnostics and stereo export tests ────────────────────────────────


class TestAudioDiagnostics:
    def test_mono_signal(self):
        audio = np.full(1000, 0.5, dtype=np.float32)
        d = audio_diagnostics(audio, "test")
        assert d["channels"] == 1
        assert d["rms_dbfs"] == pytest.approx(-6.02, abs=0.1)
        assert d["peak_dbfs"] == pytest.approx(-6.02, abs=0.1)
        assert d["label"] == "test"

    def test_stereo_signal(self):
        audio = np.column_stack([
            np.full(1000, 0.25, dtype=np.float32),
            np.full(1000, 0.5, dtype=np.float32),
        ])
        d = audio_diagnostics(audio)
        assert d["channels"] == 2

    def test_empty(self):
        d = audio_diagnostics(np.array([]))
        assert d["channels"] == 0
        assert d["rms_dbfs"] is None

    def test_silence(self):
        d = audio_diagnostics(np.zeros(1000, dtype=np.float32))
        assert d["rms_dbfs"] == -120.0
        assert d["peak_dbfs"] == -120.0


class TestStereoExcerptPreservesRMS:
    def test_stereo_segment_rms_within_half_db(self, tmp_path):
        """Stereo export via segment_audio + sf.write(PCM_16) preserves RMS."""
        import soundfile as sf

        sr = 44100
        n = sr * 2
        rng = np.random.default_rng(42)
        stereo = rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32)
        source_rms = float(np.sqrt(np.mean(stereo.astype(np.float64) ** 2)))

        seg = segment_audio(stereo, sr, start=0.0, duration=2.0)

        out_path = tmp_path / "excerpt.wav"
        sf.write(str(out_path), seg, sr, subtype="PCM_16")

        reloaded, _ = sf.read(str(out_path), dtype="float32")
        reloaded_rms = float(np.sqrt(np.mean(reloaded.astype(np.float64) ** 2)))

        source_db = 20 * np.log10(source_rms)
        reloaded_db = 20 * np.log10(reloaded_rms)
        assert abs(source_db - reloaded_db) < 0.5, (
            f"RMS drift {abs(source_db - reloaded_db):.2f} dB "
            f"(source {source_db:.1f}, reloaded {reloaded_db:.1f})"
        )
        assert reloaded.shape[1] == 2


# ── filter_onsets_for_phase tests ────────────────────────────────────────────


class TestFilterOnsetsForPhase:
    def test_no_annotations_passes_all(self):
        times = np.array([0.1, 0.5, 1.0, 1.5])
        strengths = np.array([0.8, 0.6, 0.9, 0.7])
        ft, fw = filter_onsets_for_phase(times, strengths, {})
        np.testing.assert_array_equal(ft, times)
        np.testing.assert_array_equal(fw, strengths)

    def test_ignore_excluded(self):
        times = np.array([0.1, 0.5, 1.0])
        strengths = np.array([0.8, 0.6, 0.9])
        anns = {"1": {"detected_onset_id": 1, "label": "ignore"}}
        ft, fw = filter_onsets_for_phase(times, strengths, anns)
        np.testing.assert_array_equal(ft, [0.1, 1.0])
        np.testing.assert_array_equal(fw, [0.8, 0.9])

    def test_string_noise_excluded(self):
        times = np.array([0.1, 0.5, 1.0])
        strengths = np.array([0.8, 0.6, 0.9])
        anns = {"0": {"detected_onset_id": 0, "label": "string_noise"}}
        ft, fw = filter_onsets_for_phase(times, strengths, anns)
        np.testing.assert_array_equal(ft, [0.5, 1.0])

    def test_true_attack_boosted(self):
        times = np.array([0.1, 0.5, 1.0])
        strengths = np.array([0.8, 0.6, 0.9])
        anns = {"2": {"detected_onset_id": 2, "label": "true_attack"}}
        ft, fw = filter_onsets_for_phase(times, strengths, anns)
        assert len(ft) == 3
        assert fw[2] == pytest.approx(0.9 * 3.0)
        assert fw[0] == pytest.approx(0.8)

    def test_downbeat_and_beat3_boosted(self):
        times = np.array([0.1, 0.5, 1.0])
        strengths = np.ones(3)
        anns = {
            "0": {"detected_onset_id": 0, "label": "downbeat"},
            "2": {"detected_onset_id": 2, "label": "beat3"},
        }
        ft, fw = filter_onsets_for_phase(times, strengths, anns)
        assert fw[0] == pytest.approx(3.0)
        assert fw[1] == pytest.approx(1.0)
        assert fw[2] == pytest.approx(3.0)

    def test_mixed_exclude_and_boost(self):
        times = np.array([0.1, 0.5, 1.0, 1.5])
        strengths = np.array([0.5, 0.5, 0.5, 0.5])
        anns = {
            "0": {"detected_onset_id": 0, "label": "true_attack"},
            "1": {"detected_onset_id": 1, "label": "ignore"},
            "3": {"detected_onset_id": 3, "label": "string_noise"},
        }
        ft, fw = filter_onsets_for_phase(times, strengths, anns)
        assert len(ft) == 2
        np.testing.assert_array_equal(ft, [0.1, 1.0])
        assert fw[0] == pytest.approx(0.5 * 3.0)
        assert fw[1] == pytest.approx(0.5)

    def test_out_of_range_annotation_ignored(self):
        times = np.array([0.1, 0.5])
        strengths = np.array([1.0, 1.0])
        anns = {"99": {"detected_onset_id": 99, "label": "ignore"}}
        ft, fw = filter_onsets_for_phase(times, strengths, anns)
        assert len(ft) == 2

    def test_original_strengths_not_mutated(self):
        times = np.array([0.1, 0.5])
        strengths = np.array([1.0, 1.0])
        anns = {"0": {"detected_onset_id": 0, "label": "downbeat"}}
        filter_onsets_for_phase(times, strengths, anns)
        assert strengths[0] == 1.0


# ── HTML report structure tests ──────────────────────────────────────────────


class TestReportHTMLStructure:
    """Verify generated HTML includes expected control IDs and help text."""

    @pytest.fixture()
    def report_html(self):
        sr = 44100
        audio = np.zeros(sr * 2, dtype=np.float32)
        onset_times = np.array([0.5, 1.0, 1.5])
        env_t = np.linspace(0, 2, 100)
        env_v = np.ones(100, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1)
        cls = [classify_onset_against_grid(t, 120, 4) for t in onset_times]
        return render_report(
            wav_path="test.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=2, onset_times=onset_times,
            classifications=cls, env_times=env_t, env_values=env_v,
            grid=grid, audio_src="test_excerpt.wav",
            sidecar_srcs={"stereo": "test_excerpt.wav", "bass": "test_bass.wav",
                          "song": "test_song.wav"},
        )

    def test_speed_controls(self, report_html):
        assert 'id="speed-label"' in report_html
        for rate in ["1.0", "0.75", "0.5", "0.33", "0.25"]:
            assert f'data-speed="{rate}"' in report_html

    def test_channel_selector(self, report_html):
        assert 'id="channel-select"' in report_html
        assert "test_bass.wav" in report_html
        assert "test_song.wav" in report_html

    def test_loop_controls(self, report_html):
        for eid in ["loop-toggle", "loop-set-start", "loop-set-end",
                     "loop-reset", "loop-restart",
                     "loop-start-in", "loop-end-in"]:
            assert f'id="{eid}"' in report_html

    def test_loop_region_rects(self, report_html):
        assert report_html.count('class="loop-region"') == 3

    def test_selected_status(self, report_html):
        assert 'id="selected-status"' in report_html

    def test_shortcut_help_panel(self, report_html):
        assert "Keyboard Shortcuts" in report_html
        for key_label in ["Space", "Delete", "Shift"]:
            assert key_label in report_html

    def test_hotkey_js(self, report_html):
        assert "keydown" in report_html
        assert "LABEL_KEYS" in report_html
        assert "clearAnnotation" in report_html

    def test_annotation_buttons(self, report_html):
        for lab in ["true_attack", "string_noise", "passing_note",
                     "downbeat", "beat3", "fill", "ignore", "uncertain"]:
            assert f'data-label="{lab}"' in report_html

    def test_workflow_note(self, report_html):
        assert "short windows" in report_html

    def test_overview_with_navigation(self):

        sr = 44100
        audio = np.zeros(sr * 2, dtype=np.float32)
        onset_times = np.array([0.5, 1.0])
        env_t = np.linspace(0, 2, 100)
        env_v = np.ones(100, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1)
        cls = [classify_onset_against_grid(t, 120, 4) for t in onset_times]
        ov = compute_overview(np.zeros(sr * 30, dtype=np.float32), sr)
        h = render_report(
            wav_path="test.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=8, duration=2, onset_times=onset_times,
            classifications=cls, env_times=env_t, env_values=env_v,
            grid=grid, audio_src="test.wav",
            overview=ov, prev_href="test_w0.html", next_href="test_w16.html",
            total_windows=4, window_index=1,
        )
        assert "Song Overview" in h
        assert "overview-svg" in h
        assert "test_w0.html" in h
        assert "test_w16.html" in h
        assert "Window 2/4" in h
        assert "nav-bar" in h


# ── compute_overview and window_tag tests ────────────────────────────────────


class TestComputeOverview:
    def test_basic(self):
        audio = np.random.default_rng(0).uniform(-1, 1, 44100).astype(np.float32)
        ov = compute_overview(audio, 44100, n_points=100)
        assert len(ov["maxes"]) == 100
        assert len(ov["mins"]) == 100
        assert ov["total_duration_s"] == pytest.approx(1.0, abs=0.01)

    def test_empty(self):
        ov = compute_overview(np.array([]), 44100)
        assert len(ov["maxes"]) == 0
        assert ov["total_duration_s"] == 0.0


class TestWindowTag:
    def test_integer(self):
        assert window_tag(0.0) == "w0"
        assert window_tag(8.0) == "w8"

    def test_float(self):
        assert window_tag(8.5) == "w8.5"


# ── grid settings round-trip tests ───────────────────────────────────────────


class TestGridSettings:
    def test_save_load_roundtrip(self, tmp_path):
        p = tmp_path / "gs.json"
        save_grid_settings(
            p, bpm=146, beats_per_measure=4, shuffle_fraction=0.667,
            beat_zero_s=1.05, source_file="test.wav", notes="test note",
        )
        gs = load_grid_settings(p)
        assert gs["bpm"] == 146
        assert gs["beats_per_measure"] == 4
        assert gs["shuffle_fraction"] == 0.667
        assert gs["beat_zero_s"] == pytest.approx(1.05)
        assert gs["source_file"] == "test.wav"
        assert gs["notes"] == "test note"

    def test_beat_zero_overrides_auto_phase(self):
        """Manual beat_zero_s should produce manual_beat_zero method, not auto."""

        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)
        onset_times = np.array([0.5])
        env_t = np.linspace(0, 1, 50)
        env_v = np.ones(50, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1, offset=0.25)
        cls = [classify_onset_against_grid(0.5, 120, 4, offset=0.25)]
        gs = GridSource(
            method="manual_beat_zero",
            description="beat_zero_s = 0.2500s set manually.",
        )
        h = render_report(
            wav_path="t.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=1, onset_times=onset_times,
            classifications=cls, env_times=env_t, env_values=env_v,
            grid=grid, audio_src="t.wav",
            grid_source=gs, beat_zero_s=0.25,
        )
        assert "manual_beat_zero" in h
        assert "0.2500s" in h

    def test_grid_settings_overrides_cli_defaults(self, tmp_path):
        p = tmp_path / "gs.json"
        save_grid_settings(
            p, bpm=100, beats_per_measure=3, shuffle_fraction=0.5,
            beat_zero_s=0.5, source_file="x.wav",
        )
        gs = load_grid_settings(p)
        assert gs["bpm"] == 100
        assert gs["beats_per_measure"] == 3
        assert gs["beat_zero_s"] == 0.5

    def test_report_shows_beat_zero_always(self):

        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1)
        cls = [classify_onset_against_grid(0.5, 120, 4)]
        h = render_report(
            wav_path="t.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=1, onset_times=np.array([0.5]),
            classifications=cls,
            env_times=np.linspace(0, 1, 50),
            env_values=np.ones(50, dtype=np.float32),
            grid=grid, audio_src="t.wav",
            beat_zero_s=0.0,
        )
        assert "Beat zero (M1 B1)" in h
        assert "0.0000s" in h

    def test_report_has_shift_suggestions(self):

        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1, offset=1.0)
        cls = [classify_onset_against_grid(0.5, 120, 4, offset=1.0)]
        h = render_report(
            wav_path="t.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=1, onset_times=np.array([0.5]),
            classifications=cls,
            env_times=np.linspace(0, 1, 50),
            env_values=np.ones(50, dtype=np.float32),
            grid=grid, audio_src="t.wav",
            beat_zero_s=1.0,
        )
        assert "+1 beat" in h
        assert "-1 beat" in h
        assert "+1 measure" in h
        assert "beat-zero-input" in h
        assert "export-grid-settings" in h

    def test_report_has_calibration_ui(self):

        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1)
        cls = [classify_onset_against_grid(0.5, 120, 4)]
        h = render_report(
            wav_path="t.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=1, onset_times=np.array([0.5]),
            classifications=cls,
            env_times=np.linspace(0, 1, 50),
            env_values=np.ones(50, dtype=np.float32),
            grid=grid, audio_src="t.wav",
        )
        assert 'id="grid-anchor-display"' in h
        assert 'id="grid-calibration-menu"' in h
        assert "Make this Beat 1" in h
        assert "Make this Beat 4" in h
        assert 'id="cal-from-selected"' in h
        assert "contextmenu" in h
        assert "setGridAnchor" in h
        assert "redrawGrid" in h
        assert "grid-line" in h
        assert "grid-label" in h
        assert "Advanced grid settings" in h
        assert 'id="beat-zero-input"' in h

    def test_beat_zero_hidden_in_advanced(self):

        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1)
        cls = [classify_onset_against_grid(0.5, 120, 4)]
        h = render_report(
            wav_path="t.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=1, onset_times=np.array([0.5]),
            classifications=cls,
            env_times=np.linspace(0, 1, 50),
            env_values=np.ones(50, dtype=np.float32),
            grid=grid, audio_src="t.wav", beat_zero_s=1.08,
        )
        details_start = h.find("Advanced grid settings")
        beat_zero_input = h.find('id="beat-zero-input"')
        assert details_start < beat_zero_input

    def test_report_has_suggested_phase_language(self):

        gs = GridSource(
            method="suggested_phase_from_bass_anchors",
            description=(
                "Suggested beat_zero_s = 1.05s. "
                "This is a suggested alignment and may confuse "
                "beats/subdivisions or beats 1/3 vs 2/4."
            ),
        )
        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)
        grid = make_grid(bpm=120, beats_per_measure=4, n_measures=1)
        cls = [classify_onset_against_grid(0.5, 120, 4)]
        h = render_report(
            wav_path="t.wav", bass_audio=audio, sr=sr,
            bpm=120, beats_per_measure=4, shuffle_fraction=0.667,
            start=0, duration=1, onset_times=np.array([0.5]),
            classifications=cls,
            env_times=np.linspace(0, 1, 50),
            env_values=np.ones(50, dtype=np.float32),
            grid=grid, audio_src="t.wav",
            grid_source=gs, beat_zero_s=1.05,
        )
        assert "suggested_phase_from_bass_anchors" in h
        assert "suggested alignment" in h
        assert "may confuse" in h
