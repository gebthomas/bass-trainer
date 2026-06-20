#!/usr/bin/env python3
"""Beat Microscope — inspect bass timing against a shuffle-aware beat grid.

Loads a stereo WAV, extracts the bass channel, detects onsets in a selected
time segment, and generates an interactive HTML diagnostic report with
synchronized audio playback.

Usage
-----
    python scripts/pocket_lab_inspect.py recording.wav --bpm 146
    python scripts/pocket_lab_inspect.py recording.wav --bpm 120 --beats-per-measure 3
    python scripts/pocket_lab_inspect.py recording.wav --bpm 146 --start 4.0 --duration 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pocket_lab.audio import (
    audio_diagnostics,
    compute_overview,
    load_audio,
    segment_audio,
    window_tag,
)
from pocket_lab.grid import classify_onset_against_grid, make_grid
from pocket_lab.grid_phase import (
    _PHASE_BOOST_LABELS,
    estimate_beat_zero,
    filter_onsets_for_phase,
)
from pocket_lab.grid_settings import (
    GRID_SOURCE_FIXED_BPM,
    GridSource,
    load_grid_settings,
    save_grid_settings,
)
from pocket_lab.onset import detect_onsets, novelty_envelope
from pocket_lab.report import render_report


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beat Microscope — inspect bass timing against a shuffle-aware beat grid.",
    )
    p.add_argument("wav", help="Stereo WAV file to analyze")
    p.add_argument("--bass-channel", type=int, default=0, dest="bass_channel")
    p.add_argument("--song-channel", type=int, default=1, dest="song_channel")
    p.add_argument("--bpm", type=float, default=146)
    p.add_argument("--beats-per-measure", type=int, default=4, dest="beats_per_measure")
    p.add_argument("--shuffle-fraction", type=float, default=0.667, dest="shuffle_fraction")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--output-dir", default="diagnostics", dest="output_dir")
    p.add_argument(
        "--delta", type=float, default=0.07,
        help="Librosa onset-strength threshold (default 0.07)",
    )
    p.add_argument(
        "--audio-export-mode", default="stereo", dest="audio_export_mode",
        choices=["stereo", "bass", "song", "mix"],
        help="Which audio to export as the sidecar WAV (default: stereo)",
    )
    p.add_argument(
        "--beat-zero-s", type=float, default=None, dest="beat_zero_s",
        help="Absolute song time of measure 1, beat 1 (overrides auto-phase)",
    )
    p.add_argument(
        "--grid-settings", type=str, default=None, dest="grid_settings",
        metavar="PATH",
        help="Load grid settings from JSON (overrides CLI bpm/meter/shuffle/beat-zero)",
    )
    p.add_argument(
        "--bass-anchor-beats", type=str, default=None, dest="bass_anchor_beats",
        metavar="BEATS",
        help="Comma-separated beat numbers for suggested phase alignment (e.g. 1,3)",
    )
    p.add_argument(
        "--auto-phase-from-bass-anchors", action="store_true", dest="auto_phase",
        help="Suggest beat_zero_s from bass anchor beats (overridden by --beat-zero-s)",
    )
    p.add_argument(
        "--use-annotations-for-phase", action="store_true",
        dest="use_annotations_for_phase",
        help="Filter/weight onsets by annotation labels during phase estimation",
    )
    p.add_argument(
        "--generate-all-windows", action="store_true", dest="generate_all_windows",
        help="Generate reports for every window across the full song",
    )
    p.add_argument(
        "--window-step", type=float, default=None, dest="window_step",
        help="Step between windows in seconds (default: same as --duration)",
    )
    return p.parse_args(argv)


def _report_filename(stem: str, start: float, multi: bool) -> str:
    if multi:
        return f"{stem}_microscope_{window_tag(start)}.html"
    return f"{stem}_microscope.html"


def _generate_window(
    *,
    wav_path: Path,
    raw_audio: np.ndarray,
    bass_full: np.ndarray,
    sr: int,
    is_stereo: bool,
    args: argparse.Namespace,
    win_start: float,
    win_duration: float,
    overview: dict,
    beat_zero: float,
    grid_source: GridSource,
    out_dir: Path,
    multi: bool,
    prev_href: str | None,
    next_href: str | None,
    total_windows: int,
    window_index: int,
) -> Path:
    """Generate a single-window report and its audio files. Returns HTML path."""
    import soundfile as sf

    bass_segment = segment_audio(bass_full, sr, win_start, win_duration)
    actual_duration = len(bass_segment) / sr

    onset_times = detect_onsets(bass_segment, sr, delta=args.delta)
    env_times, env_values = novelty_envelope(bass_segment, sr)

    if len(onset_times) > 0 and len(env_times) > 0:
        onset_strengths = np.interp(onset_times, env_times, env_values)
        peak_env = np.max(env_values)
        if peak_env > 0:
            onset_strengths = onset_strengths / peak_env
    else:
        onset_strengths = np.zeros(len(onset_times))

    beat_s = 60.0 / args.bpm
    n_measures = max(1, int(np.ceil(actual_duration / (beat_s * args.beats_per_measure))))

    grid = make_grid(
        bpm=args.bpm,
        beats_per_measure=args.beats_per_measure,
        n_measures=n_measures,
        shuffle_fraction=args.shuffle_fraction,
        offset=beat_zero,
    )

    classifications = [
        classify_onset_against_grid(
            t, args.bpm, args.beats_per_measure, args.shuffle_fraction,
            offset=beat_zero,
        )
        for t in onset_times
    ]

    # ── Audio export ─────────────────────────────────────────────────────
    wtag = window_tag(win_start) if multi else ""
    sfx = f"_{wtag}" if wtag else ""
    mode = args.audio_export_mode

    if is_stereo:
        source_segment = segment_audio(raw_audio, sr, win_start, win_duration)
        if mode == "stereo":
            export_audio = source_segment
        elif mode == "bass":
            export_audio = source_segment[:, args.bass_channel]
        elif mode == "song":
            export_audio = source_segment[:, args.song_channel]
        else:
            export_audio = np.mean(source_segment, axis=1)
    else:
        source_segment = bass_segment
        export_audio = bass_segment

    diag_source = audio_diagnostics(source_segment, "source segment")
    diag_export = audio_diagnostics(export_audio, "exported excerpt")

    audio_filename = f"{wav_path.stem}_excerpt{sfx}.wav"
    sf.write(str(out_dir / audio_filename), export_audio, sr, subtype="PCM_16")

    sidecar_srcs: dict[str, str] = {"stereo": audio_filename}
    if is_stereo:
        for label, data in [
            ("bass", source_segment[:, args.bass_channel]),
            ("song", source_segment[:, args.song_channel]),
        ]:
            fn = f"{wav_path.stem}_excerpt{sfx}_{label}.wav"
            sf.write(str(out_dir / fn), data, sr, subtype="PCM_16")
            sidecar_srcs[label] = fn

    storage_key = f"{wav_path.stem}_ann_s{window_tag(win_start)}_d{int(win_duration)}"

    report = render_report(
        wav_path=str(wav_path),
        bass_audio=bass_segment,
        sr=sr,
        bpm=args.bpm,
        beats_per_measure=args.beats_per_measure,
        shuffle_fraction=args.shuffle_fraction,
        start=win_start,
        duration=win_duration,
        onset_times=onset_times,
        classifications=classifications,
        env_times=env_times,
        env_values=env_values,
        grid=grid,
        audio_src=audio_filename,
        grid_source=grid_source,
        onset_strengths=onset_strengths,
        beat_zero_s=beat_zero,
        storage_key=storage_key,
        audio_diag_source=diag_source,
        audio_diag_export=diag_export,
        audio_export_mode=mode,
        sidecar_srcs=sidecar_srcs,
        overview=overview,
        prev_href=prev_href,
        next_href=next_href,
        total_windows=total_windows,
        window_index=window_index,
    )

    out_path = out_dir / _report_filename(wav_path.stem, win_start, multi)
    out_path.write_text(report, encoding="utf-8")
    return out_path


def main() -> None:
    args = _parse_args()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    if args.duration > 10:
        print("Note: Long segments may be visually dense; "
              "use shorter windows (2–6 s) for annotation.", file=sys.stderr)

    # ── Grid settings resolution (most-specific source wins) ─────────────
    # Priority: --grid-settings > --beat-zero-s > --auto-phase > defaults
    if args.grid_settings:
        gs = load_grid_settings(args.grid_settings)
        args.bpm = gs.get("bpm", args.bpm)
        args.beats_per_measure = gs.get("beats_per_measure", args.beats_per_measure)
        args.shuffle_fraction = gs.get("shuffle_fraction", args.shuffle_fraction)
        if args.beat_zero_s is None:
            args.beat_zero_s = gs.get("beat_zero_s", 0.0)
        print(f"Grid settings loaded from {args.grid_settings}")

    raw_audio, sr = load_audio(wav_path)
    is_stereo = raw_audio.ndim == 2
    bass_full = raw_audio[:, args.bass_channel] if is_stereo else raw_audio

    overview = compute_overview(bass_full, sr)
    total_dur = overview["total_duration_s"]

    # ── Beat-zero resolution ─────────────────────────────────────────────
    anchor_beats: list[int] = []
    if args.bass_anchor_beats:
        anchor_beats = [int(x.strip()) for x in args.bass_anchor_beats.split(",")]

    beat_zero = 0.0
    grid_source = GRID_SOURCE_FIXED_BPM

    if args.beat_zero_s is not None:
        beat_zero = args.beat_zero_s
        grid_source = GridSource(
            method="manual_beat_zero",
            description=(
                f"beat_zero_s = {beat_zero:.4f}s set manually via CLI or grid "
                f"settings file. BPM, meter, and shuffle fraction are user-supplied."
            ),
        )
        print(f"Manual beat_zero_s = {beat_zero:.4f}s")

    elif args.auto_phase and anchor_beats:
        full_onsets = detect_onsets(bass_full, sr, delta=args.delta)
        full_env_t, full_env_v = novelty_envelope(bass_full, sr)
        if len(full_onsets) > 0 and len(full_env_t) > 0:
            full_strengths = np.interp(full_onsets, full_env_t, full_env_v)
            pk = np.max(full_env_v)
            if pk > 0:
                full_strengths /= pk
        else:
            full_strengths = np.ones(len(full_onsets))

        phase_times, phase_weights = full_onsets, full_strengths
        used_annotations = False
        n_excluded = n_boosted = 0

        if args.use_annotations_for_phase:
            import json as _json
            ann_path = Path(args.output_dir) / f"{wav_path.stem}_annotations.json"
            loaded_anns: dict = {}
            if ann_path.exists():
                try:
                    data = _json.loads(ann_path.read_text(encoding="utf-8"))
                    for a in data.get("annotations", []):
                        loaded_anns[str(a["detected_onset_id"])] = a
                except (_json.JSONDecodeError, KeyError):
                    pass
            if loaded_anns:
                n_before = len(phase_times)
                phase_times, phase_weights = filter_onsets_for_phase(
                    full_onsets, full_strengths, loaded_anns)
                n_excluded = n_before - len(phase_times)
                n_boosted = sum(
                    1 for a in loaded_anns.values()
                    if a.get("label") in _PHASE_BOOST_LABELS
                    and int(a["detected_onset_id"]) < n_before)
                used_annotations = True

        beat_zero = estimate_beat_zero(
            phase_times, args.bpm, args.beats_per_measure,
            anchor_beats, onset_strengths=phase_weights)

        n_used = len(phase_times)
        ann_note = (
            f" Annotations applied: {n_excluded} excluded, "
            f"{n_boosted} boosted, {n_used} onsets used."
        ) if used_annotations else (
            f" {n_used} raw detected onsets used."
        )

        grid_source = GridSource(
            method="suggested_phase_from_bass_anchors",
            description=(
                f"Suggested beat_zero_s = {beat_zero:.4f}s from bass onsets "
                f"aligned to anchor beats {anchor_beats}. "
                f"This is a suggested alignment and may confuse "
                f"beats/subdivisions or beats 1/3 vs 2/4. "
                f"BPM, meter, and shuffle fraction are user-supplied."
                + ann_note),
        )
        print(f"Suggested phase: beat_zero_s = {beat_zero:.4f}s "
              f"(anchors={anchor_beats}, onsets_used={n_used})")

    # ── Export grid settings ─────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_grid_settings(
        out_dir / f"{wav_path.stem}_grid_settings.json",
        bpm=args.bpm, beats_per_measure=args.beats_per_measure,
        shuffle_fraction=args.shuffle_fraction, beat_zero_s=beat_zero,
        source_file=wav_path.name,
    )

    # ── Determine windows ────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win_dur = args.duration
    win_step = args.window_step if args.window_step is not None else win_dur

    if args.generate_all_windows:
        starts = list(np.arange(0, total_dur, win_step))
    else:
        starts = [args.start]

    multi = len(starts) > 1

    for idx, ws in enumerate(starts):
        prev_href = (
            _report_filename(wav_path.stem, starts[idx - 1], True)
            if idx > 0 else None
        )
        next_href = (
            _report_filename(wav_path.stem, starts[idx + 1], True)
            if idx < len(starts) - 1 else None
        )

        out_path = _generate_window(
            wav_path=wav_path, raw_audio=raw_audio, bass_full=bass_full,
            sr=sr, is_stereo=is_stereo, args=args,
            win_start=ws, win_duration=win_dur,
            overview=overview, beat_zero=beat_zero, grid_source=grid_source,
            out_dir=out_dir, multi=multi,
            prev_href=prev_href, next_href=next_href,
            total_windows=len(starts), window_index=idx,
        )
        print(f"[{idx + 1}/{len(starts)}] {out_path.name}  "
              f"({ws:.1f}–{ws + win_dur:.1f}s)")


if __name__ == "__main__":
    main()
