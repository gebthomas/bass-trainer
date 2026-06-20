#!/usr/bin/env python3
"""Take Comparator — compare two bass performances of the same song.

Loads two stereo WAV recordings, aligns them via song-channel
cross-correlation, detects onsets in both bass channels, matches
onsets between takes, and generates interactive HTML comparison reports
with windowed navigation.

Usage
-----
    python scripts/pocket_lab_compare.py take_a.wav take_b.wav --bpm 146
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pocket_lab.audio import (
    compute_overview,
    load_audio,
    segment_audio,
    window_tag,
)
from pocket_lab.comparator_report import render_comparator_report
from pocket_lab.match_record import ComparisonResult, OnsetRecord
from pocket_lab.onset import detect_onsets, novelty_envelope
from pocket_lab.onset_diagnostics import full_onset_diagnostic
from pocket_lab.onset_matcher import match_onsets, threshold_sweep
from pocket_lab.song_align import align_song_channels


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Take Comparator — compare two bass performances.",
    )
    p.add_argument("take_a", help="First stereo WAV file")
    p.add_argument("take_b", help="Second stereo WAV file")
    p.add_argument("--bass-channel", type=int, default=0, dest="bass_channel")
    p.add_argument("--song-channel", type=int, default=1, dest="song_channel")
    p.add_argument("--bpm", type=float, default=146)
    p.add_argument("--beats-per-measure", type=int, default=4, dest="beats_per_measure")
    p.add_argument("--shuffle-fraction", type=float, default=0.667, dest="shuffle_fraction")
    p.add_argument("--start", type=float, default=0)
    p.add_argument(
        "--duration", type=float, default=16,
        help="Window duration in seconds (default 16)",
    )
    p.add_argument("--output-dir", default="diagnostics", dest="output_dir")
    p.add_argument(
        "--delta", type=float, default=0.07,
        help="Librosa onset-strength threshold (default 0.07)",
    )
    p.add_argument(
        "--max-match-window-ms", type=float, default=50,
        dest="max_match_window_ms",
        help="Maximum time difference (ms) to consider an onset match (default 50)",
    )
    p.add_argument(
        "--max-align-offset-s", type=float, default=5.0,
        dest="max_align_offset_s",
        help="Maximum song-channel alignment offset in seconds (default 5.0)",
    )
    p.add_argument(
        "--noise-threshold", type=float, default=0.0,
        dest="noise_threshold",
        help="Onset strength below this is classified as noise (default 0.0)",
    )
    p.add_argument(
        "--window-step", type=float, default=None, dest="window_step",
        help="Step between windows in seconds (default: same as --duration)",
    )
    p.add_argument(
        "--single-window", action="store_true", dest="single_window",
        help="Generate only one window at --start instead of all windows",
    )
    return p.parse_args(argv)


def _build_onset_records(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    bass_audio: np.ndarray,
    sr: int,
    take_label: str,
    audio_sample_times: np.ndarray | None = None,
) -> list[OnsetRecord]:
    """Build OnsetRecord list with amplitude measurements.

    onset_times: times used as the record's time_s (may be in A's timeline).
    audio_sample_times: times used to index into bass_audio for amplitude
                        measurement (native recording coordinates).
                        Defaults to onset_times if not provided.
    """
    if audio_sample_times is None:
        audio_sample_times = onset_times
    records = []
    for i, t in enumerate(onset_times):
        strength = float(onset_strengths[i]) if i < len(onset_strengths) else 0.0
        native_t = float(audio_sample_times[i])
        sample_idx = int(native_t * sr)
        window = 512
        lo = max(0, sample_idx - window // 2)
        hi = min(len(bass_audio), sample_idx + window // 2)
        chunk = bass_audio[lo:hi]
        peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 1e-9
        amp_db = 20.0 * np.log10(max(peak, 1e-9))

        records.append(OnsetRecord(
            time_s=t,
            strength=strength,
            amplitude_db=amp_db,
            raw_time_s=t,
            take_label=take_label,
            onset_index=i,
        ))
    return records


def _normalize_strengths(onset_times, env_t, env_v):
    if len(onset_times) > 0 and len(env_t) > 0:
        strengths = np.interp(onset_times, env_t, env_v)
        pk = np.max(env_v)
        return strengths / pk if pk > 0 else strengths
    return np.zeros(len(onset_times))


def _filter_records_to_window(
    records: list[OnsetRecord], win_start: float, win_end: float,
) -> list[OnsetRecord]:
    """Return records whose time falls within [win_start, win_end), with times
    relative to the window start and re-indexed."""
    filtered = []
    for r in records:
        if win_start <= r.raw_time_s < win_end:
            filtered.append(OnsetRecord(
                time_s=r.time_s - win_start,
                strength=r.strength,
                amplitude_db=r.amplitude_db,
                raw_time_s=r.raw_time_s,
                take_label=r.take_label,
                onset_index=len(filtered),
            ))
    return filtered


def _report_filename(stem: str, start: float, multi: bool) -> str:
    if multi:
        return f"{stem}_comparator_{window_tag(start)}.html"
    return f"{stem}_comparator.html"


def main() -> None:
    args = _parse_args()

    path_a = Path(args.take_a)
    path_b = Path(args.take_b)
    for p in [path_a, path_b]:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {path_a.name} and {path_b.name}...")
    raw_a, sr_a = load_audio(path_a)
    raw_b, sr_b = load_audio(path_b)

    if sr_a != sr_b:
        print(f"Error: sample rates differ ({sr_a} vs {sr_b})", file=sys.stderr)
        sys.exit(1)
    sr = sr_a

    is_stereo_a = raw_a.ndim == 2
    is_stereo_b = raw_b.ndim == 2
    if not is_stereo_a or not is_stereo_b:
        print("Warning: expected stereo recordings. "
              "Mono files will use the single channel as bass.", file=sys.stderr)

    bass_a_full = raw_a[:, args.bass_channel] if is_stereo_a else raw_a
    bass_b_full = raw_b[:, args.bass_channel] if is_stereo_b else raw_b

    # ── Song-channel alignment ──────────────────────────────────────────
    # offset_s: add to B times to get A times.  Positive → B started later.
    if is_stereo_a and is_stereo_b:
        song_a_full = raw_a[:, args.song_channel]
        song_b_full = raw_b[:, args.song_channel]
        print("Aligning song channels...")
        offset_s, confidence = align_song_channels(
            song_a_full, song_b_full, sr, max_offset_s=args.max_align_offset_s)
        print(f"  Offset: {offset_s:+.4f}s, confidence: {confidence:.3f}")
        if confidence < 0.7:
            print("  Warning: low alignment confidence. "
                  "Are both recordings of the same backing track?",
                  file=sys.stderr)
    else:
        song_a_full = song_b_full = None
        offset_s, confidence = 0.0, 0.0
        print("Skipping alignment (mono input).")

    # ── Full-song duration (in A's coordinate system) ───────────────────
    dur_a = len(bass_a_full) / sr
    dur_b_aligned = len(bass_b_full) / sr + offset_s
    total_dur = max(dur_a, dur_b_aligned)

    # ── Full-song onset detection ───────────────────────────────────────
    print("Detecting onsets (full song)...")
    onsets_a = detect_onsets(bass_a_full, sr, delta=args.delta)
    onsets_b_raw = detect_onsets(bass_b_full, sr, delta=args.delta)
    # Convert B onsets to A's timeline: A_time = B_time + offset_s
    onsets_b = onsets_b_raw + offset_s
    print(f"  Take A: {len(onsets_a)} onsets")
    print(f"  Take B: {len(onsets_b)} onsets")

    env_t_a, env_v_a = novelty_envelope(bass_a_full, sr)
    env_t_b_raw, env_v_b = novelty_envelope(bass_b_full, sr)

    strengths_a = _normalize_strengths(onsets_a, env_t_a, env_v_a)
    strengths_b = _normalize_strengths(onsets_b_raw, env_t_b_raw, env_v_b)

    records_a_full = _build_onset_records(onsets_a, strengths_a, bass_a_full, sr, "A")
    records_b_full = _build_onset_records(
        onsets_b, strengths_b, bass_b_full, sr, "B",
        audio_sample_times=onsets_b_raw,
    )
    for r in records_b_full:
        r.raw_time_s = r.time_s

    # ── Onset diagnostics ──────────────────────────────────────────────
    diag_a = full_onset_diagnostic(onsets_a, strengths_a, "Take A", sr)
    diag_b = full_onset_diagnostic(onsets_b, strengths_b, "Take B", sr)

    print(f"\nOnset diagnostics:")
    qinfo = diag_a["quantization"]
    print(f"  Frame period: {qinfo['frame_period_ms']:.2f}ms "
          f"(hop={qinfo['hop_length']}, sr={qinfo['sr']})")
    print(f"  ⚠ All timing diffs are quantized to multiples of {qinfo['frame_period_ms']:.1f}ms")

    for diag in [diag_a, diag_b]:
        label = diag["label"]
        sp = diag["spacing"]
        st = diag["strength"]
        print(f"\n  {label}: {sp['count']} onsets")
        print(f"    Spacing: min={sp['min_spacing_ms']:.1f}ms, "
              f"median={sp['median_spacing_ms']:.1f}ms, "
              f"mean={sp['mean_spacing_ms']:.1f}ms")
        print(f"    Strength: min={st['min']:.4f}, max={st['max']:.4f}, "
              f"mean={st['mean']:.4f}, median={st['median']:.4f}")
        print(f"    Strength <0.01: {st['below_001']}, "
              f"<0.1: {st['below_01']}, <0.5: {st['below_05']}")

        print(f"    Close pairs:")
        for cp in diag["close_pairs"]:
            print(f"      <{cp['threshold_ms']:.0f}ms: "
                  f"{cp['count']} onsets ({cp['fraction']:.1%})")

        print(f"    Spacing histogram:")
        for h in diag["histogram"]:
            bar = "█" * min(h["count"], 50)
            print(f"      {h['label']:>12}: {h['count']:4d} {bar}")

    onset_diags = {"a": diag_a, "b": diag_b}

    # ── Threshold sweep (full song) ─────────────────────────────────────
    max_window_s = args.max_match_window_ms / 1000.0
    print("\nNoise threshold sweep:")
    sweep_rows = threshold_sweep(records_a_full, records_b_full, max_window_s)
    print(f"  {'Thr':>5}  {'Match':>6}  {'A-only':>6}  {'B-only':>6}  {'Ambig':>6}  {'Noise':>6}")
    for r in sweep_rows:
        marker = " <--" if r["threshold"] == args.noise_threshold else ""
        print(f"  {r['threshold']:5.2f}  {r['matched']:6}  {r['a_only']:6}  "
              f"{r['b_only']:6}  {r['ambiguous']:6}  {r['noise']:6}{marker}")

    # ── Full-song matching ──────────────────────────────────────────────
    print(f"\nMatching onsets (threshold={args.noise_threshold})...")
    all_matches = match_onsets(
        records_a_full, records_b_full,
        max_match_window_s=max_window_s,
        noise_strength_threshold=args.noise_threshold,
    )

    full_result = ComparisonResult(
        take_a_path=str(path_a),
        take_b_path=str(path_b),
        alignment_offset_s=offset_s,
        alignment_confidence=confidence,
        sample_rate=sr,
        matches=all_matches,
    )

    print(f"  Matched: {full_result.matched_count}")
    print(f"  A-only:  {full_result.a_only_count}")
    print(f"  B-only:  {full_result.b_only_count}")
    print(f"  Ambiguous: {full_result.ambiguous_count}")
    print(f"  Noise:   {full_result.noise_count}")

    if full_result.timing_diffs_ms:
        mean_t = np.mean(full_result.timing_diffs_ms)
        std_t = np.std(full_result.timing_diffs_ms)
        print(f"  Timing diff: {mean_t:+.1f} ± {std_t:.1f} ms")

    # ── Determine windows ───────────────────────────────────────────────
    import soundfile as sf

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    win_dur = args.duration
    win_step = args.window_step if args.window_step is not None else win_dur

    if args.single_window:
        starts = [args.start]
    else:
        starts = list(np.arange(0, total_dur, win_step))

    multi = len(starts) > 1
    stem = f"{path_a.stem}_vs_{path_b.stem}"

    overview = compute_overview(bass_a_full, sr)

    # ── Generate per-window reports ─────────────────────────────────────
    for idx, ws in enumerate(starts):
        we = ws + win_dur

        prev_href = (
            _report_filename(stem, starts[idx - 1], True)
            if multi and idx > 0 else None
        )
        next_href = (
            _report_filename(stem, starts[idx + 1], True)
            if multi and idx < len(starts) - 1 else None
        )

        # Window start in each recording's native coordinates.
        # A's coordinate system is the reference; convert to B's native.
        a_native_start = ws
        b_native_start = ws - offset_s

        # Segment audio — all sidecars cover the same musical region.
        bass_seg_a = segment_audio(bass_a_full, sr, a_native_start, win_dur)
        bass_seg_b = segment_audio(bass_b_full, sr, max(0.0, b_native_start), win_dur)

        song_seg_a = None
        song_seg_b = None
        stereo_seg_a = None
        stereo_seg_b = None
        if is_stereo_a:
            song_seg_a = segment_audio(song_a_full, sr, a_native_start, win_dur)
            stereo_seg_a = segment_audio(raw_a, sr, a_native_start, win_dur)
        if is_stereo_b:
            song_seg_b = segment_audio(song_b_full, sr, max(0.0, b_native_start), win_dur)
            stereo_seg_b = segment_audio(raw_b, sr, max(0.0, b_native_start), win_dur)

        # Filter onset records to this window
        win_records_a = _filter_records_to_window(records_a_full, ws, we)
        win_records_b = _filter_records_to_window(records_b_full, ws, we)

        # Match within window
        win_matches = match_onsets(
            win_records_a, win_records_b,
            max_match_window_s=max_window_s,
            noise_strength_threshold=args.noise_threshold,
        )

        win_result = ComparisonResult(
            take_a_path=str(path_a),
            take_b_path=str(path_b),
            alignment_offset_s=offset_s,
            alignment_confidence=confidence,
            sample_rate=sr,
            matches=win_matches,
        )

        # Export audio files — all pre-aligned, same musical time span.
        wtag = window_tag(ws) if multi else ""
        sfx = f"_{wtag}" if wtag else ""

        audio_fn_a = f"{stem}_bass_a{sfx}.wav"
        audio_fn_b = f"{stem}_bass_b{sfx}.wav"
        sf.write(str(out_dir / audio_fn_a), bass_seg_a, sr, subtype="PCM_16")
        sf.write(str(out_dir / audio_fn_b), bass_seg_b, sr, subtype="PCM_16")

        audio_fn_song = ""
        if song_seg_a is not None:
            audio_fn_song = f"{stem}_song{sfx}.wav"
            sf.write(str(out_dir / audio_fn_song), song_seg_a, sr, subtype="PCM_16")

        audio_fn_song_b = ""
        if song_seg_b is not None:
            audio_fn_song_b = f"{stem}_song_b{sfx}.wav"
            sf.write(str(out_dir / audio_fn_song_b), song_seg_b, sr, subtype="PCM_16")

        audio_fn_stereo_a = ""
        if stereo_seg_a is not None:
            audio_fn_stereo_a = f"{stem}_stereo_a{sfx}.wav"
            sf.write(str(out_dir / audio_fn_stereo_a), stereo_seg_a, sr, subtype="PCM_16")

        audio_fn_stereo_b = ""
        if stereo_seg_b is not None:
            audio_fn_stereo_b = f"{stem}_stereo_b{sfx}.wav"
            sf.write(str(out_dir / audio_fn_stereo_b), stereo_seg_b, sr, subtype="PCM_16")

        # Sync diagnostic info
        sync_info = {
            "alignment_offset_s": offset_s,
            "alignment_confidence": confidence,
            "window_start_a": a_native_start,
            "window_start_b": b_native_start,
            "bass_a_samples": len(bass_seg_a),
            "bass_b_samples": len(bass_seg_b),
            "bass_a_duration": len(bass_seg_a) / sr,
            "bass_b_duration": len(bass_seg_b) / sr,
            "song_duration": len(song_seg_a) / sr if song_seg_a is not None else 0.0,
        }

        report = render_comparator_report(
            result=win_result,
            bass_audio_a=bass_seg_a,
            bass_audio_b=bass_seg_b,
            sr=sr,
            duration=win_dur,
            audio_src_a=audio_fn_a,
            audio_src_b=audio_fn_b,
            audio_src_song=audio_fn_song,
            audio_src_song_b=audio_fn_song_b,
            audio_src_stereo_a=audio_fn_stereo_a,
            audio_src_stereo_b=audio_fn_stereo_b,
            sweep_rows=sweep_rows if idx == 0 else None,
            noise_threshold=args.noise_threshold,
            onset_diags=onset_diags if idx == 0 else None,
            sync_info=sync_info,
            overview=overview if multi else None,
            window_start=ws,
            prev_href=prev_href,
            next_href=next_href,
            total_windows=len(starts),
            window_index=idx,
        )

        out_path = out_dir / _report_filename(stem, ws, multi)
        out_path.write_text(report, encoding="utf-8")
        n_events = len(win_matches)
        print(f"[{idx + 1}/{len(starts)}] {out_path.name}  "
              f"({ws:.1f}–{we:.1f}s, {n_events} events)")

    print(f"\nDone. {len(starts)} report(s) in {out_dir}/")


if __name__ == "__main__":
    main()
