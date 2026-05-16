#!/usr/bin/env python3
"""Adaptive-timing evaluation demo — longer exercises with diagnostics and post-run summary.

Run from the project root:
    python scripts/adaptive_timing_demo.py
    python scripts/adaptive_timing_demo.py --beats 32 --bpm 72
    python scripts/adaptive_timing_demo.py --click-mode measure
    python scripts/adaptive_timing_demo.py --click-mode count-in-only
    python scripts/adaptive_timing_demo.py --device 2 --level-check

Click modes:
    beat          — click every beat (default)
    half          — click every 2 beats
    measure       — click every N beats (--time-sig, default 4)
    count-in-only — clicks only during count-in; silent thereafter
    off           — no clicks at all

Output format (per beat):
    beat N  SEVERITY  ±XX ms  [bpm=61.2 win=±5ms conf=0.82]  ●
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_pipeline import process_realtime_audio
from core.practice_session import PracticeSession
from core.tempo_tracker import TempoTracker

# ── Config ────────────────────────────────────────────────────────────────────

BLOCK_FRAMES = 512
METER_SECS   = 5.0
METER_PERIOD = 0.25

ACCENT_FREQ  = 880.0
BEAT_FREQ    = 440.0
CLICK_S      = 0.05
CLICK_AMP    = 0.7


# ── Device helpers ────────────────────────────────────────────────────────────

def _list_input_devices() -> None:
    print("Input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            sr = int(dev["default_samplerate"])
            print(f"  [{i:2d}] {dev['name']}  (default SR: {sr} Hz)")
    print()


def _resolve_device(spec: str | None) -> tuple[int | None, int]:
    if spec is None:
        dev = sd.query_devices(kind="input")
        return None, int(dev["default_samplerate"])

    try:
        idx = int(spec)
        dev = sd.query_devices(idx)
        return idx, int(dev["default_samplerate"])
    except ValueError:
        pass

    needle = spec.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
            return i, int(dev["default_samplerate"])

    print(f"Warning: no input device matching {spec!r} found; using system default.",
          file=sys.stderr)
    dev = sd.query_devices(kind="input")
    return None, int(dev["default_samplerate"])


def _run_level_meter(device, sample_rate: int) -> None:
    print(f"Input level meter — {METER_SECS:.0f}s  (play something to verify signal)\n")
    period_samples = int(sample_rate * METER_PERIOD)
    chunk_buf      = np.zeros(period_samples, dtype=np.float64)
    chunk_pos      = 0
    total_samples  = 0
    stop_samples   = int(sample_rate * METER_SECS)

    def _print_meter(samples: np.ndarray) -> None:
        rms  = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        bar  = int(min(peak, 1.0) * 40)
        print(f"  rms={rms:.4f}  peak={peak:.4f}  |{'█' * bar}{' ' * (40 - bar)}|")

    with sd.InputStream(
        device=device, samplerate=sample_rate, channels=1,
        dtype="float32", blocksize=BLOCK_FRAMES,
    ) as stream:
        while total_samples < stop_samples:
            raw, _ = stream.read(BLOCK_FRAMES)
            mono   = raw[:, 0].astype(np.float64) if raw.ndim == 2 else raw.astype(np.float64)
            remaining = period_samples - chunk_pos
            if len(mono) >= remaining:
                chunk_buf[chunk_pos:] = mono[:remaining]
                _print_meter(chunk_buf)
                chunk_pos = 0
                leftover  = mono[remaining:]
                if len(leftover):
                    chunk_buf[:len(leftover)] = leftover
                    chunk_pos = len(leftover)
            else:
                chunk_buf[chunk_pos : chunk_pos + len(mono)] = mono
                chunk_pos += len(mono)
            total_samples += len(mono)
    print()


# ── Click track generation ────────────────────────────────────────────────────

def _make_click(freq: float, sample_rate: int) -> np.ndarray:
    n = int(CLICK_S * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t / (CLICK_S * 0.3))
    return (CLICK_AMP * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)


def _make_full_click_track(
    bpm: float,
    count_in_beats: int,
    exercise_beats: int,
    click_mode: str,
    time_sig: int,
    sample_rate: int,
) -> np.ndarray:
    """Build the full click-track audio (count-in + exercise beats)."""
    beat_s        = 60.0 / bpm
    total_beats   = count_in_beats + exercise_beats
    total_samples = int(total_beats * beat_s * sample_rate)
    audio         = np.zeros(total_samples, dtype=np.float32)

    for beat in range(total_beats):
        in_count_in    = beat < count_in_beats
        exercise_beat  = beat - count_in_beats  # negative during count-in

        if click_mode == "off":
            should_click = False
        elif in_count_in:
            freq = ACCENT_FREQ if beat == 0 else BEAT_FREQ
            should_click = True
        elif click_mode == "count-in-only":
            should_click = False
        elif click_mode == "beat":
            freq = ACCENT_FREQ if exercise_beat % time_sig == 0 else BEAT_FREQ
            should_click = True
        elif click_mode == "half":
            should_click = exercise_beat % 2 == 0
            freq = ACCENT_FREQ if exercise_beat % time_sig == 0 else BEAT_FREQ
        elif click_mode == "measure":
            should_click = exercise_beat % time_sig == 0
            freq = ACCENT_FREQ
        else:
            should_click = False

        if should_click:
            click = _make_click(freq, sample_rate)
            start = int(beat * beat_s * sample_rate)
            end   = start + len(click)
            audio[start : min(end, total_samples)] = click[: min(len(click), total_samples - start)]

    return audio


# ── Beat output formatting ────────────────────────────────────────────────────

def _format_beat_line(event: dict, t_now: float) -> str:
    """Format a single beat event for real-time display."""
    idx    = event["target_index"]
    ev     = event["evaluation"]
    sev    = event["severity"].upper()
    t_err  = event["timing_error_s"]
    ms_str = f"{t_err * 1000:+.0f} ms" if t_err is not None else "-- ms"
    hit    = "●" if event.get("detected_note") is not None or ev.get("onset_found") else "·"

    line = f"beat {idx:3d}  {sev:<5}  {ms_str:>7}"

    if "timing_grid" in event:
        bpm    = event["current_bpm"]
        win_ms = event["window_shift_s"] * 1000
        conf   = event["tempo_tracker_confidence"]
        line  += f"  [bpm={bpm:.1f} win={win_ms:+.0f}ms conf={conf:.2f}]"

    line += f"  {hit}"
    return line


# ── Summary helpers (pure functions for testability) ──────────────────────────

def _fixed_timing_error(event: dict) -> float | None:
    """Derive what the fixed-grid timing error would have been from an adaptive event."""
    adj_t = event.get("adjusted_target_time_s")
    err   = event.get("timing_error_s")
    nom_t = event.get("nominal_target_time_s")
    if adj_t is None or err is None or nom_t is None:
        return None
    actual_onset_t = adj_t + err
    return actual_onset_t - nom_t


def _build_timeline(detected: list[bool], time_sig: int) -> str:
    """Build a compact text timeline: ● hit, · miss, │ bar separator."""
    parts: list[str] = []
    for i, hit in enumerate(detected):
        if i > 0 and i % time_sig == 0:
            parts.append("│")
        parts.append("●" if hit else "·")
    return " ".join(parts)


def _compute_summary_stats(
    records: list[dict],
    nominal_bpm: float,
    n_targets: int,
) -> dict:
    """Compute post-run summary statistics from collected beat records.

    Returns a dict with keys:
        n_detected, n_missed, detect_pct,
        mean_adaptive_error_ms, mean_fixed_error_ms,
        final_bpm, final_conf,
        bpm_trajectory (list[float])
    """
    n_detected = sum(1 for r in records if r["detected"])
    n_missed   = n_targets - n_detected
    detect_pct = 100.0 * n_detected / n_targets if n_targets > 0 else 0.0

    adaptive_errors = [
        abs(r["adaptive_error_s"]) for r in records
        if r["adaptive_error_s"] is not None
    ]
    fixed_errors = [
        abs(r["fixed_error_s"]) for r in records
        if r["fixed_error_s"] is not None
    ]

    mean_adaptive_ms = (
        1000.0 * sum(adaptive_errors) / len(adaptive_errors)
        if adaptive_errors else None
    )
    mean_fixed_ms = (
        1000.0 * sum(fixed_errors) / len(fixed_errors)
        if fixed_errors else None
    )

    bpm_trajectory = [r["bpm_estimate"] for r in records]
    final_bpm      = bpm_trajectory[-1] if bpm_trajectory else nominal_bpm
    final_conf     = records[-1]["conf"] if records else 0.0

    return {
        "n_detected":          n_detected,
        "n_missed":            n_missed,
        "detect_pct":          detect_pct,
        "mean_adaptive_ms":    mean_adaptive_ms,
        "mean_fixed_ms":       mean_fixed_ms,
        "final_bpm":           final_bpm,
        "final_conf":          final_conf,
        "bpm_trajectory":      bpm_trajectory,
    }


def _print_summary(
    records: list[dict],
    nominal_bpm: float,
    n_targets: int,
    time_sig: int,
    tracker: TempoTracker | None,
) -> None:
    detected = [r["detected"] for r in records]
    timeline = _build_timeline(detected, time_sig)
    stats    = _compute_summary_stats(records, nominal_bpm, n_targets)

    print("\n" + "─" * 60)
    print("POST-RUN SUMMARY")
    print("─" * 60)
    print(f"  Nominal BPM    : {nominal_bpm:.1f}")
    print(f"  Final BPM      : {stats['final_bpm']:.1f}  (conf={stats['final_conf']:.2f})")
    print(f"  Detected       : {stats['n_detected']}/{n_targets}  ({stats['detect_pct']:.0f}%)")

    if stats["mean_adaptive_ms"] is not None:
        print(f"  Mean |error|   : {stats['mean_adaptive_ms']:.1f} ms (adaptive)", end="")
        if stats["mean_fixed_ms"] is not None:
            diff = stats["mean_fixed_ms"] - stats["mean_adaptive_ms"]
            print(f"  vs {stats['mean_fixed_ms']:.1f} ms (fixed)  → adaptive saves {diff:.1f} ms", end="")
        print()

    print(f"\n  Timeline  (● hit · miss │ bar)\n  {timeline}")

    if stats["bpm_trajectory"]:
        lo  = min(stats["bpm_trajectory"])
        hi  = max(stats["bpm_trajectory"])
        print(f"\n  BPM range      : {lo:.1f} – {hi:.1f}")

    print("─" * 60)


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive timing evaluation demo")
    p.add_argument("--bpm", type=float, default=60.0,
                   help="Nominal tempo in BPM (default 60)")
    p.add_argument("--beats", type=int, default=16,
                   help="Number of exercise beats after count-in (default 16)")
    p.add_argument("--count-in", type=int, default=4,
                   help="Count-in beats before exercise starts (default 4)")
    p.add_argument("--time-sig", type=int, default=4,
                   help="Beats per measure for timeline display and measure-click (default 4)")
    p.add_argument(
        "--click-mode",
        choices=["beat", "half", "measure", "count-in-only", "off"],
        default="beat",
        help="Click pattern during exercise (default: beat)",
    )
    p.add_argument("--device", default=None,
                   help="Input device: integer index or partial name string")
    p.add_argument("--samplerate", type=int, default=None,
                   help="Sample rate in Hz (default: device default)")
    p.add_argument("--level-check", action="store_true",
                   help=f"Run a {METER_SECS:.0f}s input level meter before exercise")
    p.add_argument("--adaptive-window-shift", type=float, default=0.5, metavar="FRAC",
                   help="Window-shift fraction (0–1, default 0.5)")
    p.add_argument("--max-window-shift-beats", type=float, default=0.30, metavar="BEATS",
                   help="Max window shift as fraction of one beat (default 0.30)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    _list_input_devices()
    device, device_sr = _resolve_device(args.device)
    sample_rate       = args.samplerate if args.samplerate is not None else device_sr

    dev_info = sd.query_devices(device if device is not None else sd.default.device[0])
    print(f"Using device [{device}]: {dev_info['name']}")
    print(f"Sample rate : {sample_rate} Hz\n")

    if args.level_check:
        _run_level_meter(device, sample_rate)

    bpm          = args.bpm
    beat_s       = 60.0 / bpm
    count_in     = args.count_in
    n_beats      = args.beats
    time_sig     = args.time_sig
    count_in_s   = count_in * beat_s
    last_beat_s  = count_in_s + (n_beats - 1) * beat_s
    stop_s       = last_beat_s + 2.0

    targets      = [{"time": i} for i in range(n_beats)]
    session      = PracticeSession(targets, bpm, count_in, sample_rate)
    tracker      = TempoTracker(bpm)
    max_shift_s  = args.max_window_shift_beats * beat_s

    print(f"BPM {bpm:.1f}  |  count-in {count_in} beats  |  {n_beats} exercise beats")
    print(f"Click mode: {args.click_mode}  |  time sig: {time_sig}/4")
    print(f"Adaptive timing ON  shift={args.adaptive_window_shift:.2f}  "
          f"max={args.max_window_shift_beats:.2f} beats")
    print(f"Count-in ends at {count_in_s:.1f}s — last target at {last_beat_s:.1f}s")
    print("Ctrl-C to stop early.\n")

    click_audio = _make_full_click_track(
        bpm, count_in, n_beats, args.click_mode, time_sig, sample_rate,
    )

    max_samples = int(sample_rate * stop_s) + BLOCK_FRAMES
    buffer      = np.zeros(max_samples, dtype=np.float64)
    write_pos   = 0
    records: list[dict] = []

    try:
        with sd.InputStream(
            device=device, samplerate=sample_rate, channels=1,
            dtype="float32", blocksize=BLOCK_FRAMES,
        ) as stream:
            sd.play(click_audio, samplerate=sample_rate)

            while write_pos / sample_rate < stop_s:
                raw, _overflowed = stream.read(BLOCK_FRAMES)
                mono = (raw[:, 0].astype(np.float64)
                        if raw.ndim == 2 else raw.astype(np.float64))

                n = len(mono)
                buffer[write_pos : write_pos + n] = mono
                write_pos += n

                t_now = write_pos / sample_rate
                for event in process_realtime_audio(
                    buffer[:write_pos], write_pos, session,
                    tempo_tracker=tracker,
                    adaptive_window_shift=args.adaptive_window_shift,
                    max_window_shift_s=max_shift_s,
                ):
                    print(_format_beat_line(event, t_now))
                    records.append({
                        "beat_index":      event["target_index"],
                        "detected":        event.get("detected_note") is not None
                                           or event["evaluation"].get("onset_found", False),
                        "adaptive_error_s": event.get("timing_error_s"),
                        "fixed_error_s":   _fixed_timing_error(event),
                        "bpm_estimate":    event.get("current_bpm", bpm),
                        "conf":            event.get("tempo_tracker_confidence", 0.0),
                    })

    except KeyboardInterrupt:
        print("\nStopped early.")

    _print_summary(records, bpm, n_beats, time_sig, tracker)


if __name__ == "__main__":
    main()
