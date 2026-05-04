"""
Pitch presence diagnostic per target note window.

Usage:
    python tools/analyze_target_windows.py targets.json audio.wav
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 48000
HOP_LENGTH = 512
FRAME_LENGTH = 4096

WINDOW_PRE_SEC = 0.10
WINDOW_POST_SEC = 0.35
PITCH_MATCH_CENTS = 50.0
FOUND_THRESHOLD = 0.25   # min pitch_match_ratio to call "found"


def load_targets(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y


def compute_pitch(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("G4"),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        frame_length=FRAME_LENGTH,
    )
    times = librosa.times_like(f0, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return times, f0, voiced_flag


def analyze_target(target: dict, times: np.ndarray, f0: np.ndarray, voiced_flag: np.ndarray) -> dict:
    target_time = target["time"]
    target_note = target["note"]
    target_hz = float(librosa.note_to_hz(target_note))

    t_lo = target_time - WINDOW_PRE_SEC
    t_hi = target_time + WINDOW_POST_SEC
    window = (times >= t_lo) & (times <= t_hi)

    win_times   = times[window]
    win_f0      = f0[window]
    win_voiced  = voiced_flag[window]

    n_frames = len(win_times)
    n_voiced = int(np.sum(win_voiced))
    voiced_frame_ratio = n_voiced / n_frames if n_frames > 0 else 0.0

    # Frames with a usable pitch estimate (voiced + non-NaN + positive)
    usable = win_voiced & ~np.isnan(win_f0) & (win_f0 > 0)
    usable_f0    = win_f0[usable]
    usable_times = win_times[usable]

    if len(usable_f0) == 0:
        return _result(target_note, target_time, None, None, None,
                       voiced_frame_ratio, 0.0, "missed")

    cents_errors = 1200.0 * np.log2(usable_f0 / target_hz)
    match = np.abs(cents_errors) <= PITCH_MATCH_CENTS

    pitch_match_ratio = float(np.sum(match)) / len(usable_f0)

    if np.any(match):
        first_time      = float(usable_times[match][0])
        median_cents    = float(np.median(cents_errors[match]))
        matching_cents  = cents_errors[match]
        if len(matching_cents) > 1:
            stability = float(np.percentile(matching_cents, 75) -
                              np.percentile(matching_cents, 25))
        else:
            stability = 0.0
    else:
        first_time   = None
        median_cents = None
        stability    = None

    if pitch_match_ratio >= FOUND_THRESHOLD:
        status = "found"
    elif pitch_match_ratio > 0:
        status = "weak"
    else:
        status = "missed"

    return _result(target_note, target_time, first_time, median_cents, stability,
                   voiced_frame_ratio, pitch_match_ratio, status)


def _result(note, time, first, median_c, stability_c, voiced_ratio, match_ratio, status):
    return {
        "target_note":              note,
        "target_time":              time,
        "first_matching_pitch_time": first,
        "median_cents_error":       median_c,
        "pitch_stability_cents":    stability_c,
        "voiced_frame_ratio":       voiced_ratio,
        "pitch_match_ratio":        match_ratio,
        "status":                   status,
    }


def _fmt(value, fmt, missing="—"):
    return format(value, fmt) if value is not None else missing


def _delay_ms(r: dict) -> float | None:
    if r["first_matching_pitch_time"] is None:
        return None
    return 1000.0 * (r["first_matching_pitch_time"] - r["target_time"])


def print_report(results: list[dict], estimated_offset_ms: float | None) -> None:
    col = "{:>5}  {:>7}  {:>11}  {:>8}  {:>7}  {:>9}  {:>9}  {:>6}  {:>6}  {}"
    header = col.format("Note", "Target", "First match", "Delay ms", "Adj ms",
                        "Cents err", "Stability", "Voiced", "Match%", "Status")
    print(header)
    print("-" * len(header))
    for r in results:
        d = _delay_ms(r)
        first = (_fmt(r["first_matching_pitch_time"], ".3f") + "s"
                 if r["first_matching_pitch_time"] is not None else "—")
        delay = _fmt(d, "+.0f") + " ms" if d is not None else "—"
        adj   = (_fmt(d - estimated_offset_ms, "+.0f") + " ms"
                 if d is not None and estimated_offset_ms is not None else "—")
        cents = (_fmt(r["median_cents_error"], "+.1f") + "c"
                 if r["median_cents_error"] is not None else "—")
        stab  = (_fmt(r["pitch_stability_cents"], ".1f") + "c"
                 if r["pitch_stability_cents"] is not None else "—")
        print(col.format(
            r["target_note"],
            f"{r['target_time']:.3f}s",
            first,
            delay,
            adj,
            cents,
            stab,
            f"{r['voiced_frame_ratio']:.0%}",
            f"{r['pitch_match_ratio']:.0%}",
            r["status"],
        ))


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python tools/analyze_target_windows.py targets.json audio.wav")
        return 1

    targets = load_targets(Path(sys.argv[1]))
    y = load_audio(Path(sys.argv[2]))
    times, f0, voiced_flag = compute_pitch(y)

    results = [analyze_target(t, times, f0, voiced_flag) for t in targets]

    delays = [_delay_ms(r) for r in results
              if r["status"] in ("found", "weak") and _delay_ms(r) is not None]
    estimated_offset_ms = float(np.median(delays)) if delays else None

    print_report(results, estimated_offset_ms)

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("found", "weak", "missed")}
    print(f"\n{counts['found']} found  {counts['weak']} weak  {counts['missed']} missed"
          f"  (of {len(results)} targets)")

    if estimated_offset_ms is not None:
        print(f"Estimated offset: {estimated_offset_ms:+.0f} ms"
              f"  (median first-match delay across {len(delays)} found/weak targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
