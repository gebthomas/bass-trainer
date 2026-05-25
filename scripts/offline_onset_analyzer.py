#!/usr/bin/env python3
"""Offline onset analyzer — load a recorded WAV and detect onset times.

Uses librosa's spectral-flux onset detector, which is more robust than the
simple amplitude-threshold used in live_feedback_demo.py (no aubio required).

This is the companion to --record-wav: record a live session, then run this
script to see what a proper onset detector finds and compare it against the
beat grid.

Run from the project root:
    python scripts/offline_onset_analyzer.py session.wav
    python scripts/offline_onset_analyzer.py session.wav --bpm 60 --count-in 4 --beats 16
    python scripts/offline_onset_analyzer.py session.wav --bpm 60 --count-in 4 --beats 16 --delta 0.07
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.timing_policy import match_window_s as _canonical_match_window


# ── Pure helpers (importable for tests) ───────────────────────────────────────

def beat_grid(bpm: float, count_in: int, n_beats: int) -> np.ndarray:
    """Return absolute nominal beat times in seconds (one per target beat).

    Parameters
    ----------
    bpm        : tempo in beats per minute.
    count_in   : count-in beats before the first target.
    n_beats    : number of target beats.

    Returns
    -------
    np.ndarray of shape (n_beats,), dtype float64.
    """
    beat_s     = 60.0 / bpm
    count_in_s = count_in * beat_s
    return np.array([count_in_s + i * beat_s for i in range(n_beats)])


def match_beats_to_onsets(
    grid: np.ndarray,
    onset_times: np.ndarray,
    window_s: float,
) -> dict:
    """Greedy one-to-one matching between beat targets and detected onsets.

    Each beat is matched to at most one onset; each onset is matched to at
    most one beat.  The closest pair (by absolute timing error) is assigned
    first, then the next-closest from the remaining unassigned pairs, and so
    on.  Only pairs whose timing distance is ≤ window_s are considered.

    Parameters
    ----------
    grid        : 1-D array of nominal beat times in seconds.
    onset_times : 1-D array of detected onset times in seconds.
    window_s    : maximum |error| for a match to be valid.

    Returns
    -------
    dict with keys:

    ``beat_rows`` — list[dict], one entry per beat in grid order:
        beat_idx   int           index in grid
        beat_s     float         nominal beat time (seconds)
        onset_s    float|None    matched onset time, or None if missed
        error_ms   float|None    signed error ms (onset_s − beat_s) × 1000
        matched    bool          True if a match was found

    ``unmatched_onset_times`` — list[float]
        Onset times that were not assigned to any beat (extras).

    ``summary`` — dict:
        beats_total          int
        beats_matched        int
        beats_missed         int
        unmatched_onsets     int
        mean_signed_error_ms float|None  (None if no matched beats)
        mean_abs_error_ms    float|None
    """
    grid        = np.asarray(grid,        dtype=np.float64)
    onset_times = np.asarray(onset_times, dtype=np.float64)

    # Build all candidate (beat_idx, onset_idx) pairs within the window.
    candidates: list[tuple[float, int, int, float]] = []
    for bi in range(len(grid)):
        for oi in range(len(onset_times)):
            err_s = float(onset_times[oi]) - float(grid[bi])
            if abs(err_s) <= window_s:
                candidates.append((abs(err_s), bi, oi, err_s))

    # Greedy assignment: closest pair first.
    candidates.sort()
    used_beats:  set[int] = set()
    used_onsets: set[int] = set()
    assignments: dict[int, tuple[int, float]] = {}   # beat_idx → (onset_idx, err_s)

    for _, bi, oi, err_s in candidates:
        if bi not in used_beats and oi not in used_onsets:
            assignments[bi] = (oi, err_s)
            used_beats.add(bi)
            used_onsets.add(oi)

    # Build beat rows (one per beat, in grid order).
    beat_rows: list[dict] = []
    for bi in range(len(grid)):
        if bi in assignments:
            oi, err_s = assignments[bi]
            beat_rows.append({
                "beat_idx": bi,
                "beat_s":   float(grid[bi]),
                "onset_s":  float(onset_times[oi]),
                "error_ms": err_s * 1000.0,
                "matched":  True,
            })
        else:
            beat_rows.append({
                "beat_idx": bi,
                "beat_s":   float(grid[bi]),
                "onset_s":  None,
                "error_ms": None,
                "matched":  False,
            })

    unmatched = [
        float(onset_times[oi])
        for oi in range(len(onset_times))
        if oi not in used_onsets
    ]

    matched_errors = [r["error_ms"] for r in beat_rows if r["matched"]]
    n_matched = len(matched_errors)
    summary = {
        "beats_total":          len(grid),
        "beats_matched":        n_matched,
        "beats_missed":         len(grid) - n_matched,
        "unmatched_onsets":     len(unmatched),
        "mean_signed_error_ms": sum(matched_errors) / n_matched if n_matched else None,
        "mean_abs_error_ms":    sum(abs(e) for e in matched_errors) / n_matched if n_matched else None,
    }

    return {
        "beat_rows":             beat_rows,
        "unmatched_onset_times": unmatched,
        "summary":               summary,
    }


# ── Onset detection ───────────────────────────────────────────────────────────

def detect_onsets(audio: np.ndarray, sample_rate: int, delta: float = 0.07) -> np.ndarray:
    """Detect onset times (seconds) using librosa spectral-flux.

    Parameters
    ----------
    audio       : 1-D float32 or float64 mono audio.
    sample_rate : audio sample rate in Hz.
    delta       : onset-strength threshold passed to librosa onset_detect.
                  Lower = more sensitive; higher = only strong onsets.
                  Default 0.07 works well for plucked bass.

    Returns
    -------
    np.ndarray of onset times in seconds.

    Notes
    -----
    ``backtrack=True`` causes librosa to snap the reported onset to the
    last energy trough before the peak, giving the moment the attack
    begins rather than the moment the onset-strength peaks.
    aubio is NOT used here.
    """
    import librosa
    return librosa.onset.onset_detect(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        backtrack=True,
        delta=delta,
        units="time",
    )


# ── Formatting ────────────────────────────────────────────────────────────────

def _print_bare(onset_times: np.ndarray) -> None:
    print(f"  {'#':>3}  {'onset_s':>8}")
    print("  " + "─" * 14)
    for i, t in enumerate(onset_times):
        print(f"  {i:>3}  {t:>8.3f}")
    print("\n(Pass --bpm and --beats to compare against a beat grid.)")


def _print_match_table(result: dict) -> None:
    """Print one row per beat, with MISS for unmatched beats."""
    header = f"  {'beat':>4}  {'beat_s':>7}  {'onset_s':>8}  {'err_ms':>7}"
    sep    = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    for row in result["beat_rows"]:
        if row["matched"]:
            print(
                f"  {row['beat_idx']:>4}  {row['beat_s']:>7.3f}"
                f"  {row['onset_s']:>8.3f}  {row['error_ms']:>+7.1f}"
            )
        else:
            print(
                f"  {row['beat_idx']:>4}  {row['beat_s']:>7.3f}"
                f"  {'MISS':>8}  {'—':>7}"
            )

    unmatched = result["unmatched_onset_times"]
    if unmatched:
        times_str = "  ".join(f"{t:.3f}" for t in unmatched)
        print(f"\nUnmatched onsets ({len(unmatched)}): {times_str}")

    s = result["summary"]
    print(
        f"\nSummary:"
        f"  beats {s['beats_matched']}/{s['beats_total']} matched"
        f"  |  {s['beats_missed']} missed"
        f"  |  {s['unmatched_onsets']} unmatched onsets"
    )
    if s["mean_signed_error_ms"] is not None:
        print(
            f"         mean err {s['mean_signed_error_ms']:+.1f} ms"
            f"  |  mean |err| {s['mean_abs_error_ms']:.1f} ms"
        )


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline onset analyzer (librosa spectral-flux, no aubio)"
    )
    p.add_argument("wav", help="WAV file recorded with --record-wav")
    p.add_argument(
        "--bpm", type=float, default=None,
        help="Tempo in BPM — enables beat-grid comparison",
    )
    p.add_argument(
        "--count-in", type=int, default=2, dest="count_in",
        metavar="BEATS",
        help="Count-in beats before the first target (default 2)",
    )
    p.add_argument(
        "--beats", type=int, default=16,
        help="Number of target beats to include in the grid (default 16)",
    )
    p.add_argument(
        "--match-window", type=float, default=None, dest="match_window",
        metavar="S",
        help="Match window in seconds (default: canonical timing_policy for the given BPM)",
    )
    p.add_argument(
        "--delta", type=float, default=0.07,
        help="Librosa onset-strength threshold (lower = more sensitive; default 0.07)",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    import librosa

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration_s = len(audio) / sr
    print(
        f"Loaded  : {wav_path.name}\n"
        f"Duration: {duration_s:.2f}s  |  {sr} Hz  |  {len(audio)} samples\n"
        f"Detector: librosa spectral-flux  backtrack=True  delta={args.delta:.3f}\n"
        f"          (aubio bypassed — uses librosa onset_detect)\n"
    )

    onset_times = detect_onsets(audio, sr, delta=args.delta)
    print(f"Detected {len(onset_times)} onsets.\n")

    if args.bpm is not None:
        grid   = beat_grid(args.bpm, args.count_in, args.beats)
        win_s  = args.match_window if args.match_window is not None \
                 else _canonical_match_window(args.bpm)
        result = match_beats_to_onsets(grid, onset_times, win_s)
        print(
            f"Beat grid: BPM={args.bpm:.0f}  count_in={args.count_in}"
            f"  beats={args.beats}  match_window={win_s:.3f}s\n"
        )
        _print_match_table(result)
    else:
        _print_bare(onset_times)


if __name__ == "__main__":
    main()
