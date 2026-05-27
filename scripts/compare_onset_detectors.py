#!/usr/bin/env python3
"""Compare multiple onset-detection strategies on one recorded WAV.

Three detectors are aligned to the same beat grid so you can see where they
agree and disagree before changing live behavior.

Detectors
---------
  live     — window-by-window evaluate_window (same extraction + rise-detection
              as the live pipeline; one window per beat, no cross-beat sharing)
  spectral — librosa spectral-flux + backtrack (skipped if librosa missing)
  energy   — frame-energy derivative, pure numpy, no extra deps

Run from project root:
    python scripts/compare_onset_detectors.py session.wav --bpm 60 --count-in 4 --beats 16
    python scripts/compare_onset_detectors.py session.wav --bpm 60 --count-in 4 --beats 16 --delta 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio_windows import extract_target_window
from core.window_analyzer import evaluate_window
from core.timing_policy import match_window_s as _canonical_match_window
from scripts.offline_onset_analyzer import beat_grid, match_beats_to_onsets


# ── Detector 1: windowed live evaluator ───────────────────────────────────────

def windowed_live_detect(
    audio: np.ndarray,
    sample_rate: int,
    bpm: float,
    count_in: int,
    n_beats: int,
    pre_roll_s: float = 0.03,
    post_roll_s: float = 0.10,
) -> list[dict]:
    """Apply evaluate_window beat-by-beat, mimicking the live pipeline.

    Each beat gets its own extraction window; onset detection uses the
    rise-based dynamic threshold (same as realtime_evaluator.evaluate_window).
    There is no global onset search — beats are evaluated independently.

    Parameters
    ----------
    audio       : 1-D mono float audio array.
    sample_rate : audio sample rate in Hz.
    bpm         : tempo in beats per minute.
    count_in    : count-in beats before the first target.
    n_beats     : number of target beats.
    pre_roll_s  : seconds before the beat to include in the extraction window.
    post_roll_s : seconds after the beat to include in the extraction window.

    Returns
    -------
    list[dict], one per beat:
        beat_idx   int
        beat_s     float   nominal beat time in seconds
        onset_s    float | None
        error_ms   float | None   positive = late
        matched    bool
    """
    beat_s     = 60.0 / bpm
    count_in_s = count_in * beat_s
    rows: list[dict] = []

    for i in range(n_beats):
        target         = {"time": float(i)}
        nominal_beat_s = count_in_s + i * beat_s

        window = extract_target_window(
            audio, target, bpm, count_in, sample_rate,
            pre_roll_s=pre_roll_s, post_roll_s=post_roll_s,
        )
        ev = evaluate_window(window["audio"], sample_rate)

        if ev["onset_found"] and ev["onset_sample"] is not None:
            abs_sample = window["start_sample"] + ev["onset_sample"]
            onset_s    = abs_sample / sample_rate
            rows.append({
                "beat_idx": i,
                "beat_s":   nominal_beat_s,
                "onset_s":  onset_s,
                "error_ms": (onset_s - nominal_beat_s) * 1000.0,
                "matched":  True,
            })
        else:
            rows.append({
                "beat_idx": i,
                "beat_s":   nominal_beat_s,
                "onset_s":  None,
                "error_ms": None,
                "matched":  False,
            })

    return rows


# ── Detector 2: librosa spectral-flux ─────────────────────────────────────────

def spectral_flux_detect(
    audio: np.ndarray,
    sample_rate: int,
    delta: float = 0.07,
) -> np.ndarray:
    """Librosa spectral-flux onset detection with backtracking.

    Raises ImportError if librosa is not installed.
    Returns onset times in seconds.
    """
    import librosa
    return librosa.onset.onset_detect(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        backtrack=True,
        delta=delta,
        units="time",
    )


def _librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


# ── Detector 3: frame-energy derivative ───────────────────────────────────────

def energy_derivative_detect(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 512,
    hop_length: int = 128,
    threshold: float = 0.02,
    min_gap_s: float = 0.05,
) -> np.ndarray:
    """Frame-energy derivative onset detector (pure numpy, no extra deps).

    Divides audio into overlapping frames, computes peak absolute amplitude
    per frame, then takes the positive first-order difference as onset
    strength.  Frames where strength exceeds *threshold* with a minimum
    inter-onset gap are reported as onsets.

    Parameters
    ----------
    audio        : 1-D mono float audio array.
    sample_rate  : audio sample rate in Hz.
    frame_length : samples per analysis frame (default 512 ≈ 10 ms at 48 kHz).
    hop_length   : frame hop in samples (default 128 ≈ 2.7 ms at 48 kHz);
                   also the onset-time resolution.
    threshold    : minimum frame-energy increase to count as an onset.
    min_gap_s    : minimum time between consecutive onsets in seconds.

    Returns
    -------
    np.ndarray of onset times in seconds (start of the rising frame).
    """
    mono = np.asarray(audio, dtype=np.float64).ravel()
    if mono.size < frame_length:
        return np.array([], dtype=np.float64)

    n_frames = 1 + (len(mono) - frame_length) // hop_length
    frame_max = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop_length
        frame_max[i] = np.max(np.abs(mono[s:s + frame_length]))

    # Half-wave rectified first-order difference (onset strength proxy).
    strength = np.maximum(0.0, np.diff(frame_max))

    min_gap_frames = max(1, round(min_gap_s * sample_rate / hop_length))
    onset_times: list[float] = []
    last_frame  = -min_gap_frames

    for i, s in enumerate(strength):
        if s >= threshold and i - last_frame >= min_gap_frames:
            # Frame i+1 is the rising frame; report its start sample.
            onset_times.append((i + 1) * hop_length / sample_rate)
            last_frame = i

    return np.array(onset_times, dtype=np.float64)


# ── Table helpers ──────────────────────────────────────────────────────────────

def build_comparison_table(
    grid: np.ndarray,
    detector_results: list[tuple[str, list[dict]]],
) -> list[dict]:
    """Combine per-detector beat_rows lists into one row per beat.

    Parameters
    ----------
    grid              : 1-D array of nominal beat times in seconds.
    detector_results  : list of ``(name, beat_rows)`` pairs.  Each
                        *beat_rows* must have the same length as *grid*
                        and contain dicts with at least ``matched`` (bool)
                        and ``error_ms`` (float | None).

    Returns
    -------
    list[dict], one per beat:
        beat_idx           int
        beat_s             float
        <name>_ms          float | None   per detector
        <name>_matched     bool           per detector
    """
    grid  = np.asarray(grid, dtype=np.float64)
    rows: list[dict] = []
    for bi in range(len(grid)):
        row: dict = {"beat_idx": bi, "beat_s": float(grid[bi])}
        for name, beat_rows in detector_results:
            r = beat_rows[bi]
            row[f"{name}_ms"]      = r["error_ms"]
            row[f"{name}_matched"] = r["matched"]
        rows.append(row)
    return rows


def detector_summary(beat_rows: list[dict]) -> dict:
    """Compute summary stats from a beat_rows list.

    Returns
    -------
    dict:
        beats_total, beats_matched, beats_missed,
        mean_signed_error_ms  (None if no matches)
        mean_abs_error_ms     (None if no matches)
    """
    errors = [r["error_ms"] for r in beat_rows if r["matched"]]
    n = len(errors)
    return {
        "beats_total":          len(beat_rows),
        "beats_matched":        n,
        "beats_missed":         len(beat_rows) - n,
        "mean_signed_error_ms": sum(errors) / n if n else None,
        "mean_abs_error_ms":    sum(abs(e) for e in errors) / n if n else None,
    }


# ── Formatting ─────────────────────────────────────────────────────────────────

def _fmt_err(val: float | None) -> str:
    return "MISS" if val is None else f"{val:+.1f}"


def _print_comparison_table(
    table: list[dict],
    detector_names: list[str],
    unmatched_by_detector: dict[str, list[float]],
) -> None:
    col_w = max(8, *(len(n) + 4 for n in detector_names))
    header = f"  {'beat':>4}  {'beat_s':>7}"
    for n in detector_names:
        header += f"  {n:>{col_w}}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for row in table:
        line = f"  {row['beat_idx']:>4}  {row['beat_s']:>7.3f}"
        for n in detector_names:
            cell = _fmt_err(row[f"{n}_ms"])
            line += f"  {cell:>{col_w}}"
        print(line)

    for n, times in unmatched_by_detector.items():
        if times:
            ts = "  ".join(f"{t:.3f}" for t in times)
            print(f"\n{n} unmatched ({len(times)}): {ts}")


def _print_summary_table(
    detector_names: list[str],
    summaries: dict[str, dict],
    unmatched_by_detector: dict[str, list[float]],
) -> None:
    col_w  = max(10, *(len(n) + 2 for n in detector_names))
    header = (
        f"\n  {'Detector':<{col_w}}  {'matched':>7}  {'missed':>6}"
        f"  {'extras':>6}  {'mean_err':>9}  {'mean|err|':>9}"
    )
    print(header)
    print("  " + "─" * (len(header) - 3))

    for n in detector_names:
        s      = summaries[n]
        extras = len(unmatched_by_detector.get(n, []))
        ms_s   = f"{s['mean_signed_error_ms']:+.1f}" if s["mean_signed_error_ms"] is not None else "—"
        ms_a   = f"{s['mean_abs_error_ms']:.1f}"     if s["mean_abs_error_ms"]    is not None else "—"
        print(
            f"  {n:<{col_w}}  {s['beats_matched']:>7}  {s['beats_missed']:>6}"
            f"  {extras:>6}  {ms_s:>9}  {ms_a:>9}"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare onset detectors on a recorded WAV"
    )
    p.add_argument("wav", help="WAV file (e.g. recorded with --record-wav)")
    p.add_argument("--bpm",          type=float, required=True)
    p.add_argument("--count-in",     type=int,   default=2,    dest="count_in", metavar="BEATS")
    p.add_argument("--beats",        type=int,   default=16)
    p.add_argument("--delta",        type=float, default=0.07,
                   help="Librosa onset-strength threshold (default 0.07)")
    p.add_argument("--match-window", type=float, default=None, dest="match_window", metavar="S",
                   help="Match window in seconds (default: timing_policy for the BPM)")
    return p.parse_args()


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float64. Tries librosa then scipy."""
    if _librosa_available():
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(audio, dtype=np.float64), sr

    from scipy.io import wavfile
    sr, raw = wavfile.read(path)
    audio   = raw.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if raw.dtype == np.int16:
        audio /= 32768.0
    elif raw.dtype == np.int32:
        audio /= 2 ** 31
    return audio, sr


def main() -> None:
    args = _parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    audio, sr = _load_wav(str(wav_path))
    duration_s = len(audio) / sr
    win_s      = args.match_window if args.match_window is not None \
                 else _canonical_match_window(args.bpm)
    grid       = beat_grid(args.bpm, args.count_in, args.beats)

    print(
        f"File    : {wav_path.name}\n"
        f"Duration: {duration_s:.2f}s  |  {sr} Hz\n"
        f"Grid    : BPM={args.bpm:.0f}  count_in={args.count_in}"
        f"  beats={args.beats}  match_window={win_s:.3f}s\n"
    )

    # ── Detector 1: live windowed ──────────────────────────────────────────────
    print("Running live windowed detector …")
    live_rows = windowed_live_detect(audio, sr, args.bpm, args.count_in, args.beats)

    # ── Detector 2: librosa spectral-flux ─────────────────────────────────────
    spec_available = _librosa_available()
    spec_rows: list[dict] = []
    spec_unmatched: list[float] = []

    if spec_available:
        print("Running librosa spectral-flux detector …")
        spec_times     = spectral_flux_detect(audio, sr, delta=args.delta)
        spec_result    = match_beats_to_onsets(grid, spec_times, win_s)
        spec_rows      = spec_result["beat_rows"]
        spec_unmatched = spec_result["unmatched_onset_times"]
    else:
        print("librosa not available — skipping spectral-flux detector")

    # ── Detector 3: energy derivative ─────────────────────────────────────────
    print("Running energy-derivative detector …")
    enrg_times     = energy_derivative_detect(audio, sr)
    enrg_result    = match_beats_to_onsets(grid, enrg_times, win_s)
    enrg_rows      = enrg_result["beat_rows"]
    enrg_unmatched = enrg_result["unmatched_onset_times"]

    # ── Assemble and print ─────────────────────────────────────────────────────
    print()

    detector_results: list[tuple[str, list[dict]]] = [("live", live_rows)]
    unmatched: dict[str, list[float]] = {"live": []}

    if spec_available:
        detector_results.append(("spectral", spec_rows))
        unmatched["spectral"] = spec_unmatched

    detector_results.append(("energy", enrg_rows))
    unmatched["energy"] = enrg_unmatched

    detector_names = [n for n, _ in detector_results]
    table          = build_comparison_table(grid, detector_results)

    _print_comparison_table(table, detector_names, unmatched)

    summaries = {n: detector_summary(rows) for n, rows in detector_results}
    _print_summary_table(detector_names, summaries, unmatched)
    print()


if __name__ == "__main__":
    main()
