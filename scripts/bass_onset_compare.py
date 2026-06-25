#!/usr/bin/env python3
"""Bass-to-bass onset timing comparison in master-song time.

Compares reference bass (from demucs separation) against two recorded
takes, aligned to master time using pre-computed song-channel offsets.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent
BASS_REF = PROJECT / "music-library" / "Mr. Brightside" / "bass.wav"
TAKE_A = PROJECT / "personal recordings" / "June-22-2026" / "Mr Brightside memory.wav"
TAKE_B = PROJECT / "personal recordings" / "June-22-2026" / "Mr Brightside memory 2.wav"

OFFSET_A = 1.4307
OFFSET_B = 0.0834

ANALYSIS_START = 45.0
ANALYSIS_END = 200.0
MATCH_WINDOW_MS = 150.0


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def load_left(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    return data[:, 0], sr


def detect_onsets(audio: np.ndarray, sr: int, delta: float = 0.07) -> np.ndarray:
    return librosa.onset.onset_detect(
        y=audio.astype(np.float32), sr=sr,
        backtrack=True, delta=delta, units="time",
    )


def filter_to_window(onsets: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (onsets >= start) & (onsets <= end)
    return onsets[mask]


def match_onsets(
    onsets_a: np.ndarray,
    onsets_b: np.ndarray,
    max_window_s: float,
) -> dict:
    used_b = set()
    matched_a_idx = []
    matched_b_idx = []
    diffs_ms = []

    for i, ta in enumerate(onsets_a):
        best_j = None
        best_dist = max_window_s + 1
        for j, tb in enumerate(onsets_b):
            if j in used_b:
                continue
            dist = abs(ta - tb)
            if dist <= max_window_s and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            used_b.add(best_j)
            matched_a_idx.append(i)
            matched_b_idx.append(best_j)
            diffs_ms.append((onsets_b[best_j] - onsets_a[i]) * 1000.0)

    unmatched_a = sorted(set(range(len(onsets_a))) - set(matched_a_idx))
    unmatched_b = sorted(set(range(len(onsets_b))) - set(matched_b_idx))

    return {
        "matched_a_idx": matched_a_idx,
        "matched_b_idx": matched_b_idx,
        "diffs_ms": np.array(diffs_ms),
        "unmatched_a_idx": unmatched_a,
        "unmatched_b_idx": unmatched_b,
    }


def report_comparison(
    label_a: str, label_b: str,
    onsets_a: np.ndarray, onsets_b: np.ndarray,
    result: dict,
) -> None:
    d = result["diffs_ms"]
    n_matched = len(d)
    n_a = len(onsets_a)
    n_b = len(onsets_b)

    print(f"\n{'=' * 60}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'=' * 60}")
    print(f"  {label_a} onsets:       {n_a}")
    print(f"  {label_b} onsets:       {n_b}")
    print(f"  Matched:              {n_matched}")
    print(f"  Unmatched {label_a}:    {len(result['unmatched_a_idx'])}")
    print(f"  Unmatched {label_b}:    {len(result['unmatched_b_idx'])}")

    if n_matched == 0:
        print("  No matches — skipping statistics.")
        return

    print(f"\n  Timing differences ({label_b} − {label_a}):")
    print(f"    Mean signed:        {np.mean(d):+.1f} ms")
    print(f"    Mean absolute:      {np.mean(np.abs(d)):.1f} ms")
    print(f"    Std deviation:      {np.std(d):.1f} ms")
    print(f"    Median:             {np.median(d):+.1f} ms")
    p25, p75 = np.percentile(d, [25, 75])
    print(f"    25th percentile:    {p25:+.1f} ms")
    print(f"    75th percentile:    {p75:+.1f} ms")
    print(f"    Min:                {np.min(d):+.1f} ms")
    print(f"    Max:                {np.max(d):+.1f} ms")


def main() -> None:
    print("Loading audio...")
    bass_ref, sr_ref = load_mono(BASS_REF)
    take_a_bass, sr_a = load_left(TAKE_A)
    take_b_bass, sr_b = load_left(TAKE_B)

    print(f"  Reference bass: {len(bass_ref)/sr_ref:.1f}s @ {sr_ref} Hz")
    print(f"  Take A bass:    {len(take_a_bass)/sr_a:.1f}s @ {sr_a} Hz")
    print(f"  Take B bass:    {len(take_b_bass)/sr_b:.1f}s @ {sr_b} Hz")

    print("\nDetecting onsets...")
    ref_onsets = detect_onsets(bass_ref, sr_ref)
    a_onsets_raw = detect_onsets(take_a_bass, sr_a)
    b_onsets_raw = detect_onsets(take_b_bass, sr_b)

    print(f"  Reference: {len(ref_onsets)} onsets (full song)")
    print(f"  Take A:    {len(a_onsets_raw)} onsets (full recording)")
    print(f"  Take B:    {len(b_onsets_raw)} onsets (full recording)")

    # Convert to master time
    a_onsets_master = a_onsets_raw + OFFSET_A
    b_onsets_master = b_onsets_raw + OFFSET_B

    # Filter to analysis window
    ref = filter_to_window(ref_onsets, ANALYSIS_START, ANALYSIS_END)
    a_m = filter_to_window(a_onsets_master, ANALYSIS_START, ANALYSIS_END)
    b_m = filter_to_window(b_onsets_master, ANALYSIS_START, ANALYSIS_END)

    print(f"\nAfter filtering to {ANALYSIS_START}–{ANALYSIS_END} s (master time):")
    print(f"  Reference: {len(ref)} onsets")
    print(f"  Take A:    {len(a_m)} onsets")
    print(f"  Take B:    {len(b_m)} onsets")

    max_w = MATCH_WINDOW_MS / 1000.0

    res_ref_a = match_onsets(ref, a_m, max_w)
    res_ref_b = match_onsets(ref, b_m, max_w)
    res_a_b = match_onsets(a_m, b_m, max_w)

    report_comparison("Reference", "Take A", ref, a_m, res_ref_a)
    report_comparison("Reference", "Take B", ref, b_m, res_ref_b)
    report_comparison("Take A", "Take B", a_m, b_m, res_a_b)

    # ── Histogram figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    pairs = [
        ("Reference vs Take A", "Take A − Reference", res_ref_a),
        ("Reference vs Take B", "Take B − Reference", res_ref_b),
        ("Take A vs Take B", "Take B − Take A", res_a_b),
    ]

    for ax, (title, xlabel, res) in zip(axes, pairs):
        d = res["diffs_ms"]
        if len(d) == 0:
            ax.set_title(f"{title} — no matches")
            continue

        bins = np.arange(-MATCH_WINDOW_MS, MATCH_WINDOW_MS + 5, 5)
        ax.hist(d, bins=bins, color="#4a90d9", edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(np.mean(d), color="red", linewidth=1.5, label=f"mean = {np.mean(d):+.1f} ms")
        ax.axvline(np.median(d), color="orange", linewidth=1.5, linestyle="--",
                   label=f"median = {np.median(d):+.1f} ms")
        ax.set_title(f"{title}  ({len(d)} matched)")
        ax.set_xlabel(f"{xlabel} (ms)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")

    fig.suptitle(
        f"Bass onset timing differences — {ANALYSIS_START:.0f}–{ANALYSIS_END:.0f} s",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = PROJECT / "bass_onset_compare.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nHistogram saved to {out_path}")


if __name__ == "__main__":
    main()
