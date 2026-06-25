#!/usr/bin/env python3
"""Estimate alignment offset between master recording and each take."""

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from math import gcd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MASTER = PROJECT / "music-library" / "Mr. Brightside" / "master.wav"
TAKE_A = PROJECT / "personal recordings" / "June-22-2026" / "Mr Brightside memory.wav"
TAKE_B = PROJECT / "personal recordings" / "June-22-2026" / "Mr Brightside memory 2.wav"

START_S = 90
DURATION_S = 30


def load_mono(path):
    data, sr = sf.read(path, dtype="float64")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def load_right_channel(path):
    data, sr = sf.read(path, dtype="float64")
    return data[:, 1], sr


def resample_to(signal, sr_orig, sr_target):
    if sr_orig == sr_target:
        return signal
    g = gcd(sr_orig, sr_target)
    up = sr_target // g
    down = sr_orig // g
    return resample_poly(signal, up, down)


def normalized_cross_correlation(excerpt, full_signal):
    n = len(excerpt)
    excerpt_norm = excerpt - excerpt.mean()
    excerpt_energy = np.sqrt(np.sum(excerpt_norm ** 2))

    full_norm = full_signal - full_signal.mean()
    corr = fftconvolve(full_norm, excerpt_norm[::-1], mode="full")

    # Running energy of full_signal in windows of length n
    cumsum_sq = np.concatenate(([0], np.cumsum(full_norm ** 2)))
    # The correlation output has length len(full) + len(excerpt) - 1
    # At index k, the excerpt aligns starting at position (k - n + 1) in full_signal
    window_energy = np.zeros(len(corr))
    for k in range(len(corr)):
        start = k - n + 1
        end = start + n
        s = max(start, 0)
        e = min(end, len(full_norm))
        if e > s:
            window_energy[k] = np.sqrt(cumsum_sq[e] - cumsum_sq[s])

    valid = window_energy > 0
    ncc = np.zeros_like(corr)
    ncc[valid] = corr[valid] / (excerpt_energy * window_energy[valid])
    return ncc


def estimate_offset(master_mono, sr_master, take_song, sr_take):
    sr_target = max(sr_master, sr_take)
    master_r = resample_to(master_mono, sr_master, sr_target)
    take_r = resample_to(take_song, sr_take, sr_target)

    start_sample = int(START_S * sr_target)
    end_sample = int((START_S + DURATION_S) * sr_target)
    excerpt = master_r[start_sample:end_sample]

    ncc = normalized_cross_correlation(excerpt, take_r)

    peak_idx = np.argmax(ncc)
    confidence = ncc[peak_idx]

    n = len(excerpt)
    # At peak_idx in the full-mode correlation, the excerpt starts at
    # position (peak_idx - n + 1) in the take signal.
    take_start_sample = peak_idx - n + 1
    # The excerpt came from master at start_sample.
    # offset = (master position) - (take position) in seconds
    # Positive means the take occurs later than the master.
    offset_s = (start_sample - take_start_sample) / sr_target

    lag_axis = (np.arange(len(ncc)) - n + 1) / sr_target
    offset_axis = start_sample / sr_target - lag_axis

    return offset_s, confidence, offset_axis, ncc


def main():
    print("Loading audio files...")
    master_mono, sr_master = load_mono(MASTER)
    take_a_song, sr_a = load_right_channel(TAKE_A)
    take_b_song, sr_b = load_right_channel(TAKE_B)

    print("Estimating Take A alignment...")
    offset_a, conf_a, offset_axis_a, ncc_a = estimate_offset(
        master_mono, sr_master, take_a_song, sr_a
    )

    print("Estimating Take B alignment...")
    offset_b, conf_b, offset_axis_b, ncc_b = estimate_offset(
        master_mono, sr_master, take_b_song, sr_b
    )

    print(f"\nTake A offset: {offset_a:.4f} s")
    print(f"Take A confidence: {conf_a:.4f}")
    print(f"Take B offset: {offset_b:.4f} s")
    print(f"Take B confidence: {conf_b:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, label, offset, conf, offset_axis, ncc in [
        (axes[0], "Take A", offset_a, conf_a, offset_axis_a, ncc_a),
        (axes[1], "Take B", offset_b, conf_b, offset_axis_b, ncc_b),
    ]:
        ax.plot(offset_axis, ncc, linewidth=0.5)
        ax.axvline(offset, color="r", linestyle="--", linewidth=1.5)
        ax.set_title(f"{label} — offset = {offset:.4f} s, confidence = {conf:.4f}")
        ax.set_xlabel("Offset (s)  [positive = take later than master]")
        ax.set_ylabel("Normalized correlation")
        ax.legend(["Correlation", f"Peak @ {offset:.4f} s"], loc="upper right")

    fig.suptitle("Song-channel alignment: master vs. takes", fontsize=14)
    fig.tight_layout()

    out_path = PROJECT / "alignment_diagnostics.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nDiagnostic figure saved to {out_path}")


if __name__ == "__main__":
    main()
