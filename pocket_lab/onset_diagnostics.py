"""Onset detection diagnostics: spacing, multi-fire detection, strength distribution."""

from __future__ import annotations

import numpy as np


def onset_spacing_stats(onset_times: np.ndarray) -> dict:
    """Compute inter-onset interval statistics."""
    if len(onset_times) < 2:
        return {
            "count": len(onset_times),
            "spacings": np.array([]),
            "min_spacing_ms": 0.0,
            "median_spacing_ms": 0.0,
            "mean_spacing_ms": 0.0,
        }
    sorted_t = np.sort(onset_times)
    spacings = np.diff(sorted_t) * 1000.0
    return {
        "count": len(onset_times),
        "spacings": spacings,
        "min_spacing_ms": float(np.min(spacings)),
        "median_spacing_ms": float(np.median(spacings)),
        "mean_spacing_ms": float(np.mean(spacings)),
    }


def close_pair_counts(
    onset_times: np.ndarray,
    thresholds_ms: list[float] | None = None,
) -> list[dict]:
    """Count how many onsets have a neighbor within each threshold."""
    if thresholds_ms is None:
        thresholds_ms = [10.0, 20.0, 30.0, 50.0]
    if len(onset_times) < 2:
        return [{"threshold_ms": t, "count": 0, "fraction": 0.0} for t in thresholds_ms]

    sorted_t = np.sort(onset_times)
    spacings = np.diff(sorted_t) * 1000.0

    results = []
    for thr in thresholds_ms:
        close_mask = spacings < thr
        n_close = int(np.sum(close_mask))
        n_onsets_in_close_pair = 0
        for i, is_close in enumerate(close_mask):
            if is_close:
                n_onsets_in_close_pair += 1
                if i == 0 or not close_mask[i - 1]:
                    n_onsets_in_close_pair += 1
        n_onsets_in_close_pair = min(n_onsets_in_close_pair, len(sorted_t))
        results.append({
            "threshold_ms": thr,
            "count": n_onsets_in_close_pair,
            "fraction": n_onsets_in_close_pair / len(sorted_t),
        })
    return results


def spacing_histogram(
    onset_times: np.ndarray,
    bin_edges_ms: list[float] | None = None,
) -> list[dict]:
    """Bin inter-onset intervals into a histogram."""
    if bin_edges_ms is None:
        bin_edges_ms = [0, 15, 30, 50, 100, 200, 500, 1000, float("inf")]
    if len(onset_times) < 2:
        return []

    sorted_t = np.sort(onset_times)
    spacings_ms = np.diff(sorted_t) * 1000.0

    rows = []
    for i in range(len(bin_edges_ms) - 1):
        lo = bin_edges_ms[i]
        hi = bin_edges_ms[i + 1]
        count = int(np.sum((spacings_ms >= lo) & (spacings_ms < hi)))
        label = f"{lo:.0f}–{hi:.0f}ms" if hi < float("inf") else f"{lo:.0f}ms+"
        rows.append({"label": label, "lo": lo, "hi": hi, "count": count})
    return rows


def strength_distribution(strengths: np.ndarray) -> dict:
    """Summarize onset strength distribution."""
    if len(strengths) == 0:
        return {"count": 0, "zero_count": 0, "min": 0.0, "max": 0.0,
                "mean": 0.0, "median": 0.0, "below_001": 0, "below_01": 0,
                "below_05": 0}
    return {
        "count": len(strengths),
        "zero_count": int(np.sum(strengths == 0.0)),
        "min": float(np.min(strengths)),
        "max": float(np.max(strengths)),
        "mean": float(np.mean(strengths)),
        "median": float(np.median(strengths)),
        "below_001": int(np.sum(strengths < 0.01)),
        "below_01": int(np.sum(strengths < 0.1)),
        "below_05": int(np.sum(strengths < 0.5)),
    }


def frame_quantization_info(sr: int, hop_length: int = 512) -> dict:
    """Report the time resolution imposed by the hop size."""
    frame_s = hop_length / sr
    frame_ms = frame_s * 1000.0
    return {
        "sr": sr,
        "hop_length": hop_length,
        "frame_period_ms": frame_ms,
        "frames_per_second": sr / hop_length,
    }


def full_onset_diagnostic(
    onset_times: np.ndarray,
    strengths: np.ndarray,
    label: str,
    sr: int,
) -> dict:
    """Run all diagnostics for one take's onsets."""
    return {
        "label": label,
        "spacing": onset_spacing_stats(onset_times),
        "close_pairs": close_pair_counts(onset_times),
        "histogram": spacing_histogram(onset_times),
        "strength": strength_distribution(strengths),
        "quantization": frame_quantization_info(sr),
    }
