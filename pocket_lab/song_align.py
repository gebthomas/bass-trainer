"""Cross-correlation alignment of song channels from two recordings."""

from __future__ import annotations

import numpy as np


def align_song_channels(
    song_a: np.ndarray,
    song_b: np.ndarray,
    sr: int,
    max_offset_s: float = 5.0,
) -> tuple[float, float]:
    """Find the time offset that best aligns two song-channel recordings.

    Uses cross-correlation of the song channels (not bass channels).
    Both recordings must use the same backing track.

    Parameters
    ----------
    song_a : 1-D float array, song channel from Take A.
    song_b : 1-D float array, song channel from Take B.
    sr     : sample rate (must be same for both).
    max_offset_s : maximum allowed offset in seconds.

    Returns
    -------
    (offset_s, confidence)
        offset_s: seconds to add to Take B times to align with Take A.
                  Positive means B started later than A.
        confidence: normalized correlation peak (0-1).
    """
    from scipy.signal import fftconvolve, resample_poly

    if len(song_a) == 0 or len(song_b) == 0:
        return 0.0, 0.0

    target_sr = 8000
    if sr > target_sr:
        down = sr // target_sr
        a = resample_poly(song_a.astype(np.float64), 1, down)
        b = resample_poly(song_b.astype(np.float64), 1, down)
        effective_sr = sr / down
    else:
        a = song_a.astype(np.float64)
        b = song_b.astype(np.float64)
        effective_sr = float(sr)

    corr = fftconvolve(a, b[::-1], mode="full")

    mid = len(b) - 1
    max_lag = int(max_offset_s * effective_sr)
    lo = max(0, mid - max_lag)
    hi = min(len(corr), mid + max_lag + 1)

    restricted = corr[lo:hi]
    best_idx = int(np.argmax(restricted))
    lag = best_idx - (mid - lo)

    offset_s = lag / effective_sr

    energy_a = float(np.sum(a ** 2))
    energy_b = float(np.sum(b ** 2))
    denom = np.sqrt(energy_a * energy_b)
    confidence = float(restricted[best_idx] / denom) if denom > 0 else 0.0
    confidence = max(0.0, min(1.0, confidence))

    return offset_s, confidence
