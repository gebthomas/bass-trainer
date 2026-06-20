"""Audio loading, segmenting, and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio(wav_path: str | Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file and return (samples, sample_rate).

    Returns the raw multi-channel array as-is (shape may be (N,) for mono
    or (N, channels) for multi-channel).
    """
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32")
    return data, sr


def segment_audio(
    audio: np.ndarray,
    sr: int,
    start: float,
    duration: float,
) -> np.ndarray:
    """Extract a time segment from a 1-D or 2-D audio array."""
    n = audio.shape[0]
    s0 = int(start * sr)
    s1 = int((start + duration) * sr)
    s1 = min(s1, n)
    s0 = max(0, min(s0, n))
    return audio[s0:s1]


def audio_diagnostics(audio: np.ndarray, label: str = "") -> dict:
    """Compute RMS, peak, and channel info for an audio array."""
    if audio.size == 0:
        return {"label": label, "channels": 0, "rms_dbfs": None, "peak_dbfs": None}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    channels = audio.shape[1] if audio.ndim == 2 else 1
    return {
        "label": label,
        "channels": channels,
        "rms_dbfs": 20.0 * np.log10(rms) if rms > 0 else -120.0,
        "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
    }


def compute_overview(audio_1d: np.ndarray, sr: int, n_points: int = 900) -> dict:
    """Downsample 1-D audio to a compact envelope for overview display.

    Returns dict with keys maxes, mins (arrays of length n_points),
    and total_duration_s.
    """
    n = len(audio_1d)
    total_dur = n / sr if sr > 0 else 0.0
    if n == 0:
        return {"maxes": np.array([]), "mins": np.array([]),
                "total_duration_s": 0.0}
    bucket = max(1, n // n_points)
    nb = n // bucket
    trimmed = audio_1d[: nb * bucket].reshape(nb, bucket)
    return {
        "maxes": np.max(trimmed, axis=1),
        "mins": np.min(trimmed, axis=1),
        "total_duration_s": total_dur,
    }


def window_tag(start: float) -> str:
    """Filesystem-safe tag for a window start time."""
    if start == int(start):
        return f"w{int(start)}"
    return f"w{start:.1f}"
