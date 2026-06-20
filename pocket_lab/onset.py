"""Onset detection and novelty envelope computation."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def novelty_envelope(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute onset-strength envelope via librosa.

    Returns (times, envelope) arrays of matching length.
    """
    import librosa

    env = librosa.onset.onset_strength(
        y=np.asarray(audio, dtype=np.float32), sr=sr,
    )
    times = librosa.times_like(env, sr=sr)
    return times, env


def detect_onsets(
    audio: np.ndarray,
    sr: int,
    delta: float = 0.07,
) -> np.ndarray:
    """Detect onset times (seconds) using librosa spectral-flux."""
    import librosa

    return librosa.onset.onset_detect(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        backtrack=True,
        delta=delta,
        units="time",
    )
