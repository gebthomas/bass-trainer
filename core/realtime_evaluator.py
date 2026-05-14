"""Lightweight real-time analysis of extracted audio windows.

Pure functions only — no sounddevice, no aubio, no pitch detection.
"""

from __future__ import annotations

import numpy as np


def evaluate_window(
    audio: np.ndarray,
    sample_rate: int,
    onset_threshold: float = 0.02,
    min_rms: float = 0.005,
) -> dict:
    """Analyse an audio window for energy and onset presence.

    Parameters
    ----------
    audio            : 1-D (mono) or 2-D (frames × channels) float array.
                       Stereo (or multi-channel) input is averaged to mono
                       before analysis.
    sample_rate      : audio sample rate in Hz.
    onset_threshold  : absolute amplitude level that constitutes an onset.
    min_rms          : minimum RMS to consider meaningful audio detected.

    Returns
    -------
    dict with:
        detected      bool         True if rms >= min_rms
        rms           float        root-mean-square amplitude of the window
        peak          float        maximum absolute amplitude
        onset_found   bool         True if any sample crosses onset_threshold
        onset_sample  int | None   index of the first threshold crossing
        onset_time_s  float | None time of the first threshold crossing (s)
    """
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio.ravel()

    if mono.size == 0:
        return {
            "detected":     False,
            "rms":          0.0,
            "peak":         0.0,
            "onset_found":  False,
            "onset_sample": None,
            "onset_time_s": None,
        }

    rms  = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))

    crossings = np.where(np.abs(mono) >= onset_threshold)[0]
    if crossings.size > 0:
        onset_sample: int | None   = int(crossings[0])
        onset_time_s: float | None = onset_sample / sample_rate
        onset_found                = True
    else:
        onset_sample = None
        onset_time_s = None
        onset_found  = False

    return {
        "detected":     bool(rms >= min_rms),
        "rms":          rms,
        "peak":         peak,
        "onset_found":  onset_found,
        "onset_sample": onset_sample,
        "onset_time_s": onset_time_s,
    }
