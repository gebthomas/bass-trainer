"""Audio window energy and onset analysis.

Pure functions only — no sounddevice, no aubio, no pitch detection.
"""

from __future__ import annotations

import numpy as np


def evaluate_window(
    audio: np.ndarray,
    sample_rate: int,
    onset_threshold: float = 0.02,
    min_rms: float = 0.005,
    baseline_window_s: float = 0.015,
    rise_ratio: float = 2.5,
) -> dict:
    """Analyse an audio window for energy and onset presence.

    Parameters
    ----------
    audio              : 1-D (mono) or 2-D (frames × channels) float array.
                         Stereo (or multi-channel) input is averaged to mono
                         before analysis.
    sample_rate        : audio sample rate in Hz.
    onset_threshold    : absolute amplitude floor — the dynamic threshold is
                         always at least this high.
    min_rms            : minimum RMS to consider meaningful audio detected.
    baseline_window_s  : seconds from the start of the window used to estimate
                         local baseline energy.  Should be shorter than the
                         window's pre-roll so that sustained resonance from the
                         previous note is captured without including the new
                         attack.  Default 0.015 s (15 ms).
    rise_ratio         : the amplitude envelope must exceed
                         ``max(onset_threshold, baseline × rise_ratio)`` to be
                         counted as an onset.  A value of 2.5 means the signal
                         must rise at least 2.5× above the local baseline.
                         This rejects sustained resonance (which never rises
                         meaningfully above its own level) while detecting new
                         attacks (which produce a clear amplitude increase).

    Returns
    -------
    dict with:
        detected      bool         True if rms >= min_rms
        rms           float        root-mean-square amplitude of the window
        peak          float        maximum absolute amplitude
        onset_found   bool         True if the amplitude envelope rises
                                   sufficiently above the local baseline
        onset_sample  int | None   index of the first such rise
        onset_time_s  float | None time of that first rise (s)

    Onset detection
    ---------------
    The amplitude envelope is ``|mono|``.  A local baseline is computed as
    the mean of the envelope over the first ``baseline_window_s`` seconds.
    The detection threshold is ``max(onset_threshold, baseline × rise_ratio)``.
    A sustained signal never rises above this dynamic threshold, so it is not
    reported as an onset.  A new attack, which causes a clear rise above the
    baseline level, is detected at the first sample that crosses the threshold.
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

    env = np.abs(mono)

    baseline_n    = max(1, round(sample_rate * baseline_window_s))
    baseline_n    = min(baseline_n, len(env))
    baseline      = float(np.mean(env[:baseline_n]))
    dyn_threshold = max(onset_threshold, baseline * rise_ratio)

    crossings = np.where(env >= dyn_threshold)[0]
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
