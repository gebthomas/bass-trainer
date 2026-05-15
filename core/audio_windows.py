"""Extract analysis windows for target notes from recorded audio.

Pure functions only — no sounddevice, no pitch detection.
"""

from __future__ import annotations

import numpy as np


def extract_target_window(
    audio: np.ndarray,
    target: dict,
    bpm: float,
    count_in_beats: int,
    sample_rate: int,
    pre_roll_s: float = 0.03,
    post_roll_s: float = 0.10,
    center_time_s: float | None = None,
) -> dict:
    """Extract the audio window surrounding a target's beat position.

    Parameters
    ----------
    audio           : 1-D (mono) or 2-D (frames × channels) float array.
    target          : target dict with a ``"time"`` key in beats.
    bpm             : tempo in beats per minute.
    count_in_beats  : count-in length in beats.
    sample_rate     : audio sample rate in Hz.
    pre_roll_s      : seconds before the window center to include.
    post_roll_s     : seconds after the window center to include.
    center_time_s   : optional override for the window center (seconds).
                      When ``None`` (default) the window is centered at the
                      nominal beat position computed from *bpm* and *target*.
                      When provided the window shifts to this time, but
                      ``target_sample`` in the returned dict still reflects
                      the *nominal* beat position so that scoring references
                      remain consistent.

    Returns
    -------
    dict with:
        audio         extracted slice (same dimensionality as input)
        start_sample  first sample index, clamped to [0, n_frames]
        end_sample    one-past-last sample index, clamped to [0, n_frames]
        target_sample sample index of the nominal beat position (unclamped,
                      always based on bpm / count_in_beats / target["time"])
    """
    audio = np.asarray(audio)
    n_frames = audio.shape[0]

    beat_s            = 60.0 / bpm
    count_in_s        = count_in_beats * beat_s
    nominal_beat_time = count_in_s + target["time"] * beat_s

    target_sample = round(nominal_beat_time * sample_rate)

    center = center_time_s if center_time_s is not None else nominal_beat_time
    start_sample = max(0, min(n_frames, round((center - pre_roll_s) * sample_rate)))
    end_sample   = max(0, min(n_frames, round((center + post_roll_s) * sample_rate)))

    return {
        "audio":         audio[start_sample:end_sample],
        "start_sample":  start_sample,
        "end_sample":    end_sample,
        "target_sample": target_sample,
    }
