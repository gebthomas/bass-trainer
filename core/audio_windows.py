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
) -> dict:
    """Extract the audio window surrounding a target's beat position.

    Parameters
    ----------
    audio           : 1-D (mono) or 2-D (frames × channels) float array.
    target          : target dict with a ``"time"`` key in beats.
    bpm             : tempo in beats per minute.
    count_in_beats  : count-in length in beats.
    sample_rate     : audio sample rate in Hz.
    pre_roll_s      : seconds before the target beat to include.
    post_roll_s     : seconds after the target beat to include.

    Returns
    -------
    dict with:
        audio         extracted slice (same dimensionality as input)
        start_sample  first sample index, clamped to [0, n_frames]
        end_sample    one-past-last sample index, clamped to [0, n_frames]
        target_sample sample index of the target's beat position (unclamped)
    """
    audio = np.asarray(audio)
    n_frames = audio.shape[0]

    beat_s             = 60.0 / bpm
    count_in_s         = count_in_beats * beat_s
    target_audio_time  = count_in_s + target["time"] * beat_s

    target_sample = round(target_audio_time * sample_rate)
    start_sample  = max(0, min(n_frames, round((target_audio_time - pre_roll_s) * sample_rate)))
    end_sample    = max(0, min(n_frames, round((target_audio_time + post_roll_s) * sample_rate)))

    return {
        "audio":         audio[start_sample:end_sample],
        "start_sample":  start_sample,
        "end_sample":    end_sample,
        "target_sample": target_sample,
    }
