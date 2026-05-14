"""Compute sample-accurate analysis windows for target notes.

Pure functions only — no sounddevice, no audio hardware.
"""

from __future__ import annotations


def target_audio_time_s(
    targets: list[dict],
    idx: int,
    bpm: float,
    count_in_beats: int,
) -> float:
    """Absolute audio time (seconds) at which target *idx* should be played.

    Parameters
    ----------
    targets        : ordered target dicts; each must have a ``"time"`` key
                     in beats relative to the start of the piece.
    idx            : index into *targets*.
    bpm            : tempo in beats per minute.
    count_in_beats : count-in length in beats.
    """
    beat_s = 60.0 / bpm
    count_in_s = count_in_beats * beat_s
    return count_in_s + targets[idx]["time"] * beat_s


def target_gap_s(
    targets: list[dict],
    idx: int,
    bpm: float,
) -> float:
    """Inter-target gap in seconds for target *idx*.

    Rules
    -----
    - If there is a following target, gap = (next_time - current_time) × beat_s.
    - If *idx* is the last target and a previous target exists, reuse the
      previous gap.
    - If there is only one target, return 0.5 s as a safe default.

    Raises
    ------
    ValueError
        If the computed gap is <= 0.
    """
    beat_s = 60.0 / bpm
    n = len(targets)

    if n == 1:
        return 0.5

    if idx < n - 1:
        gap_beats = targets[idx + 1]["time"] - targets[idx]["time"]
    else:
        gap_beats = targets[idx]["time"] - targets[idx - 1]["time"]

    gap_s = gap_beats * beat_s
    if gap_s <= 0:
        raise ValueError(
            f"target gap for idx={idx} is non-positive ({gap_s:.6f} s); "
            "check that target times are strictly increasing"
        )
    return gap_s


def target_analysis_window_samples(
    targets: list[dict],
    idx: int,
    bpm: float,
    count_in_beats: int,
    sample_rate: float,
    start_offset_s: float = 0.02,
    max_window_s: float = 0.35,
    gap_fraction: float = 0.6,
) -> tuple[int, int]:
    """Return ``(start_sample, end_sample)`` for the analysis window of target *idx*.

    Parameters
    ----------
    targets          : ordered target dicts with a ``"time"`` key in beats.
    idx              : index into *targets*.
    bpm              : tempo in beats per minute (must be > 0).
    count_in_beats   : count-in length in beats.
    sample_rate      : audio sample rate in Hz (must be > 0).
    start_offset_s   : offset from the target's beat time to the window start.
    max_window_s     : maximum window duration in seconds.
    gap_fraction     : fraction of the inter-target gap used as window duration,
                       capped by *max_window_s*.

    Returns
    -------
    tuple[int, int]
        ``(start_sample, end_sample)`` as Python ints via ``round()`` —
        integer truncation with ``int()`` can produce off-by-one errors when
        floating-point multiplication lands just below an integer boundary.

    Raises
    ------
    ValueError
        If *bpm* <= 0, *sample_rate* <= 0, *idx* is out of range, the
        computed gap is <= 0, or *end_sample* <= *start_sample*.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    if idx < 0 or idx >= len(targets):
        raise ValueError(
            f"idx {idx} is out of range for targets of length {len(targets)}"
        )

    start_time = target_audio_time_s(targets, idx, bpm, count_in_beats) + start_offset_s
    duration   = min(max_window_s, gap_fraction * target_gap_s(targets, idx, bpm))
    end_time   = start_time + duration

    start_sample = round(start_time * sample_rate)
    end_sample   = round(end_time   * sample_rate)

    if end_sample <= start_sample:
        raise ValueError(
            f"end_sample ({end_sample}) <= start_sample ({start_sample}); "
            "window duration is too small for this sample rate"
        )

    return (start_sample, end_sample)
