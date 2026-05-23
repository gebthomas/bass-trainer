"""Canonical match-window policy for onset-to-target timing evaluation.

A single authoritative formula used by all matching sites in the new session
tier.  Old-tier code (matching.py / TargetMatcher) is intentionally excluded
and retains its local-gap formula until it is retired.

Typical usage
-------------
    from core.timing_policy import match_window_s

    window = match_window_s(bpm=120.0)        # → 0.25 s
    window = match_window_s(bpm=60.0)         # → 0.35 s  (upper clamp)
    window = match_window_s(bpm=400.0)        # → 0.10 s  (lower clamp)
    window = match_window_s(bpm=120.0, min_s=0.05, max_s=0.20)  # custom

Policy
------
    base = 30.0 / bpm          (half a beat in seconds)
    return clamp(base, min_s, max_s)

The default clamp range [0.10, 0.35] s is documented in
docs/match_window_policy.md §4.
"""

from __future__ import annotations

DEFAULT_MIN_MATCH_WINDOW_S: float = 0.10
DEFAULT_MAX_MATCH_WINDOW_S: float = 0.35


def match_window_s(
    bpm: float,
    *,
    min_s: float = DEFAULT_MIN_MATCH_WINDOW_S,
    max_s: float = DEFAULT_MAX_MATCH_WINDOW_S,
) -> float:
    """Return the half-beat match window in seconds for the given tempo.

    Parameters
    ----------
    bpm
        Tempo in beats per minute.  Must be > 0.
    min_s
        Lower clamp in seconds.  Must be > 0 and <= *max_s*.
    max_s
        Upper clamp in seconds.  Must be > 0 and >= *min_s*.

    Returns
    -------
    float
        ``clamp(30.0 / bpm, min_s, max_s)``

    Raises
    ------
    ValueError
        If *bpm* <= 0, or if *min_s* or *max_s* are not positive, or if
        *min_s* > *max_s*.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm!r}")
    if min_s <= 0:
        raise ValueError(f"min_s must be > 0, got {min_s!r}")
    if max_s <= 0:
        raise ValueError(f"max_s must be > 0, got {max_s!r}")
    if min_s > max_s:
        raise ValueError(f"min_s must be <= max_s, got min_s={min_s!r}, max_s={max_s!r}")

    base = 30.0 / bpm
    return max(min_s, min(max_s, base))
