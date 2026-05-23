"""Derived severity classification for session events.

Pure functions only — no audio hardware, no external dependencies.
Does not modify SessionEvent or SessionLog schemas.

Severity labels
---------------
- ``"good"``  — timing error within the tight acceptance window
- ``"warn"``  — timing error within a looser tolerance, still counts as a hit
- ``"miss"``  — timing error exceeds both windows, or no onset was detected

Default thresholds
------------------
``DEFAULT_GOOD_THRESHOLD_S`` and ``DEFAULT_WARN_THRESHOLD_S`` are the
canonical defaults.  Both functions accept keyword-only overrides so callers
can substitute exercise-specific or BPM-adaptive thresholds without touching
the defaults.

Typical usage
-------------
    from core.severity_policy import (
        event_timing_severity,
        timing_severity,
        SEVERITY_GOOD, SEVERITY_WARN, SEVERITY_MISS,
    )

    sev = timing_severity(event.value)            # raw float path
    sev = event_timing_severity(event)            # SessionEvent path
"""

from __future__ import annotations

from core.session_log import EXTRA_ONSET, TARGET_HIT, TARGET_MISS, SessionEvent

SEVERITY_GOOD = "good"
SEVERITY_WARN = "warn"
SEVERITY_MISS = "miss"

DEFAULT_GOOD_THRESHOLD_S: float = 0.05
DEFAULT_WARN_THRESHOLD_S: float = 0.12


def timing_severity(
    error_s: float,
    *,
    good_s: float = DEFAULT_GOOD_THRESHOLD_S,
    warn_s: float = DEFAULT_WARN_THRESHOLD_S,
) -> str:
    """Classify a timing error in seconds into a severity label.

    Uses ``abs(error_s)`` so early and late errors are treated symmetrically.

    Parameters
    ----------
    error_s
        Timing error in seconds (signed; onset_time - target_time).
    good_s
        Upper bound for ``"good"``; must be positive and <= *warn_s*.
    warn_s
        Upper bound for ``"warn"``; must be positive and >= *good_s*.

    Returns
    -------
    str
        ``SEVERITY_GOOD``, ``SEVERITY_WARN``, or ``SEVERITY_MISS``.

    Raises
    ------
    ValueError
        If *good_s* or *warn_s* are non-positive, or if *good_s* > *warn_s*.
    """
    if good_s <= 0:
        raise ValueError(f"good_s must be positive, got {good_s}")
    if warn_s <= 0:
        raise ValueError(f"warn_s must be positive, got {warn_s}")
    if good_s > warn_s:
        raise ValueError(
            f"good_s must be <= warn_s; got good_s={good_s}, warn_s={warn_s}"
        )
    abs_error = abs(error_s)
    if abs_error <= good_s:
        return SEVERITY_GOOD
    if abs_error <= warn_s:
        return SEVERITY_WARN
    return SEVERITY_MISS


def event_timing_severity(
    event: SessionEvent,
    *,
    good_s: float = DEFAULT_GOOD_THRESHOLD_S,
    warn_s: float = DEFAULT_WARN_THRESHOLD_S,
) -> str | None:
    """Derive a severity label from a SessionEvent without modifying it.

    Parameters
    ----------
    event
        A validated ``SessionEvent``.
    good_s
        Forwarded to ``timing_severity()`` for TARGET_HIT events.
    warn_s
        Forwarded to ``timing_severity()`` for TARGET_HIT events.

    Returns
    -------
    str | None
        - ``TARGET_HIT`` with a numeric ``value``: classify via
          ``timing_severity(event.value)``.
        - ``TARGET_MISS``: always ``SEVERITY_MISS``.
        - ``EXTRA_ONSET``: ``None`` — no target to grade against.

    Raises
    ------
    ValueError
        If ``event_type`` is ``TARGET_HIT`` but ``event.value`` is ``None``.
    """
    if event.event_type == TARGET_HIT:
        if event.value is None:
            raise ValueError(
                "event_timing_severity: TARGET_HIT event has value=None; "
                "timing error in seconds is required for severity classification"
            )
        return timing_severity(event.value, good_s=good_s, warn_s=warn_s)
    if event.event_type == TARGET_MISS:
        return SEVERITY_MISS
    if event.event_type == EXTRA_ONSET:
        return None
    raise ValueError(f"Unrecognised event_type: {event.event_type!r}")
