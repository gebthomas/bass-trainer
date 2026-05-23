"""Structured metrics computed from a canonical SessionLog.

Pure functions only — no audio hardware, no external dependencies.
Does not modify SessionEvent, SessionLog, or metrics.py.

Typical usage
-------------
    from core.log_metrics import compute_log_metrics, log_metrics_to_dict

    m = compute_log_metrics(log)
    print(m.good_hits, m.mean_abs_error_s)
    d = log_metrics_to_dict(m)   # JSON-ready dict
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from core.session_log import (
    EXTRA_ONSET,
    TARGET_HIT,
    TARGET_MISS,
    SessionLog,
    validate_session_log,
)
from core.severity_policy import SEVERITY_GOOD, SEVERITY_WARN, event_timing_severity


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class LogMetrics:
    """Aggregate metrics derived from a SessionLog.

    Parameters
    ----------
    targets_total
        Total targets addressed (hits + misses).
    targets_hit
        Number of TARGET_HIT events.
    targets_missed
        Number of TARGET_MISS events.
    extra_onsets
        Number of EXTRA_ONSET events.
    good_hits
        TARGET_HIT events whose timing severity is ``"good"``.
    warn_hits
        TARGET_HIT events whose timing severity is ``"warn"``.
    miss_severity_events
        TARGET_MISS events plus TARGET_HIT events whose timing severity is
        ``"miss"``.  A hit that was detected but badly timed counts here.
    mean_signed_error_s
        Mean signed timing error (seconds) over TARGET_HIT events that carry a
        value.  Positive = systematically late; negative = systematically early.
        ``None`` when no such events exist.
    mean_abs_error_s
        Mean absolute timing error (seconds) over the same set.
        ``None`` when no such events exist.
    """

    targets_total:        int
    targets_hit:          int
    targets_missed:       int
    extra_onsets:         int
    good_hits:            int
    warn_hits:            int
    miss_severity_events: int
    mean_signed_error_s:  Optional[float]
    mean_abs_error_s:     Optional[float]


# ── Public API ────────────────────────────────────────────────────────────────

def compute_log_metrics(log: SessionLog) -> LogMetrics:
    """Compute LogMetrics from a SessionLog.

    Validates the log before computing.

    Parameters
    ----------
    log
        A SessionLog to analyse.

    Returns
    -------
    LogMetrics

    Raises
    ------
    ValueError
        If log validation fails, or if a TARGET_HIT event has value=None.
        A hit without a timing error cannot be classified and would silently
        break the severity accounting invariant.
    """
    validate_session_log(log)

    n_hit      = 0
    n_miss     = 0
    n_extra    = 0
    n_good     = 0
    n_warn     = 0
    n_miss_sev = 0
    hit_errors: list[float] = []

    for event in log.events:
        if event.event_type == TARGET_HIT:
            n_hit += 1
            if event.value is None:
                raise ValueError(
                    "compute_log_metrics: TARGET_HIT event has value=None; "
                    "timing error in seconds is required for severity classification"
                )
            hit_errors.append(event.value)
            sev = event_timing_severity(event)
            if sev == SEVERITY_GOOD:
                n_good += 1
            elif sev == SEVERITY_WARN:
                n_warn += 1
            else:
                n_miss_sev += 1
        elif event.event_type == TARGET_MISS:
            n_miss += 1
            n_miss_sev += 1
        elif event.event_type == EXTRA_ONSET:
            n_extra += 1

    n = len(hit_errors)
    mean_signed = sum(hit_errors) / n          if n else None
    mean_abs    = sum(abs(e) for e in hit_errors) / n if n else None

    return LogMetrics(
        targets_total        = n_hit + n_miss,
        targets_hit          = n_hit,
        targets_missed       = n_miss,
        extra_onsets         = n_extra,
        good_hits            = n_good,
        warn_hits            = n_warn,
        miss_severity_events = n_miss_sev,
        mean_signed_error_s  = mean_signed,
        mean_abs_error_s     = mean_abs,
    )


def log_metrics_to_dict(metrics: LogMetrics) -> dict:
    """Return a JSON-serialisable dict representation of *metrics*."""
    return dataclasses.asdict(metrics)
