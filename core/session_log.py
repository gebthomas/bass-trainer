"""Session telemetry model for practice sessions.

Pure data modeling and validation — no audio hardware, no external dependencies.

A SessionLog is a timestamped record of what happened during a practice session.
It captures individual events (onsets, misses, marks) with their associated
metadata, and aggregate metrics computed at session end.  SessionLogs are
serialisable to JSON for persistence, replay, and future progress analytics.

Typical usage
-------------
    from core.session_log import (
        SessionLog, SessionEvent,
        append_event, session_log_to_json, save_session_log_file,
    )

    log = SessionLog(schema_version=1, started_at="2026-05-23T10:00:00")
    event = SessionEvent(time_sec=0.5, event_type=TARGET_HIT, target_index=0)
    append_event(log, event)
    save_session_log_file(log, "sessions/2026-05-23.json")

Schema version
--------------
schema_version is stored in every serialised log so that loaders can detect
incompatible formats before attempting to deserialise them.  The current
version is 1; future breaking changes will increment this value.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION: int = 1

TARGET_HIT:          str            = "target_hit"
TARGET_MISS:         str            = "target_miss"
EXTRA_ONSET:         str            = "extra_onset"
ALLOWED_EVENT_TYPES: frozenset[str] = frozenset({TARGET_HIT, TARGET_MISS, EXTRA_ONSET})


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SessionEvent:
    """A single timestamped event within a practice session.

    Parameters
    ----------
    time_sec
        Session time in seconds when the event occurred.  Must be >= 0.
    event_type
        Category string (e.g. ``"hit"``, ``"miss"``, ``"extra"``, ``"mark"``).
        Must be non-empty.
    target_index
        Index into the exercise target list, if applicable.  When provided,
        must be >= 0.
    message
        Optional human-readable description (e.g. ``"played A1, expected E1"``).
    value
        Optional numeric payload (e.g. timing error in seconds, pitch error in
        cents).
    metadata
        Arbitrary string key/value annotations.
    """

    time_sec:     float
    event_type:   str
    target_index: Optional[int]   = None
    message:      Optional[str]   = None
    value:        Optional[float] = None
    metadata:     dict[str, str]  = field(default_factory=dict)


@dataclass
class SessionLog:
    """Versioned telemetry record for a practice session.

    Parameters
    ----------
    schema_version
        Must be 1 for the current format.
    started_at
        ISO 8601 timestamp string for when the session began.  Must be non-empty.
    ended_at
        ISO 8601 timestamp string for when the session ended.  May be ``None``
        while the session is in progress.
    practice_mode_path
        Path to the PracticeMode JSON file that configured this session.
    exercise_path
        Path to the Exercise JSON file used in this session.
    alignment_path
        Path to the BeatAlignment JSON file used in this session.
    events
        Ordered list of session events, appended as the session progresses.
    metrics
        Aggregate numeric metrics computed at session end (e.g.
        ``hit_rate``, ``mean_timing_error_ms``).  Values must be int or float.
    metadata
        Arbitrary string key/value annotations.
    """

    schema_version:     int
    started_at:         str
    ended_at:           Optional[str]      = None
    practice_mode_path: Optional[str]      = None
    exercise_path:      Optional[str]      = None
    alignment_path:     Optional[str]      = None
    events:             list[SessionEvent] = field(default_factory=list)
    metrics:            dict[str, float]   = field(default_factory=dict)
    metadata:           dict[str, str]     = field(default_factory=dict)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_event(event: SessionEvent, index: int | None = None) -> None:
    ctx = f"events[{index}]." if index is not None else "event."
    if event.time_sec < 0:
        raise ValueError(f"{ctx}time_sec must be >= 0, got {event.time_sec}")
    if event.event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"{ctx}event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}, "
            f"got {event.event_type!r}"
        )
    if event.target_index is not None and event.target_index < 0:
        raise ValueError(
            f"{ctx}target_index must be >= 0, got {event.target_index}"
        )
    for k, v in event.metadata.items():
        if not isinstance(v, str):
            raise ValueError(
                f"{ctx}metadata[{k!r}] must be a str, got {type(v).__name__}"
            )


def validate_session_log(log: SessionLog) -> None:
    """Validate a SessionLog, raising ValueError on the first violation found.

    Called automatically by session_log_from_dict().  Callers that construct
    SessionLog directly should call this explicitly.
    """
    if log.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {log.schema_version}"
        )
    if not log.started_at or not log.started_at.strip():
        raise ValueError("started_at must be non-empty")
    for i, event in enumerate(log.events):
        _validate_event(event, index=i)
    for k, v in log.metrics.items():
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"metrics[{k!r}] must be an int or float, got {type(v).__name__}"
            )
    for k, v in log.metadata.items():
        if not isinstance(v, str):
            raise ValueError(
                f"metadata[{k!r}] must be a str, got {type(v).__name__}"
            )


# ── Event append ──────────────────────────────────────────────────────────────

def append_event(log: SessionLog, event: SessionEvent) -> SessionLog:
    """Validate *event* and append it to *log.events* in place.

    Returns *log* so that calls may be chained or the result captured.

    Raises
    ------
    ValueError
        If *event* fails validation.  The log is not modified on failure.
    """
    _validate_event(event)
    log.events.append(event)
    return log


# ── Serialisation ─────────────────────────────────────────────────────────────

def session_log_to_dict(log: SessionLog) -> dict:
    """Return a JSON-serialisable dict representation of *log*."""
    return dataclasses.asdict(log)


def session_log_from_dict(data: dict) -> SessionLog:
    """Construct and validate a SessionLog from a plain dict.

    Raises
    ------
    KeyError
        If a required top-level field (``schema_version``, ``started_at``) is
        missing.
    ValueError
        If validation fails.
    """
    events = [_event_from_dict(e) for e in (data.get("events") or [])]
    log = SessionLog(
        schema_version     = int(data["schema_version"]),
        started_at         = str(data["started_at"]),
        ended_at           = data.get("ended_at") or None,
        practice_mode_path = data.get("practice_mode_path") or None,
        exercise_path      = data.get("exercise_path") or None,
        alignment_path     = data.get("alignment_path") or None,
        events             = events,
        metrics            = {str(k): v for k, v in (data.get("metrics") or {}).items()},
        metadata           = {
            str(k): str(v)
            for k, v in (data.get("metadata") or {}).items()
        },
    )
    validate_session_log(log)
    return log


def session_log_to_json(log: SessionLog, *, indent: int = 2) -> str:
    """Serialise *log* to a JSON string."""
    return json.dumps(session_log_to_dict(log), indent=indent)


def session_log_from_json(text: str) -> SessionLog:
    """Parse *text* as JSON and return a validated SessionLog."""
    return session_log_from_dict(json.loads(text))


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_session_log_file(path: str | Path) -> SessionLog:
    """Load and validate a SessionLog from a JSON file."""
    return session_log_from_json(Path(path).read_text(encoding="utf-8"))


def save_session_log_file(log: SessionLog, path: str | Path) -> None:
    """Serialise *log* to a JSON file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(session_log_to_json(log), encoding="utf-8")


# ── Private helpers ───────────────────────────────────────────────────────────

def _event_from_dict(data: dict) -> SessionEvent:
    return SessionEvent(
        time_sec     = float(data["time_sec"]),
        event_type   = str(data["event_type"]),
        target_index = (
            int(data["target_index"])
            if data.get("target_index") is not None
            else None
        ),
        message  = data.get("message") or None,
        value    = (
            float(data["value"])
            if data.get("value") is not None
            else None
        ),
        metadata = {
            str(k): str(v)
            for k, v in (data.get("metadata") or {}).items()
        },
    )
