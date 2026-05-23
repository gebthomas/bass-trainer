"""Versioned exercise schema for bass practice sessions.

Pure data modeling and validation — no audio hardware, no external dependencies.

An Exercise is a self-contained, serialisable description of a practice task.
It can be loaded from JSON, validated, and passed to the session engine,
replay tools, or a future UI without modification.

Typical usage
-------------
    from core.exercise import exercise_from_json, exercise_to_json

    ex  = exercise_from_json(path.read_text())
    out = exercise_to_json(ex, indent=2)

    # Convenience constructor for timing-only exercises:
    from core.exercise import simple_timing_exercise
    ex = simple_timing_exercise("Quarter notes at 120", 120.0, 2, [1, 2, 3, 4])

Schema version
--------------
schema_version is stored in every serialised exercise so that loaders can
detect incompatible formats before attempting to deserialise them.  The
current version is 1; future breaking changes will increment this value.

Backward compatibility with existing target dicts
-------------------------------------------------
exercise_from_dict() accepts target dicts that contain at least a "time" key,
including the old {"time": float, "note": str} format used by the replay
harness.  The "note" field is mapped to expected_pitch if present.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION: int = 1


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Target:
    """A single practice target within an exercise.

    Parameters
    ----------
    time
        Beat-relative onset time after the count-in.  Must be >= 0.
        Two targets may share the same time (e.g. simultaneous chord tones).
    label
        Optional display label (e.g. "beat 1", "root", "5th").
    expected_pitch
        Expected pitch as a note name ("E1", "A2") or a frequency in Hz.
        None means timing-only; pitch is not evaluated.
    duration_beats
        Optional note duration in beats.  When provided, must be > 0.
    metadata
        Arbitrary string key/value pairs for tool-specific annotations.
    """

    time:            float
    label:           Optional[str]   = None
    expected_pitch:  Optional[str | float] = None
    duration_beats:  Optional[float] = None
    metadata:        dict[str, str]  = field(default_factory=dict)


@dataclass
class Exercise:
    """A versioned, serialisable practice exercise.

    Parameters
    ----------
    schema_version
        Must be 1 for the current format.
    name
        Human-readable exercise name.  Must be non-empty.
    bpm
        Nominal tempo in beats per minute.  Must be > 0.
    count_in_beats
        Number of count-in beats before the first target.  Must be >= 0.
    targets
        Ordered list of targets.  Must not be empty.  Times must be
        monotonically nondecreasing (equal times are allowed for simultaneous
        notes; strictly decreasing times are not permitted).
    description
        Optional prose description shown to the player or in exercise browsers.
    tags
        List of string tags for filtering and categorisation (e.g. "ii-V-I",
        "walking-bass", "beginner").
    metadata
        Arbitrary string key/value pairs for tool-specific use.
    """

    schema_version: int
    name:           str
    bpm:            float
    count_in_beats: int
    targets:        list[Target]
    description:    Optional[str]   = None
    tags:           list[str]       = field(default_factory=list)
    metadata:       dict[str, str]  = field(default_factory=dict)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_exercise(exercise: Exercise) -> None:
    """Validate an Exercise, raising ValueError on the first violation found.

    Called automatically by exercise_from_dict() and simple_timing_exercise().
    Callers that construct Exercise directly should call this explicitly if
    they need to confirm validity.
    """
    if exercise.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {exercise.schema_version}"
        )
    if not exercise.name or not exercise.name.strip():
        raise ValueError("name must be non-empty")
    if exercise.bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {exercise.bpm}")
    if exercise.count_in_beats < 0:
        raise ValueError(f"count_in_beats must be >= 0, got {exercise.count_in_beats}")
    if not exercise.targets:
        raise ValueError("targets must not be empty")

    prev_time: float | None = None
    for i, t in enumerate(exercise.targets):
        if t.time < 0:
            raise ValueError(
                f"targets[{i}].time must be >= 0, got {t.time}"
            )
        if prev_time is not None and t.time < prev_time:
            raise ValueError(
                f"target times must be monotonically nondecreasing; "
                f"targets[{i}].time={t.time} < targets[{i - 1}].time={prev_time}"
            )
        if t.duration_beats is not None and t.duration_beats <= 0:
            raise ValueError(
                f"targets[{i}].duration_beats must be > 0, got {t.duration_beats}"
            )
        for k, v in t.metadata.items():
            if not isinstance(v, str):
                raise ValueError(
                    f"targets[{i}].metadata[{k!r}] must be a str, "
                    f"got {type(v).__name__}"
                )
        prev_time = t.time

    for k, v in exercise.metadata.items():
        if not isinstance(v, str):
            raise ValueError(
                f"exercise.metadata[{k!r}] must be a str, got {type(v).__name__}"
            )


# ── Serialisation ─────────────────────────────────────────────────────────────

def exercise_to_dict(exercise: Exercise) -> dict:
    """Return a JSON-serialisable dict representation of *exercise*.

    Uses dataclasses.asdict() to recursively convert nested Target objects.
    All fields are included in the output, including optional ones that are
    None, so that the format is stable and round-trippable.
    """
    return dataclasses.asdict(exercise)


def exercise_from_dict(data: dict) -> Exercise:
    """Construct and validate an Exercise from a plain dict.

    Accepts the current schema as well as the older {"time": ..., "note": ...}
    target format used by the replay harness.  The "note" field is mapped to
    expected_pitch when expected_pitch is not already present.

    Raises
    ------
    KeyError
        If a required field is missing.
    ValueError
        If validation fails.
    """
    targets = [_target_from_dict(t) for t in data["targets"]]
    exercise = Exercise(
        schema_version = int(data["schema_version"]),
        name           = str(data["name"]),
        bpm            = float(data["bpm"]),
        count_in_beats = int(data["count_in_beats"]),
        targets        = targets,
        description    = data.get("description"),
        tags           = list(data.get("tags") or []),
        metadata       = {str(k): str(v) for k, v in (data.get("metadata") or {}).items()},
    )
    validate_exercise(exercise)
    return exercise


def exercise_to_json(exercise: Exercise, *, indent: int = 2) -> str:
    """Serialise *exercise* to a JSON string."""
    return json.dumps(exercise_to_dict(exercise), indent=indent)


def exercise_from_json(text: str) -> Exercise:
    """Parse *text* as JSON and return a validated Exercise."""
    return exercise_from_dict(json.loads(text))


# ── Convenience constructor ───────────────────────────────────────────────────

def simple_timing_exercise(
    name:           str,
    bpm:            float,
    count_in_beats: int,
    target_times:   list[float],
) -> Exercise:
    """Return a validated timing-only Exercise from a list of beat positions.

    All targets are created with time only; label, expected_pitch, and
    duration_beats are left as None.  Useful for quick construction in tests
    and the live demo.

    Parameters
    ----------
    name
        Exercise name.
    bpm
        Tempo in beats per minute.
    count_in_beats
        Number of count-in beats before the first target.
    target_times
        Beat positions (after count-in) for each target, in order.
    """
    targets = [Target(time=float(t)) for t in target_times]
    exercise = Exercise(
        schema_version = SCHEMA_VERSION,
        name           = name,
        bpm            = bpm,
        count_in_beats = count_in_beats,
        targets        = targets,
    )
    validate_exercise(exercise)
    return exercise


# ── Adapter helpers ───────────────────────────────────────────────────────────

def exercise_targets(exercise_or_targets: "Exercise | list[dict]") -> list[dict]:
    """Return a list of target dicts from an Exercise or pass through a raw list.

    When passed an Exercise, converts each Target to the dict shape expected by
    SessionEngine, replay_session_data, and the live pipeline — ``{"time": ...}``
    plus optional ``"note"``, ``"label"``, ``"duration_beats"``, ``"metadata"``.

    When passed a list, returns a shallow copy unchanged so callers need not
    branch on the type of the thing they received.
    """
    if isinstance(exercise_or_targets, Exercise):
        return [_target_to_engine_dict(t) for t in exercise_or_targets.targets]
    return list(exercise_or_targets)


def exercise_bpm(
    exercise_or_targets: "Exercise | list | None",
    fallback_bpm: float | None = None,
) -> float:
    """Return bpm from an Exercise, or *fallback_bpm* for raw lists/None.

    Raises ValueError if no Exercise is given and fallback_bpm is also None.
    """
    if isinstance(exercise_or_targets, Exercise):
        return exercise_or_targets.bpm
    if fallback_bpm is not None:
        return float(fallback_bpm)
    raise ValueError(
        "exercise_bpm: received a non-Exercise value and fallback_bpm is None"
    )


def exercise_count_in_beats(
    exercise_or_targets: "Exercise | list | None",
    fallback_count_in_beats: int | None = None,
) -> int:
    """Return count_in_beats from an Exercise, or *fallback_count_in_beats* for raw lists/None.

    Raises ValueError if no Exercise is given and the fallback is also None.
    """
    if isinstance(exercise_or_targets, Exercise):
        return exercise_or_targets.count_in_beats
    if fallback_count_in_beats is not None:
        return int(fallback_count_in_beats)
    raise ValueError(
        "exercise_count_in_beats: received a non-Exercise value and "
        "fallback_count_in_beats is None"
    )


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_exercise_file(path: str | Path) -> Exercise:
    """Load and validate an Exercise from a JSON file."""
    return exercise_from_json(Path(path).read_text(encoding="utf-8"))


def save_exercise_file(exercise: Exercise, path: str | Path) -> None:
    """Serialise *exercise* to a JSON file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(exercise_to_json(exercise), encoding="utf-8")


# ── Private helpers ───────────────────────────────────────────────────────────

def _target_to_engine_dict(target: Target) -> dict:
    """Convert a Target to the dict format expected by SessionEngine and replay.

    Uses ``"note"`` as the key for expected_pitch because that is the key read
    by feedback_event() when building the ``"expected_note"`` output field.
    """
    d: dict = {"time": target.time}
    if target.expected_pitch is not None:
        d["note"] = target.expected_pitch
    if target.label is not None:
        d["label"] = target.label
    if target.duration_beats is not None:
        d["duration_beats"] = target.duration_beats
    if target.metadata:
        d["metadata"] = dict(target.metadata)
    return d


def _target_from_dict(data: dict) -> Target:
    """Build a Target from a dict, including old-format {"time", "note"} dicts."""
    return Target(
        time           = float(data["time"]),
        label          = data.get("label"),
        expected_pitch = data.get("expected_pitch") or data.get("note") or None,
        duration_beats = data.get("duration_beats"),
        metadata       = {
            str(k): str(v)
            for k, v in (data.get("metadata") or {}).items()
        },
    )
