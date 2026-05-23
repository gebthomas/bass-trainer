"""Practice session configuration models.

Pure data modeling only — no audio playback, no sounddevice, no third-party
dependencies.

A PracticeMode is a serialisable configuration record that describes how a
practice session is set up: which assets are required, how targets are timed,
and what the user's intent is.  It does not implement any session logic itself.

Supported modes
---------------
metronome_exercise
    Timing comes from a synthetic metronome clock derived from the exercise's
    bpm and count_in_beats fields.  Only exercise_path is required.

recording_aligned_exercise
    Target times are mapped onto a backing audio recording via a BeatAlignment.
    Both exercise_path and alignment_path are required.

play_to_align
    The user plays against a recording to generate an alignment for later use.
    Only audio_file is required; no exercise or alignment exists yet.

Mode-specific asset requirements
---------------------------------
    Mode                          Required fields
    metronome_exercise            exercise_path
    recording_aligned_exercise    exercise_path, alignment_path
    play_to_align                 audio_file
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION: int = 1

_ALLOWED_MODES = frozenset({
    "metronome_exercise",
    "recording_aligned_exercise",
    "play_to_align",
})


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PracticeMode:
    """Serialisable configuration for a practice session.

    Parameters
    ----------
    schema_version
        Must be 1 for the current format.
    mode
        Session mode: ``"metronome_exercise"``, ``"recording_aligned_exercise"``,
        or ``"play_to_align"``.
    exercise_path
        Path to an Exercise JSON file.  Required for ``metronome_exercise``
        and ``recording_aligned_exercise``; optional otherwise.
    alignment_path
        Path to a BeatAlignment JSON file.  Required for
        ``recording_aligned_exercise``; optional otherwise.
    audio_file
        Path or identifier of the backing audio recording.  Required for
        ``play_to_align``; optional otherwise.
    description
        Optional human-readable description of the session configuration.
    metadata
        Arbitrary string key/value annotations.
    """

    schema_version: int
    mode:           str
    exercise_path:  Optional[str]  = None
    alignment_path: Optional[str]  = None
    audio_file:     Optional[str]  = None
    description:    Optional[str]  = None
    metadata:       dict[str, str] = field(default_factory=dict)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_practice_mode(practice_mode: PracticeMode) -> None:
    """Validate a PracticeMode, raising ValueError on the first violation.

    Called automatically by practice_mode_from_dict().  Callers that construct
    PracticeMode directly should call this explicitly.
    """
    if practice_mode.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, "
            f"got {practice_mode.schema_version}"
        )
    if practice_mode.mode not in _ALLOWED_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_ALLOWED_MODES)}, "
            f"got {practice_mode.mode!r}"
        )
    for k, v in practice_mode.metadata.items():
        if not isinstance(v, str):
            raise ValueError(
                f"metadata[{k!r}] must be a str, got {type(v).__name__}"
            )

    mode = practice_mode.mode
    if mode == "metronome_exercise":
        if not practice_mode.exercise_path or not practice_mode.exercise_path.strip():
            raise ValueError(
                "metronome_exercise requires a non-empty exercise_path"
            )
    elif mode == "recording_aligned_exercise":
        if not practice_mode.exercise_path or not practice_mode.exercise_path.strip():
            raise ValueError(
                "recording_aligned_exercise requires a non-empty exercise_path"
            )
        if not practice_mode.alignment_path or not practice_mode.alignment_path.strip():
            raise ValueError(
                "recording_aligned_exercise requires a non-empty alignment_path"
            )
    elif mode == "play_to_align":
        if not practice_mode.audio_file or not practice_mode.audio_file.strip():
            raise ValueError(
                "play_to_align requires a non-empty audio_file"
            )


# ── Asset helper ──────────────────────────────────────────────────────────────

def required_assets(practice_mode: PracticeMode) -> dict:
    """Return the assets required for *practice_mode* to run.

    The returned dict contains only the fields that are mandatory for the
    mode; optional fields are excluded.

    Returns
    -------
    dict
        ``metronome_exercise``          → ``{"exercise_path": ...}``
        ``recording_aligned_exercise``  → ``{"exercise_path": ..., "alignment_path": ...}``
        ``play_to_align``               → ``{"audio_file": ...}``

    Raises ValueError for an unrecognised mode (should not occur on a
    validated PracticeMode).
    """
    mode = practice_mode.mode
    if mode == "metronome_exercise":
        return {"exercise_path": practice_mode.exercise_path}
    if mode == "recording_aligned_exercise":
        return {
            "exercise_path":  practice_mode.exercise_path,
            "alignment_path": practice_mode.alignment_path,
        }
    if mode == "play_to_align":
        return {"audio_file": practice_mode.audio_file}
    raise ValueError(f"required_assets: unrecognised mode {mode!r}")


# ── Serialisation ─────────────────────────────────────────────────────────────

def practice_mode_to_dict(practice_mode: PracticeMode) -> dict:
    """Return a JSON-serialisable dict representation of *practice_mode*."""
    return dataclasses.asdict(practice_mode)


def practice_mode_from_dict(data: dict) -> PracticeMode:
    """Construct and validate a PracticeMode from a plain dict.

    Raises
    ------
    KeyError
        If a required top-level field (``schema_version``, ``mode``) is missing.
    ValueError
        If validation fails.
    """
    pm = PracticeMode(
        schema_version = int(data["schema_version"]),
        mode           = str(data["mode"]),
        exercise_path  = data.get("exercise_path") or None,
        alignment_path = data.get("alignment_path") or None,
        audio_file     = data.get("audio_file") or None,
        description    = data.get("description") or None,
        metadata       = {
            str(k): str(v)
            for k, v in (data.get("metadata") or {}).items()
        },
    )
    validate_practice_mode(pm)
    return pm


def practice_mode_to_json(practice_mode: PracticeMode, *, indent: int = 2) -> str:
    """Serialise *practice_mode* to a JSON string."""
    return json.dumps(practice_mode_to_dict(practice_mode), indent=indent)


def practice_mode_from_json(text: str) -> PracticeMode:
    """Parse *text* as JSON and return a validated PracticeMode."""
    return practice_mode_from_dict(json.loads(text))


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_practice_mode_file(path: str | Path) -> PracticeMode:
    """Load and validate a PracticeMode from a JSON file."""
    return practice_mode_from_json(Path(path).read_text(encoding="utf-8"))


def save_practice_mode_file(practice_mode: PracticeMode, path: str | Path) -> None:
    """Serialise *practice_mode* to a JSON file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(practice_mode_to_json(practice_mode), encoding="utf-8")
