"""Beat alignment map connecting an audio recording timeline to musical beat indices.

Pure functions only — no audio playback, no sounddevice, no third-party dependencies.

Tempo model — constant tempo only
----------------------------------
The current model is **constant-tempo**: it interpolates a single uniform beat
grid between two anchor points (first_beat_time_sec and last_beat_time_sec).
All inter-beat intervals are identical.  This is a valid approximation for
recordings that were made to a click track or have only small, acceptable tempo
drift.

This model does **not** handle:
- Recordings with intentional tempo changes (rubato, ritardando, accelerando).
- Recordings with section-level BPM shifts.
- Human performances with beat-to-beat tempo drift beyond what the uniform
  grid absorbs.

Future variable-tempo support
------------------------------
When constant-tempo is insufficient, replace the single (first, last) anchor
pair with a list of timed anchor points and use piecewise linear (or spline)
interpolation between them.  Each anchor would be ``(beat_index, audio_time_s)``,
and ``beat_time()`` would locate the surrounding segment and interpolate within
it.  The schema_version field exists precisely so that a future multi-anchor
format can be detected and deserialized differently.

Time signatures
---------------
beats_per_bar is stored for display and bar-counting purposes.  It does not
affect beat timing: beat_time() and estimated_bpm() are independent of
beats_per_bar.  A 3/4 waltz and a 4/4 march with the same first/last anchors
and beat_count produce identical beat grids; beats_per_bar only determines
how many beats group into a bar.

Typical usage
-------------
    from core.alignment import load_alignment_file, beat_time, beat_times

    align = load_alignment_file("sessions/track.alignment.json")
    t = beat_time(align, 4)          # absolute time of beat 4 in the recording
    ts = beat_times(align, align.beat_count)  # all beat times

Schema version
--------------
schema_version is stored in every serialised alignment so loaders can detect
incompatible formats.  The current version is 1.

Beat indexing
-------------
Beat indices are zero-based.  beat_time(align, 0) == align.first_beat_time_sec
and beat_time(align, align.beat_count - 1) == align.last_beat_time_sec.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION: int = 1

_ALLOWED_METHODS = frozenset({"manual_tap", "player_onsets", "imported"})


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BeatAlignment:
    """Beat grid anchored to an audio recording timeline.

    Parameters
    ----------
    schema_version
        Must be 1 for the current format.
    audio_file
        Path or identifier of the source audio file.  Must be non-empty.
    alignment_method
        How the alignment was produced: ``"manual_tap"``, ``"player_onsets"``,
        or ``"imported"``.
    first_beat_time_sec
        Absolute time in seconds of beat index 0 in the recording.  Must be >= 0.
    last_beat_time_sec
        Absolute time in seconds of the final beat.  Must be > first_beat_time_sec.
    beat_count
        Total number of beats described by this alignment.  Must be >= 2 (a grid
        requires at least two anchor points).
    beats_per_bar
        Time signature numerator — beats per bar.  Default 4.  Must be >= 1.
    confirmed_by_user
        Whether a human has explicitly verified this alignment.
    confidence
        Qualitative confidence label (e.g. ``"high"``, ``"low"``, ``"unknown"``).
        Must be non-empty.
    metadata
        Arbitrary string key/value annotations.
    """

    schema_version:     int
    audio_file:         str
    alignment_method:   str
    first_beat_time_sec: float
    last_beat_time_sec:  float
    beat_count:         int
    beats_per_bar:      int            = 4
    confirmed_by_user:  bool           = False
    confidence:         str            = "unknown"
    metadata:           dict[str, str] = field(default_factory=dict)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_alignment(alignment: BeatAlignment) -> None:
    """Validate a BeatAlignment, raising ValueError on the first violation.

    Called automatically by alignment_from_dict().  Callers that construct
    BeatAlignment directly should call this explicitly.
    """
    if alignment.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, "
            f"got {alignment.schema_version}"
        )
    if not alignment.audio_file or not alignment.audio_file.strip():
        raise ValueError("audio_file must be non-empty")
    if alignment.alignment_method not in _ALLOWED_METHODS:
        raise ValueError(
            f"alignment_method must be one of {sorted(_ALLOWED_METHODS)}, "
            f"got {alignment.alignment_method!r}"
        )
    if alignment.first_beat_time_sec < 0:
        raise ValueError(
            f"first_beat_time_sec must be >= 0, got {alignment.first_beat_time_sec}"
        )
    if alignment.last_beat_time_sec <= alignment.first_beat_time_sec:
        raise ValueError(
            f"last_beat_time_sec must be > first_beat_time_sec; "
            f"got {alignment.last_beat_time_sec} <= {alignment.first_beat_time_sec}"
        )
    if alignment.beat_count < 2:
        raise ValueError(
            f"beat_count must be >= 2, got {alignment.beat_count}"
        )
    if alignment.beats_per_bar < 1:
        raise ValueError(
            f"beats_per_bar must be >= 1, got {alignment.beats_per_bar}"
        )
    if not alignment.confidence or not alignment.confidence.strip():
        raise ValueError("confidence must be non-empty")
    for k, v in alignment.metadata.items():
        if not isinstance(v, str):
            raise ValueError(
                f"metadata[{k!r}] must be a str, got {type(v).__name__}"
            )


# ── Beat grid calculations ────────────────────────────────────────────────────

def estimate_bpm_from_first_last(
    first_beat_time_sec: float,
    last_beat_time_sec:  float,
    beat_count:          int,
) -> float:
    """Return the average BPM implied by the given anchor times and beat count.

    The grid has *beat_count* beats spanning *beat_count - 1* inter-beat
    intervals, so the beat period is::

        period = (last - first) / (beat_count - 1)

    Raises ValueError if the inputs would produce a non-positive period or
    if beat_count < 2.
    """
    if beat_count < 2:
        raise ValueError(f"beat_count must be >= 2, got {beat_count}")
    period = (last_beat_time_sec - first_beat_time_sec) / (beat_count - 1)
    if period <= 0:
        raise ValueError(
            f"beat period is non-positive ({period:.6f} s); "
            "last_beat_time_sec must be > first_beat_time_sec"
        )
    return 60.0 / period


def beat_period_sec(alignment: BeatAlignment) -> float:
    """Return the uniform inter-beat interval in seconds."""
    return (
        (alignment.last_beat_time_sec - alignment.first_beat_time_sec)
        / (alignment.beat_count - 1)
    )


def estimated_bpm(alignment: BeatAlignment) -> float:
    """Return the average BPM computed from the alignment's anchor times."""
    return 60.0 / beat_period_sec(alignment)


def beat_index_to_audio_time(alignment: BeatAlignment, beat_index: float) -> float:
    """Return the audio timeline time (seconds) for a beat index.

    Accepts fractional indices (e.g. 0.5, 1.5) for subdivisions.  Beat index 0
    maps to ``first_beat_time_sec``; ``beat_count - 1`` maps to
    ``last_beat_time_sec``.  Indices beyond ``beat_count - 1`` are extrapolated
    on the same uniform grid.

    Raises ValueError for negative beat_index.
    """
    if beat_index < 0:
        raise ValueError(f"beat_index must be >= 0, got {beat_index}")
    return alignment.first_beat_time_sec + beat_index * beat_period_sec(alignment)


def beat_time(alignment: BeatAlignment, beat_index: int) -> float:
    """Compatibility wrapper for integer beat indices — delegates to beat_index_to_audio_time().

    Retained so existing callers that pass integer indices continue to work
    without changes.  For fractional (subdivision) indices use
    beat_index_to_audio_time() directly.
    """
    return beat_index_to_audio_time(alignment, beat_index)


def beat_times(alignment: BeatAlignment, n_beats: int) -> list[float]:
    """Return absolute times in seconds for beat indices 0 through n_beats - 1."""
    period = beat_period_sec(alignment)
    return [alignment.first_beat_time_sec + i * period for i in range(n_beats)]


def exercise_target_audio_times(
    exercise_or_targets,
    alignment: BeatAlignment,
) -> list[float]:
    """Map exercise target beat positions onto the audio recording timeline.

    Normalises *exercise_or_targets* via ``core.exercise.exercise_targets()``,
    then converts each ``target["time"]`` (a beat position) into an absolute
    audio time using the alignment grid.

    Parameters
    ----------
    exercise_or_targets
        An ``Exercise`` or a raw list of target dicts.
    alignment
        Beat alignment to use for the conversion.

    Returns
    -------
    list[float]
        Absolute audio times in seconds, one per target, in target order.
    """
    from core.exercise import exercise_targets as _exercise_targets
    targets = _exercise_targets(exercise_or_targets)
    period  = beat_period_sec(alignment)
    return [alignment.first_beat_time_sec + t["time"] * period for t in targets]


def alignment_to_exercise_clock(alignment: BeatAlignment) -> dict:
    """Return a minimal clock dict for code that expects bpm and count_in_beats.

    Returns ``{"bpm": estimated_bpm(alignment), "count_in_beats": 0}``.
    Alignment-based sessions use the recording timeline directly, so there is
    no synthetic count-in offset (count_in_beats is always 0).
    """
    return {
        "bpm":            estimated_bpm(alignment),
        "count_in_beats": 0,
    }


# ── Serialisation ─────────────────────────────────────────────────────────────

def alignment_to_dict(alignment: BeatAlignment) -> dict:
    """Return a JSON-serialisable dict representation of *alignment*."""
    return dataclasses.asdict(alignment)


def alignment_from_dict(data: dict) -> BeatAlignment:
    """Construct and validate a BeatAlignment from a plain dict.

    Raises
    ------
    KeyError
        If a required field is missing.
    ValueError
        If validation fails.
    """
    alignment = BeatAlignment(
        schema_version      = int(data["schema_version"]),
        audio_file          = str(data["audio_file"]),
        alignment_method    = str(data["alignment_method"]),
        first_beat_time_sec = float(data["first_beat_time_sec"]),
        last_beat_time_sec  = float(data["last_beat_time_sec"]),
        beat_count          = int(data["beat_count"]),
        beats_per_bar       = int(data.get("beats_per_bar", 4)),
        confirmed_by_user   = bool(data.get("confirmed_by_user", False)),
        confidence          = str(data.get("confidence", "unknown")),
        metadata            = {
            str(k): str(v)
            for k, v in (data.get("metadata") or {}).items()
        },
    )
    validate_alignment(alignment)
    return alignment


def alignment_to_json(alignment: BeatAlignment, *, indent: int = 2) -> str:
    """Serialise *alignment* to a JSON string."""
    return json.dumps(alignment_to_dict(alignment), indent=indent)


def alignment_from_json(text: str) -> BeatAlignment:
    """Parse *text* as JSON and return a validated BeatAlignment."""
    return alignment_from_dict(json.loads(text))


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_alignment_file(path: str | Path) -> BeatAlignment:
    """Load and validate a BeatAlignment from a JSON file."""
    return alignment_from_json(Path(path).read_text(encoding="utf-8"))


def save_alignment_file(alignment: BeatAlignment, path: str | Path) -> None:
    """Serialise *alignment* to a JSON file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(alignment_to_json(alignment), encoding="utf-8")
