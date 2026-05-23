"""Session bundle — orchestration layer connecting practice assets.

Pure Python — no audio hardware, no playback, no sounddevice.

A SessionBundle is the resolved, in-memory union of all assets needed to run
or analyse a practice session:

    PracticeMode  (what mode, what files are referenced)
    Exercise      (what to play and at what tempo)
    BeatAlignment (how the exercise maps onto an audio recording)
    SessionLog    (what happened — added after or during the session)

``load_session_bundle()`` is the main entry point: it reads a PracticeMode
file and loads all assets required by that mode.  ``validate_session_bundle()``
checks that the combination is self-consistent.

Typical usage
-------------
    from core.session_bundle import load_session_bundle, bundle_target_audio_times

    bundle = load_session_bundle("practice_modes/my_session.json")
    times  = bundle_target_audio_times(bundle)

Path resolution
---------------
Relative paths stored in the PracticeMode (``exercise_path``,
``alignment_path``) are resolved relative to the directory that contains the
PracticeMode file.  Absolute paths are used as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.alignment import BeatAlignment, exercise_target_audio_times, load_alignment_file
from core.exercise import Exercise, exercise_targets, load_exercise_file
from core.practice_mode import PracticeMode, load_practice_mode_file
from core.session_log import SessionLog
from core.target_windows import target_audio_time_s


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SessionBundle:
    """Resolved in-memory representation of all assets for a practice session.

    Parameters
    ----------
    practice_mode
        The loaded and validated PracticeMode configuration.
    exercise
        Loaded Exercise, or None if not required by the mode.
    alignment
        Loaded BeatAlignment, or None if not required by the mode.
    session_log
        Optional SessionLog — populated after or during a session, not on load.
    """

    practice_mode: PracticeMode
    exercise:      Optional[Exercise]      = None
    alignment:     Optional[BeatAlignment] = None
    session_log:   Optional[SessionLog]    = None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_session_bundle(bundle: SessionBundle) -> None:
    """Validate a SessionBundle, raising ValueError on the first violation found.

    Checks that the loaded assets match the requirements of the mode:

    ``metronome_exercise``
        Requires ``exercise``.  ``alignment`` must be absent.

    ``recording_aligned_exercise``
        Requires both ``exercise`` and ``alignment``.

    ``play_to_align``
        ``exercise`` and ``alignment`` must both be absent.

    ``session_log``, if present, is accepted without further cross-checks for
    now.  Timestamp and path-consistency validation is deferred.
    """
    mode = bundle.practice_mode.mode

    if mode == "metronome_exercise":
        if bundle.exercise is None:
            raise ValueError("metronome_exercise requires an exercise")
        if bundle.alignment is not None:
            raise ValueError("metronome_exercise does not use an alignment")

    elif mode == "recording_aligned_exercise":
        if bundle.exercise is None:
            raise ValueError("recording_aligned_exercise requires an exercise")
        if bundle.alignment is None:
            raise ValueError("recording_aligned_exercise requires an alignment")

    elif mode == "play_to_align":
        if bundle.exercise is not None:
            raise ValueError("play_to_align does not use an exercise")
        if bundle.alignment is not None:
            raise ValueError("play_to_align does not use an alignment")


# ── Loader ────────────────────────────────────────────────────────────────────

def load_session_bundle(practice_mode_path: str | Path) -> SessionBundle:
    """Load a PracticeMode and all assets it requires, returning a SessionBundle.

    Parameters
    ----------
    practice_mode_path
        Path to a PracticeMode JSON file.

    Returns
    -------
    SessionBundle
        Validated bundle with all required assets loaded.

    Raises
    ------
    FileNotFoundError
        If a required asset file (exercise, alignment) cannot be found.
    ValueError
        If the loaded combination fails ``validate_session_bundle()``.
    """
    pm_path = Path(practice_mode_path).resolve()
    base    = pm_path.parent
    pm      = load_practice_mode_file(pm_path)

    exercise  = None
    alignment = None

    if pm.exercise_path:
        ex_path = _resolve(base, pm.exercise_path)
        if not ex_path.exists():
            raise FileNotFoundError(
                f"exercise file not found: {ex_path} "
                f"(exercise_path={pm.exercise_path!r} in {pm_path})"
            )
        exercise = load_exercise_file(ex_path)

    if pm.alignment_path:
        al_path = _resolve(base, pm.alignment_path)
        if not al_path.exists():
            raise FileNotFoundError(
                f"alignment file not found: {al_path} "
                f"(alignment_path={pm.alignment_path!r} in {pm_path})"
            )
        alignment = load_alignment_file(al_path)

    bundle = SessionBundle(
        practice_mode = pm,
        exercise      = exercise,
        alignment     = alignment,
    )
    validate_session_bundle(bundle)
    return bundle


# ── Target time helper ────────────────────────────────────────────────────────

def bundle_target_audio_times(bundle: SessionBundle) -> list[float]:
    """Return absolute audio times in seconds for each exercise target.

    ``metronome_exercise``
        Times are derived from the exercise ``bpm`` and ``count_in_beats``.

    ``recording_aligned_exercise``
        Times are derived from the alignment grid via
        :func:`~core.alignment.exercise_target_audio_times`.

    ``play_to_align``
        No targets are defined yet; returns an empty list.
    """
    mode = bundle.practice_mode.mode

    if mode == "metronome_exercise":
        ex      = bundle.exercise
        targets = exercise_targets(ex)
        return [
            target_audio_time_s(targets, i, ex.bpm, ex.count_in_beats)
            for i in range(len(targets))
        ]

    if mode == "recording_aligned_exercise":
        return list(exercise_target_audio_times(bundle.exercise, bundle.alignment))

    return []  # play_to_align


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve(base: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()
