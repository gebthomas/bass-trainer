"""Pure session executor for non-audio workflows.

Drives a SessionBundle against a provided onset stream and returns a
populated SessionLog.  No sounddevice, no playback, no hardware.

Typical usage
-------------
    from core.session_bundle import load_session_bundle
    from core.session_runner import run_session_bundle

    bundle = load_session_bundle("practice_modes/my_session.json")
    log = run_session_bundle(
        bundle,
        onset_times_sec=[1.02, 1.51, 2.00, 2.48],
        started_at="2026-05-23T10:00:00",
        ended_at="2026-05-23T10:05:00",
    )

Matching algorithm
------------------
Onsets are processed in chronological order.  For each onset the closest
unclaimed target within the match window is claimed (ties resolve to the
lower target index — identical to the rule in ``SessionEngine.on_onset``).
Unmatched targets become ``target_miss`` events; unmatched onsets become
``extra_onset`` events.  All events are returned sorted by ``time_sec``.

Match window
------------
The default window is half a beat at the session tempo:

    metronome_exercise          30.0 / exercise.bpm
    recording_aligned_exercise  30.0 / estimated_bpm(alignment)

This matches the ``SessionEngine`` default of ``30.0 / bpm``.
"""

from __future__ import annotations

from core.alignment import estimated_bpm
from core.session_bundle import SessionBundle, bundle_target_audio_times
from core.session_log import SessionEvent, SessionLog, validate_session_log


# ── Public API ────────────────────────────────────────────────────────────────

def run_session_bundle(
    bundle: SessionBundle,
    onset_times_sec: list[float],
    *,
    started_at: str,
    ended_at: str | None = None,
) -> SessionLog:
    """Execute a SessionBundle against an onset stream and return a SessionLog.

    Parameters
    ----------
    bundle
        Validated session bundle with all required assets loaded.
    onset_times_sec
        Detected onset times in seconds.  Need not be pre-sorted.
    started_at
        ISO 8601 timestamp string for when the session began.
    ended_at
        ISO 8601 timestamp string for when the session ended.  May be None.

    Returns
    -------
    SessionLog
        Populated and validated log.  Events are sorted by ``time_sec``.

        ``target_hit``
            An onset matched a target.  ``target_index`` is set; ``value``
            holds the timing error in seconds (positive = late, negative =
            early).
        ``target_miss``
            A target elapsed with no matching onset.  ``target_index`` is set;
            ``time_sec`` is the nominal target time; ``value`` is None.
        ``extra_onset``
            An onset did not match any target.  ``target_index`` is None;
            ``value`` is None.

    Metrics
    -------
    ``targets_total``, ``targets_hit``, ``targets_missed``, ``extra_onsets``
    """
    pm           = bundle.practice_mode
    target_times = bundle_target_audio_times(bundle)
    onsets       = sorted(onset_times_sec)

    if target_times and onsets:
        window             = _default_match_window(bundle)
        target_to_onset, matched_onset_idxs = _match(target_times, onsets, window)
    else:
        target_to_onset      = {}
        matched_onset_idxs   = set()

    events: list[SessionEvent] = []

    for t_idx, t_time in enumerate(target_times):
        if t_idx in target_to_onset:
            o_idx   = target_to_onset[t_idx]
            onset_t = onsets[o_idx]
            events.append(SessionEvent(
                time_sec     = onset_t,
                event_type   = "target_hit",
                target_index = t_idx,
                value        = onset_t - t_time,
            ))
        else:
            events.append(SessionEvent(
                time_sec     = t_time,
                event_type   = "target_miss",
                target_index = t_idx,
            ))

    extra_onset_idxs = set(range(len(onsets))) - matched_onset_idxs
    for o_idx in sorted(extra_onset_idxs):
        events.append(SessionEvent(
            time_sec   = onsets[o_idx],
            event_type = "extra_onset",
        ))

    events.sort(key=lambda e: e.time_sec)

    n_targets = len(target_times)
    n_hits    = len(target_to_onset)
    n_misses  = n_targets - n_hits
    n_extras  = len(extra_onset_idxs)

    log = SessionLog(
        schema_version = 1,
        started_at     = started_at,
        ended_at       = ended_at,
        exercise_path  = pm.exercise_path,
        alignment_path = pm.alignment_path,
        events         = events,
        metrics        = {
            "targets_total":  n_targets,
            "targets_hit":    n_hits,
            "targets_missed": n_misses,
            "extra_onsets":   n_extras,
        },
    )
    validate_session_log(log)
    return log


# ── Private helpers ───────────────────────────────────────────────────────────

def _default_match_window(bundle: SessionBundle) -> float:
    """Return half-a-beat match window in seconds for *bundle*."""
    mode = bundle.practice_mode.mode
    if mode == "metronome_exercise" and bundle.exercise is not None:
        return 30.0 / bundle.exercise.bpm
    if mode == "recording_aligned_exercise" and bundle.alignment is not None:
        return 30.0 / estimated_bpm(bundle.alignment)
    return 0.25


def _match(
    target_times: list[float],
    onsets: list[float],
    window_s: float,
) -> tuple[dict[int, int], set[int]]:
    """Greedy nearest-target match — mirrors ``SessionEngine.on_onset`` rule.

    For each onset in chronological order, claims the closest unclaimed target
    within *window_s*.  Ties resolve to the lower target index (strict
    ``delta < best_delta`` comparison).

    Returns
    -------
    target_to_onset : dict[int, int]
        Maps each matched target index to the onset index that claimed it.
    matched_onset_idxs : set[int]
        Indices into *onsets* that were matched to a target.
    """
    target_to_onset:    dict[int, int] = {}
    matched_onset_idxs: set[int]       = set()

    for o_idx, onset_t in enumerate(onsets):
        best_t_idx = None
        best_delta = float("inf")
        for t_idx, t_time in enumerate(target_times):
            if t_idx in target_to_onset:
                continue
            delta = abs(onset_t - t_time)
            if delta <= window_s and delta < best_delta:
                best_delta = delta
                best_t_idx = t_idx
        if best_t_idx is not None:
            target_to_onset[best_t_idx] = o_idx
            matched_onset_idxs.add(o_idx)

    return target_to_onset, matched_onset_idxs
