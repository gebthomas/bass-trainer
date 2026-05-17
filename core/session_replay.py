"""Deterministic replay of timestamped onset streams for SessionEngine.

Pure Python — no audio hardware, no sounddevice.

Intended uses
-------------
- Offline replay of captured onset logs without audio hardware.
- Regression testing: store onset streams + expected results as JSON fixtures
  and verify that future engine changes do not silently alter feedback output.

Typical usage
-------------
    import json
    from core.session_replay import replay_session_data, summarize_replay

    data   = json.loads(Path("session.json").read_text())
    events = replay_session_data(data)
    stats  = summarize_replay(events)

Session file schema (JSON)
--------------------------
    {
      "version": 1,
      "bpm": 120.0,
      "count_in_beats": 0,
      "targets": [{"time": 1, "note": "?"}, ...],
      "engine":  {"match_window_s": 0.25},   // optional; omit to use computed default
      "tracker": {"phase_alpha": 0.10, ...}, // optional; omit to use class defaults
      "onsets":  [0.512, 1.008, ...]         // seconds; need not be pre-sorted
      "expected": {                          // optional; used by regression tests
        "total": 4, "hits": 3, "misses": 1, "good": 2, "warn": 1
      },
      "golden_events": [...]                 // optional; full deterministic check
    }

Note on determinism
-------------------
replay_session_data() produces identical output across runs as long as:
  * the same session file is used (onsets are sorted on load),
  * engine and tracker parameters come from the file rather than code defaults,
  * the host Python version is the same (IEEE-754 arithmetic is deterministic
    per platform but not guaranteed cross-platform).

When engine/tracker keys are absent, the module falls back to class defaults,
which means a code-default change can silently alter the replay for those
fixtures. Prefer storing explicit parameters in production fixture files.
"""

from __future__ import annotations

from core.session_engine import SessionEngine
from core.target_windows import target_audio_time_s
from core.tempo_tracker import TempoTracker


def replay_session_data(data: dict) -> list[dict]:
    """Replay an onset stream from a session dict and return feedback events.

    Parameters
    ----------
    data
        Session dict matching the schema described in the module docstring.

    Returns
    -------
    list[dict]
        Feedback event dicts in emission order, each extended with a
        ``replay_time_s`` field (the simulated time at which the event fired).
        Hit events also carry ``onset_time_s`` (added by SessionEngine).
    """
    bpm            = data["bpm"]
    count_in_beats = data["count_in_beats"]
    targets        = data["targets"]
    onsets         = sorted(data["onsets"])

    engine_cfg  = data.get("engine",  {})
    tracker_cfg = data.get("tracker", {})

    tracker = TempoTracker(bpm, **tracker_cfg)
    engine  = SessionEngine(
        targets,
        bpm=bpm,
        count_in_beats=count_in_beats,
        tracker=tracker,
        **engine_cfg,
    )

    events: list[dict] = []

    for onset_t in onsets:
        # on_onset before update_time: mirrors the live audio loop and ensures
        # an onset at the exact deadline is matched rather than declared a miss.
        for ev in engine.on_onset(onset_t):
            ev["replay_time_s"] = onset_t
            events.append(ev)
        for ev in engine.update_time(onset_t):
            ev["replay_time_s"] = onset_t
            events.append(ev)

    # Advance past the last target's deadline to flush any remaining misses.
    if targets:
        last_nom   = target_audio_time_s(targets, len(targets) - 1, bpm, count_in_beats)
        end_time_s = last_nom + engine.match_window_s + 1e-6
        for ev in engine.update_time(end_time_s):
            ev["replay_time_s"] = end_time_s
            events.append(ev)

    return events


def summarize_replay(events: list[dict]) -> dict:
    """Aggregate a replay event list into session-level counts and statistics.

    Parameters
    ----------
    events
        List returned by :func:`replay_session_data`.

    Returns
    -------
    dict with keys:
        total, hits, misses, good, warn, mean_abs_timing_s
    """
    hits   = [e for e in events if e["detected_note"] is not None]
    misses = [e for e in events if e["detected_note"] is None]
    t_abs  = [abs(e["timing_error_s"]) for e in hits if e["timing_error_s"] is not None]

    return {
        "total":             len(events),
        "hits":              len(hits),
        "misses":            len(misses),
        "good":              sum(1 for e in events if e["severity"] == "good"),
        "warn":              sum(1 for e in events if e["severity"] == "warn"),
        "mean_abs_timing_s": sum(t_abs) / len(t_abs) if t_abs else 0.0,
    }
