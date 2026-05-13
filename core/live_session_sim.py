"""Simulate a live feedback session through time without audio hardware.

Pure functions only — no sounddevice, no audio hardware.

Time is advanced in steps of tick_s using index-based multiplication
(t = tick_index * tick_s) to avoid cumulative floating-point drift.
"""

from __future__ import annotations

from core.live_feedback_controller import process_ready_feedback


def simulate_live_session(
    targets: list[dict],
    bpm: float,
    count_in_beats: int,
    sample_rate: int,
    duration_s: float,
    tick_s: float,
    evaluation_schedule: dict[int, dict],
    margin_s: float = 0.15,
) -> dict:
    """Advance simulated time from 0 to duration_s in steps of tick_s.

    evaluation_schedule maps target index → result dict.  Results are treated
    as immediately available once the target's analysis window has passed.

    Returns
    -------
    dict with:
        events            list[dict]  all feedback events in chronological order
        evaluated_indices set[int]   all indices evaluated during the session
        tick_count        int        number of ticks processed (always >= 1)
        duration_s        float      the requested duration (= parameter)
    """
    all_events: list[dict] = []
    evaluated:  set[int]   = set()
    tick_count = 0
    t = 0.0

    while t <= duration_s + 1e-9:
        current_sample = round(t * sample_rate)
        out = process_ready_feedback(
            targets, bpm, count_in_beats, sample_rate,
            current_sample, evaluated, evaluation_schedule, margin_s,
        )
        all_events.extend(out["events"])
        evaluated = out["all_evaluated_indices"]
        tick_count += 1
        t = tick_count * tick_s   # index-based: no float drift

    return {
        "events":            all_events,
        "evaluated_indices": evaluated,
        "tick_count":        tick_count,
        "duration_s":        duration_s,
    }


def simulate_until_complete(
    targets: list[dict],
    bpm: float,
    count_in_beats: int,
    sample_rate: int,
    evaluation_schedule: dict[int, dict],
    tick_s: float = 0.02,
    max_duration_s: float = 30.0,
    margin_s: float = 0.15,
) -> dict:
    """Simulate until all scheduled targets are evaluated or max_duration_s elapses.

    'Scheduled' means indices present in evaluation_schedule that are valid
    into targets (0 <= index < len(targets)).  Invalid keys are ignored for
    the completion check.

    Returns
    -------
    dict with:
        events            list[dict]  all feedback events in chronological order
        evaluated_indices set[int]   all indices evaluated
        tick_count        int        number of ticks processed
        duration_s        float      simulated time of the last processed tick
        completed         bool       True if all schedulable targets were evaluated
    """
    schedulable = {i for i in evaluation_schedule if 0 <= i < len(targets)}

    all_events: list[dict] = []
    evaluated:  set[int]   = set()
    tick_count = 0
    t = 0.0

    while True:
        current_sample = round(t * sample_rate)
        out = process_ready_feedback(
            targets, bpm, count_in_beats, sample_rate,
            current_sample, evaluated, evaluation_schedule, margin_s,
        )
        all_events.extend(out["events"])
        evaluated = out["all_evaluated_indices"]
        tick_count += 1
        last_t = t

        if schedulable <= evaluated:
            return {
                "events":            all_events,
                "evaluated_indices": evaluated,
                "tick_count":        tick_count,
                "duration_s":        last_t,
                "completed":         True,
            }

        t = tick_count * tick_s
        if t > max_duration_s + 1e-9:
            return {
                "events":            all_events,
                "evaluated_indices": evaluated,
                "tick_count":        tick_count,
                "duration_s":        last_t,
                "completed":         False,
            }
