"""Coordinate real-time per-target feedback without audio hardware.

Pure functions only — no sounddevice, no audio hardware.
"""

from __future__ import annotations

from core.live_feedback import ready_targets
from core.feedback_events import feedback_event


def process_ready_feedback(
    targets: list[dict],
    bpm: float,
    count_in_beats: int,
    sample_rate: int,
    current_sample: int,
    evaluated_indices: set[int] | list[int],
    evaluation_results_by_index: dict[int, dict],
    margin_s: float = 0.15,
) -> dict:
    """Find newly ready targets and emit feedback events for those with results.

    Parameters
    ----------
    targets                     : full ordered list of target dicts
    bpm                         : tempo in beats per minute
    count_in_beats              : count-in length in beats
    sample_rate                 : audio sample rate in Hz
    current_sample              : current position in the audio buffer
    evaluated_indices           : targets already evaluated — not mutated
    evaluation_results_by_index : result dicts keyed by target index;
                                  entries for non-ready or invalid indices
                                  are silently ignored
    margin_s                    : analysis window margin in seconds

    Returns
    -------
    dict with:
        events                   list[dict]  feedback events in target order
        newly_evaluated_indices  set[int]    indices newly evaluated this call
        all_evaluated_indices    set[int]    prior ∪ newly evaluated
    """
    prior = set(evaluated_indices)

    ready = ready_targets(
        targets, bpm, count_in_beats, sample_rate,
        current_sample, prior, margin_s,
    )

    events: list[dict] = []
    newly:  set[int]   = set()

    for idx in ready:
        result = evaluation_results_by_index.get(idx)
        if result is None:
            continue
        events.append(feedback_event(idx, targets[idx], result))
        newly.add(idx)

    return {
        "events":                  events,
        "newly_evaluated_indices": newly,
        "all_evaluated_indices":   prior | newly,
    }
