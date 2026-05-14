"""Coordinate real-time practice evaluation.

Pure orchestration — no sounddevice, no UI.
"""

from __future__ import annotations

import numpy as np

from core.audio_windows import extract_target_window
from core.feedback_events import feedback_event
from core.practice_session import PracticeSession
from core.realtime_evaluator import evaluate_window


def process_realtime_audio(
    audio: np.ndarray,
    current_sample: int,
    session: PracticeSession,
) -> list[dict]:
    """Advance session state and evaluate any newly ready targets.

    Parameters
    ----------
    audio          : full recorded audio buffer (1-D mono or 2-D frames×channels).
    current_sample : current position in the audio buffer (samples).
    session        : mutable ``PracticeSession``; ``evaluated_indices`` is
                     updated in place.

    Returns
    -------
    list[dict]
        One entry per newly ready target, in ascending target-index order::

            {
                "target_index":     int,
                "evaluation":       dict,         # raw output from evaluate_window
                "severity":         str,          # "good" | "warn" | "miss"
                "timing_error_s":   float | None, # signed, beat-relative; positive = late
                "messages":         list[str],
                "expected_note":    str | None,
                "detected_note":    str | None,   # "?" = onset found, pitch not measured
                "pitch_error_cents": float | None,
                "confidence":       None,
            }

        Empty list if no targets became ready this call.
    """
    audio = np.asarray(audio)

    newly_ready = session.update(current_sample)

    events = []
    for idx in newly_ready:
        window = extract_target_window(
            audio,
            session.targets[idx],
            session.bpm,
            session.count_in_beats,
            session.sample_rate,
        )
        evaluation = evaluate_window(window["audio"], session.sample_rate)
        result = _evaluation_to_result(window, evaluation, session.sample_rate)
        ev = feedback_event(idx, session.targets[idx], result)
        events.append({
            "target_index":      idx,
            "evaluation":        evaluation,
            "severity":          ev["severity"],
            "timing_error_s":    ev["timing_error_s"],
            "messages":          ev["messages"],
            "expected_note":     ev["expected_note"],
            "detected_note":     ev["detected_note"],
            "pitch_error_cents": ev["pitch_error_cents"],
            "confidence":        ev["confidence"],
        })

    return events


def _evaluation_to_result(window: dict, evaluation: dict, sample_rate: int) -> dict:
    """Translate evaluator output + window metadata into a feedback_event result dict.

    timing_error_s is beat-relative (positive = late), computed from the absolute
    sample positions so pre_roll does not bias the score.
    """
    if evaluation.get("onset_found") and evaluation["onset_sample"] is not None:
        abs_onset      = window["start_sample"] + evaluation["onset_sample"]
        timing_error_s = (abs_onset - window["target_sample"]) / sample_rate
        return {
            "detected_note":     "?",  # onset detected; pitch not measured in realtime
            "timing_error_s":    timing_error_s,
            "pitch_error_cents": None,
        }
    return {
        "detected_note":     None,
        "timing_error_s":    None,
        "pitch_error_cents": None,
    }
