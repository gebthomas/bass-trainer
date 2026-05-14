"""Coordinate real-time practice evaluation.

Pure orchestration — no sounddevice, no UI.
"""

from __future__ import annotations

import numpy as np

from core.audio_windows import extract_target_window
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
                "target_index": int,
                "evaluation":   dict,   # from evaluate_window
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
        events.append({"target_index": idx, "evaluation": evaluation})

    return events
