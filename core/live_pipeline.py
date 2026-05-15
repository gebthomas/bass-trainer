"""Coordinate real-time practice evaluation.

Pure orchestration — no sounddevice, no UI.
"""

from __future__ import annotations

import numpy as np

from core.audio_windows import extract_target_window
from core.feedback_events import feedback_event
from core.practice_session import PracticeSession
from core.realtime_evaluator import evaluate_window
from core.tempo_tracker import TempoTracker

# Fraction of one beat used as the default maximum window shift.
_DEFAULT_MAX_SHIFT_FRAC = 0.30


def process_realtime_audio(
    audio: np.ndarray,
    current_sample: int,
    session: PracticeSession,
    tempo_tracker: TempoTracker | None = None,
    adaptive_window_shift: float = 0.5,
    max_window_shift_s: float | None = None,
) -> list[dict]:
    """Advance session state and evaluate any newly ready targets.

    Parameters
    ----------
    audio                : full recorded audio buffer (1-D mono or 2-D frames×channels).
    current_sample       : current position in the audio buffer (samples).
    session              : mutable ``PracticeSession``; ``evaluated_indices`` is
                           updated in place.
    tempo_tracker        : optional :class:`~core.tempo_tracker.TempoTracker`.
                           When provided the pipeline runs in *adaptive mode*:
                           extraction windows are shifted toward the tracker's
                           adjusted target time and timing evaluation scores
                           against that adjusted reference.  ``None`` (default)
                           preserves the original fixed-grid behaviour.
    adaptive_window_shift: fraction of the gap between nominal and adjusted
                           target times to apply as a window shift.  ``0.5``
                           (default) moves the window half-way; ``0.0`` gives
                           no shift; ``1.0`` moves it fully to the adjusted
                           time.  Only used when *tempo_tracker* is provided.
    max_window_shift_s   : hard cap on the window shift in seconds.  When
                           ``None`` (default) the cap is
                           ``0.30 × nominal_beat_duration``, approximately
                           30 % of one beat.  Only used when *tempo_tracker*
                           is provided.

    Returns
    -------
    list[dict]
        One entry per newly ready target, in ascending target-index order.

        Fixed-grid fields (always present)::

            {
                "target_index":      int,
                "evaluation":        dict,         # raw output from evaluate_window
                "severity":          str,          # "good" | "warn" | "miss"
                "timing_error_s":    float | None, # signed, beat-relative; +ve = late
                "messages":          list[str],
                "expected_note":     str | None,
                "detected_note":     str | None,   # "?" = onset found, pitch not measured
                "pitch_error_cents": float | None,
                "confidence":        None,
            }

        Additional fields present only when *tempo_tracker* is supplied::

            {
                "timing_grid":               "adaptive",
                "nominal_target_time_s":     float,
                "adjusted_target_time_s":    float | None,
                "window_center_time_s":      float,
                "window_shift_s":            float,
                "tempo_ratio":               float,
                "current_bpm":               float,
                "tempo_tracker_confidence":  float,
            }

        Empty list if no targets became ready this call.
    """
    audio = np.asarray(audio)

    beat_s     = 60.0 / session.bpm
    count_in_s = session.count_in_beats * beat_s
    _max_shift = (max_window_shift_s
                  if max_window_shift_s is not None
                  else _DEFAULT_MAX_SHIFT_FRAC * beat_s)

    newly_ready = session.update(current_sample)

    events = []
    for idx in newly_ready:
        nominal_beat_time = count_in_s + session.targets[idx]["time"] * beat_s

        if tempo_tracker is not None:
            window_center_time_s, window_shift_s = _compute_adaptive_window(
                nominal_beat_time, tempo_tracker, adaptive_window_shift, _max_shift,
            )
        else:
            window_center_time_s = nominal_beat_time
            window_shift_s = 0.0

        window = extract_target_window(
            audio,
            session.targets[idx],
            session.bpm,
            session.count_in_beats,
            session.sample_rate,
            center_time_s=window_center_time_s if tempo_tracker is not None else None,
        )
        evaluation = evaluate_window(window["audio"], session.sample_rate)
        result = _evaluation_to_result(window, evaluation, session.sample_rate, tempo_tracker)
        ev = feedback_event(idx, session.targets[idx], result)

        event: dict = {
            "target_index":      idx,
            "evaluation":        evaluation,
            "severity":          ev["severity"],
            "timing_error_s":    ev["timing_error_s"],
            "messages":          ev["messages"],
            "expected_note":     ev["expected_note"],
            "detected_note":     ev["detected_note"],
            "pitch_error_cents": ev["pitch_error_cents"],
            "confidence":        ev["confidence"],
        }

        if tempo_tracker is not None:
            event["timing_grid"]              = "adaptive"
            event["nominal_target_time_s"]    = nominal_beat_time
            event["adjusted_target_time_s"]   = result["_adjusted_beat_time_s"]
            event["window_center_time_s"]     = window_center_time_s
            event["window_shift_s"]           = window_shift_s
            event["tempo_ratio"]              = tempo_tracker.tempo_ratio
            event["current_bpm"]              = tempo_tracker.current_tempo_bpm()
            event["tempo_tracker_confidence"] = tempo_tracker.confidence()

        events.append(event)

    return events


def _compute_adaptive_window(
    nominal_beat_time: float,
    tempo_tracker: TempoTracker,
    shift_fraction: float,
    max_shift_s: float,
) -> tuple[float, float]:
    """Return (window_center_time_s, window_shift_s) for adaptive mode.

    The shift is ``shift_fraction × (adjusted − nominal)``, clamped to
    ``±max_shift_s``.  When the tracker has no anchor yet,
    ``adjusted == nominal`` so the shift is zero.
    """
    adjusted = tempo_tracker.adjusted_target_time(nominal_beat_time)
    raw_shift = shift_fraction * (adjusted - nominal_beat_time)
    clamped   = max(-max_shift_s, min(max_shift_s, raw_shift))
    return nominal_beat_time + clamped, clamped


def _evaluation_to_result(
    window: dict,
    evaluation: dict,
    sample_rate: int,
    tempo_tracker: TempoTracker | None = None,
) -> dict:
    """Translate evaluator output + window metadata into a feedback_event result dict.

    timing_error_s is beat-relative (positive = late).  In fixed-grid mode it
    is computed relative to window["target_sample"] (the nominal beat position).
    In adaptive mode it is computed relative to the tracker's adjusted target
    time; the tracker is then updated via ``observe()`` so subsequent calls
    benefit from the new data point.

    The extraction window may already be shifted (see ``_compute_adaptive_window``
    and the ``center_time_s`` parameter of ``extract_target_window``).  Regardless
    of where the window sits, ``window["target_sample"]`` always reflects the
    nominal beat position, so the nominal reference for scoring is stable.

    The returned dict includes a private ``_adjusted_beat_time_s`` key used by
    the caller to populate the ``adjusted_target_time_s`` event field.
    """
    if evaluation.get("onset_found") and evaluation["onset_sample"] is not None:
        abs_onset         = window["start_sample"] + evaluation["onset_sample"]
        nominal_beat_time = window["target_sample"] / sample_rate
        actual_onset_time = abs_onset / sample_rate

        if tempo_tracker is not None:
            adjusted_beat_time   = tempo_tracker.adjusted_target_time(nominal_beat_time)
            adjusted_target_samp = round(adjusted_beat_time * sample_rate)
            timing_error_s       = (abs_onset - adjusted_target_samp) / sample_rate
            tempo_tracker.observe(nominal_beat_time, actual_onset_time)
        else:
            timing_error_s     = (abs_onset - window["target_sample"]) / sample_rate
            adjusted_beat_time = None

        return {
            "detected_note":         "?",  # onset found; pitch not measured in realtime
            "timing_error_s":        timing_error_s,
            "pitch_error_cents":     None,
            "_adjusted_beat_time_s": adjusted_beat_time,
        }

    return {
        "detected_note":         None,
        "timing_error_s":        None,
        "pitch_error_cents":     None,
        "_adjusted_beat_time_s": None,
    }
