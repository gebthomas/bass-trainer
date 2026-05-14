"""Convert evaluator measurements into simple timing feedback.

Pure functions only — no audio hardware.
"""

from __future__ import annotations


def score_timing(
    evaluation: dict,
    target_index: int | None = None,
    tolerance_s: float = 0.050,
) -> dict:
    """Classify an onset as on_time, early, late, or miss.

    Parameters
    ----------
    evaluation   : dict returned by ``evaluate_window``; keys consulted are
                   ``detected``, ``onset_found``, and ``onset_time_s``.
    target_index : optional caller-supplied index passed through unchanged.
    tolerance_s  : half-window (seconds) around zero that counts as on_time.

    Returns
    -------
    dict with:
        status        "on_time" | "early" | "late" | "miss"
        offset_s      float | None   signed timing offset in seconds
        offset_ms     float | None   signed timing offset in milliseconds
        target_index  int | None
    """
    if not evaluation.get("detected") or not evaluation.get("onset_found"):
        return {
            "status":       "miss",
            "offset_s":     None,
            "offset_ms":    None,
            "target_index": target_index,
        }

    offset_s: float = evaluation["onset_time_s"]

    if offset_s < -tolerance_s:
        status = "early"
    elif offset_s > tolerance_s:
        status = "late"
    else:
        status = "on_time"

    return {
        "status":       status,
        "offset_s":     offset_s,
        "offset_ms":    offset_s * 1000.0,
        "target_index": target_index,
    }
