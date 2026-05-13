"""Convert per-target evaluation results into real-time feedback events.

Pure functions only — no audio hardware, no sounddevice.

Severity rules
--------------
Timing (timing_error_s, positive = late):
    |error| <= 0.05 s  → good
    |error| <= 0.12 s  → warn
    otherwise          → miss

Pitch (pitch_error_cents, positive = sharp):
    |error| <= 25 c    → good
    |error| <= 50 c    → warn
    otherwise          → miss

No detected note      → miss
Confidence < 0.5      → at least warn

Overall severity is the worst of all individual dimensions.
"""

from __future__ import annotations

_TIMING_GOOD = 0.05
_TIMING_WARN = 0.12
_PITCH_GOOD  = 25.0
_PITCH_WARN  = 50.0
_LOW_CONF    = 0.5

_RANK = {"good": 0, "warn": 1, "miss": 2}
_NAME = {0: "good", 1: "warn", 2: "miss"}


# ── Public API ────────────────────────────────────────────────────────────────

def feedback_event(target_idx: int, target: dict, result: dict) -> dict:
    """Return a feedback event dict for one evaluated target.

    result may contain any subset of:
        detected_note     : str | None
        timing_error_s    : float | None  (positive = late)
        pitch_error_cents : float | None  (positive = sharp)
        confidence        : float | None  (0..1)
    """
    detected_note     = result.get("detected_note")
    timing_error_s    = result.get("timing_error_s")
    pitch_error_cents = result.get("pitch_error_cents")
    confidence        = result.get("confidence")

    rank: int = 0
    messages: list[str] = []

    if detected_note is None:
        rank = 2
        messages.append("No note detected")
    else:
        if timing_error_s is not None:
            abs_t = abs(timing_error_s)
            if abs_t <= _TIMING_GOOD:
                messages.append("Good timing")
            elif abs_t <= _TIMING_WARN:
                rank = max(rank, 1)
                messages.append("A little late" if timing_error_s > 0 else "A little early")
            else:
                rank = max(rank, 2)
                messages.append("Very late" if timing_error_s > 0 else "Very early")

        if pitch_error_cents is not None:
            abs_p = abs(pitch_error_cents)
            if abs_p > _PITCH_GOOD:
                rank = max(rank, 2 if abs_p > _PITCH_WARN else 1)
                messages.append("Pitch high" if pitch_error_cents > 0 else "Pitch low")

        if confidence is not None and confidence < _LOW_CONF:
            rank = max(rank, 1)

    return {
        "target_idx":          target_idx,
        "expected_note":       target.get("note"),
        "detected_note":       detected_note,
        "timing_error_s":      timing_error_s,
        "pitch_error_cents":   pitch_error_cents,
        "confidence":          confidence,
        "severity":            _NAME[rank],
        "messages":            messages,
    }


def summarize_feedback(events: list[dict]) -> dict:
    """Aggregate a list of feedback events into session-level statistics.

    None fields (e.g. no-detection misses) are excluded from means.
    """
    good = sum(1 for e in events if e["severity"] == "good")
    warn = sum(1 for e in events if e["severity"] == "warn")
    miss = sum(1 for e in events if e["severity"] == "miss")

    t_abs = [abs(e["timing_error_s"])    for e in events if e["timing_error_s"]    is not None]
    p_abs = [abs(e["pitch_error_cents"]) for e in events if e["pitch_error_cents"] is not None]

    return {
        "total":                      len(events),
        "good_count":                 good,
        "warn_count":                 warn,
        "miss_count":                 miss,
        "mean_abs_timing_error_s":    sum(t_abs) / len(t_abs) if t_abs else 0.0,
        "mean_abs_pitch_error_cents": sum(p_abs) / len(p_abs) if p_abs else 0.0,
    }
