"""Real-time target evaluation: match detected onsets against expected targets.

Pure functions only — no audio hardware, no sounddevice, no threading.

Pipeline position
-----------------
This module sits between onset detection and higher-level feedback/logging::

    onset_times_s  ──▶  evaluate_targets()  ──▶  list[TargetEvaluation]
    target_times_s ──▶

Callers are responsible for converting beat-position targets to absolute
seconds (see core.target_windows.target_audio_time_s) and for computing a
BPM-appropriate tolerance window (see core.timing_policy.match_window_s).

Matching policy
---------------
Targets are processed left-to-right in time order.  For each target, all
unmatched onsets whose absolute timing error falls within *tolerance_s* are
collected.  The nearest such onset (smallest |error|) is selected and
permanently consumed — it cannot match any subsequent target.  If no onset
falls within the window the target is a miss.

This is a greedy, target-ordered, nearest-match policy:

  * "greedy" — decisions are made left-to-right with no backtracking.
  * "target-ordered" — earlier targets have priority over later ones when
    both are within range of the same onset.
  * "nearest-match" — among valid candidates the closest onset wins.
  * Ties (equal |error|) are broken by onset time: the earlier onset wins,
    because ``sorted()`` is stable and ``min()`` returns the first minimum.

Out-of-order onset input is silently normalised: onsets are sorted by time
before matching begins.  Original caller indices are preserved in
``matched_onset_index`` so the caller can cross-reference their input list.

Signed error convention
-----------------------
    signed_error_ms = (actual_time_s − expected_time_s) × 1000
    positive  → onset arrived late
    negative  → onset arrived early

Classification thresholds (both boundaries inclusive)
------------------------------------------------------
    |signed_error_ms| ≤ on_time_threshold_s × 1000  → "on_time"
    signed_error_ms  < −on_time_threshold_ms         → "early"
    signed_error_ms  > +on_time_threshold_ms         → "late"
    no onset matched                                 → "miss"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetEvaluation:
    """Immutable result of matching one target against the detected onset list.

    Fields
    ------
    target_index
        Position of this target in the original ``target_times_s`` input.
    expected_time_s
        Nominal time the target was expected (seconds).
    actual_time_s
        Time of the matched onset (seconds), or ``None`` for a miss.
    signed_error_ms
        ``(actual_time_s − expected_time_s) × 1000``.
        Positive = late; negative = early.  ``None`` for a miss.
    classification
        One of ``"on_time"``, ``"early"``, ``"late"``, ``"miss"``.
    matched_onset_index
        Index into the *caller's* ``onset_times_s`` list of the matched
        onset, or ``None`` for a miss.  Reflects the original input order,
        not the internal sorted order.
    """

    target_index:        int
    expected_time_s:     float
    actual_time_s:       float | None
    signed_error_ms:     float | None
    classification:      str
    matched_onset_index: int | None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_timing(signed_error_ms: float, on_time_threshold_ms: float) -> str:
    """Return the timing classification string for a matched onset.

    The on-time band is symmetric and inclusive on both sides.
    """
    if abs(signed_error_ms) <= on_time_threshold_ms:
        return "on_time"
    return "late" if signed_error_ms > 0 else "early"


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_targets(
    target_times_s: Sequence[float],
    onset_times_s: Sequence[float],
    tolerance_s: float = 0.08,
    on_time_threshold_s: float = 0.03,
) -> list[TargetEvaluation]:
    """Match detected onsets against expected targets.

    Parameters
    ----------
    target_times_s
        Ordered sequence of expected target times in seconds.  Should be
        in ascending time order; this is not enforced but matches the
        documented greedy left-to-right policy.
    onset_times_s
        Detected onset times in seconds.  May be in any order; they are
        sorted internally before matching.
    tolerance_s
        Half-width of the acceptance window in seconds.  An onset at time
        *t* is a candidate for a target at *T* iff ``|t − T| ≤ tolerance_s``.
        The boundary is inclusive.  Default: 0.08 s (80 ms).
    on_time_threshold_s
        Maximum absolute error (seconds) for a matched onset to be
        classified as ``"on_time"`` rather than ``"early"`` or ``"late"``.
        The boundary is inclusive.  Default: 0.03 s (30 ms).

    Returns
    -------
    list[TargetEvaluation]
        One :class:`TargetEvaluation` per target, in the same order as
        ``target_times_s``.  The list is empty when ``target_times_s`` is
        empty.

    Notes
    -----
    * Each onset may match at most one target.
    * Unmatched onsets (no target within tolerance, or already consumed) are
      silently ignored — they do not produce output entries.
    * When ``on_time_threshold_s >= tolerance_s`` the classifications
      ``"early"`` and ``"late"`` can never be produced.
    """
    if not target_times_s:
        return []

    on_time_threshold_ms = on_time_threshold_s * 1000.0

    # Sort onsets by time, keeping original indices for reporting.
    sorted_onsets: list[tuple[int, float]] = sorted(
        enumerate(onset_times_s), key=lambda x: x[1]
    )

    consumed: set[int] = set()   # original onset indices already matched

    results: list[TargetEvaluation] = []

    for target_idx, target_t in enumerate(target_times_s):
        candidates = [
            (orig_idx, t)
            for orig_idx, t in sorted_onsets
            if orig_idx not in consumed and abs(t - target_t) <= tolerance_s
        ]

        if not candidates:
            results.append(TargetEvaluation(
                target_index=target_idx,
                expected_time_s=target_t,
                actual_time_s=None,
                signed_error_ms=None,
                classification="miss",
                matched_onset_index=None,
            ))
            continue

        # Nearest candidate wins; ties resolved by iteration order (time-sorted),
        # so the earlier onset takes priority.
        best_orig_idx, best_t = min(candidates, key=lambda x: abs(x[1] - target_t))
        consumed.add(best_orig_idx)

        signed_error_ms = (best_t - target_t) * 1000.0
        classification  = _classify_timing(signed_error_ms, on_time_threshold_ms)

        results.append(TargetEvaluation(
            target_index=target_idx,
            expected_time_s=target_t,
            actual_time_s=best_t,
            signed_error_ms=signed_error_ms,
            classification=classification,
            matched_onset_index=best_orig_idx,
        ))

    return results
