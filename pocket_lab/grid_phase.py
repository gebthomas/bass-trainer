"""Beat-zero phase estimation from onset data."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

_PHASE_EXCLUDE_LABELS = {"ignore", "string_noise"}
_PHASE_BOOST_LABELS = {"true_attack", "downbeat", "beat3"}
_PHASE_BOOST_FACTOR = 3.0


def estimate_beat_zero(
    onset_times: np.ndarray,
    bpm: float,
    beats_per_measure: int,
    anchor_beats: List[int],
    onset_strengths: np.ndarray | None = None,
) -> float:
    """Estimate beat_zero_s by finding the phase that best aligns onsets with anchor beats.

    For each onset and each anchor beat, computes the implied grid phase.
    The phase with the most weighted onset support wins.

    Parameters
    ----------
    onset_times        : detected onset times in seconds.
    bpm                : tempo.
    beats_per_measure  : meter.
    anchor_beats       : 1-based beat numbers expected to carry strong bass (e.g. [1, 3]).
    onset_strengths    : optional per-onset weights (higher = more influence).

    Returns
    -------
    float : estimated beat_zero_s (phase offset in seconds, within one measure).
    """
    if len(onset_times) == 0 or len(anchor_beats) == 0:
        return 0.0

    beat_s = 60.0 / bpm
    measure_s = beats_per_measure * beat_s
    weights = onset_strengths if onset_strengths is not None else np.ones(len(onset_times))

    anchor_positions = np.array([(b - 1) * beat_s for b in anchor_beats])

    candidates = []
    candidate_weights = []
    for i, t in enumerate(onset_times):
        for ap in anchor_positions:
            phi = (t - ap) % measure_s
            candidates.append(phi)
            candidate_weights.append(float(weights[i]))

    candidates = np.array(candidates)
    candidate_weights = np.array(candidate_weights)

    tolerance = beat_s * 0.15
    n_steps = max(200, int(measure_s / 0.001))
    best_phi = 0.0
    best_score = -1.0
    best_total_dist = float("inf")

    for step in range(n_steps):
        phi = step * measure_s / n_steps
        dists = np.abs(candidates - phi)
        dists = np.minimum(dists, measure_s - dists)
        within = dists < tolerance
        score = float(np.sum(candidate_weights[within]))
        total_dist = float(np.sum(dists[within])) if np.any(within) else float("inf")
        if score > best_score or (score == best_score and total_dist < best_total_dist):
            best_score = score
            best_phi = phi
            best_total_dist = total_dist

    return float(best_phi)


def filter_onsets_for_phase(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    annotations: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply annotation-based filtering and weighting for phase estimation.

    Onsets labeled ignore/string_noise are excluded.
    Onsets labeled true_attack/downbeat/beat3 get boosted weight.
    Unannotated onsets pass through unchanged.

    Returns (filtered_times, filtered_strengths) with matching lengths.
    """
    keep_mask = np.ones(len(onset_times), dtype=bool)
    weights = onset_strengths.copy()

    for id_str, ann in annotations.items():
        idx = int(id_str)
        if idx >= len(onset_times):
            continue
        label = ann.get("label", "")
        if label in _PHASE_EXCLUDE_LABELS:
            keep_mask[idx] = False
        elif label in _PHASE_BOOST_LABELS:
            weights[idx] *= _PHASE_BOOST_FACTOR

    return onset_times[keep_mask], weights[keep_mask]
