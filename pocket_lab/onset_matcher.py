"""Onset pairing algorithm for comparing two takes."""

from __future__ import annotations

from pocket_lab.match_record import MatchCategory, MatchRecord, OnsetRecord


def match_onsets(
    onsets_a: list[OnsetRecord],
    onsets_b: list[OnsetRecord],
    max_match_window_s: float = 0.050,
    noise_strength_threshold: float = 0.0,
    ambiguity_ratio: float = 2.0,
) -> list[MatchRecord]:
    """Pair onsets between two takes using nearest-neighbor with constraints.

    Parameters
    ----------
    onsets_a, onsets_b : sorted by time_s (aligned coordinates).
    max_match_window_s : maximum time difference to consider a match.
    noise_strength_threshold : onsets below this strength are NOISE.
    ambiguity_ratio : if second-best is within this factor of best distance,
                      classify as AMBIGUOUS.
    """
    results: list[MatchRecord] = []

    noise_a: set[int] = set()
    noise_b: set[int] = set()
    for i, o in enumerate(onsets_a):
        if o.strength < noise_strength_threshold:
            noise_a.add(i)
            results.append(MatchRecord(category=MatchCategory.NOISE, onset_a=o))
    for i, o in enumerate(onsets_b):
        if o.strength < noise_strength_threshold:
            noise_b.add(i)
            results.append(MatchRecord(category=MatchCategory.NOISE, onset_b=o))

    active_a = [o for i, o in enumerate(onsets_a) if i not in noise_a]
    active_b = [o for i, o in enumerate(onsets_b) if i not in noise_b]

    claims: dict[int, list[tuple[int, float]]] = {}

    for ai, oa in enumerate(active_a):
        candidates = []
        for bi, ob in enumerate(active_b):
            dt = abs(oa.time_s - ob.time_s)
            if dt <= max_match_window_s:
                candidates.append((bi, dt))

        if not candidates:
            results.append(MatchRecord(category=MatchCategory.A_ONLY, onset_a=oa))
            continue

        candidates.sort(key=lambda x: x[1])

        if len(candidates) >= 2:
            best_dist = candidates[0][1]
            second_dist = candidates[1][1]
            if best_dist == 0 or second_dist / max(best_dist, 1e-12) < ambiguity_ratio:
                results.append(MatchRecord(
                    category=MatchCategory.AMBIGUOUS,
                    onset_a=oa,
                    onset_b=active_b[candidates[0][0]],
                    candidates_b=[active_b[c[0]] for c in candidates],
                    confidence=0.5,
                ))
                continue

        best_bi, best_dist = candidates[0]
        if best_bi not in claims:
            claims[best_bi] = []
        claims[best_bi].append((ai, best_dist))

    claimed_a: set[int] = set()
    for bi, claimants in claims.items():
        if len(claimants) == 1:
            ai, dist = claimants[0]
            oa = active_a[ai]
            ob = active_b[bi]
            results.append(MatchRecord(
                category=MatchCategory.MATCHED,
                onset_a=oa,
                onset_b=ob,
                timing_diff_ms=(oa.time_s - ob.time_s) * 1000.0,
                amplitude_diff_db=oa.amplitude_db - ob.amplitude_db,
            ))
            claimed_a.add(ai)
        else:
            claimants.sort(key=lambda x: x[1])
            winner_ai, winner_dist = claimants[0]
            oa = active_a[winner_ai]
            ob = active_b[bi]
            results.append(MatchRecord(
                category=MatchCategory.MATCHED,
                onset_a=oa,
                onset_b=ob,
                timing_diff_ms=(oa.time_s - ob.time_s) * 1000.0,
                amplitude_diff_db=oa.amplitude_db - ob.amplitude_db,
            ))
            claimed_a.add(winner_ai)
            for loser_ai, _ in claimants[1:]:
                if loser_ai not in claimed_a:
                    results.append(MatchRecord(
                        category=MatchCategory.A_ONLY,
                        onset_a=active_a[loser_ai],
                    ))
                    claimed_a.add(loser_ai)

    claimed_b = set(claims.keys())
    ambiguous_b = set()
    for r in results:
        if r.category == MatchCategory.AMBIGUOUS:
            for c in r.candidates_b:
                for bi, ob in enumerate(active_b):
                    if ob is c:
                        ambiguous_b.add(bi)

    for bi, ob in enumerate(active_b):
        if bi not in claimed_b and bi not in noise_b and bi not in ambiguous_b:
            results.append(MatchRecord(category=MatchCategory.B_ONLY, onset_b=ob))

    results.sort(key=lambda r: r.time_s)
    return results


_SWEEP_THRESHOLDS = [0.0, 0.01, 0.02, 0.05, 0.10]


def threshold_sweep(
    onsets_a: list[OnsetRecord],
    onsets_b: list[OnsetRecord],
    max_match_window_s: float = 0.050,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Run match_onsets at multiple noise thresholds and return summary counts."""
    if thresholds is None:
        thresholds = _SWEEP_THRESHOLDS
    rows = []
    for thr in thresholds:
        matches = match_onsets(onsets_a, onsets_b, max_match_window_s, thr)
        counts = {cat: 0 for cat in MatchCategory}
        for m in matches:
            counts[m.category] += 1
        rows.append({
            "threshold": thr,
            "matched": counts[MatchCategory.MATCHED],
            "a_only": counts[MatchCategory.A_ONLY],
            "b_only": counts[MatchCategory.B_ONLY],
            "ambiguous": counts[MatchCategory.AMBIGUOUS],
            "noise": counts[MatchCategory.NOISE],
        })
    return rows
