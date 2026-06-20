"""Beat grid generation and onset classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GridLine:
    """A single line in the beat grid (measure, beat, or subdivision)."""
    time: float
    kind: str       # "measure", "beat", or "subdivision"
    measure: int    # 1-based
    beat: int       # 1-based within measure
    subdivision: int  # 0 = on-beat, 1 = shuffle subdivision


def make_grid(
    bpm: float,
    beats_per_measure: int,
    n_measures: int,
    shuffle_fraction: float = 2 / 3,
    offset: float = 0.0,
) -> List[GridLine]:
    """Build a beat grid with measure/beat/subdivision lines.

    Parameters
    ----------
    bpm               : tempo in beats per minute.
    beats_per_measure  : beats in each measure (e.g. 4 for 4/4, 3 for 3/4).
    n_measures         : number of measures to generate.
    shuffle_fraction   : position of the shuffle subdivision within each beat
                         (0.667 = triplet swing). Set to 0.5 for straight.
    offset             : time offset in seconds for the first beat.

    Returns a list of GridLine sorted by time.
    """
    beat_s = 60.0 / bpm
    lines: List[GridLine] = []

    for m in range(n_measures):
        for b in range(beats_per_measure):
            beat_index = m * beats_per_measure + b
            t = offset + beat_index * beat_s
            kind = "measure" if b == 0 else "beat"
            lines.append(GridLine(
                time=t, kind=kind,
                measure=m + 1, beat=b + 1, subdivision=0,
            ))
            sub_t = t + shuffle_fraction * beat_s
            lines.append(GridLine(
                time=sub_t, kind="subdivision",
                measure=m + 1, beat=b + 1, subdivision=1,
            ))

    lines.sort(key=lambda g: g.time)
    return lines


@dataclass
class OnsetClassification:
    """Result of classifying a detected onset against the grid."""
    onset_time: float
    nearest_measure: int
    nearest_beat: int
    offset_ms: float
    beat_fraction: float
    label: str


def classify_onset_against_grid(
    onset_time: float,
    bpm: float,
    beats_per_measure: int,
    shuffle_fraction: float = 2 / 3,
    offset: float = 0.0,
) -> OnsetClassification:
    """Classify a single onset time relative to the beat grid.

    Returns an OnsetClassification with nearest measure/beat, signed offset
    in ms, beat fraction, and a preliminary label.

    ``beat_fraction`` is the position within the beat interval (0 = on-beat,
    1 = next beat), computed from the beat the onset falls after (floor).
    ``offset_ms`` and ``nearest_measure/beat`` are relative to the nearest
    beat (round), so a negative offset_ms means early.
    """
    beat_s = 60.0 / bpm
    rel = onset_time - offset
    beat_index_exact = rel / beat_s

    beat_index_floor = max(0, int(np.floor(beat_index_exact)))
    beat_fraction = beat_index_exact - beat_index_floor

    beat_index_nearest = max(0, int(round(beat_index_exact)))
    measure = beat_index_nearest // beats_per_measure + 1
    beat_in_measure = beat_index_nearest % beats_per_measure + 1

    nearest_beat_time = offset + beat_index_nearest * beat_s
    offset_ms = (onset_time - nearest_beat_time) * 1000.0

    on_beat_dist = min(beat_fraction, 1.0 - beat_fraction)
    shuffle_dist = abs(beat_fraction - shuffle_fraction)
    threshold = 0.08

    if on_beat_dist < threshold:
        label = "on-beat"
    elif shuffle_dist < threshold:
        label = "shuffle"
    elif beat_fraction > 0.5 and on_beat_dist < shuffle_dist:
        label = "early"
    elif beat_fraction < 0.5 and on_beat_dist < shuffle_dist:
        label = "late"
    else:
        label = "between"

    return OnsetClassification(
        onset_time=onset_time,
        nearest_measure=measure,
        nearest_beat=beat_in_measure,
        offset_ms=offset_ms,
        beat_fraction=beat_fraction,
        label=label,
    )
