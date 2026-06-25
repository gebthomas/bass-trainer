"""BPM estimation from tap times, beat-grid projection, and timing-error computation.

Pure Python — standard library only (``math``, ``statistics``).
No audio hardware, no NumPy.

Grid model
----------
The beat grid is parameterised by:

  bpm        : tempo in beats per minute
  phase_s    : a reference beat time in the same epoch as onset times
  subdivision: 1, 2, or 4  (quarter / eighth / sixteenth)
  feel       : 'straight' or 'shuffle'  (only differs at subdivision=2)

Within each beat starting at ``beat_start``, grid points are placed at::

    candidate = beat_start + frac * beat_s

where ``frac`` comes from ``grid_fractions(subdivision, feel)``::

    subdivision=1  (quarter, feel ignored)      : [0.0]
    subdivision=2, feel='straight' (eighth)     : [0.0, 0.5]
    subdivision=2, feel='shuffle'               : [0.0, 2/3]
    subdivision=4  (sixteenth, feel ignored)    : [0.0, 0.25, 0.5, 0.75]

The grid extends indefinitely in both directions from ``phase_s``.

Typical usage
-------------
    # --- tap capture (wall-clock times) ---
    bpm, anchor_wall_s, n_intervals = estimate_bpm_from_taps(tap_wall_times)

    # --- session start ---
    session_wall_start = time.perf_counter()
    phase_s = anchor_wall_s - session_wall_start   # convert to session time

    # --- per onset (session time, after latency compensation) ---
    error_s, grid_idx = nearest_grid_error_s(
        onset_s, bpm, phase_s, subdivision=2, feel='straight'
    )

    # --- post session ---
    stats = grid_session_stats([e * 1000 for e in error_s_list])
"""

from __future__ import annotations

import math
import statistics

# ── Public constants ──────────────────────────────────────────────────────────

VALID_SUBDIVISIONS: frozenset[int] = frozenset({1, 2, 4})
VALID_FEELS: frozenset[str] = frozenset({'straight', 'shuffle'})

# Minimum taps required by estimate_bpm_from_taps
MIN_TAPS: int = 4

# Outlier rejection: threshold = max(MAD_FACTOR * MAD, MEDIAN_FLOOR * median)
# The median-relative floor catches the doubled/halved-interval case even
# when all remaining intervals are perfectly uniform (MAD == 0).
_OUTLIER_MAD_FACTOR: float = 2.0
_OUTLIER_MEDIAN_FLOOR: float = 0.30
_MIN_CLEAN_INTERVALS: int = 3


# ── Grid geometry ─────────────────────────────────────────────────────────────

def grid_fractions(subdivision: int, feel: str = 'straight') -> list[float]:
    """Return the beat fractions for each grid point within one beat.

    Parameters
    ----------
    subdivision : int
        1 = quarter note, 2 = eighth note, 4 = sixteenth note.
    feel : str
        ``'straight'`` or ``'shuffle'``.  Only affects ``subdivision=2``;
        for 1 and 4 the feel is ignored and straight fractions are returned.

    Returns
    -------
    list[float]
        Beat fractions in ``[0.0, 1.0)`` in ascending order.

    Raises
    ------
    ValueError
        If *subdivision* or *feel* are not recognised values.
    """
    if subdivision not in VALID_SUBDIVISIONS:
        raise ValueError(
            f"subdivision must be one of {sorted(VALID_SUBDIVISIONS)}, got {subdivision!r}"
        )
    if feel not in VALID_FEELS:
        raise ValueError(
            f"feel must be one of {sorted(VALID_FEELS)}, got {feel!r}"
        )

    if subdivision == 1:
        return [0.0]
    if subdivision == 4:
        return [0.0, 0.25, 0.5, 0.75]
    # subdivision == 2
    if feel == 'shuffle':
        return [0.0, 2.0 / 3.0]
    return [0.0, 0.5]


def grid_unit_name(subdivision: int, feel: str = 'straight') -> str:
    """Return a human-readable name for this grid configuration.

    Returns
    -------
    str
        One of ``'quarter'``, ``'eighth'``, ``'shuffle_eighth'``,
        ``'sixteenth'``.

    Raises
    ------
    ValueError
        If *subdivision* is not recognised.
    """
    if subdivision not in VALID_SUBDIVISIONS:
        raise ValueError(
            f"subdivision must be one of {sorted(VALID_SUBDIVISIONS)}, got {subdivision!r}"
        )
    if subdivision == 1:
        return 'quarter'
    if subdivision == 4:
        return 'sixteenth'
    # subdivision == 2
    return 'shuffle_eighth' if feel == 'shuffle' else 'eighth'


# ── Tap BPM estimation ────────────────────────────────────────────────────────

def estimate_bpm_from_taps(tap_times: list[float]) -> tuple[float, float, int]:
    """Estimate BPM and phase anchor from a sequence of tap timestamps.

    Parameters
    ----------
    tap_times : list[float]
        Monotonically increasing timestamps in seconds (e.g. from
        ``time.perf_counter()``).  Must contain at least ``MIN_TAPS`` values.

    Returns
    -------
    tuple[float, float, int]
        ``(bpm, anchor_s, n_intervals_used)`` where

        * ``bpm``               — estimated tempo in beats per minute.
        * ``anchor_s``          — first tap time; a reference beat on the grid.
          Callers convert this to session time by subtracting
          ``session_wall_start`` before passing to ``nearest_grid_error_s``.
        * ``n_intervals_used``  — number of inter-tap intervals that survived
          outlier rejection and contributed to the BPM estimate.

    Raises
    ------
    ValueError
        If fewer than ``MIN_TAPS`` tap times are provided, or if outlier
        rejection leaves fewer than ``_MIN_CLEAN_INTERVALS`` clean intervals.

    Algorithm
    ---------
    1. Compute consecutive inter-tap intervals.
    2. Find outliers using median absolute deviation (MAD).  The rejection
       threshold is ``max(2×MAD, 0.30×median)``, which safely catches
       doubled/halved intervals even when all remaining intervals are
       perfectly uniform (MAD == 0).
    3. Beat period = mean of clean intervals.
    """
    if len(tap_times) < MIN_TAPS:
        raise ValueError(
            f"At least {MIN_TAPS} tap times required, got {len(tap_times)}"
        )

    intervals = [tap_times[i + 1] - tap_times[i] for i in range(len(tap_times) - 1)]

    median_iv = statistics.median(intervals)
    deviations = [abs(iv - median_iv) for iv in intervals]
    mad = statistics.median(deviations)

    threshold = max(_OUTLIER_MAD_FACTOR * mad, _OUTLIER_MEDIAN_FLOOR * median_iv)

    clean_intervals = [
        iv for iv, dev in zip(intervals, deviations) if dev <= threshold
    ]

    if len(clean_intervals) < _MIN_CLEAN_INTERVALS:
        raise ValueError(
            f"Too many outlier intervals: only {len(clean_intervals)} of "
            f"{len(intervals)} survived rejection (need ≥ {_MIN_CLEAN_INTERVALS})"
        )

    beat_s = statistics.mean(clean_intervals)
    bpm = 60.0 / beat_s
    anchor_s = tap_times[0]

    return bpm, anchor_s, len(clean_intervals)


# ── Grid matching ─────────────────────────────────────────────────────────────

def nearest_grid_error_s(
    onset_time_s: float,
    bpm: float,
    phase_s: float,
    subdivision: int = 2,
    feel: str = 'straight',
) -> tuple[float, int]:
    """Return the timing error and grid index of the nearest grid point.

    The beat grid starts at ``phase_s`` and extends indefinitely in both
    directions.  Within each beat, grid points are placed at
    ``beat_start + frac * beat_s`` for each fraction in
    ``grid_fractions(subdivision, feel)``.

    Parameters
    ----------
    onset_time_s : float
        Onset time in seconds (any consistent epoch; must match ``phase_s``).
    bpm : float
        Tempo in beats per minute.
    phase_s : float
        A reference beat time in the same epoch as ``onset_time_s``.
        Typically negative when the tap phase preceded session recording.
    subdivision : int
        1, 2, or 4.  Default 2 (eighth-note grid).
    feel : str
        ``'straight'`` or ``'shuffle'``.  Default ``'straight'``.

    Returns
    -------
    tuple[float, int]
        ``(error_s, grid_index)`` where

        * ``error_s`` = ``onset_time_s − nearest_grid_time``
          (negative = played early, positive = played late).
        * ``grid_index`` is the absolute grid-slot number counting from the
          slot at ``phase_s`` (slot 0).  Can be negative for onsets before
          ``phase_s``.

    Notes
    -----
    When two grid points are equidistant from the onset, the earlier
    (lower-index) slot wins, giving deterministic tie-breaking.
    """
    beat_s = 60.0 / bpm
    fractions = grid_fractions(subdivision, feel)
    n_per_beat = len(fractions)

    # Approximate beat number; ±1 search range guarantees we find the winner
    # for any subdivision and any feel (worst-case nearest candidate is
    # always within one beat of the rounded estimate).
    n_approx = round((onset_time_s - phase_s) / beat_s)

    best_error = math.inf
    best_grid_index = 0

    for n in range(n_approx - 1, n_approx + 2):
        beat_start = phase_s + n * beat_s
        for sub_i, frac in enumerate(fractions):
            candidate = beat_start + frac * beat_s
            error = onset_time_s - candidate
            if abs(error) < abs(best_error):
                best_error = error
                best_grid_index = n * n_per_beat + sub_i

    return best_error, best_grid_index


# ── Session statistics ────────────────────────────────────────────────────────

def grid_session_stats(timing_errors_ms: list[float]) -> dict:
    """Compute aggregate timing statistics for a grid practice session.

    Parameters
    ----------
    timing_errors_ms : list[float]
        Timing errors in milliseconds (negative = early, positive = late).

    Returns
    -------
    dict
        Keys:

        * ``n_onsets``         — number of detected onsets (int)
        * ``mean_signed_ms``   — mean signed error in ms (float | None)
        * ``mean_abs_ms``      — mean absolute error in ms (float | None)
        * ``std_ms``           — population standard deviation in ms (float | None)
        * ``pct_within_30ms``  — percent of onsets within ±30 ms (float | None)
        * ``pct_within_60ms``  — percent of onsets within ±60 ms (float | None)
        * ``pct_within_100ms`` — percent of onsets within ±100 ms (float | None)

        All float fields are ``None`` when *timing_errors_ms* is empty.
    """
    n = len(timing_errors_ms)
    if n == 0:
        return {
            'n_onsets': 0,
            'mean_signed_ms': None,
            'mean_abs_ms': None,
            'std_ms': None,
            'pct_within_30ms': None,
            'pct_within_60ms': None,
            'pct_within_100ms': None,
        }

    mean_signed = statistics.mean(timing_errors_ms)
    mean_abs = statistics.mean([abs(e) for e in timing_errors_ms])
    std = statistics.pstdev(timing_errors_ms)

    def _pct(limit: float) -> float:
        return 100.0 * sum(1 for e in timing_errors_ms if abs(e) <= limit) / n

    return {
        'n_onsets': n,
        'mean_signed_ms': mean_signed,
        'mean_abs_ms': mean_abs,
        'std_ms': std,
        'pct_within_30ms': _pct(30.0),
        'pct_within_60ms': _pct(60.0),
        'pct_within_100ms': _pct(100.0),
    }


# ── Grid offset scan ──────────────────────────────────────────────────────────

def scan_grid_offsets(
    onset_times_s: list[float],
    bpm: float,
    phase_s: float,
    subdivision: int = 2,
    feel: str = 'straight',
    *,
    fine_step_ms: float | None = None,
    top_n: int = 5,
    sort_by: str = 'std_ms',
) -> list[dict]:
    """Evaluate onset timing errors under multiple candidate phase offsets.

    Shifts the beat grid by a range of candidate offsets, recomputes all
    timing errors for each candidate, and returns the best *top_n* sorted
    by the chosen metric.

    Candidate offsets
    -----------------
    When *fine_step_ms* is ``None`` (default), five coarse offsets are tested:
    ``−beat_s``, ``−beat_s/2``, ``0``, ``+beat_s/2``, ``+beat_s``.

    When *fine_step_ms* is a positive number, the coarse candidates are
    replaced by a fine sweep from ``−beat_s`` to ``+beat_s`` inclusive in
    steps of *fine_step_ms* milliseconds.

    Parameters
    ----------
    onset_times_s : list[float]
        Latency-compensated onset times in session seconds.
    bpm : float
        Tempo in beats per minute.
    phase_s : float
        Current anchor phase in session seconds (from ``estimate_bpm_from_taps``).
    subdivision : int
        1, 2, or 4.  Default 2.
    feel : str
        ``'straight'`` or ``'shuffle'``.  Default ``'straight'``.
    fine_step_ms : float | None
        Step size in milliseconds for the fine sweep.  ``None`` → coarse only.
    top_n : int
        Maximum number of candidates to return.  Default 5.
    sort_by : str
        Primary sort key: ``'std_ms'`` (default) or ``'mean_abs_ms'``.
        The other metric is always used as the tiebreaker.

    Returns
    -------
    list[dict]
        Each dict contains:

        * ``offset_ms``     — candidate offset in milliseconds (positive = later anchor)
        * ``phase_s``       — shifted phase = *phase_s* + *offset_ms* / 1000
        * all keys from :func:`grid_session_stats`

        Sorted by *sort_by* ascending (tiebreaker: the other metric), then
        trimmed to *top_n* entries.

    Raises
    ------
    ValueError
        If *sort_by* is not ``'std_ms'`` or ``'mean_abs_ms'``.
    """
    if sort_by not in ('std_ms', 'mean_abs_ms'):
        raise ValueError(
            f"sort_by must be 'std_ms' or 'mean_abs_ms', got {sort_by!r}"
        )

    beat_s = 60.0 / bpm

    if fine_step_ms is not None and fine_step_ms > 0:
        step_s  = fine_step_ms / 1000.0
        n_steps = round(2.0 * beat_s / step_s)
        offsets_s = [-beat_s + i * step_s for i in range(n_steps + 1)]
    else:
        half      = beat_s / 2.0
        offsets_s = [-beat_s, -half, 0.0, half, beat_s]

    results: list[dict] = []
    for offset_s in offsets_s:
        phase_candidate = phase_s + offset_s
        errors_ms = [
            nearest_grid_error_s(t, bpm, phase_candidate, subdivision, feel)[0] * 1000.0
            for t in onset_times_s
        ]
        stats = grid_session_stats(errors_ms)
        stats['offset_ms'] = round(offset_s * 1000.0, 6)
        stats['phase_s']   = phase_candidate
        results.append(stats)

    alt = 'mean_abs_ms' if sort_by == 'std_ms' else 'std_ms'

    def _key(r: dict) -> tuple[float, float]:
        return (
            r[sort_by] if r[sort_by] is not None else math.inf,
            r[alt]     if r[alt]     is not None else math.inf,
        )

    results.sort(key=_key)
    return results[:top_n]
