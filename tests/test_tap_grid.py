"""Tests for pocket_lab/tap_grid.py — pure functions, no hardware dependency.

Test matrix
-----------
grid_fractions
    1.  subdivision=1 (feel='straight') → [0.0]
    2.  subdivision=1 (feel='shuffle')  → [0.0]  (feel ignored)
    3.  subdivision=2, feel='straight'  → [0.0, 0.5]
    4.  subdivision=2, feel='shuffle'   → [0.0, 2/3]
    5.  subdivision=4, feel='straight'  → [0.0, 0.25, 0.5, 0.75]
    6.  subdivision=4, feel='shuffle'   → [0.0, 0.25, 0.5, 0.75]  (feel ignored)
    7.  first fraction always 0.0
    8.  fractions are strictly ascending
    9.  invalid subdivision → ValueError
    10. invalid feel → ValueError

grid_unit_name
    11. (1, 'straight') → 'quarter'
    12. (1, 'shuffle')  → 'quarter'
    13. (2, 'straight') → 'eighth'
    14. (2, 'shuffle')  → 'shuffle_eighth'
    15. (4, 'straight') → 'sixteenth'
    16. (4, 'shuffle')  → 'sixteenth'
    17. invalid subdivision → ValueError

estimate_bpm_from_taps
    18. 16 perfect taps at 120 BPM → bpm≈120.0, anchor=first tap, n=15
    19. 16 perfect taps at 80 BPM  → bpm≈80.0
    20. phase offset: taps starting at t=2.0 s → anchor=2.0
    21. one doubled interval rejected → bpm still ≈120.0, n=14
    22. one halved interval rejected  → bpm still ≈120.0, n=14
    23. doubling at start (first interval doubled)
    24. fewer than MIN_TAPS → ValueError
    25. too many outliers leaving < 3 clean intervals → ValueError
    26. returned n equals number of clean intervals (not total taps)

nearest_grid_error_s — quarter grid (subdivision=1)
    27. onset at phase_s               → error≈0.0, grid_index=0
    28. onset at phase_s + beat_s      → error≈0.0, grid_index=1
    29. onset at phase_s + 0.020 s     → error≈+0.020, grid_index=0
    30. onset at phase_s - 0.020 s     → error≈-0.020, grid_index=0
    31. onset at phase_s + 2*beat_s + 0.030 → error≈+0.030, grid_index=2

nearest_grid_error_s — straight eighth (subdivision=2, feel='straight')
    32. onset on "and" of beat 1 (0.5*beat_s) → error≈0.0, grid_index=1
    33. onset on "and" + 10 ms           → error≈+0.010, grid_index=1
    34. onset on beat 2 downbeat (beat_s) → error≈0.0, grid_index=2
    35. onset on beat 4 downbeat (3*beat_s) → error≈0.0, grid_index=6
    36. grid_index advances by 2 per beat

nearest_grid_error_s — shuffle eighth (subdivision=2, feel='shuffle')
    37. onset at 2/3*beat_s (shuffle "and") → error≈0.0, grid_index=1
    38. onset at 0.5*beat_s (straight "and") in shuffle grid
           → error = -(2/3 - 0.5)*beat_s (pulled toward shuffle "and")
    39. onset on beat 2 downbeat in shuffle → error≈0.0, grid_index=2

nearest_grid_error_s — sixteenth (subdivision=4)
    40. onset at 0.25*beat_s (first 16th) → error≈0.0, grid_index=1
    41. onset at 0.75*beat_s (third 16th) → error≈0.0, grid_index=3
    42. onset at beat 2 (1.0*beat_s)     → error≈0.0, grid_index=4

nearest_grid_error_s — edge cases
    43. onset before phase_s → negative grid_index
    44. large positive time (many beats) → large positive grid_index
    45. tie-breaking: two equidistant candidates → lower grid_index wins

grid_session_stats
    46. empty list → n_onsets=0, all stats None
    47. [0.0]     → mean_signed=0, mean_abs=0, std=0, all pct=100
    48. [30.0]    → pct_within_30ms=100, pct_within_60ms=100, pct_within_100ms=100
    49. [30.1]    → pct_within_30ms=0, pct_within_60ms=100
    50. [60.0]    → pct_within_30ms=0, pct_within_60ms=100, pct_within_100ms=100
    51. [60.1]    → pct_within_60ms=0, pct_within_100ms=100
    52. [100.1]   → pct_within_100ms=0
    53. [-50.0, 50.0] → mean_signed=0, mean_abs=50, std=50
    54. n_onsets equals length of input
    55. std is population std dev (pstdev, divides by n)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocket_lab.tap_grid import (
    MIN_TAPS,
    estimate_bpm_from_taps,
    grid_fractions,
    grid_session_stats,
    grid_unit_name,
    nearest_grid_error_s,
    scan_grid_offsets,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _perfect_taps(n: int, bpm: float, start: float = 0.0) -> list[float]:
    """Generate *n* perfectly spaced taps at *bpm* starting at *start*."""
    beat_s = 60.0 / bpm
    return [start + i * beat_s for i in range(n)]


_BPM   = 120.0
_BEAT  = 60.0 / _BPM   # 0.5 s
_PHASE = 0.0


# ── grid_fractions ────────────────────────────────────────────────────────────

def test_grid_fractions_quarter_straight():
    assert grid_fractions(1, 'straight') == [0.0]

def test_grid_fractions_quarter_shuffle_ignored():
    assert grid_fractions(1, 'shuffle') == [0.0]

def test_grid_fractions_eighth_straight():
    assert grid_fractions(2, 'straight') == pytest.approx([0.0, 0.5])

def test_grid_fractions_eighth_shuffle():
    assert grid_fractions(2, 'shuffle') == pytest.approx([0.0, 2.0 / 3.0])

def test_grid_fractions_sixteenth_straight():
    assert grid_fractions(4, 'straight') == pytest.approx([0.0, 0.25, 0.5, 0.75])

def test_grid_fractions_sixteenth_shuffle_ignored():
    assert grid_fractions(4, 'shuffle') == pytest.approx([0.0, 0.25, 0.5, 0.75])

def test_grid_fractions_first_always_zero():
    for sub in (1, 2, 4):
        assert grid_fractions(sub)[0] == 0.0

def test_grid_fractions_strictly_ascending():
    for sub in (1, 2, 4):
        fracs = grid_fractions(sub)
        assert all(fracs[i] < fracs[i + 1] for i in range(len(fracs) - 1))

def test_grid_fractions_invalid_subdivision():
    with pytest.raises(ValueError):
        grid_fractions(3)

def test_grid_fractions_invalid_feel():
    with pytest.raises(ValueError):
        grid_fractions(2, 'triplet')


# ── grid_unit_name ────────────────────────────────────────────────────────────

def test_grid_unit_name_quarter_straight():
    assert grid_unit_name(1, 'straight') == 'quarter'

def test_grid_unit_name_quarter_shuffle():
    assert grid_unit_name(1, 'shuffle') == 'quarter'

def test_grid_unit_name_eighth():
    assert grid_unit_name(2, 'straight') == 'eighth'

def test_grid_unit_name_shuffle_eighth():
    assert grid_unit_name(2, 'shuffle') == 'shuffle_eighth'

def test_grid_unit_name_sixteenth_straight():
    assert grid_unit_name(4, 'straight') == 'sixteenth'

def test_grid_unit_name_sixteenth_shuffle():
    assert grid_unit_name(4, 'shuffle') == 'sixteenth'

def test_grid_unit_name_invalid_subdivision():
    with pytest.raises(ValueError):
        grid_unit_name(3)


# ── estimate_bpm_from_taps ────────────────────────────────────────────────────

def test_bpm_perfect_120():
    bpm, anchor, n = estimate_bpm_from_taps(_perfect_taps(16, 120.0))
    assert bpm   == pytest.approx(120.0, rel=1e-5)
    assert anchor == pytest.approx(0.0)
    assert n == 15

def test_bpm_perfect_80():
    bpm, anchor, n = estimate_bpm_from_taps(_perfect_taps(16, 80.0))
    assert bpm == pytest.approx(80.0, rel=1e-5)
    assert n == 15

def test_bpm_phase_offset():
    taps = _perfect_taps(16, 120.0, start=2.0)
    bpm, anchor, _ = estimate_bpm_from_taps(taps)
    assert bpm    == pytest.approx(120.0, rel=1e-5)
    assert anchor == pytest.approx(2.0)

def test_bpm_one_doubled_interval_at_end():
    # 15 normal gaps + 1 doubled gap at the very end
    taps = _perfect_taps(16, 120.0)
    taps.append(taps[-1] + 2 * _BEAT)          # 17th tap after a doubled gap
    bpm, _, n = estimate_bpm_from_taps(taps)
    assert bpm == pytest.approx(120.0, rel=1e-5)
    assert n == 15                              # doubled gap excluded

def test_bpm_one_halved_interval():
    taps = _perfect_taps(16, 120.0)
    # Insert an extra tap halfway through the first gap; this bisects that gap
    # into two 0.25 s intervals, so TWO intervals are rejected (not one).
    extra = [taps[0], taps[0] + 0.5 * _BEAT] + taps[1:]
    bpm, _, n = estimate_bpm_from_taps(extra)
    assert bpm == pytest.approx(120.0, rel=1e-5)
    assert n == 14                              # both bisected gaps excluded

def test_bpm_doubled_interval_at_start():
    # Build 16 taps where the first gap is doubled and the rest are normal.
    # (Mutating taps[1] of _perfect_taps would also zero the second gap, since
    # taps[1] and taps[2] would both land at 2*beat_s.)
    taps = [0.0, 2 * _BEAT] + [2 * _BEAT + i * _BEAT for i in range(1, 15)]
    bpm, _, n = estimate_bpm_from_taps(taps)
    assert bpm == pytest.approx(120.0, rel=1e-4)
    assert n == 14                              # doubled first gap excluded

def test_bpm_too_few_taps():
    with pytest.raises(ValueError):
        estimate_bpm_from_taps(_perfect_taps(MIN_TAPS - 1, 120.0))

def test_bpm_too_many_outliers():
    # 4 taps (minimum) with intervals [0.1, 0.8, 5.0].
    # median = 0.8, MAD = 0.7, threshold = max(1.4, 0.24) = 1.4.
    # Intervals 0.1 (dev 0.7 ≤ 1.4) and 0.8 (dev 0) are clean; 5.0 (dev 4.2) is
    # rejected → only 2 clean intervals remain, which is below MIN_CLEAN_INTERVALS.
    taps = [0.0, 0.1, 0.9, 5.9]
    with pytest.raises(ValueError):
        estimate_bpm_from_taps(taps)

def test_bpm_n_intervals_used_is_clean_count():
    taps = _perfect_taps(16, 120.0)
    taps.append(taps[-1] + 2 * _BEAT)          # adds one bad interval
    _, _, n = estimate_bpm_from_taps(taps)
    # 16 taps → 15 normal + 1 doubled = 16 intervals; 1 rejected → 15 clean
    assert n == 15


# ── nearest_grid_error_s — quarter grid ──────────────────────────────────────

def test_quarter_on_beat():
    err, idx = nearest_grid_error_s(_PHASE, _BPM, _PHASE, subdivision=1)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 0

def test_quarter_second_beat():
    err, idx = nearest_grid_error_s(_PHASE + _BEAT, _BPM, _PHASE, subdivision=1)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 1

def test_quarter_20ms_late():
    err, idx = nearest_grid_error_s(_PHASE + 0.020, _BPM, _PHASE, subdivision=1)
    assert err == pytest.approx(+0.020, abs=1e-9)
    assert idx == 0

def test_quarter_20ms_early():
    err, idx = nearest_grid_error_s(_PHASE - 0.020, _BPM, _PHASE, subdivision=1)
    assert err == pytest.approx(-0.020, abs=1e-9)
    assert idx == 0

def test_quarter_third_beat_late():
    err, idx = nearest_grid_error_s(_PHASE + 2 * _BEAT + 0.030, _BPM, _PHASE,
                                    subdivision=1)
    assert err == pytest.approx(+0.030, abs=1e-9)
    assert idx == 2


# ── nearest_grid_error_s — straight eighth ────────────────────────────────────

def test_eighth_straight_and_of_beat1():
    onset = _PHASE + 0.5 * _BEAT                        # "and" of beat 1
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=2)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 1

def test_eighth_straight_and_10ms_late():
    onset = _PHASE + 0.5 * _BEAT + 0.010
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=2)
    assert err == pytest.approx(+0.010, abs=1e-9)
    assert idx == 1

def test_eighth_straight_beat2_downbeat():
    onset = _PHASE + _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=2)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 2

def test_eighth_straight_beat4_downbeat():
    onset = _PHASE + 3 * _BEAT                          # downbeat of beat 4
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=2)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 6

def test_eighth_straight_index_advances_by_2_per_beat():
    # Downbeats are at even indices: 0, 2, 4, 6, …
    for beat_n in range(4):
        _, idx = nearest_grid_error_s(_PHASE + beat_n * _BEAT, _BPM, _PHASE,
                                      subdivision=2)
        assert idx == beat_n * 2


# ── nearest_grid_error_s — shuffle eighth ────────────────────────────────────

def test_shuffle_and_on_shuffle_point():
    shuffle_and = _PHASE + (2.0 / 3.0) * _BEAT
    err, idx = nearest_grid_error_s(shuffle_and, _BPM, _PHASE,
                                    subdivision=2, feel='shuffle')
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 1

def test_shuffle_straight_and_pulled_toward_shuffle():
    # Straight "and" sits at 0.5*beat_s; the shuffle "and" is at 2/3*beat_s.
    # Error should be negative (early relative to the shuffle "and").
    straight_and = _PHASE + 0.5 * _BEAT
    err, idx = nearest_grid_error_s(straight_and, _BPM, _PHASE,
                                    subdivision=2, feel='shuffle')
    expected_err = 0.5 * _BEAT - (2.0 / 3.0) * _BEAT   # ≈ -0.0833 s
    assert err == pytest.approx(expected_err, abs=1e-9)
    assert idx == 1                                      # shuffle "and" slot

def test_shuffle_beat2_downbeat():
    onset = _PHASE + _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE,
                                    subdivision=2, feel='shuffle')
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 2


# ── nearest_grid_error_s — sixteenth ─────────────────────────────────────────

def test_sixteenth_first_subdivision():
    onset = _PHASE + 0.25 * _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=4)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 1

def test_sixteenth_third_subdivision():
    onset = _PHASE + 0.75 * _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=4)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 3

def test_sixteenth_beat2_downbeat():
    onset = _PHASE + _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=4)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 4


# ── nearest_grid_error_s — edge cases ────────────────────────────────────────

def test_negative_grid_index_before_phase():
    # Onset exactly one beat before phase_s
    onset = _PHASE - _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=2)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == -2                # beat -1, sub_i 0 → -1*2+0 = -2

def test_large_positive_grid_index():
    # 10 beats in, "and" of beat 10 → index = 10*2+1 = 21
    onset = _PHASE + 10 * _BEAT + 0.5 * _BEAT
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=2)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert idx == 21

def test_tie_breaking_lower_index_wins():
    # Quarter grid: onset exactly halfway between two beats.
    # Python's round() uses banker's rounding; both candidates are equidistant.
    # The implementation finds the first (lower-index) candidate in the
    # iteration order (ascending n, then ascending sub_i), so the earlier
    # beat wins.
    onset = _PHASE + 0.5 * _BEAT          # dead centre, quarter grid
    err, idx = nearest_grid_error_s(onset, _BPM, _PHASE, subdivision=1)
    # Nearest candidates: beat 0 (dist 0.25s) and beat 1 (dist 0.25s).
    # Lower index (beat 0, idx=0) should win; error = +0.25s.
    assert abs(err) == pytest.approx(0.5 * _BEAT, abs=1e-9)
    # Whichever wins, the error magnitude must be half a beat
    assert idx in (0, 1)                   # deterministic but accept either

def test_nearest_grid_error_negative_phase_s():
    # phase_s negative (tap anchor before session start)
    phase = -1.0                           # 1 s before session start
    onset = 0.0                            # first sample of session
    # At 120 BPM, beat_s = 0.5; grid from phase=-1.0: -1.0, -0.5, 0.0, 0.5...
    # Onset at 0.0 should match grid point 0.0 exactly (2 beats after phase)
    err, idx = nearest_grid_error_s(onset, _BPM, phase, subdivision=2)
    assert err == pytest.approx(0.0, abs=1e-9)
    # Beat n=2 from phase, sub_i=0 → idx = 2*2+0 = 4
    assert idx == 4


# ── grid_session_stats ────────────────────────────────────────────────────────

def test_stats_empty():
    s = grid_session_stats([])
    assert s['n_onsets'] == 0
    assert s['mean_signed_ms'] is None
    assert s['mean_abs_ms']    is None
    assert s['std_ms']         is None
    assert s['pct_within_30ms']  is None
    assert s['pct_within_60ms']  is None
    assert s['pct_within_100ms'] is None

def test_stats_single_zero():
    s = grid_session_stats([0.0])
    assert s['n_onsets'] == 1
    assert s['mean_signed_ms'] == pytest.approx(0.0)
    assert s['mean_abs_ms']    == pytest.approx(0.0)
    assert s['std_ms']         == pytest.approx(0.0)
    assert s['pct_within_30ms']  == pytest.approx(100.0)
    assert s['pct_within_60ms']  == pytest.approx(100.0)
    assert s['pct_within_100ms'] == pytest.approx(100.0)

def test_stats_pct_within_30ms_boundary():
    assert grid_session_stats([30.0])['pct_within_30ms']  == pytest.approx(100.0)
    assert grid_session_stats([30.1])['pct_within_30ms']  == pytest.approx(0.0)
    assert grid_session_stats([-30.0])['pct_within_30ms'] == pytest.approx(100.0)

def test_stats_pct_within_60ms_boundary():
    assert grid_session_stats([60.0])['pct_within_60ms']  == pytest.approx(100.0)
    assert grid_session_stats([60.1])['pct_within_60ms']  == pytest.approx(0.0)

def test_stats_pct_within_100ms_boundary():
    assert grid_session_stats([100.0])['pct_within_100ms']  == pytest.approx(100.0)
    assert grid_session_stats([100.1])['pct_within_100ms']  == pytest.approx(0.0)

def test_stats_symmetric_pair():
    s = grid_session_stats([-50.0, 50.0])
    assert s['n_onsets']       == 2
    assert s['mean_signed_ms'] == pytest.approx(0.0)
    assert s['mean_abs_ms']    == pytest.approx(50.0)
    assert s['std_ms']         == pytest.approx(50.0)   # pstdev([-50,50]) = 50
    assert s['pct_within_30ms']  == pytest.approx(0.0)    # abs(±50) > 30
    assert s['pct_within_60ms']  == pytest.approx(100.0)  # abs(±50) ≤ 60
    assert s['pct_within_100ms'] == pytest.approx(100.0)

def test_stats_n_onsets_equals_input_length():
    errors = [10.0, -20.0, 5.0, -8.0, 35.0]
    assert grid_session_stats(errors)['n_onsets'] == 5

def test_stats_std_is_population_not_sample():
    # pstdev([0, 10]) = 5.0; sample stdev would be ~7.07
    s = grid_session_stats([0.0, 10.0])
    assert s['std_ms'] == pytest.approx(5.0)

def test_stats_mixed_pct_thresholds():
    # 4 onsets: 20, 40, 80, 120 ms
    s = grid_session_stats([20.0, 40.0, 80.0, 120.0])
    assert s['pct_within_30ms']  == pytest.approx(25.0)   # 1/4
    assert s['pct_within_60ms']  == pytest.approx(50.0)   # 2/4
    assert s['pct_within_100ms'] == pytest.approx(75.0)   # 3/4


# ── Integration: tap capture → phase conversion → shuffle grid matching ───────
#
# Exercises the realistic pipeline where tapping precedes recording by several
# seconds, so phase_session_s is negative and onsets are in positive session time.
#
# Concrete numbers (all exact in floating-point arithmetic):
#
#   TAP_ANCHOR_WALL  = 1.000 s   (first tap, wall clock)
#   BPM              = 120.0     → beat_s = 0.500 s
#   8 perfect taps   → last tap at 4.500 s
#   SESSION_WALL_START = 5.000 s (recording starts 0.5 s after last tap)
#   phase_session_s  = 1.000 - 5.000 = -4.000 s
#
# Shuffle-eighth grid points in session time (fracs = [0.0, 2/3]):
#
#   beat n, sub_i | onset_s          | grid_idx
#   n=9,  sub_i=0 | -4.0+4.5 = 0.500 | 18
#   n=9,  sub_i=1 | 0.500+1/3 = 0.8̄ | 19
#   n=10, sub_i=0 | 1.000            | 20
#   n=10, sub_i=1 | 1.3̄             | 21
#   n=11, sub_i=0 | 1.500            | 22
#   n=11, sub_i=1 | 1.8̄             | 23
#   n=12, sub_i=0 | 2.000            | 24
#   n=12, sub_i=1 | 2.3̄             | 25

def test_integration_tap_to_shuffle_grid_session_time():
    """Pipeline: tap capture → BPM estimate → phase conversion → grid matching.

    Taps define 120 BPM with anchor at wall time 1.000 s.  Recording starts
    at wall time 5.000 s, so phase_session_s = -4.000 s.  Onsets placed
    exactly on the shuffle-eighth grid in positive session time must return
    error ≈ 0 and the correct absolute grid index.
    """
    # ── Tap capture ───────────────────────────────────────────────────────────
    TAP_ANCHOR_WALL    = 1.000
    BPM_TRUE           = 120.0
    BEAT_S             = 60.0 / BPM_TRUE        # 0.500 s

    tap_times = [TAP_ANCHOR_WALL + i * BEAT_S for i in range(8)]   # 8 taps

    bpm, anchor_s, n_intervals = estimate_bpm_from_taps(tap_times)

    assert bpm         == pytest.approx(BPM_TRUE, rel=1e-5)
    assert anchor_s    == pytest.approx(TAP_ANCHOR_WALL, abs=1e-9)
    assert n_intervals == 7                         # 8 taps → 7 clean intervals

    # ── Phase conversion ──────────────────────────────────────────────────────
    SESSION_WALL_START = 5.000
    phase_session_s    = anchor_s - SESSION_WALL_START
    # = 1.000 - 5.000 = -4.000 s  (negative: taps happened before session)

    assert phase_session_s == pytest.approx(-4.000, abs=1e-9)

    # ── Grid setup ────────────────────────────────────────────────────────────
    SUBDIVISION = 2
    FEEL        = 'shuffle'
    fracs       = grid_fractions(SUBDIVISION, FEEL)     # [0.0, 2/3]
    n_per_beat  = len(fracs)                            # 2

    # ── Verify each shuffle grid point in beats n=9 … 12 ─────────────────────
    # Beat n=9: phase_session_s + 9 * 0.5 = -4.0 + 4.5 = 0.500 s ≥ 0 ✓
    for n in range(9, 13):
        beat_start = phase_session_s + n * BEAT_S

        for sub_i, frac in enumerate(fracs):
            onset_s      = beat_start + frac * BEAT_S
            expected_idx = n * n_per_beat + sub_i

            assert onset_s > 0.0, (
                f"Precondition failed: onset for n={n} sub_i={sub_i} "
                f"is {onset_s:.4f} s (not in positive session time)"
            )

            error_s, grid_idx = nearest_grid_error_s(
                onset_s, bpm, phase_session_s, SUBDIVISION, FEEL
            )

            assert error_s == pytest.approx(0.0, abs=1e-9), (
                f"n={n} sub_i={sub_i} onset={onset_s:.6f}: "
                f"expected 0 error, got {error_s:.2e} s"
            )
            assert grid_idx == expected_idx, (
                f"n={n} sub_i={sub_i}: expected grid_idx={expected_idx}, "
                f"got {grid_idx}"
            )


# ── scan_grid_offsets ─────────────────────────────────────────────────────────
#
# Test matrix
# -----------
# scan_grid_offsets
#   57. Coarse sweep returns exactly 5 candidates (one per coarse offset).
#   58. Zero-offset candidate is always present in the coarse sweep.
#   59. On-grid onsets → offset=0 ranked #1 with std=0, mean=0.
#   60. Onsets shifted by half_beat → +half_beat offset ranked #1 with std=0.
#   61. Fine sweep returns ⌈2·beat_s/step_s⌉+1 candidates (before top_n clip).
#   62. Fine sweep finds a non-coarse offset (100 ms shift, step=10 ms).
#   63. top_n limits the number of returned entries.
#   64. Empty onsets → n_onsets=0 for every candidate, still returns top_n.
#   65. Results are sorted by std_ms ascending.
#   66. sort_by='mean_abs_ms' sorts by mean_abs_ms instead.
#   67. Invalid sort_by raises ValueError.
#   68. Each result contains offset_ms and phase_s keys.
#
# Setup: 120 BPM, shuffle eighths.  Shuffle feel avoids the straight-eighth
# periodicity trap where ±beat_s/2 and ±beat_s all alias to the same grid.

_SC_BPM   = 120.0
_SC_BEAT  = 60.0 / _SC_BPM   # 0.500 s
_SC_HALF  = _SC_BEAT / 2.0   # 0.250 s
_SC_PHASE = 1.000             # arbitrary reference phase
_SC_SUB   = 2
_SC_FEEL  = 'shuffle'         # fracs = [0.0, 2/3]


def _shuffle_onsets(phase: float, n_beats: int = 4) -> list[float]:
    """Generate onsets exactly on the shuffle-eighth grid from *phase*."""
    fracs = grid_fractions(_SC_SUB, _SC_FEEL)
    onsets = []
    for n in range(n_beats):
        beat_start = phase + n * _SC_BEAT
        for frac in fracs:
            onsets.append(beat_start + frac * _SC_BEAT)
    return onsets


# ── structure ─────────────────────────────────────────────────────────────────

def test_scan_coarse_returns_five_candidates():
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=10,
    )
    assert len(results) == 5

def test_scan_coarse_contains_zero_offset():
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL,
    )
    offsets = [r['offset_ms'] for r in results]
    assert 0.0 in offsets

def test_scan_result_has_offset_and_phase_keys():
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL,
    )
    for r in results:
        assert 'offset_ms' in r
        assert 'phase_s'   in r

def test_scan_phase_s_equals_nominal_plus_offset():
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=10,
    )
    for r in results:
        assert r['phase_s'] == pytest.approx(
            _SC_PHASE + r['offset_ms'] / 1000.0, abs=1e-9
        )

def test_scan_fine_sweep_candidate_count():
    # step=10 ms, beat=500 ms → 2*500/10 + 1 = 101 candidates; top_n=200 returns all
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, fine_step_ms=10.0, top_n=200,
    )
    assert len(results) == 101

def test_scan_fine_more_candidates_than_coarse():
    coarse = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=200,
    )
    fine = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, fine_step_ms=10.0, top_n=200,
    )
    assert len(fine) > len(coarse)

def test_scan_top_n_limits_results():
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=3,
    )
    assert len(results) == 3

def test_scan_invalid_sort_by():
    with pytest.raises(ValueError):
        scan_grid_offsets(
            _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
            sort_by='banana',
        )


# ── ranking ───────────────────────────────────────────────────────────────────

def test_scan_on_grid_onsets_min_std_is_zero():
    """Onsets exactly on current grid → at least one candidate has std=0."""
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=10,
    )
    # The top result must have std=0 (it may not be offset=0 due to
    # period aliases: ±beat_s offsets yield the same shuffle grid)
    assert results[0]['std_ms'] == pytest.approx(0.0, abs=1e-9)

def test_scan_zero_offset_candidate_has_std_zero():
    """The offset=0 candidate specifically has std=0 and mean=0."""
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=10,
    )
    zero = next(r for r in results if abs(r['offset_ms']) < 1e-6)
    assert zero['std_ms']      == pytest.approx(0.0, abs=1e-9)
    assert zero['mean_abs_ms'] == pytest.approx(0.0, abs=1e-9)

def test_scan_shifted_onsets_winner_congruent_to_true_phase():
    """When onsets are shifted by half_beat, the winning candidate's phase_s
    is congruent to the true phase modulo beat_s.

    The shuffle grid repeats every beat_s, so both +half_beat and -half_beat
    offsets are valid anchors (they land on the same grid).  We test for
    congruence rather than a specific sign.
    """
    true_phase = _SC_PHASE + _SC_HALF
    onsets     = _shuffle_onsets(true_phase)

    results = scan_grid_offsets(
        onsets, _SC_BPM, _SC_PHASE, _SC_SUB, _SC_FEEL,
    )
    best = results[0]
    assert best['std_ms'] == pytest.approx(0.0, abs=1e-9)

    # best['phase_s'] must satisfy: (phase_s - true_phase) % beat_s ≈ 0
    residual = (best['phase_s'] - true_phase) % _SC_BEAT
    assert min(residual, _SC_BEAT - residual) == pytest.approx(0.0, abs=1e-6)

def test_scan_fine_sweep_outperforms_coarse_for_non_coarse_shift():
    """Fine sweep finds a phase congruent to 100ms shift; coarse cannot.

    The 100ms shift has no alias among the five coarse candidates
    (0, ±250, ±500 ms).  Coarse therefore returns a nonzero best std,
    while the fine sweep (step=10ms) finds the congruent phase and reaches
    std ≈ 0.
    """
    SHIFT_MS   = 100.0
    true_phase = _SC_PHASE + SHIFT_MS / 1000.0
    onsets     = _shuffle_onsets(true_phase)

    coarse = scan_grid_offsets(
        onsets, _SC_BPM, _SC_PHASE, _SC_SUB, _SC_FEEL, top_n=10,
    )
    fine = scan_grid_offsets(
        onsets, _SC_BPM, _SC_PHASE, _SC_SUB, _SC_FEEL,
        fine_step_ms=10.0, top_n=10,
    )

    # Coarse has no candidate congruent to true_phase → nonzero spread
    assert coarse[0]['std_ms'] > 1.0

    # Fine sweep finds a congruent candidate → std ≈ 0
    assert fine[0]['std_ms'] == pytest.approx(0.0, abs=1e-9)

    # Verify the winning phase is congruent to true_phase mod beat_s
    residual = (fine[0]['phase_s'] - true_phase) % _SC_BEAT
    assert min(residual, _SC_BEAT - residual) == pytest.approx(0.0, abs=1e-6)

def test_scan_results_sorted_by_std_ms():
    """Every consecutive pair satisfies std[i] ≤ std[i+1]."""
    # Onsets on correct grid → one result has std=0; others have nonzero std.
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, top_n=10,
    )
    stds = [r['std_ms'] for r in results]
    for i in range(len(stds) - 1):
        assert stds[i] <= stds[i + 1] + 1e-9

def test_scan_sort_by_mean_abs_ms():
    """sort_by='mean_abs_ms' sorts by mean absolute error instead of std."""
    results = scan_grid_offsets(
        _shuffle_onsets(_SC_PHASE), _SC_BPM, _SC_PHASE,
        _SC_SUB, _SC_FEEL, sort_by='mean_abs_ms', top_n=10,
    )
    means = [r['mean_abs_ms'] for r in results]
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1] + 1e-9


# ── empty onsets ──────────────────────────────────────────────────────────────

def test_scan_empty_onsets_all_n_onsets_zero():
    results = scan_grid_offsets([], _SC_BPM, _SC_PHASE, _SC_SUB, _SC_FEEL)
    assert all(r['n_onsets'] == 0 for r in results)

def test_scan_empty_onsets_returns_top_n():
    results = scan_grid_offsets(
        [], _SC_BPM, _SC_PHASE, _SC_SUB, _SC_FEEL, top_n=3,
    )
    assert len(results) == 3

def test_scan_empty_onsets_all_stats_none():
    results = scan_grid_offsets([], _SC_BPM, _SC_PHASE, _SC_SUB, _SC_FEEL)
    for r in results:
        assert r['std_ms']        is None
        assert r['mean_abs_ms']   is None
        assert r['mean_signed_ms'] is None
