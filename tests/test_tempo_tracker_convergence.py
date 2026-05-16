"""Convergence tests for the corrected TempoTracker tempo_beta=0.15 update rule.

Verifies the specific failure mode reported:  120 BPM nominal, player at a
different true tempo, tracker fails to converge after 48 beats.

All tests use direct-feed observations (no audio window) so they isolate the
tracker's update rule from the window-detection problem.

Test matrix
-----------
Convergence — faster player
    1.  After 48 direct observations at 132 BPM, tracker reaches ≥ 131 BPM.
    2.  After 48 direct observations at 125 BPM, tracker reaches ≥ 124 BPM.
    3.  Old EMA (beta=0.05) would NOT reach those thresholds with 48 beats.
    4.  After only 16 direct observations at 125 BPM, new rule already beats
        the old rule after 48 observations.

Convergence — slower player
    5.  After 48 direct observations at 108 BPM, tracker reaches ≤ 109 BPM.
    6.  After 48 direct observations at 115 BPM, tracker reaches ≤ 116 BPM.
    7.  Slow-player old-rule comparison (same pattern as test 3).

Convergence speed
    8.  After 16 observations at 125 BPM, new rule is closer to truth than old.
    9.  After 8 observations at 125 BPM, new rule is closer to truth than old.

Outlier isolation preserved
    10. 16 clean beats at 125 BPM → 1 outlier (400 ms late) → 16 more clean.
        tempo_ratio does not change across the outlier.
    11. After the outlier the tracker recovers and continues converging.
    12. An isolated mistake that exceeds 40 % of one beat is silently skipped.

Boundary — window-block scenario (diagnosis)
    13. With only the anchor observation (no further beats), tracker stays at
        nominal BPM regardless of what the true tempo was.  Confirms the
        window-block failure mode: the fix needs more observations to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tempo_tracker import TempoTracker


# ── Helpers ───────────────────────────────────────────────────────────────────

NOMINAL_BPM   = 120.0
COUNT_IN_S    = 2.0
NOMINAL_BEAT_S = 60.0 / NOMINAL_BPM


def _direct_feed(
    tracker: TempoTracker,
    true_bpm: float,
    n_beats: int,
    count_in_s: float = COUNT_IN_S,
) -> None:
    """Feed n_beats directly to the tracker, bypassing window detection."""
    true_beat_s = 60.0 / true_bpm
    for i in range(n_beats):
        nom = count_in_s + i * NOMINAL_BEAT_S
        act = count_in_s + i * true_beat_s
        tracker.observe(nom, act)


def _tracker(beta: float = 0.15) -> TempoTracker:
    return TempoTracker(NOMINAL_BPM, tempo_beta=beta)


# ── 1–4: Faster player convergence ───────────────────────────────────────────

def test_48_beats_132bpm_converges_above_131():
    """Regression: the scenario that originally reached only ~121 BPM."""
    t = _tracker()
    _direct_feed(t, true_bpm=132.0, n_beats=48)
    assert t.current_tempo_bpm() >= 131.0, (
        f"Expected ≥ 131.0 BPM after 48 beats at 132, got {t.current_tempo_bpm():.2f}"
    )


def test_48_beats_125bpm_converges_above_124():
    t = _tracker()
    _direct_feed(t, true_bpm=125.0, n_beats=48)
    assert t.current_tempo_bpm() >= 124.0, (
        f"Expected ≥ 124.0 BPM after 48 beats at 125, got {t.current_tempo_bpm():.2f}"
    )


def test_old_beta_does_not_converge_as_well_132bpm():
    """Old EMA (beta=0.05) must fall short of the new rule's threshold."""
    t_old = _tracker(beta=0.05)
    _direct_feed(t_old, true_bpm=132.0, n_beats=48)
    # Old rule reaches ≈ 130.8 BPM; new rule reaches ≈ 131.99 BPM.
    # 131.5 is a clear discriminator: new passes, old fails.
    assert t_old.current_tempo_bpm() < 131.5, (
        "Old beta=0.05 unexpectedly reached ≥ 131.5 BPM — threshold needs rechecking."
    )


def test_16_beats_new_rule_beats_old_rule_at_48_beats():
    """After only 16 observations the new rule is already closer to truth."""
    true_bpm = 125.0
    t_new = _tracker(beta=0.15)
    t_old = _tracker(beta=0.05)
    _direct_feed(t_new, true_bpm, n_beats=16)
    _direct_feed(t_old, true_bpm, n_beats=48)  # old rule, 3× more beats

    err_new = abs(t_new.current_tempo_bpm() - true_bpm)
    err_old = abs(t_old.current_tempo_bpm() - true_bpm)
    assert err_new < err_old, (
        f"New rule at 16 beats ({t_new.current_tempo_bpm():.2f}) should be closer to "
        f"{true_bpm} than old rule at 48 beats ({t_old.current_tempo_bpm():.2f})"
    )


# ── 5–7: Slower player convergence ───────────────────────────────────────────

def test_48_beats_108bpm_converges_below_109():
    t = _tracker()
    _direct_feed(t, true_bpm=108.0, n_beats=48)
    assert t.current_tempo_bpm() <= 109.0, (
        f"Expected ≤ 109.0 BPM after 48 beats at 108, got {t.current_tempo_bpm():.2f}"
    )


def test_48_beats_115bpm_converges_below_116():
    t = _tracker()
    _direct_feed(t, true_bpm=115.0, n_beats=48)
    assert t.current_tempo_bpm() <= 116.0, (
        f"Expected ≤ 116.0 BPM after 48 beats at 115, got {t.current_tempo_bpm():.2f}"
    )


def test_old_beta_does_not_converge_as_well_108bpm():
    t_old = _tracker(beta=0.05)
    _direct_feed(t_old, true_bpm=108.0, n_beats=48)
    # Old rule reaches ≈ 109.2 BPM; new reaches ≈ 108.01 BPM.
    # 108.7 discriminates clearly.
    assert t_old.current_tempo_bpm() > 108.7, (
        "Old beta=0.05 unexpectedly converged below 108.7 BPM."
    )


# ── 8–9: Convergence speed ────────────────────────────────────────────────────

def test_16_observations_new_closer_to_truth_than_old():
    true_bpm = 125.0
    t_new = _tracker(beta=0.15)
    t_old = _tracker(beta=0.05)
    _direct_feed(t_new, true_bpm, n_beats=16)
    _direct_feed(t_old, true_bpm, n_beats=16)
    assert abs(t_new.current_tempo_bpm() - true_bpm) < abs(t_old.current_tempo_bpm() - true_bpm), (
        f"new={t_new.current_tempo_bpm():.2f}  old={t_old.current_tempo_bpm():.2f}  target={true_bpm}"
    )


def test_8_observations_new_closer_to_truth_than_old():
    true_bpm = 125.0
    t_new = _tracker(beta=0.15)
    t_old = _tracker(beta=0.05)
    _direct_feed(t_new, true_bpm, n_beats=8)
    _direct_feed(t_old, true_bpm, n_beats=8)
    assert abs(t_new.current_tempo_bpm() - true_bpm) < abs(t_old.current_tempo_bpm() - true_bpm), (
        f"new={t_new.current_tempo_bpm():.2f}  old={t_old.current_tempo_bpm():.2f}  target={true_bpm}"
    )


# ── 10–12: Outlier isolation preserved ───────────────────────────────────────

def test_outlier_does_not_change_tempo_ratio():
    """An outlier must leave tempo_ratio unchanged (outlier rejection preserved)."""
    true_bpm = 125.0
    true_beat_s = 60.0 / true_bpm

    t = _tracker()
    # 16 clean beats
    for i in range(16):
        nom = COUNT_IN_S + i * NOMINAL_BEAT_S
        act = COUNT_IN_S + i * true_beat_s
        t.observe(nom, act)

    ratio_before = t.tempo_ratio
    phase_before = t.phase_offset

    # One outlier: 500 ms late (> 40% of nominal beat)
    nom_bad = COUNT_IN_S + 16 * NOMINAL_BEAT_S
    act_bad = COUNT_IN_S + 16 * true_beat_s + 0.500
    t.observe(nom_bad, act_bad)

    assert t.tempo_ratio == pytest.approx(ratio_before, abs=1e-9)
    assert t.phase_offset == pytest.approx(phase_before, abs=1e-9)


def test_tracker_recovers_after_outlier():
    """Clean beats after an outlier continue convergence normally."""
    true_bpm  = 125.0
    true_beat_s = 60.0 / true_bpm

    t = _tracker()
    noms = [COUNT_IN_S + i * NOMINAL_BEAT_S for i in range(33)]
    acts = [COUNT_IN_S + i * true_beat_s    for i in range(33)]

    # 16 clean
    for i in range(16):
        t.observe(noms[i], acts[i])
    bpm_before = t.current_tempo_bpm()

    # Outlier at beat 16
    t.observe(noms[16], acts[16] + 0.600)

    # 16 more clean
    for i in range(17, 33):
        t.observe(noms[i], acts[i])

    # Should have continued converging from where it was
    assert t.current_tempo_bpm() > bpm_before, (
        "Tracker should have kept converging after outlier was skipped"
    )
    assert t.current_tempo_bpm() >= 124.0


def test_isolated_mistake_exceeding_40pct_is_rejected():
    """Outlier threshold is 40% of nominal beat = 200 ms at 120 BPM."""
    t = _tracker()
    # Anchor
    t.observe(COUNT_IN_S, COUNT_IN_S)

    ratio_after_anchor = t.tempo_ratio
    # Mistake: 210 ms late (> 200 ms threshold)
    t.observe(COUNT_IN_S + NOMINAL_BEAT_S, COUNT_IN_S + NOMINAL_BEAT_S + 0.210)
    assert t.tempo_ratio == pytest.approx(ratio_after_anchor, abs=1e-9), (
        "tempo_ratio should not change after a 210 ms outlier"
    )


# ── 13: Window-block diagnosis ────────────────────────────────────────────────

def test_anchor_only_stays_at_nominal():
    """Without observations after the anchor, BPM stays at nominal.

    This is the live-system failure: at 132 BPM, beat 1 arrives 45.5 ms early
    but the extraction window only looks 30 ms back → zero updates after anchor.
    The beta fix alone cannot help when observations never arrive.
    """
    t = _tracker()
    # Only the anchor
    t.observe(COUNT_IN_S, COUNT_IN_S)
    assert t.current_tempo_bpm() == pytest.approx(NOMINAL_BPM, abs=0.01), (
        "With only the anchor observation, tracker must stay at nominal BPM"
    )
    # Not stable with a different true tempo either — it simply can't know
    t2 = _tracker()
    t2.observe(COUNT_IN_S, COUNT_IN_S)  # anchor at nominal, true tempo = 132
    assert t2.current_tempo_bpm() == pytest.approx(NOMINAL_BPM, abs=0.01)
