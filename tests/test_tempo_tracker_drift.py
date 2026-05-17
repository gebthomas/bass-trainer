"""Tests for the TempoTracker drift-detection policy.

The drift-detection feature widens the outlier acceptance window when the last
``drift_window`` accepted observations are all in the same direction AND their
smallest absolute error exceeds ``drift_min_frac × nominal_beat_s``.

This allows gradual acceleration / deceleration to continue training the
tracker without being incorrectly rejected as outliers, while preserving
strict outlier rejection after stable playing.

Test matrix
-----------
effective_outlier_limit() unit tests
    1.  No accepted errors → returns base limit.
    2.  Fewer than drift_window entries → returns base limit.
    3.  Exactly drift_window entries, all same sign, all significant → widened.
    4.  Same sign but NOT significant (below floor) → base limit.
    5.  Significant but mixed signs → base limit.
    6.  Scale is applied correctly (limit = base × drift_threshold_scale).
    7.  drift_window=0 disables drift detection entirely.

Outlier isolation preserved
    8.  Isolated +350 ms outlier after stable playing is rejected (errors are
        tiny, below the significance floor → no widening).
    9.  Isolated outlier after 4 same-sign tiny errors is rejected (errors
        below floor even though same-sign).
    10. After a correctly rejected outlier, tracker BPM unchanged.

Drift detection activates correctly
    11. After drift_window beats of large same-sign errors, effective limit
        is widened.
    12. When direction reverses (sign flip in one entry), limit returns to base.
    13. Drift detection is symmetric: works for both consistent early and
        consistent late playing.

Gradual acceleration/deceleration via direct-feed
    14. Direct-feed 120→132 over 48 beats: 0 rejections (was 11 before fix).
    15. Direct-feed 120→132 over 48 beats: final BPM ≥ 130.
    16. Direct-feed 132→120 over 48 beats: 0 rejections.
    17. Direct-feed 132→120 over 48 beats: final BPM ≤ 122.

Stable tempo / small drift unaffected
    18. Exact 120 BPM: all accepted, final BPM stays at 120.
    19. 121 BPM (small drift, errors stay < floor): all accepted, no change
        in rejection behaviour.

Replay-tool integration
    20. accel scenario via run_replay: 0 rejections.
    21. accel scenario via run_replay: final BPM ≥ 130.
    22. decel scenario via run_replay: 0 rejections.
    23. decel scenario via run_replay: final BPM ≤ 122.
    24. outlier_late after stable playing: still "reject" in run_replay output.
    25. Stable tempo (exact): no rejections in run_replay output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tempo_tracker import TempoTracker
from tools.replay_tempo_tracking import (
    run_replay,
    scenario_exact_nominal,
    scenario_gradual_change,
    scenario_steady_offset,
    scenario_with_outlier,
)


# ── Shared constants ──────────────────────────────────────────────────────────

NOM_BPM   = 120.0
BEAT_S    = 60.0 / NOM_BPM        # 0.5 s
COUNT_IN  = 2.0                   # seconds
BASE_LIMIT = 0.40 * BEAT_S        # 0.20 s = 200 ms


def _tracker(**kw) -> TempoTracker:
    return TempoTracker(NOM_BPM, **kw)


def _feed_errors(tracker: TempoTracker, errors_s: list[float]) -> None:
    """Feed beats at nominal timing; anchor first, then beats displaced by errors.

    Because observe() updates the tracker on each step, the actual values stored
    in _accepted_errors will differ from errors_s after the first observation.
    Use _force_accepted_errors when you need exact control of _accepted_errors.
    """
    tracker.observe(COUNT_IN, COUNT_IN)          # anchor
    for i, err in enumerate(errors_s, start=1):
        nom = COUNT_IN + i * BEAT_S
        act = nom + err
        tracker.observe(nom, act)


def _force_accepted_errors(tracker: TempoTracker, errors_s: list[float]) -> None:
    """Bypass observe() and write exactly these values into _accepted_errors.

    Used exclusively for unit-testing effective_outlier_limit() in isolation,
    where we need precise control over the drift-detection window contents.
    """
    tracker._accepted_errors.clear()
    for e in errors_s[-tracker._drift_window:]:
        tracker._accepted_errors.append(e)


# ── 1–7: effective_outlier_limit() unit tests ─────────────────────────────────

def test_no_errors_returns_base_limit():
    t = _tracker()
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT, abs=1e-9)


def test_fewer_than_window_returns_base():
    t = _tracker(drift_window=4)
    t.observe(COUNT_IN, COUNT_IN)                # anchor
    t.observe(COUNT_IN + BEAT_S, COUNT_IN + BEAT_S + 0.08)   # 1 accepted error
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT, abs=1e-9)


def test_full_window_same_sign_significant_widens():
    """drift_window=4 entries, all positive, all > floor → limit widens."""
    t = _tracker(drift_window=4, drift_threshold_scale=2.0, drift_min_frac=0.10)
    # Directly inject 4 large positive errors (all well above the 50 ms floor).
    _force_accepted_errors(t, [0.060, 0.070, 0.080, 0.090])
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT * 2.0, abs=1e-9)


def test_same_sign_but_below_floor_no_widening():
    """All errors same sign but all below the magnitude floor → no widening."""
    t = _tracker(drift_window=4, drift_min_frac=0.10)
    # Errors of 5 ms each — well below 50 ms floor
    _feed_errors(t, [0.005, 0.005, 0.005, 0.005])
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT, abs=1e-9)


def test_significant_but_mixed_signs_no_widening():
    """Large errors that alternate sign → no widening."""
    t = _tracker(drift_window=4, drift_min_frac=0.10)
    # Errors alternate: +80 ms, -80 ms, +80 ms, -80 ms
    _feed_errors(t, [0.08, -0.08, 0.08, -0.08])
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT, abs=1e-9)


def test_drift_scale_applied_correctly():
    """drift_threshold_scale=3.0 → effective limit = 3× base."""
    t = _tracker(drift_window=4, drift_threshold_scale=3.0, drift_min_frac=0.10)
    _force_accepted_errors(t, [0.08, 0.09, 0.10, 0.11])
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT * 3.0, abs=1e-9)


def test_drift_window_zero_disables():
    """drift_window=0 → always return base limit."""
    t = _tracker(drift_window=0)
    t.observe(COUNT_IN, COUNT_IN)
    # Even if we feed large same-sign errors directly, should have no effect
    for i in range(10):
        nom = COUNT_IN + (i + 1) * BEAT_S
        t.observe(nom, nom + 0.09)
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT, abs=1e-9)


# ── 8–10: Outlier isolation preserved ────────────────────────────────────────

def test_isolated_outlier_after_stable_playing_rejected():
    """350ms outlier after stable playing must still be rejected.

    After exact-timing beats the accepted errors are ~0, well below the
    50ms floor, so drift detection does not activate.
    """
    t = _tracker()
    for i in range(8):
        t.observe(COUNT_IN + i * BEAT_S, COUNT_IN + i * BEAT_S)  # exact

    ratio_before = t.tempo_ratio
    nom_out = COUNT_IN + 8 * BEAT_S
    t.observe(nom_out, nom_out + 0.350)
    assert t.tempo_ratio == pytest.approx(ratio_before, abs=1e-9)


def test_outlier_after_tiny_consistent_errors_rejected():
    """Outlier after 4 small same-sign errors (-5 ms each) is rejected.

    Errors are same-sign but 5 ms < 50 ms floor → no widening → 350 ms rejected.
    """
    t = _tracker()
    _feed_errors(t, [-0.005, -0.005, -0.005, -0.005])

    ratio_before = t.tempo_ratio
    nom_out = COUNT_IN + 5 * BEAT_S
    t.observe(nom_out, nom_out + 0.350)   # 350 ms positive outlier
    assert t.tempo_ratio == pytest.approx(ratio_before, abs=1e-9)


def test_outlier_rejection_leaves_bpm_unchanged():
    t = _tracker()
    for i in range(8):
        t.observe(COUNT_IN + i * BEAT_S, COUNT_IN + i * BEAT_S)
    bpm_before = t.current_tempo_bpm()
    t.observe(COUNT_IN + 8 * BEAT_S, COUNT_IN + 8 * BEAT_S + 0.350)
    assert t.current_tempo_bpm() == pytest.approx(bpm_before, abs=0.01)


# ── 11–13: Drift detection activates correctly ───────────────────────────────

def test_widening_after_drift_window_large_same_sign():
    t = _tracker(drift_window=4, drift_min_frac=0.10)
    _force_accepted_errors(t, [0.10, 0.12, 0.14, 0.16])
    assert t.effective_outlier_limit() > BASE_LIMIT


def test_widening_resets_when_sign_flips():
    """After drift is detected, one opposite-sign entry resets to base limit."""
    t = _tracker(drift_window=4, drift_min_frac=0.10)
    # 3 large positive errors then 1 large negative
    _feed_errors(t, [0.10, 0.12, 0.14, -0.10])
    assert t.effective_outlier_limit() == pytest.approx(BASE_LIMIT, abs=1e-9)


def test_drift_detection_works_for_early_playing():
    """Consistent early playing (negative errors) also triggers widening."""
    t = _tracker(drift_window=4, drift_min_frac=0.10)
    _force_accepted_errors(t, [-0.10, -0.12, -0.14, -0.16])
    assert t.effective_outlier_limit() > BASE_LIMIT


# ── 14–17: Gradual accel/decel direct-feed ───────────────────────────────────

def _gradual_feed(
    tracker: TempoTracker,
    start_bpm: float,
    end_bpm: float,
    n_beats: int,
    count_in_s: float = COUNT_IN,
) -> int:
    """Feed n_beats of linearly changing tempo; return rejection count."""
    nom_beat_s = 60.0 / NOM_BPM
    rejected = 0
    actual_t = count_in_s
    for i in range(n_beats):
        nom = count_in_s + i * nom_beat_s
        limit_before = tracker.effective_outlier_limit()
        predicted = tracker.adjusted_target_time(nom)
        error = actual_t - predicted
        if i > 0 and abs(error) > limit_before:
            rejected += 1
        tracker.observe(nom, actual_t)
        frac = i / max(1, n_beats - 1)
        current_bpm = start_bpm + frac * (end_bpm - start_bpm)
        actual_t += 60.0 / current_bpm
    return rejected


def test_accel_direct_feed_zero_rejections():
    t = _tracker()
    rejected = _gradual_feed(t, 120.0, 132.0, 48)
    assert rejected == 0, f"Expected 0 rejections, got {rejected}"


def test_accel_direct_feed_bpm_converges():
    t = _tracker()
    _gradual_feed(t, 120.0, 132.0, 48)
    assert t.current_tempo_bpm() >= 130.0, (
        f"Expected ≥ 130 BPM after 48-beat accel, got {t.current_tempo_bpm():.2f}"
    )


def test_decel_direct_feed_zero_rejections():
    t = _tracker()
    rejected = _gradual_feed(t, 132.0, 120.0, 48)
    assert rejected == 0, f"Expected 0 rejections, got {rejected}"


def test_decel_direct_feed_bpm_converges():
    t = _tracker()
    _gradual_feed(t, 132.0, 120.0, 48)
    assert t.current_tempo_bpm() <= 122.0, (
        f"Expected ≤ 122 BPM after 48-beat decel, got {t.current_tempo_bpm():.2f}"
    )


# ── 18–19: Stable / small-drift unaffected ───────────────────────────────────

def test_exact_tempo_no_rejections():
    t = _tracker()
    for i in range(32):
        t.observe(COUNT_IN + i * BEAT_S, COUNT_IN + i * BEAT_S)
    assert t.current_tempo_bpm() == pytest.approx(NOM_BPM, abs=0.1)


def test_small_drift_121bpm_no_rejections():
    """121 BPM → errors per beat ≈ 4 ms, well below 50 ms floor.  No widening,
    but also no spurious rejections since errors stay far below base limit."""
    t = _tracker()
    player_beat_s = 60.0 / 121.0
    rejected = 0
    for i in range(32):
        nom = COUNT_IN + i * BEAT_S
        act = COUNT_IN + i * player_beat_s
        limit = t.effective_outlier_limit()
        predicted = t.adjusted_target_time(nom)
        if i > 0 and abs(act - predicted) > limit:
            rejected += 1
        t.observe(nom, act)
    assert rejected == 0


# ── 20–25: Replay-tool integration ───────────────────────────────────────────

def test_replay_accel_zero_rejections():
    onsets = scenario_gradual_change(NOM_BPM, 120.0, 132.0, 48, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    rejects = [r for r in results if r["status"] == "reject"]
    assert len(rejects) == 0, f"Expected 0 rejections, got {len(rejects)}"


def test_replay_accel_bpm_converges():
    onsets = scenario_gradual_change(NOM_BPM, 120.0, 132.0, 48, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    assert results[-1]["estimated_bpm"] >= 130.0, (
        f"Expected ≥ 130 BPM, got {results[-1]['estimated_bpm']:.2f}"
    )


def test_replay_decel_zero_rejections():
    onsets = scenario_gradual_change(NOM_BPM, 132.0, 120.0, 48, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    rejects = [r for r in results if r["status"] == "reject"]
    assert len(rejects) == 0, f"Expected 0 rejections, got {len(rejects)}"


def test_replay_decel_bpm_converges():
    onsets = scenario_gradual_change(NOM_BPM, 132.0, 120.0, 48, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    assert results[-1]["estimated_bpm"] <= 122.0, (
        f"Expected ≤ 122 BPM after decel, got {results[-1]['estimated_bpm']:.2f}"
    )


def test_replay_outlier_after_stable_still_rejected():
    onsets = scenario_with_outlier(NOM_BPM, 20, 10, 0.350, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    assert results[10]["status"] == "reject"


def test_replay_exact_no_rejections():
    onsets = scenario_exact_nominal(NOM_BPM, 32, COUNT_IN)
    results = run_replay(onsets, NOM_BPM)
    rejects = [r for r in results if r["status"] == "reject"]
    assert len(rejects) == 0
