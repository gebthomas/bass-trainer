"""Unit tests for core.tempo_tracker.TempoTracker.

All tests use synthetic onset times — no audio hardware involved.

Scenario matrix
---------------
1. Before first observation  : adjusted_target_time is identity
2. Stable tempo              : grid stays at nominal; confidence rises to near-1
3. Constant phase offset     : player always late; anchor absorbs it; predictions match
4. Gradual speed-up          : player's BPM drifts higher; tracker detects it
5. Gradual slow-down         : player's BPM drifts lower; tracker detects it
6. Isolated timing mistake   : one far-off onset; grid barely moves
7. Sparse observations       : every-other-beat; tempo ratio still estimated correctly
8. Sustained phase drift     : consistent late drift after anchor; phase_offset grows
"""

from __future__ import annotations

import pytest

from core.tempo_tracker import TempoTracker


# ── helpers ───────────────────────────────────────────────────────────────────

def _feed(tracker: TempoTracker, beats: list[tuple[float, float]]) -> None:
    for nom, act in beats:
        tracker.observe(nom, act)


def _nominal_beats(n: int, bpm: float, count_in_s: float = 2.0) -> list[float]:
    """Return n nominal beat times starting after count_in_s."""
    beat_s = 60.0 / bpm
    return [count_in_s + i * beat_s for i in range(n)]


# ── 1. before first observation ───────────────────────────────────────────────

def test_no_observation_returns_nominal():
    tracker = TempoTracker(60.0)
    assert not tracker.has_anchor
    assert tracker.adjusted_target_time(2.0) == pytest.approx(2.0)
    assert tracker.adjusted_target_time(3.5) == pytest.approx(3.5)
    assert tracker.current_tempo_bpm() == pytest.approx(60.0)
    assert tracker.confidence() == pytest.approx(0.0)


# ── 2. stable tempo ───────────────────────────────────────────────────────────

def test_stable_tempo_stays_near_nominal():
    """16 beats played exactly on the nominal grid — tracker should stay calibrated."""
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(16, 60.0)
    _feed(tracker, [(t, t) for t in noms])

    assert tracker.current_tempo_bpm() == pytest.approx(60.0, abs=0.05)
    assert tracker.tempo_ratio == pytest.approx(1.0, abs=0.001)
    assert tracker.phase_offset == pytest.approx(0.0, abs=0.001)


def test_stable_tempo_raises_confidence():
    """Confidence should climb to near-1 after many consistent observations."""
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(12, 60.0)
    _feed(tracker, [(t, t) for t in noms])
    assert tracker.confidence() > 0.95


# ── 3. constant phase offset (player always late / early) ─────────────────────

def test_constant_offset_predictions_match_player():
    """Player always 60 ms late — adjusted grid should predict player timing."""
    offset = 0.060  # 60 ms
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(16, 60.0)
    _feed(tracker, [(t, t + offset) for t in noms])

    # Predictions for future beats should match the player's timing
    for i in range(16, 20):
        nom = 2.0 + i * 1.0
        adj = tracker.adjusted_target_time(nom)
        assert abs(adj - (nom + offset)) < 0.010, (
            f"beat {i}: expected {nom + offset:.3f}, got {adj:.3f}"
        )


def test_constant_early_offset_predictions_match_player():
    """Player always 40 ms early."""
    offset = -0.040
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(16, 60.0)
    _feed(tracker, [(t, t + offset) for t in noms])

    for i in range(16, 20):
        nom = 2.0 + i * 1.0
        adj = tracker.adjusted_target_time(nom)
        assert abs(adj - (nom + offset)) < 0.010


# ── 4. gradual speed-up ───────────────────────────────────────────────────────

def test_gradual_speedup_detected():
    """Player runs at 64 BPM vs 60 BPM nominal — tracker should detect faster tempo."""
    nominal_bpm = 60.0
    player_bpm  = 64.0
    tracker = TempoTracker(nominal_bpm)

    noms = _nominal_beats(28, nominal_bpm)
    acts = [noms[0] + i * (60.0 / player_bpm) for i in range(len(noms))]
    _feed(tracker, list(zip(noms, acts)))

    assert tracker.current_tempo_bpm() > nominal_bpm, (
        f"Expected BPM > {nominal_bpm}, got {tracker.current_tempo_bpm():.2f}"
    )
    # Adaptation is gradual — should not fully converge but should move toward truth
    assert tracker.current_tempo_bpm() < player_bpm + 0.5


def test_gradual_speedup_adjusts_predictions():
    """Adjusted target times should shift earlier as the tracker detects speedup."""
    nominal_bpm = 60.0
    player_bpm  = 63.0
    tracker = TempoTracker(nominal_bpm)

    noms = _nominal_beats(24, nominal_bpm)
    acts = [noms[0] + i * (60.0 / player_bpm) for i in range(len(noms))]
    _feed(tracker, list(zip(noms, acts)))

    # Adjusted prediction for beat 24 should be earlier than nominal
    nom_24 = 2.0 + 24 * 1.0
    adj_24 = tracker.adjusted_target_time(nom_24)
    # Player at beat 24: noms[0] + 24 * player_beat_s
    player_24 = noms[0] + 24 * (60.0 / player_bpm)
    # Tracker prediction should be closer to player_24 than nom_24 is
    assert abs(adj_24 - player_24) < abs(nom_24 - player_24)


# ── 5. gradual slow-down ──────────────────────────────────────────────────────

def test_gradual_slowdown_detected():
    """Player runs at 57 BPM vs 60 BPM nominal — tracker should detect slower tempo."""
    nominal_bpm = 60.0
    player_bpm  = 57.0
    tracker = TempoTracker(nominal_bpm)

    noms = _nominal_beats(28, nominal_bpm)
    acts = [noms[0] + i * (60.0 / player_bpm) for i in range(len(noms))]
    _feed(tracker, list(zip(noms, acts)))

    assert tracker.current_tempo_bpm() < nominal_bpm, (
        f"Expected BPM < {nominal_bpm}, got {tracker.current_tempo_bpm():.2f}"
    )
    assert tracker.current_tempo_bpm() > player_bpm - 0.5


def test_gradual_slowdown_adjusts_predictions():
    nominal_bpm = 60.0
    player_bpm  = 57.0
    tracker = TempoTracker(nominal_bpm)

    noms = _nominal_beats(24, nominal_bpm)
    acts = [noms[0] + i * (60.0 / player_bpm) for i in range(len(noms))]
    _feed(tracker, list(zip(noms, acts)))

    nom_24  = 2.0 + 24 * 1.0
    adj_24  = tracker.adjusted_target_time(nom_24)
    player_24 = noms[0] + 24 * (60.0 / player_bpm)
    assert abs(adj_24 - player_24) < abs(nom_24 - player_24)


# ── 6. isolated timing mistake ────────────────────────────────────────────────

def test_isolated_mistake_ignored():
    """One beat is 0.6 s late (> 40% of beat at 60 BPM) — grid should not move."""
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(20, 60.0)

    # Feed 8 clean beats
    _feed(tracker, [(t, t) for t in noms[:8]])
    ratio_before  = tracker.tempo_ratio
    phase_before  = tracker.phase_offset

    # Isolated gross mistake on beat 8
    tracker.observe(noms[8], noms[8] + 0.60)  # 600 ms late — outlier

    ratio_after = tracker.tempo_ratio
    phase_after = tracker.phase_offset

    assert ratio_after == pytest.approx(ratio_before, abs=1e-9), "tempo_ratio changed after outlier"
    assert phase_after == pytest.approx(phase_before, abs=1e-9), "phase_offset changed after outlier"


def test_isolated_mistake_recovery():
    """After a mistake, subsequent good beats continue to track correctly."""
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(20, 60.0)

    _feed(tracker, [(t, t) for t in noms[:8]])
    tracker.observe(noms[8], noms[8] + 0.70)   # outlier
    _feed(tracker, [(t, t) for t in noms[9:]])  # back to normal

    assert tracker.current_tempo_bpm() == pytest.approx(60.0, abs=0.1)
    assert tracker.confidence() > 0.8


# ── 7. sparse observations (every other beat) ─────────────────────────────────

def test_sparse_observations_detect_tempo():
    """Observing only even beats still allows tempo tracking."""
    nominal_bpm = 60.0
    player_bpm  = 64.0
    tracker = TempoTracker(nominal_bpm)

    noms = _nominal_beats(32, nominal_bpm)
    acts = [noms[0] + i * (60.0 / player_bpm) for i in range(len(noms))]

    # Only observe every other beat
    for i in range(0, len(noms), 2):
        tracker.observe(noms[i], acts[i])

    assert tracker.current_tempo_bpm() > nominal_bpm, (
        "Sparse observations should still detect faster tempo"
    )


def test_sparse_observations_predictions():
    """Sparse-trained tracker should still improve predictions for un-observed beats."""
    nominal_bpm = 60.0
    player_bpm  = 63.0
    tracker = TempoTracker(nominal_bpm)

    noms = _nominal_beats(24, nominal_bpm)
    acts = [noms[0] + i * (60.0 / player_bpm) for i in range(len(noms))]

    for i in range(0, len(noms), 2):
        tracker.observe(noms[i], acts[i])

    nom_24  = 2.0 + 24 * 1.0
    adj_24  = tracker.adjusted_target_time(nom_24)
    player_24 = noms[0] + 24 * (60.0 / player_bpm)
    assert abs(adj_24 - player_24) < abs(nom_24 - player_24)


# ── 8. sustained phase drift ──────────────────────────────────────────────────

def test_sustained_phase_drift_after_anchor():
    """Player starts on time then consistently lags — phase_offset should grow."""
    tracker = TempoTracker(60.0)
    noms = _nominal_beats(20, 60.0)

    # First beat: anchor (no drift yet)
    tracker.observe(noms[0], noms[0])

    # Subsequent beats: consistently 80 ms late relative to anchor-based prediction
    drift = 0.080
    for nom in noms[1:]:
        # The anchor-based prediction is 'nom'; player plays at nom + drift
        tracker.observe(nom, nom + drift)

    # phase_offset should have grown toward drift
    assert tracker.phase_offset > 0.0, "Phase offset should be positive (player late)"
    assert tracker.phase_offset < drift * 1.5, "Phase offset should not overshoot"


# ── edge cases ────────────────────────────────────────────────────────────────

def test_invalid_bpm_raises():
    with pytest.raises(ValueError):
        TempoTracker(0.0)
    with pytest.raises(ValueError):
        TempoTracker(-60.0)


def test_single_observation_no_crash():
    """A single observation sets the anchor but doesn't update tempo or phase."""
    tracker = TempoTracker(120.0)
    tracker.observe(1.0, 1.05)
    assert tracker.has_anchor
    assert tracker.tempo_ratio == pytest.approx(1.0)
    assert tracker.phase_offset == pytest.approx(0.0)
    assert tracker.confidence() == pytest.approx(0.0)


def test_adjusted_time_before_anchor_is_identity():
    tracker = TempoTracker(90.0)
    for t in [0.0, 1.5, 10.0, 100.0]:
        assert tracker.adjusted_target_time(t) == pytest.approx(t)
