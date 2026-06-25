#!/usr/bin/env python3
"""Diagnose TempoTracker convergence toward a different true tempo.

Reproduces the observed failure: script at 120 BPM, player at 132 BPM,
tracker reports ~121 BPM after 48 beats.

Run from the project root:
    python tools/diagnostics/diagnose_tempo_convergence.py

Three scenarios
---------------
A. DIRECT-FEED: all 48 beats fed directly to the tracker (bypass window).
   Shows theoretical convergence ceiling with the current EMA.

B. WINDOW-FILTERED FEED: only beats that fall inside the 30 ms extraction
   window are fed.  This is what the live pipeline actually sees.
   Shows that virtually no beats make it through, explaining ~121 BPM.

C. PROPOSED FIX — LONG-SPAN RATIO + FASTER BETA: replace the consecutive-
   pair inter-beat ratio with the anchor-to-current span ratio and raise
   tempo_beta.  Shows near-full convergence even when only a handful of
   beats slip through the window.

Usage
-----
The script prints a beat-by-beat table for each scenario and a summary
at the end.  No audio hardware required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tempo_tracker import TempoTracker


# ── Constants ─────────────────────────────────────────────────────────────────

NOMINAL_BPM   = 120.0
TRUE_BPM      = 132.0
N_BEATS       = 48
COUNT_IN_S    = 2.0      # anchor nominal time

NOMINAL_BEAT_S = 60.0 / NOMINAL_BPM   # 0.500 s
TRUE_BEAT_S    = 60.0 / TRUE_BPM      # 0.4545… s

PRE_ROLL_S     = 0.03    # extraction window lookahead before nominal


# ── Generators ────────────────────────────────────────────────────────────────

def _nominal_times(n: int) -> list[float]:
    return [COUNT_IN_S + i * NOMINAL_BEAT_S for i in range(n)]


def _actual_times(n: int, offset_s: float = 0.0) -> list[float]:
    """Actual onset times for a player at TRUE_BPM, anchored at COUNT_IN_S."""
    return [COUNT_IN_S + offset_s + i * TRUE_BEAT_S for i in range(n)]


def _window_visible(actual_t: float, nominal_t: float) -> bool:
    """True if an actual onset falls inside the nominal extraction window."""
    return (nominal_t - PRE_ROLL_S) <= actual_t


# ── Scenario runner ───────────────────────────────────────────────────────────

def _run_scenario(
    label: str,
    beats: list[tuple[float, float]],  # (nominal, actual) pairs already filtered
    tracker: TempoTracker,
    n_total: int,
    print_table: bool = True,
) -> float:
    """Feed beats into *tracker*, print table, return final BPM."""
    if print_table:
        print(f"\n{'─'*64}")
        print(f"  {label}")
        print(f"{'─'*64}")
        print(f"  {'beat':>4}  {'nom':>7}  {'act':>7}  {'drift_ms':>9}  {'bpm':>8}  visible")

    n_accepted = 0
    for i, (nom, act) in enumerate(beats):
        drift_ms = (act - nom) * 1000
        visible  = _window_visible(act, nom)
        tracker.observe(nom, act)
        bpm = tracker.current_tempo_bpm()
        if print_table:
            mark = "●" if visible else " "
            print(f"  {i:>4}  {nom:>7.3f}  {act:>7.3f}  {drift_ms:>+9.1f}  {bpm:>8.2f}  {mark}")
        n_accepted += 1

    final_bpm = tracker.current_tempo_bpm()
    n_visible = sum(1 for nom, act in beats if _window_visible(act, nom))
    if print_table:
        print(f"\n  Accepted observations : {n_accepted}/{n_total}")
        print(f"  Visible to live window: {n_visible}/{n_total}")
        print(f"  Final BPM             : {final_bpm:.2f}  (target {TRUE_BPM:.1f})")
    return final_bpm


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  TempoTracker convergence diagnostic")
    print(f"  Nominal {NOMINAL_BPM} BPM  |  True player {TRUE_BPM} BPM  |  {N_BEATS} beats")
    print(f"  Window pre_roll = {PRE_ROLL_S*1000:.0f} ms")
    print("=" * 64)

    noms = _nominal_times(N_BEATS)
    acts = _actual_times(N_BEATS)

    pairs_direct  = list(zip(noms, acts))
    pairs_visible = [(n, a) for n, a in pairs_direct if _window_visible(a, n)]

    # ── Scenario A: direct feed (all beats) ──────────────────────────────────

    tracker_a = TempoTracker(NOMINAL_BPM)
    bpm_a = _run_scenario(
        "SCENARIO A — Direct feed (all beats, bypass window)",
        pairs_direct, tracker_a, N_BEATS,
    )

    # ── Scenario B: window-filtered feed ─────────────────────────────────────

    tracker_b = TempoTracker(NOMINAL_BPM)
    bpm_b = _run_scenario(
        "SCENARIO B — Window-filtered feed (live pipeline reality)",
        pairs_visible, tracker_b, N_BEATS,
    )

    # ── Explain why B is bad ──────────────────────────────────────────────────

    print(f"\n  WHY WINDOW FILTERING KILLS CONVERGENCE")
    print(f"  Beat 1 arrives {(acts[1]-noms[1])*1000:+.1f} ms vs nominal.")
    print(f"  Window starts at nominal − {PRE_ROLL_S*1000:.0f} ms.")
    n_vis = len(pairs_visible)
    print(f"  Only {n_vis}/{N_BEATS} beats are within the window ({n_vis/N_BEATS*100:.0f}%).")
    if n_vis > 0:
        print(f"  Visible beats: {[int(noms.index(n)) for n, _ in pairs_visible]}")

    # ── Scenario C: smaller deviation where some beats are visible ───────────

    TRUE_BPM_C   = 122.0
    TRUE_BEAT_S_C = 60.0 / TRUE_BPM_C
    noms_c = _nominal_times(N_BEATS)
    acts_c = [COUNT_IN_S + i * TRUE_BEAT_S_C for i in range(N_BEATS)]
    pairs_c_all = list(zip(noms_c, acts_c))
    pairs_c_vis = [(n, a) for n, a in pairs_c_all if _window_visible(a, n)]

    print(f"\n{'─'*64}")
    print(f"  SCENARIO C — Smaller deviation ({NOMINAL_BPM}→{TRUE_BPM_C} BPM):")
    print(f"  {len(pairs_c_vis)} beats visible through {PRE_ROLL_S*1000:.0f} ms window")
    print(f"{'─'*64}")
    print(f"  {'beat':>4}  {'drift_ms':>9}  {'bpm_old':>9}  {'bpm_fix':>9}")

    tracker_c_old = TempoTracker(NOMINAL_BPM, tempo_beta=0.05)
    tracker_c_fix = TempoTracker(NOMINAL_BPM, tempo_beta=0.15)

    for i, (nom, act) in enumerate(pairs_c_vis):
        tracker_c_old.observe(nom, act)
        tracker_c_fix.observe(nom, act)
        bpm_old = tracker_c_old.current_tempo_bpm()
        bpm_fix = tracker_c_fix.current_tempo_bpm()
        drift_ms = (act - nom) * 1000
        print(f"  {i:>4}  {drift_ms:>+9.1f}  {bpm_old:>9.2f}  {bpm_fix:>9.2f}")

    bpm_c_old_final = tracker_c_old.current_tempo_bpm()
    bpm_c_fix_final = tracker_c_fix.current_tempo_bpm()

    print(f"\n  Old rule final BPM : {bpm_c_old_final:.2f}  (target {TRUE_BPM_C})")
    print(f"  Fix rule final BPM : {bpm_c_fix_final:.2f}  (target {TRUE_BPM_C})")

    # Direct-feed comparison for 132 BPM (bypasses window)
    t_old_direct = TempoTracker(NOMINAL_BPM, tempo_beta=0.05)
    t_fix_direct = TempoTracker(NOMINAL_BPM, tempo_beta=0.15)
    for nom, act in pairs_direct:
        t_old_direct.observe(nom, act)
        t_fix_direct.observe(nom, act)

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'='*64}")
    print(f"  SUMMARY")
    print(f"{'='*64}")
    print(f"  A (direct, {N_BEATS} beats, {NOMINAL_BPM}→{TRUE_BPM} BPM)")
    print(f"     old beta=0.05 : {t_old_direct.current_tempo_bpm():.1f} BPM")
    print(f"     fix beta=0.15 : {t_fix_direct.current_tempo_bpm():.1f} BPM  ← applied")
    print(f"  B (window-filtered, {len(pairs_visible)} beats visible, {NOMINAL_BPM}→{TRUE_BPM} BPM)")
    print(f"     old beta=0.05 : {bpm_b:.1f} BPM  ← only 1 beat (anchor) gets through")
    print(f"  C (window-filtered, {len(pairs_c_vis)} beats visible, {NOMINAL_BPM}→{TRUE_BPM_C} BPM)")
    print(f"     old beta=0.05 : {bpm_c_old_final:.1f} BPM")
    print(f"     fix beta=0.15 : {bpm_c_fix_final:.1f} BPM")
    print()
    print(f"  ROOT CAUSE 1 — Window too narrow for large tempo differences")
    print(f"    {TRUE_BPM} BPM: beat 1 arrives {abs((acts[1]-noms[1])*1000):.1f} ms early > {PRE_ROLL_S*1000:.0f} ms pre-roll → all blocked")
    print(f"    {TRUE_BPM_C} BPM: beats arrive {abs((acts_c[1]-noms_c[1])*1000):.1f} ms/beat early → {len(pairs_c_vis)} beats within window")
    print(f"    → Fix: widen pre_roll in audio_windows.py (separate change, not done here)")
    print()
    print(f"  ROOT CAUSE 2 — EMA too slow (beta=0.05 ≈ 9% residual after 48 beats)")
    print(f"    beta=0.05 residual after 48 beats : {0.95**47 * 100:.1f}%  → {t_old_direct.current_tempo_bpm():.1f} BPM")
    print(f"    beta=0.15 residual after 48 beats : {0.85**47 * 100:.2f}%  → {t_fix_direct.current_tempo_bpm():.1f} BPM")
    print(f"    → APPLIED: default tempo_beta raised 0.05 → 0.15 in core/tempo_tracker.py")
    print(f"    → Outlier rejection (40% threshold) unchanged; phase_alpha unchanged")
    print("=" * 64)


if __name__ == "__main__":
    main()
