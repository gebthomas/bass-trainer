#!/usr/bin/env python3
"""Simulate TempoTracker behaviour with a synthetic player.

No audio hardware involved — generates fake onset times and shows how the
tracker adapts over time.

Usage
-----
    python scripts/tempo_tracker_sim.py
    python scripts/tempo_tracker_sim.py --scenario slowdown
    python scripts/tempo_tracker_sim.py --scenario mistake
    python scripts/tempo_tracker_sim.py --scenario sparse
    python scripts/tempo_tracker_sim.py --list-scenarios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tempo_tracker import TempoTracker


# ── Scenario definitions ──────────────────────────────────────────────────────

def _scenario_stable(n: int = 24, bpm: float = 60.0):
    """Player perfectly on the grid at nominal BPM."""
    beat_s = 60.0 / bpm
    anchor = 2.0
    for i in range(n):
        nom = anchor + i * beat_s
        yield nom, nom, f"beat {i:2d}"


def _scenario_speedup(n: int = 24, nominal_bpm: float = 60.0, player_bpm: float = 64.0):
    """Player gradually running faster than nominal."""
    nom_beat_s = 60.0 / nominal_bpm
    act_beat_s = 60.0 / player_bpm
    anchor = 2.0
    for i in range(n):
        nom = anchor + i * nom_beat_s
        act = anchor + i * act_beat_s
        yield nom, act, f"beat {i:2d}"


def _scenario_slowdown(n: int = 24, nominal_bpm: float = 60.0, player_bpm: float = 56.0):
    """Player gradually running slower than nominal."""
    nom_beat_s = 60.0 / nominal_bpm
    act_beat_s = 60.0 / player_bpm
    anchor = 2.0
    for i in range(n):
        nom = anchor + i * nom_beat_s
        act = anchor + i * act_beat_s
        yield nom, act, f"beat {i:2d}"


def _scenario_mistake(n: int = 24, bpm: float = 60.0):
    """Stable player with one gross mistake at beat 8."""
    beat_s = 60.0 / bpm
    anchor = 2.0
    for i in range(n):
        nom = anchor + i * beat_s
        if i == 8:
            act = nom + 0.65   # ≫ outlier threshold — should be ignored
            label = f"beat {i:2d}  ← MISTAKE (+650 ms)"
        else:
            act = nom
            label = f"beat {i:2d}"
        yield nom, act, label


def _scenario_sparse(n: int = 24, nominal_bpm: float = 60.0, player_bpm: float = 63.0):
    """Only even beats played; player runs faster than nominal."""
    nom_beat_s = 60.0 / nominal_bpm
    act_beat_s = 60.0 / player_bpm
    anchor = 2.0
    for i in range(n):
        if i % 2 == 0:
            nom = anchor + i * nom_beat_s
            act = anchor + i * act_beat_s
            yield nom, act, f"beat {i:2d}"


def _scenario_drift(n: int = 32, bpm: float = 60.0):
    """Player starts on time then slowly drifts later each beat (+8 ms per beat)."""
    beat_s = 60.0 / bpm
    anchor = 2.0
    for i in range(n):
        nom = anchor + i * beat_s
        drift = i * 0.008          # 8 ms later per beat
        act = nom + drift
        yield nom, act, f"beat {i:2d}  drift={drift*1000:+.0f}ms"


SCENARIOS = {
    "stable":   (_scenario_stable,   "Perfect timing at nominal BPM"),
    "speedup":  (_scenario_speedup,  "Player runs ~6% faster (64 BPM vs 60)"),
    "slowdown": (_scenario_slowdown, "Player runs ~7% slower (56 BPM vs 60)"),
    "mistake":  (_scenario_mistake,  "Stable player with one gross timing error"),
    "sparse":   (_scenario_sparse,   "Every other beat only; player slightly fast"),
    "drift":    (_scenario_drift,    "Player drifts progressively later (+8 ms/beat)"),
}


# ── Display helpers ───────────────────────────────────────────────────────────

_HDR = (
    f"{'beat':<22} {'nom_t':>7} {'act_t':>7} {'raw_err':>9}"
    f"  {'adj_t':>7} {'adj_err':>9}  {'bpm':>6} {'conf':>5}"
)
_SEP = "-" * len(_HDR)


def _run_scenario(name: str, gen, nominal_bpm: float = 60.0) -> None:
    print(f"\n{'='*len(_HDR)}")
    print(f"Scenario: {name}  —  {SCENARIOS[name][1]}")
    print(f"Nominal BPM: {nominal_bpm}")
    print(_SEP)
    print(_HDR)
    print(_SEP)

    tracker = TempoTracker(nominal_bpm)

    for nom, act, label in gen:
        adj  = tracker.adjusted_target_time(nom)
        raw_err_ms = (act - nom) * 1000
        adj_err_ms = (act - adj) * 1000

        print(
            f"{label:<22} {nom:7.3f} {act:7.3f} {raw_err_ms:+8.1f}ms"
            f"  {adj:7.3f} {adj_err_ms:+8.1f}ms  "
            f"{tracker.current_tempo_bpm():6.2f} {tracker.confidence():5.3f}"
        )
        tracker.observe(nom, act)

    print(_SEP)
    print(
        f"Final:  BPM={tracker.current_tempo_bpm():.2f}  "
        f"ratio={tracker.tempo_ratio:.4f}  "
        f"phase={tracker.phase_offset*1000:+.1f}ms  "
        f"conf={tracker.confidence():.3f}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TempoTracker simulation")
    p.add_argument(
        "--scenario", default="speedup",
        choices=list(SCENARIOS),
        help="Which scenario to run (default: speedup)",
    )
    p.add_argument("--all", action="store_true", help="Run all scenarios")
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--bpm", type=float, default=60.0, help="Nominal BPM (default 60)")
    return p.parse_args()


def main() -> None:
    args = _parse()

    if args.list_scenarios:
        print("Available scenarios:")
        for name, (_, desc) in SCENARIOS.items():
            print(f"  {name:<10} {desc}")
        return

    names = list(SCENARIOS) if args.all else [args.scenario]
    for name in names:
        gen_fn, _ = SCENARIOS[name]
        _run_scenario(name, gen_fn(), nominal_bpm=args.bpm)


if __name__ == "__main__":
    main()
