"""Replay synthetic onset sequences through fixed-grid and adaptive tempo tracking.

Compares how much timing error each approach accumulates beat-by-beat across a
set of built-in scenarios.  All logic is synthetic — no audio hardware required.

Usage
-----
    python tools/diagnostics/replay_tempo_tracking.py                   # default: fast scenario
    python tools/diagnostics/replay_tempo_tracking.py --scenario all    # every scenario
    python tools/diagnostics/replay_tempo_tracking.py --scenario fast --actual-bpm 130 --beats 32
    python tools/diagnostics/replay_tempo_tracking.py --scenario accel --start-bpm 100 --end-bpm 132
    python tools/diagnostics/replay_tempo_tracking.py --scenario outlier_late --outlier-offset-ms 350
    python tools/diagnostics/replay_tempo_tracking.py --scenario missed --missed-beats 5,10,15
    python tools/diagnostics/replay_tempo_tracking.py --scenario all --csv /tmp/timing_report.csv
    python tools/diagnostics/replay_tempo_tracking.py --scenario all --summary-only
    python tools/diagnostics/replay_tempo_tracking.py --scenario fast --no-summary

Scenarios
---------
exact           Player follows nominal tempo exactly.
fast            Player plays at a steady tempo faster than nominal.
slow            Player plays at a steady tempo slower than nominal.
accel           Player gradually accelerates beat-by-beat.
decel           Player gradually decelerates beat-by-beat.
outlier_early   One beat arrives far too early (outlier rejection check).
outlier_late    One beat arrives far too late (outlier rejection check).
missed          A subset of beats produce no detected onset.
all             Every scenario above, in sequence.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tempo_tracker import TempoTracker


# ── Onset entry ───────────────────────────────────────────────────────────────

def _onset(beat_index: int, nominal_time_s: float, actual_time_s: float | None) -> dict:
    return {
        "beat_index":     beat_index,
        "nominal_time_s": nominal_time_s,
        "actual_time_s":  actual_time_s,
    }


# ── Scenario generators ───────────────────────────────────────────────────────

def scenario_exact_nominal(
    nominal_bpm: float,
    n_beats: int,
    count_in_s: float = 2.0,
) -> list[dict]:
    """Player follows the nominal grid exactly."""
    beat_s = 60.0 / nominal_bpm
    return [
        _onset(i, count_in_s + i * beat_s, count_in_s + i * beat_s)
        for i in range(n_beats)
    ]


def scenario_steady_offset(
    nominal_bpm: float,
    actual_bpm: float,
    n_beats: int,
    count_in_s: float = 2.0,
) -> list[dict]:
    """Player plays at a constant tempo different from nominal.

    Beat 0 is shared (both start at count_in_s); subsequent beats accumulate
    drift because the player's inter-onset interval differs from nominal.
    """
    nom_beat_s = 60.0 / nominal_bpm
    act_beat_s = 60.0 / actual_bpm
    return [
        _onset(i, count_in_s + i * nom_beat_s, count_in_s + i * act_beat_s)
        for i in range(n_beats)
    ]


def scenario_gradual_change(
    nominal_bpm: float,
    start_bpm: float,
    end_bpm: float,
    n_beats: int,
    count_in_s: float = 2.0,
) -> list[dict]:
    """Player accelerates or decelerates linearly across the sequence.

    The inter-onset interval is linearly interpolated from ``start_bpm`` to
    ``end_bpm`` beat-by-beat.  Beat 0 is always at ``count_in_s``.
    """
    nom_beat_s = 60.0 / nominal_bpm
    onsets: list[dict] = []
    actual_time = count_in_s
    for i in range(n_beats):
        nom = count_in_s + i * nom_beat_s
        onsets.append(_onset(i, nom, actual_time))
        frac = i / max(1, n_beats - 1)
        current_bpm = start_bpm + frac * (end_bpm - start_bpm)
        actual_time += 60.0 / current_bpm
    return onsets


def scenario_with_outlier(
    nominal_bpm: float,
    n_beats: int,
    outlier_beat: int,
    outlier_offset_s: float,
    count_in_s: float = 2.0,
) -> list[dict]:
    """All beats at exactly nominal timing, except one is offset by *outlier_offset_s*."""
    beat_s = 60.0 / nominal_bpm
    onsets: list[dict] = []
    for i in range(n_beats):
        nom = count_in_s + i * beat_s
        act = nom + (outlier_offset_s if i == outlier_beat else 0.0)
        onsets.append(_onset(i, nom, act))
    return onsets


def scenario_with_missed_beats(
    nominal_bpm: float,
    n_beats: int,
    missed: set[int],
    count_in_s: float = 2.0,
) -> list[dict]:
    """Exact nominal timing for all beats; specified beats produce no onset."""
    beat_s = 60.0 / nominal_bpm
    return [
        _onset(i, count_in_s + i * beat_s, None if i in missed else count_in_s + i * beat_s)
        for i in range(n_beats)
    ]


# ── Replay engine ─────────────────────────────────────────────────────────────

def run_replay(
    onsets: list[dict],
    nominal_bpm: float,
    tempo_beta: float = 0.30,
    phase_alpha: float = 0.10,
    outlier_threshold: float = 0.40,
) -> list[dict]:
    """Feed *onsets* through a TempoTracker and return per-beat result rows.

    Each entry in *onsets* is a dict with keys:
        beat_index      : int
        nominal_time_s  : float   — metronome position (seconds)
        actual_time_s   : float | None  — player onset; None = missed beat

    Returns one result dict per onset.  Fields:

        beat                  : int
        nominal_time_s        : float
        actual_onset_time_s   : float | None
        adjusted_target_time_s: float | None  (None when no onset)
        fixed_error_ms        : float | None
        adaptive_error_ms     : float | None
        estimated_bpm         : float         (current tracker estimate)
        confidence            : float         (0–1 stability score)
        status                : str           "anchor" | "accept" | "reject" | "miss"
    """
    tracker = TempoTracker(
        nominal_bpm,
        tempo_beta=tempo_beta,
        phase_alpha=phase_alpha,
        outlier_threshold=outlier_threshold,
    )

    results: list[dict] = []

    for onset in onsets:
        nom = onset["nominal_time_s"]
        act = onset["actual_time_s"]

        # Capture effective limit before any state change; used for summary.
        effective_limit_ms = tracker.effective_outlier_limit() * 1000.0

        if act is None:
            # Missed beat — no observation, tracker unchanged.
            results.append({
                "beat":                      onset["beat_index"],
                "nominal_time_s":            nom,
                "actual_onset_time_s":       None,
                "adjusted_target_time_s":    None,
                "fixed_error_ms":            None,
                "adaptive_error_ms":         None,
                "estimated_bpm":             tracker.current_tempo_bpm(),
                "confidence":                tracker.confidence(),
                "status":                    "miss",
                "effective_outlier_limit_ms": effective_limit_ms,
            })
            continue

        # Compute adjusted target before updating the tracker.
        adjusted = tracker.adjusted_target_time(nom)
        raw_error_s = act - adjusted
        fixed_error_s = act - nom

        # Determine acceptance status using the same effective limit that
        # observe() will apply (includes drift-detection widening).
        if not tracker.has_anchor:
            status = "anchor"
        elif abs(raw_error_s) <= tracker.effective_outlier_limit():
            status = "accept"
        else:
            status = "reject"

        # Update tracker (internally mirrors the same accept/reject logic).
        tracker.observe(nom, act)

        results.append({
            "beat":                      onset["beat_index"],
            "nominal_time_s":            nom,
            "actual_onset_time_s":       act,
            "adjusted_target_time_s":    adjusted,
            "fixed_error_ms":            fixed_error_s * 1000.0,
            "adaptive_error_ms":         raw_error_s * 1000.0,
            "estimated_bpm":             tracker.current_tempo_bpm(),
            "confidence":                tracker.confidence(),
            "status":                    status,
            "effective_outlier_limit_ms": effective_limit_ms,
        })

    return results


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_table(results: list[dict], title: str = "") -> str:
    """Return a human-readable fixed-width table string."""
    header = (
        f"{'Beat':>5}  {'Nominal(s)':>10}  {'Actual(s)':>9}  "
        f"{'Adjusted(s)':>11}  {'Fixed(ms)':>9}  {'Adapt(ms)':>9}  "
        f"{'Est.BPM':>7}  {'Conf':>5}  Status"
    )
    rule = (
        f"{'─'*5}  {'─'*10}  {'─'*9}  {'─'*11}  "
        f"{'─'*9}  {'─'*9}  {'─'*7}  {'─'*5}  {'─'*6}"
    )

    def _fmt_f(v: float | None, fmt: str, signed: bool = False) -> str:
        if v is None:
            return "—"
        return format(v, ('+' if signed else '') + fmt)

    rows = []
    for r in results:
        rows.append(
            f"{r['beat']:>5}  "
            f"{_fmt_f(r['nominal_time_s'],    '9.3f'):>10}  "
            f"{_fmt_f(r['actual_onset_time_s'],'8.3f'):>9}  "
            f"{_fmt_f(r['adjusted_target_time_s'],'10.3f'):>11}  "
            f"{_fmt_f(r['fixed_error_ms'],    '8.1f', signed=True):>9}  "
            f"{_fmt_f(r['adaptive_error_ms'], '8.1f', signed=True):>9}  "
            f"{r['estimated_bpm']:>7.1f}  "
            f"{r['confidence']:>5.3f}  "
            f"{r['status']}"
        )

    lines = []
    if title:
        lines.append(title)
        lines.append("")
    lines.append(header)
    lines.append(rule)
    lines.extend(rows)
    return "\n".join(lines)


def export_csv(results: list[dict], path: str) -> None:
    """Write *results* to a CSV file at *path*."""
    columns = [
        "beat", "nominal_time_s", "actual_onset_time_s",
        "adjusted_target_time_s", "fixed_error_ms", "adaptive_error_ms",
        "estimated_bpm", "confidence", "status",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in columns})


# ── Summary helpers ──────────────────────────────────────────────────────────

def summarize_results(results: list[dict]) -> dict:
    """Compute aggregate statistics from a ``run_replay`` result list.

    Returns a dict with the following keys:

        accepted              : int    — beats with status "anchor" or "accept"
        rejected              : int    — beats with status "reject"
        missed                : int    — beats with status "miss"
        final_bpm             : float
        bpm_min               : float
        bpm_max               : float
        mean_abs_fixed_ms     : float | None  — over beats that have an onset
        mean_abs_adapt_ms     : float | None
        max_abs_adapt_ms      : float | None
        adapt_improvement_ms  : float | None  — mean_abs_fixed − mean_abs_adapt
        adapt_improvement_pct : float | None  — improvement as % of fixed mean
        outlier_limit_min_ms  : float | None
        outlier_limit_max_ms  : float | None
    """
    accepted = sum(1 for r in results if r["status"] in ("anchor", "accept"))
    rejected = sum(1 for r in results if r["status"] == "reject")
    missed   = sum(1 for r in results if r["status"] == "miss")

    bpms      = [r["estimated_bpm"] for r in results]
    final_bpm = bpms[-1] if bpms else 0.0
    bpm_min   = min(bpms) if bpms else 0.0
    bpm_max   = max(bpms) if bpms else 0.0

    fixed_abs  = [abs(r["fixed_error_ms"])   for r in results if r["fixed_error_ms"]   is not None]
    adapt_abs  = [abs(r["adaptive_error_ms"]) for r in results if r["adaptive_error_ms"] is not None]

    mean_abs_fixed_ms = sum(fixed_abs) / len(fixed_abs) if fixed_abs else None
    mean_abs_adapt_ms = sum(adapt_abs) / len(adapt_abs) if adapt_abs else None
    max_abs_adapt_ms  = max(adapt_abs)                   if adapt_abs else None

    if mean_abs_fixed_ms is not None and mean_abs_adapt_ms is not None:
        adapt_improvement_ms = mean_abs_fixed_ms - mean_abs_adapt_ms
        adapt_improvement_pct = (
            100.0 * adapt_improvement_ms / mean_abs_fixed_ms
            if mean_abs_fixed_ms > 0 else None
        )
    else:
        adapt_improvement_ms  = None
        adapt_improvement_pct = None

    limits = [
        r["effective_outlier_limit_ms"]
        for r in results
        if r.get("effective_outlier_limit_ms") is not None
    ]
    outlier_limit_min_ms = min(limits) if limits else None
    outlier_limit_max_ms = max(limits) if limits else None

    return {
        "accepted":              accepted,
        "rejected":              rejected,
        "missed":                missed,
        "final_bpm":             final_bpm,
        "bpm_min":               bpm_min,
        "bpm_max":               bpm_max,
        "mean_abs_fixed_ms":     mean_abs_fixed_ms,
        "mean_abs_adapt_ms":     mean_abs_adapt_ms,
        "max_abs_adapt_ms":      max_abs_adapt_ms,
        "adapt_improvement_ms":  adapt_improvement_ms,
        "adapt_improvement_pct": adapt_improvement_pct,
        "outlier_limit_min_ms":  outlier_limit_min_ms,
        "outlier_limit_max_ms":  outlier_limit_max_ms,
    }


def format_summary(summary: dict, title: str = "") -> str:
    """Return a compact multi-line summary string for one scenario."""

    def _f(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    sep = "─" * 60

    lines = [sep]
    if title:
        lines.append(f"  {title}")
        lines.append("")

    lines.append(
        f"  Observations          : "
        f"{summary['accepted']} accepted  "
        f"{summary['rejected']} rejected  "
        f"{summary['missed']} missed"
    )
    lines.append(
        f"  Final BPM             : {_f(summary['final_bpm'], '.1f')}"
        f"  (range {_f(summary['bpm_min'], '.1f')}–{_f(summary['bpm_max'], '.1f')})"
    )
    lines.append(
        f"  Mean error (fixed)    : {_f(summary['mean_abs_fixed_ms'], '.1f')} ms"
    )
    lines.append(
        f"  Mean error (adaptive) : {_f(summary['mean_abs_adapt_ms'], '.1f')} ms"
        f"  max {_f(summary['max_abs_adapt_ms'], '.1f')} ms"
    )

    imp_ms  = summary["adapt_improvement_ms"]
    imp_pct = summary["adapt_improvement_pct"]
    if imp_ms is not None:
        pct_str = f"  ({_f(imp_pct, '+.1f')}%)" if imp_pct is not None else ""
        lines.append(
            f"  Adaptive improvement  : {_f(imp_ms, '+.1f')} ms{pct_str}"
        )

    lim_min = summary["outlier_limit_min_ms"]
    lim_max = summary["outlier_limit_max_ms"]
    if lim_min is not None:
        if abs(lim_min - lim_max) < 0.1:
            lim_str = f"{lim_min:.0f} ms (constant)"
        else:
            lim_str = f"{lim_min:.0f}–{lim_max:.0f} ms (widened by drift detection)"
        lines.append(f"  Effective outlier lim : {lim_str}")

    lines.append(sep)
    return "\n".join(lines)


# ── Scenario registry ─────────────────────────────────────────────────────────

def build_scenarios(args: argparse.Namespace) -> list[tuple[str, list[dict]]]:
    """Return a list of (title, onsets) pairs to replay."""
    n        = args.beats
    nom      = args.nominal_bpm
    cin_s    = args.count_in_beats * 60.0 / nom
    act      = args.actual_bpm
    start    = args.start_bpm
    end      = args.end_bpm
    ob       = args.outlier_beat if args.outlier_beat >= 0 else n // 2
    off_s    = args.outlier_offset_ms / 1000.0
    missed   = set(int(x) for x in args.missed_beats.split(",") if x.strip()) if args.missed_beats else {n // 4, n // 2}

    all_scenarios = [
        (
            f"exact  (nominal={nom:.0f} BPM, {n} beats)",
            scenario_exact_nominal(nom, n, cin_s),
        ),
        (
            f"fast   (nominal={nom:.0f} BPM → actual={act:.0f} BPM, {n} beats)",
            scenario_steady_offset(nom, act, n, cin_s),
        ),
        (
            f"slow   (nominal={nom:.0f} BPM → actual={nom - (act - nom):.0f} BPM, {n} beats)",
            scenario_steady_offset(nom, nom - (act - nom), n, cin_s),
        ),
        (
            f"accel  (nominal={nom:.0f} BPM, player {start:.0f}→{end:.0f} BPM, {n} beats)",
            scenario_gradual_change(nom, start, end, n, cin_s),
        ),
        (
            f"decel  (nominal={nom:.0f} BPM, player {end:.0f}→{start:.0f} BPM, {n} beats)",
            scenario_gradual_change(nom, end, start, n, cin_s),
        ),
        (
            f"outlier_early  (nominal={nom:.0f} BPM, beat {ob} offset {-abs(off_s)*1000:.0f} ms)",
            scenario_with_outlier(nom, n, ob, -abs(off_s), cin_s),
        ),
        (
            f"outlier_late   (nominal={nom:.0f} BPM, beat {ob} offset +{abs(off_s)*1000:.0f} ms)",
            scenario_with_outlier(nom, n, ob, abs(off_s), cin_s),
        ),
        (
            f"missed (nominal={nom:.0f} BPM, beats {sorted(missed)} missing, {n} beats)",
            scenario_with_missed_beats(nom, n, missed, cin_s),
        ),
    ]

    if args.scenario == "all":
        return all_scenarios

    name_map = {
        "exact":         all_scenarios[0],
        "fast":          all_scenarios[1],
        "slow":          all_scenarios[2],
        "accel":         all_scenarios[3],
        "decel":         all_scenarios[4],
        "outlier_early": all_scenarios[5],
        "outlier_late":  all_scenarios[6],
        "missed":        all_scenarios[7],
    }
    if args.scenario not in name_map:
        print(f"Unknown scenario '{args.scenario}'. "
              f"Choose from: {', '.join(name_map)} or 'all'.", file=sys.stderr)
        sys.exit(1)
    return [name_map[args.scenario]]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scenario",       default="fast",
                   help="Scenario to run (default: fast).")
    p.add_argument("--nominal-bpm",    type=float, default=120.0,
                   dest="nominal_bpm", metavar="BPM",
                   help="Reference tempo in BPM (default: 120).")
    p.add_argument("--actual-bpm",     type=float, default=132.0,
                   dest="actual_bpm",  metavar="BPM",
                   help="Actual player tempo for fast/slow scenarios (default: 132).")
    p.add_argument("--start-bpm",      type=float, default=120.0,
                   dest="start_bpm",   metavar="BPM",
                   help="Starting BPM for gradual scenarios (default: 120).")
    p.add_argument("--end-bpm",        type=float, default=132.0,
                   dest="end_bpm",     metavar="BPM",
                   help="Ending BPM for gradual scenarios (default: 132).")
    p.add_argument("--beats",          type=int,   default=32,
                   help="Number of beats to simulate (default: 32).")
    p.add_argument("--count-in-beats", type=int,   default=4,
                   dest="count_in_beats",
                   help="Count-in beats before the sequence (default: 4).")
    p.add_argument("--outlier-beat",   type=int,   default=-1,
                   dest="outlier_beat", metavar="IDX",
                   help="Beat index to displace in outlier scenarios (default: n/2).")
    p.add_argument("--outlier-offset-ms", type=float, default=350.0,
                   dest="outlier_offset_ms", metavar="MS",
                   help="Signed outlier displacement in ms (default: 350).")
    p.add_argument("--missed-beats",   default="",
                   dest="missed_beats", metavar="LIST",
                   help="Comma-separated beat indices to mark as missed.")
    p.add_argument("--csv",            default="",
                   metavar="PATH",
                   help="Optional path to write CSV output.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--summary-only", action="store_true", default=False, dest="summary_only",
        help="Print only the summary; suppress the per-beat table.",
    )
    mode.add_argument(
        "--no-summary", action="store_true", default=False, dest="no_summary",
        help="Print only the per-beat table; suppress the summary (original behavior).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    scenarios = build_scenarios(args)

    show_table   = not args.summary_only
    show_summary = not args.no_summary

    all_results: list[dict] = []

    for title, onsets in scenarios:
        results = run_replay(onsets, args.nominal_bpm)
        if show_table:
            print(format_table(results, title=title))
            print()
        if show_summary:
            print(format_summary(summarize_results(results), title=title))
            print()
        if args.csv:
            for r in results:
                r["_scenario"] = title
            all_results.extend(results)

    if args.csv:
        # Add scenario column to the column list when exporting multi-scenario.
        columns = [
            "beat", "nominal_time_s", "actual_onset_time_s",
            "adjusted_target_time_s", "fixed_error_ms", "adaptive_error_ms",
            "estimated_bpm", "confidence", "status",
        ]
        if len(scenarios) > 1:
            columns = ["_scenario"] + columns
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in all_results:
                writer.writerow({
                    k: ("" if row.get(k) is None else row.get(k))
                    for k in columns
                })
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
