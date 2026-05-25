#!/usr/bin/env python3
"""Synthetic timing diagnostic — verify timing-error accuracy with known onsets.

Injects onsets at known offsets from target beat times and checks whether the
engine reports timing errors that match those offsets exactly.

NOTE: aubio onset detection is BYPASSED entirely.  Onset times are fed
      directly to SessionEngine via session_replay.replay_session_data(),
      so only the matching and timing-error computation are exercised —
      not the audio DSP chain used in live_feedback_demo.py.

Run from the project root:
    python scripts/synthetic_timing_diagnostic.py
    python scripts/synthetic_timing_diagnostic.py --bpm 120
    python scripts/synthetic_timing_diagnostic.py --bpm 120 --beats 1 --count-in 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_replay import replay_session_data
from core.timing_policy import match_window_s as _match_window_s

BPM        = 60
COUNT_IN   = 2
BEATS      = 4
OFFSETS_MS = [-150, -80, -30, 0, +30, +80, +150]


# ── Pure helper (importable for tests) ───────────────────────────────────────

def run_offset_check(
    bpm: float,
    count_in: int,
    targets: list[dict],
    offset_ms: float,
) -> list[dict]:
    """Inject all targets with a constant timing offset and return result rows.

    Each target gets an onset at ``nominal_beat_time + offset_s``.  The result
    row reports whether the engine recovered that offset correctly.

    Parameters
    ----------
    bpm, count_in, targets
        Session parameters forwarded to replay_session_data.
    offset_ms
        Signed offset in milliseconds applied uniformly to all onset times.

    Returns
    -------
    list[dict]
        One entry per target, in target order, with keys:
            target_idx, target_time_s, onset_time_s,
            injected_ms, reported_ms, delta_ms, severity
        If an onset fell outside the match window (missed), ``reported_ms``
        and ``delta_ms`` are None and ``severity`` is ``"MISSED"``.
    """
    beat_s     = 60.0 / bpm
    count_in_s = count_in * beat_s
    offset_s   = offset_ms / 1000.0

    onsets = [
        count_in_s + t["time"] * beat_s + offset_s
        for t in targets
    ]

    data   = {"bpm": bpm, "count_in_beats": count_in, "targets": targets, "onsets": onsets}
    events = replay_session_data(data)

    hit_events = {
        ev["target_idx"]: ev
        for ev in events
        if ev["detected_note"] is not None
    }

    rows = []
    for i, t in enumerate(targets):
        nominal_s = count_in_s + t["time"] * beat_s
        ev        = hit_events.get(i)
        if ev is not None:
            reported_ms = ev["timing_error_s"] * 1000.0
            rows.append({
                "target_idx":    i,
                "target_time_s": nominal_s,
                "onset_time_s":  nominal_s + offset_s,
                "injected_ms":   offset_ms,
                "reported_ms":   reported_ms,
                "delta_ms":      reported_ms - offset_ms,
                "severity":      ev["severity"],
            })
        else:
            rows.append({
                "target_idx":    i,
                "target_time_s": nominal_s,
                "onset_time_s":  nominal_s + offset_s,
                "injected_ms":   offset_ms,
                "reported_ms":   None,
                "delta_ms":      None,
                "severity":      "MISSED",
            })

    return rows


# ── Formatting ────────────────────────────────────────────────────────────────

def _print_table(all_rows: list[dict]) -> None:
    header = (
        f"  {'tgt':>3}  {'target_s':>8}  {'onset_s':>9}  "
        f"{'inj_ms':>7}  {'rep_ms':>7}  {'delta_ms':>8}  severity"
    )
    sep = "  " + "─" * (len(header) - 2)

    last_offset: float | None = None
    for row in all_rows:
        if row["injected_ms"] != last_offset:
            print(f"\nOffset {row['injected_ms']:+.0f} ms")
            print(header)
            print(sep)
            last_offset = row["injected_ms"]

        rep_s   = f"{row['reported_ms']:+7.1f}" if row["reported_ms"] is not None else f"{'—':>7}"
        delta_s = f"{row['delta_ms']:+8.3f}"    if row["delta_ms"]    is not None else f"{'—':>8}"
        print(
            f"  {row['target_idx']:>3}  {row['target_time_s']:>8.3f}  {row['onset_time_s']:>9.3f}  "
            f"{row['injected_ms']:>+7.1f}  {rep_s}  {delta_s}  {row['severity']}"
        )


def _print_summary(all_rows: list[dict]) -> None:
    print("\nSummary (per offset):")
    print(f"  {'inj_ms':>7}  {'hits':>4}  {'misses':>6}  {'max|Δ|ms':>9}  {'mean|Δ|ms':>10}")
    print("  " + "─" * 44)

    offsets = sorted({r["injected_ms"] for r in all_rows})
    for off in offsets:
        group  = [r for r in all_rows if r["injected_ms"] == off]
        hits   = [r for r in group if r["delta_ms"] is not None]
        misses = len(group) - len(hits)
        deltas = [abs(r["delta_ms"]) for r in hits]
        max_d  = f"{max(deltas):.3f}"              if deltas else "—"
        mean_d = f"{sum(deltas)/len(deltas):.3f}"  if deltas else "—"
        print(f"  {off:>+7.0f}  {len(hits):>4}  {misses:>6}  {max_d:>9}  {mean_d:>10}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic timing diagnostic (no audio hardware)"
    )
    p.add_argument("--bpm",      type=float, default=float(BPM),
                   help=f"Tempo in BPM (default {BPM})")
    p.add_argument("--beats",    type=int,   default=BEATS,
                   help=f"Number of targets (default {BEATS})")
    p.add_argument("--count-in", type=int,   default=COUNT_IN, dest="count_in",
                   help=f"Count-in beats before first target (default {COUNT_IN})")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = _parse_args()
    targets = [{"time": i} for i in range(args.beats)]
    mw_s    = _match_window_s(args.bpm)

    print(
        f"Synthetic timing diagnostic\n"
        f"  BPM={args.bpm:.0f}  beats={args.beats}  count_in={args.count_in}\n"
        f"  match_window = {mw_s:.3f} s ({mw_s*1000:.0f} ms)  "
        f"— onsets beyond ±{mw_s*1000:.0f} ms are misses\n"
        f"\n"
        f"  NOTE: aubio onset detection is BYPASSED.  Onset times are injected\n"
        f"  directly into SessionEngine.on_onset(); only matching and timing-\n"
        f"  error computation are exercised, not the live audio DSP chain.\n"
    )

    all_rows: list[dict] = []
    for offset_ms in OFFSETS_MS:
        all_rows.extend(run_offset_check(args.bpm, args.count_in, targets, offset_ms))

    _print_table(all_rows)
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
