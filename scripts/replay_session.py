#!/usr/bin/env python3
"""Replay a captured onset stream through SessionEngine and print feedback.

Loads a session JSON file produced by a live capture or written by hand,
drives SessionEngine deterministically without audio hardware, and prints
the same feedback output as session_engine_demo.py.

Run from the project root:
    python scripts/replay_session.py --file path/to/session.json
    python scripts/replay_session.py --file path/to/session.json --verbose
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_replay import replay_session_data, summarize_replay

# ── Display helpers ────────────────────────────────────────────────────────────

SEVERITY_ICON = {"good": "✓", "warn": "~", "miss": "✗"}


def _print_event(ev: dict, verbose: bool) -> None:
    icon    = SEVERITY_ICON[ev["severity"]]
    err     = ev["timing_error_s"]
    err_str = f"{err * 1000:+.0f} ms" if err is not None else "MISS"
    t_str   = f"{ev['replay_time_s']:5.2f}s" if ev.get("replay_time_s") is not None else "  ???s"
    line    = f"  [{t_str}] beat {ev['target_idx']}  {icon}  {err_str:>8}"
    if verbose and ev["messages"]:
        line += f"  {'  '.join(ev['messages'])}"
    print(line)


def _print_summary(stats: dict, data: dict) -> None:
    bpm = data["bpm"]
    print("\n── Replay summary ───────────────────────────────────")
    print(f"  BPM      : {bpm:.1f}  (nominal)")
    print(f"  Targets  : {stats['total']}")
    print(f"  Hits     : {stats['hits']}  "
          f"(good: {stats['good']}  warn: {stats['warn']})")
    print(f"  Misses   : {stats['misses']}")
    if stats["hits"] > 0:
        print(f"  Mean |error| : {stats['mean_abs_timing_s'] * 1000:.0f} ms")
    print("─────────────────────────────────────────────────────")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a session onset stream")
    p.add_argument("--file", required=True,
                   help="Path to session JSON file")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Include feedback messages in event lines")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text())

    print(f"File    : {path}")
    print(f"BPM     : {data['bpm']}")
    print(f"Targets : {len(data['targets'])}")
    print(f"Onsets  : {len(data['onsets'])}\n")

    events = replay_session_data(data)

    for ev in events:
        _print_event(ev, verbose=args.verbose)

    _print_summary(summarize_replay(events), data)


if __name__ == "__main__":
    main()
