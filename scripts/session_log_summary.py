#!/usr/bin/env python3
"""Print a human-readable summary of a .session.json file.

Reads an existing session log, computes additional diagnostics beyond what
log_metrics provides, and prints them in a structured format.

Run from the project root:
    python scripts/session_log_summary.py sessions/2026-05-25/foo.session.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.log_metrics import compute_log_metrics
from core.session_log import (
    TARGET_HIT,
    TARGET_MISS,
    SessionLog,
    load_session_log_file,
)
from core.severity_policy import event_timing_severity


# ── Pure summary helpers (importable for tests) ────────────────────────────────

def hit_errors_ordered(log: SessionLog) -> list[float]:
    """Return TARGET_HIT timing errors (seconds) sorted by target_index.

    Events with ``value=None`` or ``target_index=None`` are skipped.
    The result is ordered by target_index so consecutive-run analysis
    reflects the beat sequence, not event-arrival order.
    """
    hits = [
        (e.target_index, e.value)
        for e in log.events
        if e.event_type == TARGET_HIT
        and e.value is not None
        and e.target_index is not None
    ]
    hits.sort(key=lambda x: x[0])
    return [err for _, err in hits]


def split_half_means(
    errors: list[float],
) -> tuple[float | None, float | None]:
    """Split *errors* at the midpoint and return (first_mean, second_mean).

    The split is ``errors[:n//2]`` / ``errors[n//2:]``.  Either mean is
    ``None`` when the corresponding slice is empty (n < 2).
    """
    n = len(errors)
    if n == 0:
        return None, None
    mid    = n // 2
    first  = errors[:mid]
    second = errors[mid:]
    mean_f = sum(first)  / len(first)  if first  else None
    mean_s = sum(second) / len(second) if second else None
    return mean_f, mean_s


def longest_run(errors: list[float], predicate) -> int:
    """Return the length of the longest consecutive run where *predicate* is True.

    Parameters
    ----------
    errors    : list of timing-error floats, in beat order.
    predicate : callable(float) → bool.

    Returns
    -------
    int — 0 when *errors* is empty or no element satisfies *predicate*.
    """
    max_run = 0
    current = 0
    for v in errors:
        if predicate(v):
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def missed_target_indices(log: SessionLog) -> list[int]:
    """Return a sorted list of target_index values for all TARGET_MISS events.

    Events with ``target_index=None`` are excluded.
    """
    return sorted(
        e.target_index
        for e in log.events
        if e.event_type == TARGET_MISS and e.target_index is not None
    )


def severity_counts(log: SessionLog) -> dict[str, int]:
    """Count TARGET_HIT events by their timing severity label.

    Returns dict with keys ``"good"``, ``"warn"``, ``"miss"`` and integer
    values.  Only TARGET_HIT events are classified; TARGET_MISS and
    EXTRA_ONSET events are excluded.
    """
    counts: dict[str, int] = {"good": 0, "warn": 0, "miss": 0}
    for ev in log.events:
        if ev.event_type == TARGET_HIT:
            sev = event_timing_severity(ev)
            if sev in counts:
                counts[sev] += 1
    return counts


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"({n / total * 100:.1f}%)" if total > 0 else ""


def _ms(value_s: float | None, sign: bool = False) -> str:
    if value_s is None:
        return "—"
    ms = value_s * 1000.0
    return f"{ms:+.1f} ms" if sign else f"{ms:.1f} ms"


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("─" * len(title))


def _print_summary(log: SessionLog, path: Path) -> None:
    m      = compute_log_metrics(log)
    errors = hit_errors_ordered(log)
    first_mean, second_mean = split_half_means(errors)
    late_run  = longest_run(errors, lambda e: e > 0)
    early_run = longest_run(errors, lambda e: e < 0)
    missed    = missed_target_indices(log)
    sev       = severity_counts(log)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"File    : {path.name}")
    if log.started_at:
        print(f"Started : {log.started_at}")
    if log.ended_at:
        print(f"Ended   : {log.ended_at}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    if log.metadata:
        _print_section("Metadata")
        key_w = max(len(k) for k in log.metadata) if log.metadata else 8
        for k, v in log.metadata.items():
            print(f"  {k:<{key_w}} : {v}")

    # ── Counts ────────────────────────────────────────────────────────────────
    _print_section("Counts")
    total = m.targets_total
    print(f"  targets : {total}")
    print(f"  hits    : {m.targets_hit} / {total}  {_pct(m.targets_hit, total)}")
    print(f"  misses  : {m.targets_missed}")
    if m.extra_onsets:
        print(f"  extras  : {m.extra_onsets}")

    # ── Severity ──────────────────────────────────────────────────────────────
    n_hit = m.targets_hit
    _print_section(f"Severity  (of {n_hit} hits)")
    print(f"  good      : {sev['good']:>3}  {_pct(sev['good'], n_hit)}")
    print(f"  warn      : {sev['warn']:>3}  {_pct(sev['warn'], n_hit)}")
    print(f"  miss-tier : {sev['miss']:>3}  {_pct(sev['miss'], n_hit)}")

    # ── Timing errors ─────────────────────────────────────────────────────────
    _print_section("Timing errors  (hits only)")
    print(f"  mean signed  : {_ms(m.mean_signed_error_s, sign=True)}")
    print(f"  mean |err|   : {_ms(m.mean_abs_error_s)}")

    n = len(errors)
    if n >= 2:
        mid = n // 2
        print(
            f"  first half   : {_ms(first_mean, sign=True)}"
            f"  ({mid} hit{'s' if mid != 1 else ''})"
        )
        print(
            f"  second half  : {_ms(second_mean, sign=True)}"
            f"  ({n - mid} hit{'s' if n - mid != 1 else ''})"
        )
    elif n == 1:
        print("  (only 1 hit — no half split)")

    # ── Runs ──────────────────────────────────────────────────────────────────
    _print_section("Consecutive runs  (in beat order)")
    print(f"  longest late run  : {late_run}")
    print(f"  longest early run : {early_run}")

    # ── Missed beats ──────────────────────────────────────────────────────────
    _print_section("Missed beat indices")
    if missed:
        print("  " + "  ".join(str(i) for i in missed))
    else:
        print("  (none)")

    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise a .session.json file")
    p.add_argument("session_file", help="Path to the .session.json file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.session_file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        log = load_session_log_file(path)
    except Exception as exc:
        print(f"Error loading session log: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(log, path)


if __name__ == "__main__":
    main()
