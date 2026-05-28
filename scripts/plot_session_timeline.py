#!/usr/bin/env python3
"""Offline timing diagnostic visualization for bass-trainer.

Generates a PNG showing target positions, onset positions, matched pairs,
and per-target timing error for one or more hardcoded scenarios.  All
evaluation logic is delegated to core.realtime_evaluator.evaluate_targets();
nothing is duplicated here.

Run from the project root:
    python scripts/plot_session_timeline.py
    python scripts/plot_session_timeline.py --output figs/diag.png
    python scripts/plot_session_timeline.py --scenario perfect mixed
    python scripts/plot_session_timeline.py --tolerance-ms 100 --on-time-ms 20

Output: a single PNG file (default: diagnostic_timeline.png).

Extensions
----------
The plotting helpers (plot_timeline, plot_error_bars) are designed to be called
independently on a pre-created Axes pair.  Future additions (tempo curve, rolling
error, groove metrics) follow the same pattern: new function, new Axes column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # non-interactive; file output only
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime_evaluator import TargetEvaluation, evaluate_targets


# ── Visual constants ──────────────────────────────────────────────────────────

_COLORS: dict[str, str] = {
    "on_time": "#2ecc71",   # green
    "early":   "#3498db",   # blue
    "late":    "#e67e22",   # orange
    "miss":    "#e74c3c",   # red
}

_DEFAULT_TOLERANCE_S = 0.08   # 80 ms
_DEFAULT_ON_TIME_S   = 0.03   # 30 ms

_Y_TARGET = 1.0               # vertical position of the target lane
_Y_ONSET  = 0.0               # vertical position of the onset lane


# ── Scenario builders ─────────────────────────────────────────────────────────
# Each function returns (target_times_s, onset_times_s, title, description).
# Scenarios use direct float arrays — no audio, no session files.

def _scenario_perfect(bpm: float = 120.0) -> tuple:
    """Eight beats, all onsets exactly on time."""
    beat_s  = 60.0 / bpm
    targets = [i * beat_s for i in range(8)]
    return targets, list(targets), "Perfect timing", "8 beats, zero error on every hit"


def _scenario_gradual_drift(bpm: float = 120.0) -> tuple:
    """Onset drifts progressively later by 8 ms per beat."""
    beat_s  = 60.0 / bpm
    drift   = 8.0              # ms per beat
    targets = [i * beat_s for i in range(8)]
    onsets  = [t + i * drift / 1000.0 for i, t in enumerate(targets)]
    return targets, onsets, "Gradual drift", f"Accumulating +{drift:.0f} ms/beat (common in real playing)"


def _scenario_mixed(bpm: float = 120.0) -> tuple:
    """Mix of early, late, on_time, and a miss at the end."""
    beat_s  = 60.0 / bpm
    offsets = [-50, -20, 0, +25, +55, +70, -65, +10]   # ms
    targets = [i * beat_s for i in range(len(offsets))]
    onsets  = [t + o / 1000.0 for t, o in zip(targets, offsets)]
    return targets, onsets, "Mixed early/late", "Various offsets; classifications span all four categories"


def _scenario_misses(bpm: float = 120.0) -> tuple:
    """Only beats 0, 1, 3, 5 are played; the rest are misses."""
    beat_s  = 60.0 / bpm
    n       = 8
    targets = [i * beat_s for i in range(n)]
    hit_idx = [0, 1, 3, 5]
    onsets  = [targets[i] + 0.015 for i in hit_idx]
    return targets, onsets, "Missed targets", "Beats 0,1,3,5 played (15 ms late); beats 2,4,6,7 missed"


def _scenario_double_hit(bpm: float = 120.0) -> tuple:
    """Beat 1 has two nearby onsets (double-hit); beat 3 arrives outside the window."""
    beat_s  = 60.0 / bpm
    targets = [i * beat_s for i in range(5)]
    onsets  = [
        targets[0] + 0.005,    # beat 0: on time
        targets[1] - 0.025,    # beat 1: first hit (25 ms early)
        targets[1] + 0.015,    # beat 1: second hit (15 ms late) — nearer → matched
        targets[2] + 0.000,    # beat 2: exact
        targets[3] + 0.120,    # beat 3: 120 ms late → outside 80 ms window → miss
        targets[4] + 0.010,    # beat 4: on time
    ]
    return (
        targets, onsets,
        "Double-hit & miss",
        "Beat 1: two onsets, nearest matched; beat 3: outside tolerance window → miss",
    )


_ALL_SCENARIOS: dict[str, object] = {
    "perfect":       _scenario_perfect,
    "gradual_drift": _scenario_gradual_drift,
    "mixed":         _scenario_mixed,
    "misses":        _scenario_misses,
    "double_hit":    _scenario_double_hit,
}


# ── Plotting: two-lane timeline ───────────────────────────────────────────────

def plot_timeline(
    ax: plt.Axes,
    target_times_s: list[float],
    onset_times_s: list[float],
    evaluations: list[TargetEvaluation],
    *,
    tolerance_s: float = _DEFAULT_TOLERANCE_S,
    on_time_s: float = _DEFAULT_ON_TIME_S,
    title: str = "",
    description: str = "",
) -> None:
    """Draw the target/onset timeline on *ax*.

    Two horizontal lanes
    --------------------
    y=1  targets — expected beat positions
    y=0  onsets  — detected onset positions

    Visual encodings
    ----------------
    Grey band      : ±tolerance_s acceptance window around each target.
    Green band     : ±on_time_s on-time zone (subset of grey).
    Coloured line  : connects each matched target to its onset.
    Triangle (^)   : matched target, coloured by classification.
    Cross (x)      : missed target, red.
    Circle (o)     : onset; coloured if matched, grey if unmatched.

    The horizontal displacement of each connecting line directly encodes
    the signed timing error — leftward lean = early, rightward = late.
    """
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([_Y_ONSET, _Y_TARGET])
    ax.set_yticklabels(["onsets", "targets"], fontsize=9)
    ax.axhline(_Y_TARGET, color="#cccccc", lw=0.6, zorder=0)
    ax.axhline(_Y_ONSET,  color="#cccccc", lw=0.6, zorder=0)

    label = title + (f"\n{description}" if description else "")
    ax.set_title(label, fontsize=9, loc="left", pad=3)

    # Background bands around each target
    for t in target_times_s:
        ax.axvspan(t - tolerance_s, t + tolerance_s,
                   alpha=0.045, color="#888888", zorder=0)
        ax.axvspan(t - on_time_s, t + on_time_s,
                   alpha=0.08, color="#2ecc71", zorder=0)

    # Match lines (drawn first so markers sit on top)
    for ev in evaluations:
        if ev.classification != "miss":
            ax.plot(
                [ev.expected_time_s, ev.actual_time_s],
                [_Y_TARGET, _Y_ONSET],
                color=_COLORS[ev.classification], lw=1.6, alpha=0.75, zorder=1,
            )

    # Target markers
    for ev in evaluations:
        if ev.classification == "miss":
            ax.plot(ev.expected_time_s, _Y_TARGET, "x",
                    color=_COLORS["miss"], ms=11, mew=2.2, zorder=3)
        else:
            ax.plot(ev.expected_time_s, _Y_TARGET, "^",
                    color=_COLORS[ev.classification], ms=9, zorder=3)

    # Onset markers — coloured by the target they matched, grey if unmatched
    consumed = {ev.matched_onset_index for ev in evaluations
                if ev.matched_onset_index is not None}
    for i, t in enumerate(onset_times_s):
        if i in consumed:
            ev    = next(e for e in evaluations if e.matched_onset_index == i)
            color = _COLORS[ev.classification]
        else:
            color = "#aaaaaa"
        ax.plot(t, _Y_ONSET, "o", color=color, ms=7, zorder=3)

    # Hit-rate summary in top-right corner
    n_hit = sum(1 for e in evaluations if e.classification != "miss")
    n_tot = len(evaluations)
    pct   = 100 * n_hit // max(n_tot, 1)
    ax.text(0.99, 0.97, f"{n_hit}/{n_tot}  ({pct}%)",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color="#555555")

    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plotting: error bar chart ─────────────────────────────────────────────────

def plot_error_bars(
    ax: plt.Axes,
    evaluations: list[TargetEvaluation],
    *,
    on_time_s: float = _DEFAULT_ON_TIME_S,
) -> None:
    """Draw signed timing error (ms) per target index on *ax*.

    Bars are coloured by classification.  Missed targets show a red ✗ at
    zero.  Dashed green lines mark the ±on_time threshold.
    """
    thr_ms = on_time_s * 1000.0
    ax.axhline(0,        color="#999999", lw=0.8, zorder=0)
    ax.axhline(+thr_ms,  color="#2ecc71", lw=0.8, ls="--", zorder=0)
    ax.axhline(-thr_ms,  color="#2ecc71", lw=0.8, ls="--", zorder=0)

    for ev in evaluations:
        if ev.classification == "miss":
            ax.text(ev.target_index, 0, "✗",
                    ha="center", va="center", fontsize=10,
                    color=_COLORS["miss"], zorder=3)
        else:
            ax.bar(ev.target_index, ev.signed_error_ms,
                   color=_COLORS[ev.classification], alpha=0.85,
                   width=0.7, zorder=2)

    idxs = [ev.target_index for ev in evaluations]
    if idxs:
        ax.set_xlim(min(idxs) - 0.5, max(idxs) + 0.5)
    ax.set_xlabel("target #", fontsize=8)
    ax.set_ylabel("error (ms)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure assembly ───────────────────────────────────────────────────────────

def build_figure(
    scenario_rows: list[tuple[list, list, str, str]],
    *,
    tolerance_s: float = _DEFAULT_TOLERANCE_S,
    on_time_s: float = _DEFAULT_ON_TIME_S,
) -> plt.Figure:
    """Build a multi-scenario diagnostic figure.

    Parameters
    ----------
    scenario_rows
        List of ``(target_times_s, onset_times_s, title, description)`` tuples.
        :func:`~core.realtime_evaluator.evaluate_targets` is called here —
        callers provide raw time arrays, not pre-computed evaluations.
    tolerance_s
        Match tolerance forwarded to evaluate_targets and the timeline plotter.
    on_time_s
        On-time threshold forwarded to evaluate_targets and both plotters.

    Returns
    -------
    plt.Figure
        Fully rendered figure, ready for savefig().

    Figure layout
    -------------
    One row per scenario.  Each row has two columns:
      [0] Timeline  (3× wide) — two-lane target/onset view
      [1] Error bars (1× wide) — signed error in ms per target
    """
    n = len(scenario_rows)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(15, 3.0 * n),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if n == 1:
        axes = [axes]   # normalise shape

    fig.suptitle("Bass Trainer — Timing Evaluation Diagnostic",
                 fontsize=12, fontweight="bold", y=1.01)

    for row_axes, (targets, onsets, title, desc) in zip(axes, scenario_rows):
        ax_tl, ax_err = row_axes

        evals = evaluate_targets(
            targets, onsets,
            tolerance_s=tolerance_s,
            on_time_threshold_s=on_time_s,
        )

        plot_timeline(ax_tl, targets, onsets, evals,
                      tolerance_s=tolerance_s, on_time_s=on_time_s,
                      title=title, description=desc)
        plot_error_bars(ax_err, evals, on_time_s=on_time_s)

    # Shared legend below all rows
    handles = [
        mpatches.Patch(color=_COLORS[k], label=k.replace("_", " "))
        for k in ("on_time", "early", "late", "miss")
    ]
    handles.append(
        plt.Line2D([0], [0], color="#aaaaaa", marker="o", linestyle="none",
                   markersize=6, label="unmatched onset")
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline timing diagnostic visualization for bass-trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario", nargs="+",
        choices=list(_ALL_SCENARIOS) + ["all"],
        default=["all"], metavar="NAME",
        help=(
            "Scenarios to plot: "
            + ", ".join(_ALL_SCENARIOS)
            + ", all"
        ),
    )
    p.add_argument(
        "--output", "-o",
        default="diagnostic_timeline.png",
        help="Output PNG path",
    )
    p.add_argument(
        "--tolerance-ms", type=float,
        default=_DEFAULT_TOLERANCE_S * 1000, dest="tolerance_ms",
        help="Match tolerance in ms",
    )
    p.add_argument(
        "--on-time-ms", type=float,
        default=_DEFAULT_ON_TIME_S * 1000, dest="on_time_ms",
        help="On-time classification threshold in ms",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    keys = list(_ALL_SCENARIOS) if "all" in args.scenario else args.scenario
    rows = [_ALL_SCENARIOS[k]() for k in keys]

    fig = build_figure(
        rows,
        tolerance_s=args.tolerance_ms / 1000.0,
        on_time_s=args.on_time_ms / 1000.0,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
