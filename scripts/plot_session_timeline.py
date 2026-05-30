#!/usr/bin/env python3
"""Offline timing diagnostic visualization for bass-trainer.

Generates a PNG showing target positions, onset positions, matched pairs,
and per-target timing error.  All evaluation logic is delegated to
core.realtime_evaluator.evaluate_targets(); nothing is duplicated here.

Usage
-----
Demo mode (built-in synthetic scenarios):
    python scripts/plot_session_timeline.py
    python scripts/plot_session_timeline.py --scenario perfect mixed
    python scripts/plot_session_timeline.py --output figs/diag.png

Session-file mode:
    python scripts/plot_session_timeline.py diagnostics/example.session.json
    python scripts/plot_session_timeline.py sessions/2026-05-26/foo.session.json -o out.png

The --tolerance-ms and --on-time-ms flags apply in both modes.  Because
evaluate_targets() is called fresh on the extracted arrays, the displayed
classifications may differ from the original log if a different tolerance
is used.  This is intentional — it lets you explore how matching policy
changes would affect the session outcome.

Extensions
----------
The plotting helpers (plot_timeline, plot_error_bars) are designed to be
called independently on a pre-created Axes pair.  Future additions (tempo
curve, rolling error, groove metrics) follow the same pattern: new function,
new Axes column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive; file output only
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime_evaluator import TargetEvaluation, evaluate_targets
from core.session_log import (
    EXTRA_ONSET,
    TARGET_HIT,
    TARGET_MISS,
    SessionLog,
    load_session_log_file,
)
from core.timing_policy import match_window_s


# ── Data model ────────────────────────────────────────────────────────────────

class TimingScenario(NamedTuple):
    """Inputs for one timeline row.

    A NamedTuple so it can be passed to build_figure() — which tuple-unpacks
    each row — without breaking callers that still use plain 4-tuples.
    """

    target_times_s: list[float]
    onset_times_s:  list[float]
    title:          str
    description:    str


class EvaluationSummary(NamedTuple):
    """Aggregate statistics for one evaluated scenario.

    Error fields (mean_signed_error_ms, mean_abs_error_ms, max_abs_error_ms)
    are computed over matched (non-miss) evaluations only.  All three are
    ``None`` when every target is a miss or the evaluation list is empty.
    """

    n_targets:            int
    n_on_time:            int
    n_early:              int
    n_late:               int
    n_miss:               int
    n_unmatched_onsets:   int
    mean_signed_error_ms: float | None
    mean_abs_error_ms:    float | None
    max_abs_error_ms:     float | None


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


# ── Built-in demo scenario builders ──────────────────────────────────────────
# Each returns a TimingScenario (target_times_s, onset_times_s, title, desc).

def _scenario_perfect(bpm: float = 120.0) -> TimingScenario:
    beat_s  = 60.0 / bpm
    targets = [i * beat_s for i in range(8)]
    return TimingScenario(targets, list(targets),
                          "Perfect timing", "8 beats, zero error on every hit")


def _scenario_gradual_drift(bpm: float = 120.0) -> TimingScenario:
    beat_s  = 60.0 / bpm
    drift   = 8.0              # ms per beat
    targets = [i * beat_s for i in range(8)]
    onsets  = [t + i * drift / 1000.0 for i, t in enumerate(targets)]
    return TimingScenario(targets, onsets,
                          "Gradual drift",
                          f"Accumulating +{drift:.0f} ms/beat (common in real playing)")


def _scenario_mixed(bpm: float = 120.0) -> TimingScenario:
    beat_s  = 60.0 / bpm
    offsets = [-50, -20, 0, +25, +55, +70, -65, +10]   # ms
    targets = [i * beat_s for i in range(len(offsets))]
    onsets  = [t + o / 1000.0 for t, o in zip(targets, offsets)]
    return TimingScenario(targets, onsets,
                          "Mixed early/late",
                          "Various offsets; classifications span all four categories")


def _scenario_misses(bpm: float = 120.0) -> TimingScenario:
    beat_s  = 60.0 / bpm
    n       = 8
    targets = [i * beat_s for i in range(n)]
    hit_idx = [0, 1, 3, 5]
    onsets  = [targets[i] + 0.015 for i in hit_idx]
    return TimingScenario(targets, onsets,
                          "Missed targets",
                          "Beats 0,1,3,5 played (15 ms late); beats 2,4,6,7 missed")


def _scenario_double_hit(bpm: float = 120.0) -> TimingScenario:
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
    return TimingScenario(
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


# ── Session-file extraction ───────────────────────────────────────────────────

def scenario_from_session_log(log: SessionLog) -> TimingScenario:
    """Extract timing data from a SessionLog into a TimingScenario.

    Extraction rules
    ----------------
    target_hit  : ``time_sec`` is the onset time; ``value`` is the timing
                  error in seconds (onset − expected).  Therefore:
                  ``expected_time = time_sec − value``
                  ``onset_time    = time_sec``

    target_miss : ``time_sec`` is the nominal (expected) target time.
                  No onset is produced.

    extra_onset : ``time_sec`` is an unmatched onset time.
                  No target is produced.

    Hit events with a None value field are skipped (malformed log).

    Targets are sorted by target_index before building the output list.
    Onset times are sorted ascending.

    Note on re-evaluation
    ---------------------
    The returned arrays are passed to evaluate_targets() inside build_figure().
    Because evaluate_targets() uses its own tolerance window, the displayed
    classifications may differ from the original session-log outcomes when
    different tolerances are configured.  This is intentional: the script is
    a diagnostic tool, not a faithful replay.
    """
    target_times: dict[int, float] = {}
    onset_times: list[float] = []

    for event in log.events:
        if event.event_type == TARGET_HIT:
            if event.value is not None and event.target_index is not None:
                target_times[event.target_index] = event.time_sec - event.value
                onset_times.append(event.time_sec)
            # Skip malformed hits (value=None) rather than crashing.
        elif event.event_type == TARGET_MISS:
            if event.target_index is not None:
                target_times[event.target_index] = event.time_sec
        elif event.event_type == EXTRA_ONSET:
            onset_times.append(event.time_sec)

    sorted_targets = [t for _, t in sorted(target_times.items())]

    # Build a human-readable label from whatever metadata the log carries.
    date_str = log.started_at[:10] if log.started_at else "unknown date"
    n_targets = log.metadata.get("beats", str(len(target_times)))
    bpm       = log.metadata.get("bpm", "")
    device    = log.metadata.get("device", "")

    title = f"Session {date_str}"
    parts = [f"{n_targets} targets"]
    if bpm:
        parts.append(f"{bpm} BPM")
    if device:
        parts.append(device)
    description = " · ".join(parts)

    return TimingScenario(
        target_times_s = sorted_targets,
        onset_times_s  = sorted(onset_times),
        title          = title,
        description    = description,
    )


def scenario_from_session_file(path: str | Path) -> TimingScenario:
    """Load a .session.json file and return a TimingScenario.

    Uses core.session_log.load_session_log_file() — no manual JSON parsing.
    """
    return scenario_from_session_log(load_session_log_file(path))


def session_tolerance_s(log: SessionLog) -> float | None:
    """Return the best available match tolerance for *log*, or ``None``.

    Resolution order (first valid value wins):

    1. ``log.metadata["match_window_s"]`` — the exact window persisted at
       session-creation time.
    2. ``match_window_s(log.metadata["bpm"])`` — BPM-derived fallback for
       older logs that pre-date the ``match_window_s`` metadata key.

    Returns ``None`` without raising when neither key yields a valid positive
    number.
    """
    # 1. Explicit match_window_s stored at session creation.
    raw_mw = log.metadata.get("match_window_s", "")
    if raw_mw:
        try:
            w = float(raw_mw)
            if w > 0:
                return w
        except (ValueError, TypeError):
            pass

    # 2. BPM-derived fallback.
    raw_bpm = log.metadata.get("bpm", "")
    if not raw_bpm:
        return None
    try:
        bpm = float(raw_bpm)
    except (ValueError, TypeError):
        return None
    if bpm <= 0:
        return None
    return match_window_s(bpm)


# ── Evaluation summary ────────────────────────────────────────────────────────

def summarize_evaluations(
    evals: list[TargetEvaluation],
    onset_times_s: list[float],
) -> EvaluationSummary:
    """Compute aggregate statistics over a list of TargetEvaluation results.

    Parameters
    ----------
    evals
        Output of ``evaluate_targets()``.
    onset_times_s
        The original onset list passed to ``evaluate_targets()``.  Used to
        count unmatched (extra) onsets: any onset whose index does not appear
        in the ``matched_onset_index`` fields of *evals*.

    Notes
    -----
    Error statistics (mean_signed_error_ms, mean_abs_error_ms,
    max_abs_error_ms) are computed over matched (non-miss) targets only.
    They are ``None`` when all targets are misses or *evals* is empty.
    """
    n_on_time = sum(1 for e in evals if e.classification == "on_time")
    n_early   = sum(1 for e in evals if e.classification == "early")
    n_late    = sum(1 for e in evals if e.classification == "late")
    n_miss    = sum(1 for e in evals if e.classification == "miss")

    consumed = {e.matched_onset_index for e in evals if e.matched_onset_index is not None}
    n_unmatched = len(onset_times_s) - len(consumed)

    errors = [e.signed_error_ms for e in evals if e.signed_error_ms is not None]
    if errors:
        mean_signed = sum(errors) / len(errors)
        abs_errors  = [abs(e) for e in errors]
        mean_abs    = sum(abs_errors) / len(abs_errors)
        max_abs     = max(abs_errors)
    else:
        mean_signed = mean_abs = max_abs = None

    return EvaluationSummary(
        n_targets            = len(evals),
        n_on_time            = n_on_time,
        n_early              = n_early,
        n_late               = n_late,
        n_miss               = n_miss,
        n_unmatched_onsets   = n_unmatched,
        mean_signed_error_ms = mean_signed,
        mean_abs_error_ms    = mean_abs,
        max_abs_error_ms     = max_abs,
    )


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

    The horizontal displacement of each connecting line encodes timing error
    direction: leftward lean = early, rightward = late.
    """
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([_Y_ONSET, _Y_TARGET])
    ax.set_yticklabels(["onsets", "targets"], fontsize=9)
    ax.axhline(_Y_TARGET, color="#cccccc", lw=0.6, zorder=0)
    ax.axhline(_Y_ONSET,  color="#cccccc", lw=0.6, zorder=0)

    label = title + (f"\n{description}" if description else "")
    ax.set_title(label, fontsize=9, loc="left", pad=3)

    for t in target_times_s:
        ax.axvspan(t - tolerance_s, t + tolerance_s,
                   alpha=0.045, color="#888888", zorder=0)
        ax.axvspan(t - on_time_s, t + on_time_s,
                   alpha=0.08, color="#2ecc71", zorder=0)

    for ev in evaluations:
        if ev.classification != "miss":
            ax.plot(
                [ev.expected_time_s, ev.actual_time_s],
                [_Y_TARGET, _Y_ONSET],
                color=_COLORS[ev.classification], lw=1.6, alpha=0.75, zorder=1,
            )

    for ev in evaluations:
        if ev.classification == "miss":
            ax.plot(ev.expected_time_s, _Y_TARGET, "x",
                    color=_COLORS["miss"], ms=11, mew=2.2, zorder=3)
        else:
            ax.plot(ev.expected_time_s, _Y_TARGET, "^",
                    color=_COLORS[ev.classification], ms=9, zorder=3)

    consumed = {ev.matched_onset_index for ev in evaluations
                if ev.matched_onset_index is not None}
    for i, t in enumerate(onset_times_s):
        if i in consumed:
            ev    = next(e for e in evaluations if e.matched_onset_index == i)
            color = _COLORS[ev.classification]
        else:
            color = "#aaaaaa"
        ax.plot(t, _Y_ONSET, "o", color=color, ms=7, zorder=3)

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


# ── Plotting: summary text block ─────────────────────────────────────────────

def plot_summary(
    ax: plt.Axes,
    summary: EvaluationSummary,
) -> None:
    """Render an EvaluationSummary as a fixed-width text block on *ax*.

    The axes frame and ticks are hidden.  Signed mean error uses a ±
    prefix so direction is clear at a glance; absolute stats use plain
    decimal notation.
    """
    ax.axis("off")

    def _s(v: float | None) -> str:
        return f"{v:+.1f}" if v is not None else "—"

    def _a(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "—"

    lines = [
        f"{'Targets:':<10} {summary.n_targets:>4}",
        f"{'On time:':<10} {summary.n_on_time:>4}",
        f"{'Early:':<10} {summary.n_early:>4}",
        f"{'Late:':<10} {summary.n_late:>4}",
        f"{'Miss:':<10} {summary.n_miss:>4}",
        f"{'Extra:':<10} {summary.n_unmatched_onsets:>4}",
        "",
        f"{'Mean err:':<10} {_s(summary.mean_signed_error_ms):>7} ms",
        f"{'MAE:':<10} {_a(summary.mean_abs_error_ms):>7} ms",
        f"{'Max |e|:':<10} {_a(summary.max_abs_error_ms):>7} ms",
    ]

    ax.text(
        0.08, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  alpha=0.8, edgecolor="#cccccc", lw=0.8),
    )


# ── Figure assembly ───────────────────────────────────────────────────────────

def build_figure(
    scenario_rows: list[TimingScenario],
    *,
    tolerance_s: float = _DEFAULT_TOLERANCE_S,
    on_time_s: float = _DEFAULT_ON_TIME_S,
) -> plt.Figure:
    """Build a multi-scenario diagnostic figure.

    Parameters
    ----------
    scenario_rows
        List of TimingScenario (or plain 4-tuples).
        evaluate_targets() is called here — callers provide raw arrays.
    tolerance_s
        Match tolerance forwarded to evaluate_targets and the plotters.
    on_time_s
        On-time threshold forwarded to evaluate_targets and both plotters.

    Figure layout
    -------------
    One row per scenario:
      [0] Timeline   (3× wide) — two-lane target/onset view
      [1] Error bars (1× wide) — signed error in ms per target
      [2] Summary    (1× wide) — aggregate statistics text block
    """
    n = len(scenario_rows)
    fig, axes = plt.subplots(
        n, 3,
        figsize=(16, 3.0 * n),
        gridspec_kw={"width_ratios": [3, 1, 1]},
    )
    if n == 1:
        axes = [axes]

    fig.suptitle("Bass Trainer — Timing Evaluation Diagnostic",
                 fontsize=12, fontweight="bold", y=1.01)

    for row_axes, (targets, onsets, title, desc) in zip(axes, scenario_rows):
        ax_tl, ax_err, ax_sum = row_axes

        evals = evaluate_targets(
            targets, onsets,
            tolerance_s=tolerance_s,
            on_time_threshold_s=on_time_s,
        )

        plot_timeline(ax_tl, targets, onsets, evals,
                      tolerance_s=tolerance_s, on_time_s=on_time_s,
                      title=title, description=desc)
        plot_error_bars(ax_err, evals, on_time_s=on_time_s)
        plot_summary(ax_sum, summarize_evaluations(evals, onsets))

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


# ── File output ──────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    """Save *fig* to *output_path*, creating parent directories as needed.

    Parameters
    ----------
    fig
        The matplotlib figure to save.
    output_path
        Destination file path.  The format is inferred from the extension
        (``".png"``, ``".pdf"``, etc.).

    Returns
    -------
    Path
        Resolved absolute path of the saved file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out.resolve()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv
        Argument list.  When None (default), sys.argv[1:] is used — the
        standard argparse behaviour.  Pass an explicit list for testing.
    """
    p = argparse.ArgumentParser(
        description="Offline timing diagnostic visualization for bass-trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "file",
        nargs="?",
        default=None,
        metavar="FILE.session.json",
        help="Session JSON file to visualize.  Omit to plot built-in scenarios.",
    )
    p.add_argument(
        "--scenario", nargs="+",
        choices=list(_ALL_SCENARIOS) + ["all"],
        default=["all"], metavar="NAME",
        help=(
            "Demo scenarios to plot (ignored when FILE is given): "
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
        default=None, dest="tolerance_ms",
        help=(
            "Match tolerance in ms.  When omitted in session-file mode, "
            "defaults to half a beat derived from the session BPM "
            f"(or {_DEFAULT_TOLERANCE_S * 1000:.0f} ms when BPM is unavailable).  "
            "In demo mode defaults to "
            f"{_DEFAULT_TOLERANCE_S * 1000:.0f} ms."
        ),
    )
    p.add_argument(
        "--on-time-ms", type=float,
        default=_DEFAULT_ON_TIME_S * 1000, dest="on_time_ms",
        help="On-time classification threshold in ms",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    on_time_s = args.on_time_ms / 1000.0

    if args.file is not None:
        # ── Session-file mode ──────────────────────────────────────────────
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        log  = load_session_log_file(path)
        rows = [scenario_from_session_log(log)]

        if args.tolerance_ms is not None:
            tolerance_s = args.tolerance_ms / 1000.0
        else:
            tol = session_tolerance_s(log)
            tolerance_s = tol if tol is not None else _DEFAULT_TOLERANCE_S
    else:
        # ── Demo mode ─────────────────────────────────────────────────────
        keys = list(_ALL_SCENARIOS) if "all" in args.scenario else args.scenario
        rows = [_ALL_SCENARIOS[k]() for k in keys]
        tolerance_s = (
            args.tolerance_ms / 1000.0
            if args.tolerance_ms is not None
            else _DEFAULT_TOLERANCE_S
        )

    fig = build_figure(rows, tolerance_s=tolerance_s, on_time_s=on_time_s)
    out = save_figure(fig, args.output)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
