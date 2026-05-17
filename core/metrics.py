"""Practice-session statistics computed from SessionEngine feedback event streams.

Pure Python, stdlib only — no audio hardware, no file I/O, no sounddevice.

Input is any list of feedback event dicts as produced by
``core.feedback_events.feedback_event()`` and emitted by
``core.session_engine.SessionEngine``.  The replay harness in
``core.session_replay`` is the typical source for offline analysis.

Key distinction
---------------
``detected_note is None`` identifies a true no-show (the player did not
play anything for that target).  ``severity == "miss"`` is a broader
category that also includes detected notes that were very late (within the
match window but outside the 120 ms timing-miss threshold).  The metrics
layer tracks both signals separately rather than collapsing them.

Typical usage
-------------
    from core.metrics import compute_session_metrics, format_session_metrics

    events  = replay_session_data(data)
    metrics = compute_session_metrics(events)
    print(format_session_metrics(metrics, bpm=data["bpm"]))
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimingStats:
    """Statistics over events that have a non-None timing_error_s value."""

    count:            int    # number of events contributing to these stats
    mean_error_s:     float  # signed mean; positive = player runs late
    mean_abs_error_s: float  # average magnitude — the "how precise" number
    std_error_s:      float  # population std dev — the "how consistent" number
    max_abs_error_s:  float  # worst individual timing error (still matched)


@dataclass(frozen=True)
class PitchStats:
    """Statistics over events that have a non-None pitch_error_cents value.

    Currently None for all push-path sessions (OnsetAdapter does no pitch
    analysis).  Structure is ready for when pitch detection is added.
    """

    count:                int
    mean_error_cents:     float  # signed; positive = player runs sharp
    mean_abs_error_cents: float
    in_tune_rate:         float  # fraction where |error| <= 25 cents


@dataclass(frozen=True)
class StreakStats:
    """Consecutive-event streak lengths across the session."""

    longest_good_streak:  int  # most consecutive severity=="good" events
    longest_clean_streak: int  # most consecutive detected events (any severity)
    longest_miss_streak:  int  # most consecutive no-show (detected_note is None)
    current_good_streak:  int  # good streak at end of session


@dataclass(frozen=True)
class RollingWindow:
    """Metrics computed over one sliding window of events."""

    end_idx:           int          # index (in the full event list) of the window's last event
    hit_rate:          float        # detected / window_size
    mean_abs_timing_s: float | None # None if no detected events in window
    good_rate:         float | None # good_count / detected; None if no detected events


@dataclass(frozen=True)
class SessionMetrics:
    """Aggregated statistics for a complete practice session."""

    # ── Detection counts ──────────────────────────────────────────────────────
    total:      int    # total targets evaluated
    detected:   int    # targets where the player played something
    undetected: int    # targets where nothing was detected (true no-shows)
    hit_rate:   float  # detected / total; 0.0 when total == 0

    # ── Severity counts (across all events) ───────────────────────────────────
    good_count:     int  # severity == "good"
    warn_count:     int  # severity == "warn"
    miss_count:     int  # severity == "miss" (no-shows + very-late hits)
    late_hit_count: int  # detected but severity == "miss" (played but very late)

    # ── Severity rates (over detected events only) ────────────────────────────
    good_rate: float | None  # good_count / detected; None when detected == 0
    warn_rate: float | None  # warn_count / detected; None when detected == 0

    # ── Per-dimension statistics ──────────────────────────────────────────────
    timing: TimingStats | None  # None when no events have timing data
    pitch:  PitchStats  | None  # None when no events have pitch data

    # ── Streak analysis ───────────────────────────────────────────────────────
    streaks: StreakStats

    # ── Missed target positions ───────────────────────────────────────────────
    missed_indices: tuple[int, ...]  # target_idx of each no-show, in event order


# ── Public API ────────────────────────────────────────────────────────────────

def compute_session_metrics(events: list[dict]) -> SessionMetrics:
    """Compute final session statistics from a complete feedback event list.

    Parameters
    ----------
    events
        Feedback event dicts as produced by ``feedback_event()`` and optionally
        extended by ``SessionEngine`` (``onset_time_s``) or the replay harness
        (``replay_time_s``).  Extra keys are ignored.

    Returns
    -------
    SessionMetrics
        Frozen result; safe to cache or compare across runs.
    """
    if not events:
        return SessionMetrics(
            total=0, detected=0, undetected=0, hit_rate=0.0,
            good_count=0, warn_count=0, miss_count=0, late_hit_count=0,
            good_rate=None, warn_rate=None,
            timing=None, pitch=None,
            streaks=StreakStats(
                longest_good_streak=0, longest_clean_streak=0,
                longest_miss_streak=0, current_good_streak=0,
            ),
            missed_indices=(),
        )

    total      = len(events)
    detected   = sum(1 for e in events if e["detected_note"] is not None)
    undetected = total - detected
    hit_rate   = detected / total

    good_count     = sum(1 for e in events if e["severity"] == "good")
    warn_count     = sum(1 for e in events if e["severity"] == "warn")
    miss_count     = sum(1 for e in events if e["severity"] == "miss")
    late_hit_count = sum(
        1 for e in events
        if e["detected_note"] is not None and e["severity"] == "miss"
    )

    good_rate = good_count / detected if detected > 0 else None
    warn_rate = warn_count / detected if detected > 0 else None

    timing = _timing_stats(events)
    pitch  = _pitch_stats(events)
    streaks = _streak_stats(events)

    missed_indices = tuple(
        e["target_idx"] for e in events if e["detected_note"] is None
    )

    return SessionMetrics(
        total=total,
        detected=detected,
        undetected=undetected,
        hit_rate=hit_rate,
        good_count=good_count,
        warn_count=warn_count,
        miss_count=miss_count,
        late_hit_count=late_hit_count,
        good_rate=good_rate,
        warn_rate=warn_rate,
        timing=timing,
        pitch=pitch,
        streaks=streaks,
        missed_indices=missed_indices,
    )


def compute_rolling_metrics(
    events: list[dict],
    window: int = 8,
) -> list[RollingWindow]:
    """Compute rolling window metrics over an event sequence.

    Returns one :class:`RollingWindow` per event starting at index
    ``window - 1`` (the first full window).  The result list is empty when
    ``window > len(events)``.

    Parameters
    ----------
    events
        Feedback event list, ordered as emitted.
    window
        Number of events per window.

    Returns
    -------
    list[RollingWindow]
        One entry per event position from ``window - 1`` to ``len(events) - 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > len(events):
        return []

    results: list[RollingWindow] = []

    for end in range(window - 1, len(events)):
        win = events[end - window + 1 : end + 1]

        det = [e for e in win if e["detected_note"] is not None]
        hit_rate = len(det) / window

        t_abs = [abs(e["timing_error_s"]) for e in det if e.get("timing_error_s") is not None]
        mean_abs_timing = sum(t_abs) / len(t_abs) if t_abs else None

        good_in_win = sum(1 for e in win if e["severity"] == "good")
        good_rate = good_in_win / len(det) if det else None

        results.append(RollingWindow(
            end_idx=end,
            hit_rate=hit_rate,
            mean_abs_timing_s=mean_abs_timing,
            good_rate=good_rate,
        ))

    return results


def format_session_metrics(
    metrics: SessionMetrics,
    bpm: float | None = None,
) -> str:
    """Render a :class:`SessionMetrics` as a human-readable string.

    Parameters
    ----------
    metrics
        Result of :func:`compute_session_metrics`.
    bpm
        When provided, timing errors are annotated with their beat-fraction
        equivalent (e.g. "28 ms (0.056 beats)").

    Returns
    -------
    str
        Multi-line formatted summary.
    """
    lines: list[str] = []
    lines.append("── Session metrics ───────────────────────────────────")

    lines.append(f"  Targets   : {metrics.total}")
    lines.append(
        f"  Hit rate  : {metrics.hit_rate * 100:.1f}%"
        f"  ({metrics.detected} detected, {metrics.undetected} no-shows)"
    )

    if metrics.detected > 0:
        good_pct = (metrics.good_rate or 0.0) * 100
        warn_pct = (metrics.warn_rate or 0.0) * 100
        lines.append(f"  Good      : {metrics.good_count}  ({good_pct:.1f}% of hits)")
        lines.append(f"  Warn      : {metrics.warn_count}  ({warn_pct:.1f}% of hits)")
        if metrics.late_hit_count > 0:
            lines.append(f"  Late hits : {metrics.late_hit_count}  (detected but > 120 ms off)")

    if metrics.timing is not None:
        t = metrics.timing
        mean_ms  = t.mean_error_s     * 1000
        abs_ms   = t.mean_abs_error_s * 1000
        std_ms   = t.std_error_s      * 1000
        worst_ms = t.max_abs_error_s  * 1000
        beat_note = ""
        if bpm is not None and bpm > 0:
            beat_s     = 60.0 / bpm
            abs_beats  = t.mean_abs_error_s / beat_s
            beat_note  = f"  ({abs_beats:.3f} beats)"
        lines.append(
            f"  Timing    : mean {mean_ms:+.0f} ms"
            f"  |mean| {abs_ms:.0f} ms"
            f"  σ {std_ms:.0f} ms"
            f"  worst {worst_ms:.0f} ms"
            f"{beat_note}"
        )

    s = metrics.streaks
    lines.append(
        f"  Streaks   : best good run {s.longest_good_streak}"
        f"  /  longest miss run {s.longest_miss_streak}"
    )

    if metrics.missed_indices:
        idx_str = " ".join(str(i) for i in metrics.missed_indices)
        lines.append(f"  Missed    : target {idx_str}")

    if metrics.pitch is not None:
        p = metrics.pitch
        lines.append(
            f"  Pitch     : mean {p.mean_error_cents:+.0f} ¢"
            f"  |mean| {p.mean_abs_error_cents:.0f} ¢"
            f"  in tune {p.in_tune_rate * 100:.0f}%"
        )

    lines.append("─────────────────────────────────────────────────────")
    return "\n".join(lines)


# ── Private helpers ───────────────────────────────────────────────────────────

def _timing_stats(events: list[dict]) -> TimingStats | None:
    errors = [e["timing_error_s"] for e in events if e.get("timing_error_s") is not None]
    if not errors:
        return None
    n        = len(errors)
    mean_e   = sum(errors) / n
    abs_errs = [abs(e) for e in errors]
    mean_abs = sum(abs_errs) / n
    variance = sum((e - mean_e) ** 2 for e in errors) / n
    return TimingStats(
        count=n,
        mean_error_s=mean_e,
        mean_abs_error_s=mean_abs,
        std_error_s=math.sqrt(variance),
        max_abs_error_s=max(abs_errs),
    )


def _pitch_stats(events: list[dict]) -> PitchStats | None:
    errors = [e["pitch_error_cents"] for e in events if e.get("pitch_error_cents") is not None]
    if not errors:
        return None
    n         = len(errors)
    mean_p    = sum(errors) / n
    abs_errs  = [abs(e) for e in errors]
    mean_abs  = sum(abs_errs) / n
    in_tune   = sum(1 for e in abs_errs if e <= 25.0)  # matches _PITCH_GOOD threshold
    return PitchStats(
        count=n,
        mean_error_cents=mean_p,
        mean_abs_error_cents=mean_abs,
        in_tune_rate=in_tune / n,
    )


def _streak_stats(events: list[dict]) -> StreakStats:
    longest_good  = 0
    longest_clean = 0
    longest_miss  = 0
    cur_good  = 0
    cur_clean = 0
    cur_miss  = 0

    for e in events:
        is_detected = e["detected_note"] is not None
        is_good     = e["severity"] == "good"

        if is_detected:
            cur_miss   = 0
            cur_clean += 1
            longest_clean = max(longest_clean, cur_clean)
        else:
            cur_miss  += 1
            cur_clean  = 0
            longest_miss = max(longest_miss, cur_miss)

        if is_good:
            cur_good += 1
            longest_good = max(longest_good, cur_good)
        else:
            cur_good = 0

    return StreakStats(
        longest_good_streak=longest_good,
        longest_clean_streak=longest_clean,
        longest_miss_streak=longest_miss,
        current_good_streak=cur_good,
    )
