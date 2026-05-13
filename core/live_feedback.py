"""Near-realtime target readiness logic.

Pure functions only — no audio hardware, no sounddevice.

Timing model
------------
beat_s            = 60 / bpm
count_in_seconds  = count_in_beats * beat_s
target_audio_time = count_in_seconds + target["time"] * beat_s

Analysis window (approximate)
------------------------------
window_start = target_audio_time + 0.02
window_end   = target_audio_time + min(WIN_END_MAX, WIN_END_FRAC * inter_target_gap_s)
ready_threshold = window_end + margin_s
"""

from __future__ import annotations

_WIN_START_OFFSET = 0.02
_WIN_END_MAX      = 0.35
_WIN_END_FRAC     = 0.6


# ── Internal helpers ──────────────────────────────────────────────────────────

def _beat_s(bpm: float) -> float:
    return 60.0 / bpm


def _count_in_s(count_in_beats: int, beat_s: float) -> float:
    return count_in_beats * beat_s


def _target_audio_time(target: dict, beat_s: float, count_in_s: float) -> float:
    return count_in_s + target["time"] * beat_s


def _inter_target_gap_s(targets: list[dict], idx: int, beat_s: float) -> float:
    """Seconds between targets[idx] and the next target.

    For the last target, reuses the previous inter-target gap.
    Falls back to beat_s when there is only one target.
    """
    if idx < len(targets) - 1:
        return (targets[idx + 1]["time"] - targets[idx]["time"]) * beat_s
    if len(targets) >= 2:
        return (targets[-1]["time"] - targets[-2]["time"]) * beat_s
    return beat_s


def _window_end(target_audio_time: float, gap_s: float) -> float:
    return target_audio_time + min(_WIN_END_MAX, _WIN_END_FRAC * gap_s)


def _ready_threshold(targets: list[dict], idx: int, beat_s: float,
                     count_in_s: float, margin_s: float) -> float:
    tat  = _target_audio_time(targets[idx], beat_s, count_in_s)
    gap  = _inter_target_gap_s(targets, idx, beat_s)
    return _window_end(tat, gap) + margin_s


# ── Public API ────────────────────────────────────────────────────────────────

def target_state(
    targets: list[dict],
    idx: int,
    bpm: float,
    count_in_beats: int,
    sample_rate: int,
    current_sample: int,
    evaluated: set[int],
    margin_s: float = 0.15,
) -> str:
    """Return 'pending', 'ready', or 'evaluated' for the target at idx.

    'evaluated' — already in the evaluated set; will not be returned again.
    'ready'     — analysis window has passed (current_time >= window_end + margin_s).
    'pending'   — window has not yet ended.
    """
    if idx in evaluated:
        return "evaluated"
    bs          = _beat_s(bpm)
    cis         = _count_in_s(count_in_beats, bs)
    threshold   = _ready_threshold(targets, idx, bs, cis, margin_s)
    current_t   = current_sample / sample_rate
    return "ready" if current_t >= threshold else "pending"


def ready_targets(
    targets: list[dict],
    bpm: float,
    count_in_beats: int,
    sample_rate: int,
    current_sample: int,
    evaluated_indices: set[int] | list[int],
    margin_s: float = 0.15,
) -> list[int]:
    """Return indices of targets whose analysis window has passed and are unevaluated.

    Results are in ascending index order.
    """
    if not targets:
        return []
    evaluated = set(evaluated_indices)
    bs        = _beat_s(bpm)
    cis       = _count_in_s(count_in_beats, bs)
    current_t = current_sample / sample_rate

    result = []
    for i in range(len(targets)):
        if i in evaluated:
            continue
        if current_t >= _ready_threshold(targets, i, bs, cis, margin_s):
            result.append(i)
    return result
