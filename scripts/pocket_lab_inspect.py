#!/usr/bin/env python3
"""Beat Microscope — inspect bass timing against a shuffle-aware beat grid.

Loads a stereo WAV, extracts the bass channel, detects onsets in a selected
time segment, and generates an interactive HTML diagnostic report with
synchronized audio playback.

Usage
-----
    python scripts/pocket_lab_inspect.py recording.wav --bpm 146
    python scripts/pocket_lab_inspect.py recording.wav --bpm 120 --beats-per-measure 3
    python scripts/pocket_lab_inspect.py recording.wav --bpm 146 --start 4.0 --duration 16
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ── Pure helpers (importable for tests) ───────────────────────────────────────


def load_audio(wav_path: str | Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file and return (samples, sample_rate).

    Returns the raw multi-channel array as-is (shape may be (N,) for mono
    or (N, channels) for multi-channel).
    """
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32")
    return data, sr


def segment_audio(
    audio: np.ndarray,
    sr: int,
    start: float,
    duration: float,
) -> np.ndarray:
    """Extract a time segment from a 1-D or 2-D audio array."""
    n = audio.shape[0]
    s0 = int(start * sr)
    s1 = int((start + duration) * sr)
    s1 = min(s1, n)
    s0 = max(0, min(s0, n))
    return audio[s0:s1]


def novelty_envelope(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute onset-strength envelope via librosa.

    Returns (times, envelope) arrays of matching length.
    """
    import librosa

    env = librosa.onset.onset_strength(
        y=np.asarray(audio, dtype=np.float32), sr=sr,
    )
    times = librosa.times_like(env, sr=sr)
    return times, env


def detect_onsets(
    audio: np.ndarray,
    sr: int,
    delta: float = 0.07,
) -> np.ndarray:
    """Detect onset times (seconds) using librosa spectral-flux."""
    import librosa

    return librosa.onset.onset_detect(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        backtrack=True,
        delta=delta,
        units="time",
    )


def audio_diagnostics(audio: np.ndarray, label: str = "") -> dict:
    """Compute RMS, peak, and channel info for an audio array."""
    if audio.size == 0:
        return {"label": label, "channels": 0, "rms_dbfs": None, "peak_dbfs": None}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    channels = audio.shape[1] if audio.ndim == 2 else 1
    return {
        "label": label,
        "channels": channels,
        "rms_dbfs": 20.0 * np.log10(rms) if rms > 0 else -120.0,
        "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
    }


def compute_overview(audio_1d: np.ndarray, sr: int, n_points: int = 900) -> dict:
    """Downsample 1-D audio to a compact envelope for overview display.

    Returns dict with keys maxes, mins (arrays of length n_points),
    and total_duration_s.
    """
    n = len(audio_1d)
    total_dur = n / sr if sr > 0 else 0.0
    if n == 0:
        return {"maxes": np.array([]), "mins": np.array([]),
                "total_duration_s": 0.0}
    bucket = max(1, n // n_points)
    nb = n // bucket
    trimmed = audio_1d[: nb * bucket].reshape(nb, bucket)
    return {
        "maxes": np.max(trimmed, axis=1),
        "mins": np.min(trimmed, axis=1),
        "total_duration_s": total_dur,
    }


def window_tag(start: float) -> str:
    """Filesystem-safe tag for a window start time."""
    if start == int(start):
        return f"w{int(start)}"
    return f"w{start:.1f}"


@dataclass
class GridLine:
    """A single line in the beat grid (measure, beat, or subdivision)."""
    time: float
    kind: str       # "measure", "beat", or "subdivision"
    measure: int    # 1-based
    beat: int       # 1-based within measure
    subdivision: int  # 0 = on-beat, 1 = shuffle subdivision


def make_grid(
    bpm: float,
    beats_per_measure: int,
    n_measures: int,
    shuffle_fraction: float = 2 / 3,
    offset: float = 0.0,
) -> List[GridLine]:
    """Build a beat grid with measure/beat/subdivision lines.

    Parameters
    ----------
    bpm               : tempo in beats per minute.
    beats_per_measure  : beats in each measure (e.g. 4 for 4/4, 3 for 3/4).
    n_measures         : number of measures to generate.
    shuffle_fraction   : position of the shuffle subdivision within each beat
                         (0.667 = triplet swing). Set to 0.5 for straight.
    offset             : time offset in seconds for the first beat.

    Returns a list of GridLine sorted by time.
    """
    beat_s = 60.0 / bpm
    lines: List[GridLine] = []

    for m in range(n_measures):
        for b in range(beats_per_measure):
            beat_index = m * beats_per_measure + b
            t = offset + beat_index * beat_s
            kind = "measure" if b == 0 else "beat"
            lines.append(GridLine(
                time=t, kind=kind,
                measure=m + 1, beat=b + 1, subdivision=0,
            ))
            sub_t = t + shuffle_fraction * beat_s
            lines.append(GridLine(
                time=sub_t, kind="subdivision",
                measure=m + 1, beat=b + 1, subdivision=1,
            ))

    lines.sort(key=lambda g: g.time)
    return lines


@dataclass
class OnsetClassification:
    """Result of classifying a detected onset against the grid."""
    onset_time: float
    nearest_measure: int
    nearest_beat: int
    offset_ms: float
    beat_fraction: float
    label: str


def classify_onset_against_grid(
    onset_time: float,
    bpm: float,
    beats_per_measure: int,
    shuffle_fraction: float = 2 / 3,
    offset: float = 0.0,
) -> OnsetClassification:
    """Classify a single onset time relative to the beat grid.

    Returns an OnsetClassification with nearest measure/beat, signed offset
    in ms, beat fraction, and a preliminary label.

    ``beat_fraction`` is the position within the beat interval (0 = on-beat,
    1 = next beat), computed from the beat the onset falls after (floor).
    ``offset_ms`` and ``nearest_measure/beat`` are relative to the nearest
    beat (round), so a negative offset_ms means early.
    """
    beat_s = 60.0 / bpm
    rel = onset_time - offset
    beat_index_exact = rel / beat_s

    beat_index_floor = max(0, int(np.floor(beat_index_exact)))
    beat_fraction = beat_index_exact - beat_index_floor

    beat_index_nearest = max(0, int(round(beat_index_exact)))
    measure = beat_index_nearest // beats_per_measure + 1
    beat_in_measure = beat_index_nearest % beats_per_measure + 1

    nearest_beat_time = offset + beat_index_nearest * beat_s
    offset_ms = (onset_time - nearest_beat_time) * 1000.0

    on_beat_dist = min(beat_fraction, 1.0 - beat_fraction)
    shuffle_dist = abs(beat_fraction - shuffle_fraction)
    threshold = 0.08

    if on_beat_dist < threshold:
        label = "on-beat"
    elif shuffle_dist < threshold:
        label = "shuffle"
    elif beat_fraction > 0.5 and on_beat_dist < shuffle_dist:
        label = "early"
    elif beat_fraction < 0.5 and on_beat_dist < shuffle_dist:
        label = "late"
    else:
        label = "between"

    return OnsetClassification(
        onset_time=onset_time,
        nearest_measure=measure,
        nearest_beat=beat_in_measure,
        offset_ms=offset_ms,
        beat_fraction=beat_fraction,
        label=label,
    )


# ── Grid phase estimation ─────────────────────────────────────────────────────


def estimate_beat_zero(
    onset_times: np.ndarray,
    bpm: float,
    beats_per_measure: int,
    anchor_beats: List[int],
    onset_strengths: np.ndarray | None = None,
) -> float:
    """Estimate beat_zero_s by finding the phase that best aligns onsets with anchor beats.

    For each onset and each anchor beat, computes the implied grid phase.
    The phase with the most weighted onset support wins.

    Parameters
    ----------
    onset_times        : detected onset times in seconds.
    bpm                : tempo.
    beats_per_measure  : meter.
    anchor_beats       : 1-based beat numbers expected to carry strong bass (e.g. [1, 3]).
    onset_strengths    : optional per-onset weights (higher = more influence).

    Returns
    -------
    float : estimated beat_zero_s (phase offset in seconds, within one measure).
    """
    if len(onset_times) == 0 or len(anchor_beats) == 0:
        return 0.0

    beat_s = 60.0 / bpm
    measure_s = beats_per_measure * beat_s
    weights = onset_strengths if onset_strengths is not None else np.ones(len(onset_times))

    anchor_positions = np.array([(b - 1) * beat_s for b in anchor_beats])

    candidates = []
    candidate_weights = []
    for i, t in enumerate(onset_times):
        for ap in anchor_positions:
            phi = (t - ap) % measure_s
            candidates.append(phi)
            candidate_weights.append(float(weights[i]))

    candidates = np.array(candidates)
    candidate_weights = np.array(candidate_weights)

    tolerance = beat_s * 0.15
    n_steps = max(200, int(measure_s / 0.001))
    best_phi = 0.0
    best_score = -1.0
    best_total_dist = float("inf")

    for step in range(n_steps):
        phi = step * measure_s / n_steps
        dists = np.abs(candidates - phi)
        dists = np.minimum(dists, measure_s - dists)
        within = dists < tolerance
        score = float(np.sum(candidate_weights[within]))
        total_dist = float(np.sum(dists[within])) if np.any(within) else float("inf")
        if score > best_score or (score == best_score and total_dist < best_total_dist):
            best_score = score
            best_phi = phi
            best_total_dist = total_dist

    return float(best_phi)


_PHASE_EXCLUDE_LABELS = {"ignore", "string_noise"}
_PHASE_BOOST_LABELS = {"true_attack", "downbeat", "beat3"}
_PHASE_BOOST_FACTOR = 3.0


def filter_onsets_for_phase(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    annotations: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply annotation-based filtering and weighting for phase estimation.

    Onsets labeled ignore/string_noise are excluded.
    Onsets labeled true_attack/downbeat/beat3 get boosted weight.
    Unannotated onsets pass through unchanged.

    Returns (filtered_times, filtered_strengths) with matching lengths.
    """
    keep_mask = np.ones(len(onset_times), dtype=bool)
    weights = onset_strengths.copy()

    for id_str, ann in annotations.items():
        idx = int(id_str)
        if idx >= len(onset_times):
            continue
        label = ann.get("label", "")
        if label in _PHASE_EXCLUDE_LABELS:
            keep_mask[idx] = False
        elif label in _PHASE_BOOST_LABELS:
            weights[idx] *= _PHASE_BOOST_FACTOR

    return onset_times[keep_mask], weights[keep_mask]


# ── Grid source metadata ─────────────────────────────────────────────────────


@dataclass
class GridSource:
    """Describes how the beat grid was generated.

    Future grid sources may include:
      - fixed_bpm              (current: user-supplied tempo and meter)
      - manually_adjusted      (user-corrected beat_zero_s)
      - song_audio_beat_track  (estimated from song channel)
      - annotated_beat_map     (hand-annotated beat positions)
    """
    method: str
    description: str


GRID_SOURCE_FIXED_BPM = GridSource(
    method="fixed_bpm",
    description=(
        "Beat locations are generated from user-supplied BPM, meter, "
        "shuffle fraction, and beat-zero time. They are not yet "
        "estimated from the song audio."
    ),
)


def load_grid_settings(path: str | Path) -> dict:
    """Load grid settings from a JSON file."""
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_grid_settings(
    path: str | Path,
    *,
    bpm: float,
    beats_per_measure: int,
    shuffle_fraction: float,
    beat_zero_s: float,
    source_file: str,
    notes: str = "",
) -> None:
    """Write grid settings to a JSON file."""
    import json
    data = {
        "bpm": bpm,
        "beats_per_measure": beats_per_measure,
        "shuffle_fraction": shuffle_fraction,
        "beat_zero_s": beat_zero_s,
        "source_file": source_file,
        "notes": notes,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


ANNOTATION_LABELS = [
    "true_attack", "string_noise", "passing_note", "downbeat",
    "beat3", "fill", "ignore", "uncertain",
]


# ── SVG layout constants ─────────────────────────────────────────────────────

_SVG_W = 900
_SVG_H = 185
_PLOT_L = 55
_PLOT_R = 15
_PLOT_T = 5
_PLOT_B = 22
_PLOT_W = _SVG_W - _PLOT_L - _PLOT_R
_PLOT_H = _SVG_H - _PLOT_T - _PLOT_B

_TL_SVG_H = 155
_TL_PLOT_T = 20
_TL_PLOT_B = 22
_TL_PLOT_H = _TL_SVG_H - _TL_PLOT_T - _TL_PLOT_B


# ── SVG helpers ──────────────────────────────────────────────────────────────


def _time_ticks(duration: float) -> np.ndarray:
    if duration <= 4:
        step = 0.5
    elif duration <= 10:
        step = 1.0
    elif duration <= 30:
        step = 2.0
    else:
        step = 5.0
    return np.arange(0, duration + step / 2, step)


def _time_axis_svg(duration: float, svg_h: float) -> str:
    ticks = _time_ticks(duration)
    y_top = svg_h - _PLOT_B
    y_bot = y_top + 4
    y_label = y_bot + 13
    parts: list[str] = []
    for t in ticks:
        x = _PLOT_L + (t / duration) * _PLOT_W if duration > 0 else _PLOT_L
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bot}" '
            f'stroke="#888" stroke-width="0.5" />'
            f'<text x="{x:.1f}" y="{y_label}" fill="#888" font-size="9" '
            f'text-anchor="middle">{t:.3g}s</text>'
        )
    return "\n".join(parts)


def _cursor_line(svg_h: float, plot_t: float, plot_h: float) -> str:
    return (
        f'<line class="cursor" x1="{_PLOT_L}" y1="{plot_t}" '
        f'x2="{_PLOT_L}" y2="{plot_t + plot_h}" '
        f'stroke="#fff" stroke-width="1.5" opacity="0" />'
    )


def _y_label_svg(label: str, mid_y: float) -> str:
    return (
        f'<text x="14" y="{mid_y + 4}" fill="#888" font-size="10" '
        f'transform="rotate(-90,14,{mid_y})" text-anchor="middle">{label}</text>'
    )


def _svg_waveform(audio: np.ndarray, sr: int, duration: float) -> str:
    n = len(audio)
    if n == 0:
        return "<p>No audio data.</p>"

    bucket_size = max(1, n // _PLOT_W)
    n_buckets = n // bucket_size
    trimmed = audio[: n_buckets * bucket_size].reshape(n_buckets, bucket_size)
    maxes = np.max(trimmed, axis=1)
    mins = np.min(trimmed, axis=1)
    peak = max(np.max(np.abs(maxes)), np.max(np.abs(mins)), 1e-9)
    mid_y = _PLOT_T + _PLOT_H / 2

    points_top = []
    points_bot = []
    for i in range(n_buckets):
        x = _PLOT_L + i * _PLOT_W / n_buckets
        y_top = mid_y - (maxes[i] / peak) * (_PLOT_H / 2 - 2)
        y_bot = mid_y - (mins[i] / peak) * (_PLOT_H / 2 - 2)
        points_top.append(f"{x:.1f},{y_top:.1f}")
        points_bot.append(f"{x:.1f},{y_bot:.1f}")

    points_bot.reverse()
    poly = " ".join(points_top + points_bot)

    return (
        f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="plot-svg" id="svg-waveform" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;">'
        f'<line x1="{_PLOT_L}" y1="{mid_y}" x2="{_PLOT_L + _PLOT_W}" y2="{mid_y}" '
        f'stroke="#666" stroke-width="0.5" />'
        f'<polygon points="{poly}" fill="#4cc9f0" opacity="0.8" />'
        + _y_label_svg("Amplitude", mid_y)
        + _time_axis_svg(duration, _SVG_H)
        + f'<rect class="loop-region" x="{_PLOT_L}" y="{_PLOT_T}" '
        + f'width="{_PLOT_W}" height="{_PLOT_H}" opacity="0" />'
        + _cursor_line(_SVG_H, _PLOT_T, _PLOT_H)
        + '</svg>'
    )


def _svg_envelope(
    env_times: np.ndarray,
    env_values: np.ndarray,
    onset_times: np.ndarray,
    classifications: List[OnsetClassification],
    onset_strengths: np.ndarray,
    duration: float,
) -> str:
    if len(env_times) == 0:
        return "<p>No envelope data.</p>"

    peak = max(np.max(env_values), 1e-9)

    points = []
    for i in range(len(env_times)):
        x = _PLOT_L + (env_times[i] / duration) * _PLOT_W if duration > 0 else _PLOT_L
        y = _PLOT_T + _PLOT_H - (env_values[i] / peak) * (_PLOT_H - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    onset_parts: list[str] = []

    for idx, ot in enumerate(onset_times):
        x = _PLOT_L + (ot / duration) * _PLOT_W if duration > 0 else _PLOT_L
        onset_parts.append(
            f'<line x1="{x:.1f}" y1="{_PLOT_T}" x2="{x:.1f}" '
            f'y2="{_PLOT_T + _PLOT_H}" stroke="#ff6b6b" stroke-width="1" opacity="0.35" />'
        )
        env_val = float(np.interp(ot, env_times, env_values))
        cy = _PLOT_T + _PLOT_H - (env_val / peak) * (_PLOT_H - 4) - 2
        strength = float(onset_strengths[idx]) if idx < len(onset_strengths) else 0.0
        if idx < len(classifications):
            c = classifications[idx]
            onset_parts.append(
                f'<circle class="onset-marker" data-id="{idx}" cx="{x:.1f}" cy="{cy:.1f}" r="4.5" '
                f'fill="#ff6b6b" stroke="#fff" stroke-width="0.5" opacity="0.9" '
                f'data-time="{c.onset_time:.4f}" data-measure="{c.nearest_measure}" '
                f'data-beat="{c.nearest_beat}" data-offset="{c.offset_ms:+.1f}" '
                f'data-fraction="{c.beat_fraction:.3f}" '
                f'data-label="{html.escape(c.label)}" '
                f'data-strength="{strength:.2f}" />'
            )

    mid_y = _PLOT_T + _PLOT_H / 2
    return (
        f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="plot-svg" id="svg-envelope" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;">'
        f'<polyline points="{polyline}" fill="none" stroke="#4cc9f0" stroke-width="1.5" />'
        + "\n".join(onset_parts)
        + _y_label_svg("Onset strength", mid_y)
        + _time_axis_svg(duration, _SVG_H)
        + f'<rect class="loop-region" x="{_PLOT_L}" y="{_PLOT_T}" '
        + f'width="{_PLOT_W}" height="{_PLOT_H}" opacity="0" />'
        + _cursor_line(_SVG_H, _PLOT_T, _PLOT_H)
        + '</svg>'
    )


def _svg_timeline(
    grid: List[GridLine],
    onset_times: np.ndarray,
    classifications: List[OnsetClassification],
    onset_strengths: np.ndarray,
    duration: float,
) -> str:
    colors = {"measure": "#ff9f1c", "beat": "#4895ef", "subdivision": "#666"}
    line_widths = {"measure": 2, "beat": 1, "subdivision": 1}
    dashes = {"measure": "", "beat": "", "subdivision": 'stroke-dasharray="3,3"'}

    elements: list[str] = []
    for g in grid:
        if g.time < 0 or g.time > duration:
            continue
        x = _PLOT_L + (g.time / duration) * _PLOT_W if duration > 0 else _PLOT_L
        c = colors[g.kind]
        w = line_widths[g.kind]
        d = dashes[g.kind]
        label = ""
        if g.kind == "measure":
            label = (
                f'<text class="grid-label" x="{x:.1f}" y="{_TL_PLOT_T - 5}" '
                f'fill="{c}" font-size="10" text-anchor="middle" '
                f'data-grid-time="{g.time:.6f}">M{g.measure}</text>'
            )
        elements.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{_TL_PLOT_T}" x2="{x:.1f}" '
            f'y2="{_TL_PLOT_T + _TL_PLOT_H}" '
            f'stroke="{c}" stroke-width="{w}" {d} opacity="0.8" '
            f'data-grid-time="{g.time:.6f}" data-grid-kind="{g.kind}" '
            f'data-grid-beat="{g.beat}" />'
            + label
        )

    cy = _TL_PLOT_T + _TL_PLOT_H - 15
    for idx, ot in enumerate(onset_times):
        if ot < 0 or ot > duration:
            continue
        x = _PLOT_L + (ot / duration) * _PLOT_W if duration > 0 else _PLOT_L
        strength = float(onset_strengths[idx]) if idx < len(onset_strengths) else 0.0
        if idx < len(classifications):
            c = classifications[idx]
            elements.append(
                f'<circle class="onset-marker" data-id="{idx}" cx="{x:.1f}" cy="{cy}" r="5" '
                f'fill="#ff6b6b" stroke="#fff" stroke-width="0.5" opacity="0.9" '
                f'data-time="{c.onset_time:.4f}" data-measure="{c.nearest_measure}" '
                f'data-beat="{c.nearest_beat}" data-offset="{c.offset_ms:+.1f}" '
                f'data-fraction="{c.beat_fraction:.3f}" '
                f'data-label="{html.escape(c.label)}" '
                f'data-strength="{strength:.2f}" />'
            )
        elements.append(
            f'<line x1="{x:.1f}" y1="{cy - 8}" x2="{x:.1f}" y2="{cy + 8}" '
            f'stroke="#ff6b6b" stroke-width="1.5" />'
        )

    return (
        f'<svg viewBox="0 0 {_SVG_W} {_TL_SVG_H}" class="plot-svg" id="svg-timeline" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;">'
        + "\n".join(elements)
        + _time_axis_svg(duration, _TL_SVG_H)
        + f'<rect class="loop-region" x="{_PLOT_L}" y="{_TL_PLOT_T}" '
        + f'width="{_PLOT_W}" height="{_TL_PLOT_H}" opacity="0" />'
        + _cursor_line(_TL_SVG_H, _TL_PLOT_T, _TL_PLOT_H)
        + '</svg>'
    )


def _svg_overview(
    overview: dict,
    window_start: float,
    window_duration: float,
) -> str:
    """Compact full-song overview with highlighted viewport rectangle."""
    maxes = overview["maxes"]
    mins = overview["mins"]
    total_dur = overview["total_duration_s"]
    n = len(maxes)
    if n == 0 or total_dur <= 0:
        return ""

    height = 60
    pt = 2
    ph = height - 4
    peak = max(float(np.max(np.abs(maxes))), float(np.max(np.abs(mins))), 1e-9)
    mid = pt + ph / 2

    pts_top: list[str] = []
    pts_bot: list[str] = []
    for i in range(n):
        x = _PLOT_L + i * _PLOT_W / n
        yt = mid - (float(maxes[i]) / peak) * (ph / 2 - 1)
        yb = mid - (float(mins[i]) / peak) * (ph / 2 - 1)
        pts_top.append(f"{x:.1f},{yt:.1f}")
        pts_bot.append(f"{x:.1f},{yb:.1f}")
    pts_bot.reverse()
    poly = " ".join(pts_top + pts_bot)

    vp_x = _PLOT_L + (window_start / total_dur) * _PLOT_W
    vp_w = min((window_duration / total_dur) * _PLOT_W, _PLOT_W - (vp_x - _PLOT_L))

    ticks = []
    if total_dur <= 60:
        step = 5.0
    elif total_dur <= 300:
        step = 30.0
    else:
        step = 60.0
    for t in np.arange(0, total_dur + step / 2, step):
        tx = _PLOT_L + (t / total_dur) * _PLOT_W
        ticks.append(
            f'<line x1="{tx:.0f}" y1="{height - 2}" x2="{tx:.0f}" '
            f'y2="{height}" stroke="#888" stroke-width="0.5" />'
            f'<text x="{tx:.0f}" y="{height - 3}" fill="#888" font-size="7" '
            f'text-anchor="middle">{t:.0f}s</text>'
        )

    win_label = (
        f'<text x="{vp_x + vp_w / 2:.0f}" y="{pt + 9}" fill="#4cc9f0" '
        f'font-size="8" text-anchor="middle">'
        f'{window_start:.1f}–{window_start + window_duration:.1f}s</text>'
    )

    return (
        f'<svg viewBox="0 0 {_SVG_W} {height}" class="overview-svg" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;'
        f'border-radius:4px;margin:0.3em 0;">'
        f'<polygon points="{poly}" fill="#4cc9f0" opacity="0.5" />'
        f'<line x1="{_PLOT_L}" y1="{mid}" x2="{_PLOT_L + _PLOT_W}" y2="{mid}" '
        f'stroke="#666" stroke-width="0.3" />'
        f'<rect x="{vp_x:.1f}" y="{pt}" width="{vp_w:.1f}" height="{ph}" '
        f'fill="#4cc9f0" opacity="0.12" stroke="#4cc9f0" stroke-width="1.5" rx="2" />'
        + win_label
        + "\n".join(ticks)
        + '</svg>'
    )


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """\
body {
    font-family: 'Menlo','Consolas',monospace;
    background: #0f0f23; color: #ccc;
    padding: 20px; max-width: 960px; margin: auto;
}
h1 { color: #4cc9f0; margin-bottom: 0.3em; }
h2 { color: #ff9f1c; margin-top: 1.5em; margin-bottom: 0.4em; }
h3 { color: #ff9f1c; margin: 0 0 8px; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
th, td { border: 1px solid #333; padding: 5px 8px; text-align: left; }
th { background: #1a1a2e; color: #4cc9f0; }
tr:nth-child(even) { background: #16162a; }
svg.plot-svg {
    display: block; width: 100%; margin: 0.3em 0;
    border-radius: 4px; cursor: pointer;
}
.audio-bar {
    position: sticky; top: 0; z-index: 50;
    background: #0f0f23; padding: 6px 0 4px; border-bottom: 1px solid #333;
}
.audio-bar audio { width: 100%; }
.transport-row {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    padding: 4px 0; font-size: 12px;
}
.transport-row label { color: #888; }
.transport-row select, .transport-row input[type=number] {
    background: #1a1a2e; color: #ccc; border: 1px solid #333;
    padding: 2px 6px; border-radius: 3px; font-family: inherit;
    font-size: 11px; width: 65px;
}
.speed-label-text { color: #888; }
#speed-label { color: #4cc9f0; font-weight: bold; min-width: 38px; }
.tbtn {
    background: #1a1a2e; color: #ccc; border: 1px solid #333;
    padding: 3px 8px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit;
}
.tbtn:hover { border-color: #4cc9f0; color: #4cc9f0; }
.tbtn.active { background: #4cc9f0; color: #0f0f23; border-color: #4cc9f0; }
.onset-marker { cursor: pointer; }
#tooltip {
    display: none; position: fixed;
    background: #1a1a2e; border: 1px solid #4cc9f0;
    color: #ccc; padding: 8px 12px; border-radius: 4px;
    font-size: 11px; pointer-events: none; z-index: 100;
    line-height: 1.6; max-width: 260px;
}
.legend-box {
    display: flex; flex-wrap: wrap; gap: 16px; padding: 10px 14px;
    background: #1a1a2e; border-radius: 4px; margin: 0.5em 0;
}
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
details { margin: 1em 0; }
details table { max-height: 400px; overflow-y: auto; display: block; }
summary {
    cursor: pointer; color: #ff9f1c; font-size: 1.1em; font-weight: bold;
    padding: 6px 0;
}
summary:hover { color: #4cc9f0; }
.explanation { color: #999; font-size: 12px; margin: 0.8em 0; line-height: 1.6; }
.explanation b { color: #bbb; }
.grid-source-box {
    background: #1a1a2e; border-left: 3px solid #ff9f1c;
    padding: 12px 16px; margin: 1em 0; border-radius: 0 4px 4px 0;
    font-size: 13px; line-height: 1.5;
}
.grid-source-box .method { color: #4cc9f0; }
#annotation-panel {
    background: #1a1a2e; border: 1px solid #4cc9f0; border-radius: 4px;
    padding: 12px 16px; margin: 1em 0;
}
.ann-buttons { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.ann-btn {
    background: #0f0f23; color: #ccc; border: 1px solid #555;
    padding: 4px 10px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit;
}
.ann-btn:hover { border-color: #4cc9f0; }
.ann-btn.active { background: #4cc9f0; color: #0f0f23; font-weight: bold; }
#ann-note {
    background: #0f0f23; color: #ccc; border: 1px solid #555;
    padding: 5px 8px; border-radius: 3px; width: 100%; box-sizing: border-box;
    font-family: inherit; font-size: 12px; margin: 4px 0;
}
.ann-actions { margin-top: 8px; }
.ann-actions button, .annotation-io button, .ann-import-label {
    background: #1a1a2e; color: #ccc; border: 1px solid #555;
    padding: 4px 10px; border-radius: 3px; cursor: pointer;
    font-family: inherit; font-size: 12px;
}
.ann-actions button:hover, .annotation-io button:hover,
.ann-import-label:hover { border-color: #4cc9f0; color: #4cc9f0; }
.annotation-io {
    display: flex; align-items: center; gap: 12px; margin: 0.5em 0;
    font-size: 12px;
}
#ann-status { color: #888; font-style: italic; }
#selected-status {
    background: #1a1a2e; padding: 6px 12px; border-radius: 4px;
    font-size: 12px; margin: 0.3em 0; min-height: 1.4em;
}
.loop-region {
    fill: #4cc9f0; opacity: 0.08;
}
.shortcut-table td:first-child {
    color: #4cc9f0; font-weight: bold; white-space: nowrap; width: 110px;
}
.overview-svg { display: block; width: 100%; cursor: default; }
#grid-anchor-display {
    font-size: 14px; padding: 8px 12px;
    background: #1a1a2e; border-radius: 4px; margin: 0.5em 0;
}
#grid-anchor-display b { color: #4cc9f0; }
#grid-calibration-menu {
    display: none; position: fixed; background: #1a1a2e;
    border: 1px solid #4cc9f0; border-radius: 4px; padding: 4px 0;
    z-index: 200; font-size: 12px; min-width: 160px;
}
#grid-calibration-menu .cal-item {
    padding: 6px 14px; cursor: pointer; color: #ccc;
}
#grid-calibration-menu .cal-item:hover {
    background: #4cc9f0; color: #0f0f23;
}
.nav-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; font-size: 12px;
}
.nav-bar a, .nav-bar span.disabled {
    padding: 4px 12px; border-radius: 3px; text-decoration: none;
    font-family: inherit; font-size: 12px;
}
.nav-bar a {
    background: #1a1a2e; color: #4cc9f0; border: 1px solid #333;
}
.nav-bar a:hover { border-color: #4cc9f0; }
.nav-bar span.disabled {
    background: #111; color: #555; border: 1px solid #222;
}
.nav-bar .win-label { color: #888; }
"""


# ── JavaScript ───────────────────────────────────────────────────────────────

_JS_TEMPLATE = """\
(function() {
    var audio = document.getElementById('audio-player');
    var duration = %DURATION%;
    var plotLeft = %PLOT_LEFT%;
    var plotWidth = %PLOT_WIDTH%;
    var STORAGE_KEY = '%STORAGE_KEY%';
    var SOURCE_FILE = '%SOURCE_FILE%';
    var BPM = %BPM%;
    var BPM_INT = %BEATS_PER_MEASURE%;
    var SHUF = %SHUFFLE_FRACTION%;
    var beatZero = %BEAT_ZERO%;
    var winStart = %WINDOW_START%;
    var beatS = 60.0 / BPM;
    var cursors = document.querySelectorAll('.cursor');
    var svgs = document.querySelectorAll('.plot-svg');
    var tooltip = document.getElementById('tooltip');
    var loopRects = document.querySelectorAll('.loop-region');

    function timeToX(t) { return plotLeft + (t / duration) * plotWidth; }
    function xToTime(x) {
        return Math.max(0, Math.min(((x - plotLeft) / plotWidth) * duration, duration));
    }

    /* ── Onset id list (sorted) for arrow-key nav ────────────────────── */
    var onsetIds = [];
    var seen = {};
    var allMarkers = document.querySelectorAll('.onset-marker');
    for (var oi = 0; oi < allMarkers.length; oi++) {
        var oid = allMarkers[oi].dataset.id;
        if (!seen[oid]) { onsetIds.push(parseInt(oid)); seen[oid] = 1; }
    }
    onsetIds.sort(function(a,b){return a-b;});
    var markers = allMarkers;

    /* ── Cursor sync ─────────────────────────────────────────────────── */
    function updateCursor() {
        if (!audio) return;
        var x = timeToX(audio.currentTime);
        for (var i = 0; i < cursors.length; i++) {
            cursors[i].setAttribute('x1', x);
            cursors[i].setAttribute('x2', x);
            cursors[i].setAttribute('opacity', '0.8');
        }
        if (!audio.paused) requestAnimationFrame(updateCursor);
    }
    if (audio) {
        audio.addEventListener('play', function() { requestAnimationFrame(updateCursor); });
        audio.addEventListener('seeked', updateCursor);
        audio.addEventListener('pause', updateCursor);
        audio.addEventListener('timeupdate', function() {
            updateCursor();
            if (loopEnabled && audio.currentTime >= loopEnd)
                audio.currentTime = loopStart;
        });
    }

    /* ── Click-to-seek ───────────────────────────────────────────────── */
    for (var s = 0; s < svgs.length; s++) {
        (function(svg) {
            svg.addEventListener('click', function(e) {
                if (!audio) return;
                var pt = svg.createSVGPoint();
                pt.x = e.clientX; pt.y = e.clientY;
                var svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
                audio.currentTime = xToTime(svgPt.x);
                updateCursor();
            });
        })(svgs[s]);
    }

    /* ── Speed controls ──────────────────────────────────────────────── */
    var speedLabel = document.getElementById('speed-label');
    var speedBtns = document.querySelectorAll('.speed-btn');
    for (var sb = 0; sb < speedBtns.length; sb++) {
        (function(btn) {
            btn.addEventListener('click', function() {
                if (!audio) return;
                audio.playbackRate = parseFloat(btn.dataset.speed);
                if (speedLabel) speedLabel.textContent = btn.dataset.speed + 'x';
                for (var j = 0; j < speedBtns.length; j++)
                    speedBtns[j].classList.remove('active');
                btn.classList.add('active');
            });
        })(speedBtns[sb]);
    }

    /* ── Channel switching ───────────────────────────────────────────── */
    var channelSel = document.getElementById('channel-select');
    if (channelSel && audio) {
        channelSel.addEventListener('change', function() {
            var cur = audio.currentTime;
            var playing = !audio.paused;
            audio.src = channelSel.value;
            audio.currentTime = cur;
            if (playing) audio.play();
        });
    }

    /* ── Loop controls ───────────────────────────────────────────────── */
    var loopEnabled = false;
    var loopStart = 0;
    var loopEnd = duration;
    var loopBtn = document.getElementById('loop-toggle');
    var loopStartIn = document.getElementById('loop-start-in');
    var loopEndIn = document.getElementById('loop-end-in');

    function updateLoopDisplay() {
        if (loopBtn) loopBtn.classList.toggle('active', loopEnabled);
        if (loopStartIn) loopStartIn.value = loopStart.toFixed(2);
        if (loopEndIn) loopEndIn.value = loopEnd.toFixed(2);
        for (var lr = 0; lr < loopRects.length; lr++) {
            var r = loopRects[lr];
            if (loopEnabled) {
                r.setAttribute('x', timeToX(loopStart));
                r.setAttribute('width', timeToX(loopEnd) - timeToX(loopStart));
                r.setAttribute('opacity', '0.08');
            } else {
                r.setAttribute('opacity', '0');
            }
        }
    }
    function setLoopStart(t) { loopStart = Math.max(0, Math.min(t, loopEnd - 0.05)); updateLoopDisplay(); }
    function setLoopEnd(t) { loopEnd = Math.max(loopStart + 0.05, Math.min(t, duration)); updateLoopDisplay(); }

    if (loopBtn) loopBtn.addEventListener('click', function() {
        loopEnabled = !loopEnabled; updateLoopDisplay();
    });
    var lsBtn = document.getElementById('loop-set-start');
    var leBtn = document.getElementById('loop-set-end');
    var lrBtn = document.getElementById('loop-reset');
    var lrestartBtn = document.getElementById('loop-restart');
    if (lsBtn) lsBtn.addEventListener('click', function() { if(audio) setLoopStart(audio.currentTime); });
    if (leBtn) leBtn.addEventListener('click', function() { if(audio) setLoopEnd(audio.currentTime); });
    if (lrBtn) lrBtn.addEventListener('click', function() { loopStart=0; loopEnd=duration; updateLoopDisplay(); });
    if (lrestartBtn) lrestartBtn.addEventListener('click', function() {
        if(audio){ audio.currentTime = loopEnabled ? loopStart : 0; updateCursor(); }
    });
    if (loopStartIn) loopStartIn.addEventListener('change', function() { setLoopStart(parseFloat(this.value)||0); });
    if (loopEndIn) loopEndIn.addEventListener('change', function() { setLoopEnd(parseFloat(this.value)||duration); });
    updateLoopDisplay();

    /* ── Tooltip ──────────────────────────────────────────────────────── */
    for (var m = 0; m < markers.length; m++) {
        (function(el) {
            el.addEventListener('mouseenter', function(e) {
                var d = el.dataset;
                var ann = annotations[d.id];
                var extra = ann ? '<br>Annotation: <b>' + ann.label + '</b>' : '';
                tooltip.innerHTML =
                    '<b>Onset #' + d.id + ' at ' + d.time + 's</b><br>' +
                    'Measure ' + d.measure + ', Beat ' + d.beat + '<br>' +
                    'Offset: ' + d.offset + ' ms<br>' +
                    'Beat fraction: ' + d.fraction + '<br>' +
                    'Label: <b>' + d.label + '</b><br>' +
                    'Strength: ' + d.strength + extra;
                tooltip.style.display = 'block';
                moveTooltip(e);
            });
            el.addEventListener('mousemove', moveTooltip);
            el.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
            });
        })(markers[m]);
    }
    function moveTooltip(e) {
        var x = e.clientX + 14, y = e.clientY - 14;
        if (x + 270 > window.innerWidth) x = e.clientX - 270;
        if (y < 0) y = e.clientY + 20;
        tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
    }

    /* ── Annotations ─────────────────────────────────────────────────── */
    var LABEL_COLORS = {
        'true_attack':'#2ecc71', 'string_noise':'#f1c40f',
        'passing_note':'#9b59b6', 'downbeat':'#ff9f1c',
        'beat3':'#4895ef', 'fill':'#4cc9f0',
        'ignore':'#555', 'uncertain':'#aaa'
    };
    var LABEL_KEYS = ['true_attack','string_noise','passing_note','downbeat',
                      'beat3','fill','ignore','uncertain'];
    var annotations = {};
    var selectedOnsetId = null;
    var panel = document.getElementById('annotation-panel');
    var annInfo = document.getElementById('ann-info');
    var annNote = document.getElementById('ann-note');
    var annStatus = document.getElementById('ann-status');
    var selStatus = document.getElementById('selected-status');

    function applyAnnotationColors() {
        for (var k = 0; k < markers.length; k++) {
            var el = markers[k], id = el.dataset.id;
            if (annotations[id]) {
                el.setAttribute('fill', LABEL_COLORS[annotations[id].label]||'#ff6b6b');
                el.setAttribute('stroke', '#fff');
                el.setAttribute('stroke-width', '1.5');
            } else {
                el.setAttribute('fill', '#ff6b6b');
                el.setAttribute('stroke', '#fff');
                el.setAttribute('stroke-width', '0.5');
            }
        }
    }

    function updateSelStatus() {
        if (!selStatus) return;
        if (selectedOnsetId === null) { selStatus.textContent = 'No onset selected'; return; }
        var el = document.querySelector('.onset-marker[data-id="'+selectedOnsetId+'"]');
        if (!el) { selStatus.textContent = 'No onset selected'; return; }
        var d = el.dataset, ann = annotations[selectedOnsetId];
        selStatus.innerHTML = 'Selected: <b>#'+selectedOnsetId+'</b> t='+d.time+
            's M'+d.measure+' B'+d.beat+
            (ann ? ' — <b>'+ann.label+'</b>' : ' — unannotated')+
            (ann && ann.note ? ' "'+ann.note+'"' : '');
    }

    function selectOnset(id) {
        var prev = document.querySelectorAll('.onset-marker[data-id="'+selectedOnsetId+'"]');
        for (var p = 0; p < prev.length; p++)
            prev[p].setAttribute('stroke-width', annotations[selectedOnsetId] ? '1.5' : '0.5');
        selectedOnsetId = id;
        var els = document.querySelectorAll('.onset-marker[data-id="'+id+'"]');
        for (var q = 0; q < els.length; q++) els[q].setAttribute('stroke-width', '3');
        var el = els[0]; if (!el) return;
        var d = el.dataset;
        if (annInfo) annInfo.innerHTML = '<b>Onset #'+id+'</b> at '+d.time+
            's — M'+d.measure+' B'+d.beat+' ('+d.label+')';
        if (annNote) annNote.value = annotations[id] ? (annotations[id].note||'') : '';
        var btns = document.querySelectorAll('.ann-btn');
        for (var b = 0; b < btns.length; b++) {
            btns[b].classList.remove('active');
            if (annotations[id] && btns[b].dataset.label === annotations[id].label)
                btns[b].classList.add('active');
        }
        if (panel) panel.style.display = 'block';
        updateSelStatus();
    }

    function assignLabel(label) {
        if (selectedOnsetId === null) return;
        var el = document.querySelector('.onset-marker[data-id="'+selectedOnsetId+'"]');
        if (!el) return;
        annotations[selectedOnsetId] = {
            time_s: parseFloat(el.dataset.time),
            detected_onset_id: parseInt(selectedOnsetId),
            label: label, note: annNote ? annNote.value : '',
            created_by: 'user', created_at: new Date().toISOString()
        };
        var btns = document.querySelectorAll('.ann-btn');
        for (var j = 0; j < btns.length; j++) {
            btns[j].classList.remove('active');
            if (btns[j].dataset.label === label) btns[j].classList.add('active');
        }
        applyAnnotationColors(); saveToStorage(); updateSelStatus();
        if (annStatus) annStatus.textContent = 'Saved: #'+selectedOnsetId+' \\u2192 '+label;
    }

    function clearAnnotation() {
        if (selectedOnsetId === null) return;
        delete annotations[selectedOnsetId];
        var btns = document.querySelectorAll('.ann-btn');
        for (var j = 0; j < btns.length; j++) btns[j].classList.remove('active');
        applyAnnotationColors(); saveToStorage(); updateSelStatus();
        if (annStatus) annStatus.textContent = 'Cleared #'+selectedOnsetId;
    }

    for (var mc = 0; mc < markers.length; mc++) {
        (function(el) {
            el.addEventListener('click', function(e) {
                e.stopPropagation(); selectOnset(el.dataset.id);
            });
        })(markers[mc]);
    }

    var annBtns = document.querySelectorAll('.ann-btn');
    for (var ab = 0; ab < annBtns.length; ab++) {
        (function(btn) {
            btn.addEventListener('click', function() { assignLabel(btn.dataset.label); });
        })(annBtns[ab]);
    }

    if (annNote) annNote.addEventListener('change', function() {
        if (selectedOnsetId !== null && annotations[selectedOnsetId]) {
            annotations[selectedOnsetId].note = annNote.value;
            saveToStorage(); updateSelStatus();
        }
    });

    var cancelBtn = document.getElementById('ann-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', function() {
        if (panel) panel.style.display = 'none';
        var prev = document.querySelectorAll('.onset-marker[data-id="'+selectedOnsetId+'"]');
        for (var p = 0; p < prev.length; p++)
            prev[p].setAttribute('stroke-width', annotations[selectedOnsetId] ? '1.5' : '0.5');
        selectedOnsetId = null; updateSelStatus();
    });

    function saveToStorage() {
        var data = {version:1, source_file:SOURCE_FILE,
            annotations: Object.keys(annotations).map(function(k){ return annotations[k]; })};
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch(e) {}
    }

    try {
        var stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            var parsed = JSON.parse(stored);
            if (parsed.annotations) {
                for (var li = 0; li < parsed.annotations.length; li++) {
                    var a = parsed.annotations[li];
                    annotations[a.detected_onset_id] = a;
                }
                applyAnnotationColors();
            }
        }
    } catch(e) {}

    var exportBtn = document.getElementById('ann-export');
    if (exportBtn) exportBtn.addEventListener('click', function() {
        var data = {version:1, source_file:SOURCE_FILE,
            annotations: Object.keys(annotations).map(function(k){ return annotations[k]; })};
        var blob = new Blob([JSON.stringify(data,null,2)], {type:'application/json'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a'); a.href = url;
        a.download = STORAGE_KEY+'.json';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        if (annStatus) annStatus.textContent = 'Exported '+Object.keys(annotations).length+' annotations.';
    });

    var importInput = document.getElementById('ann-import');
    if (importInput) importInput.addEventListener('change', function(e) {
        var file = e.target.files[0]; if (!file) return;
        var reader = new FileReader();
        reader.onload = function(ev) {
            try {
                var data = JSON.parse(ev.target.result);
                if (data.annotations) {
                    annotations = {};
                    for (var i = 0; i < data.annotations.length; i++)
                        annotations[data.annotations[i].detected_onset_id] = data.annotations[i];
                    applyAnnotationColors(); saveToStorage();
                    if (annStatus) annStatus.textContent = 'Imported '+data.annotations.length+' annotations.';
                }
            } catch(err) { if (annStatus) annStatus.textContent = 'Import error: '+err.message; }
        };
        reader.readAsText(file);
    });

    /* ── Hotkeys ──────────────────────────────────────────────────────── */
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' ||
            e.target.tagName === 'SELECT') return;
        var k = e.key;
        if (k === ' ') { e.preventDefault(); if(audio){audio.paused?audio.play():audio.pause();} }
        else if (k === 'r' || k === 'R') {
            e.preventDefault();
            if(audio){audio.currentTime=loopEnabled?loopStart:0; updateCursor();}
        }
        else if (k === 'l' || k === 'L') {
            e.preventDefault(); loopEnabled=!loopEnabled; updateLoopDisplay();
        }
        else if (k === '[') { e.preventDefault(); if(audio) setLoopStart(audio.currentTime); }
        else if (k === ']') { e.preventDefault(); if(audio) setLoopEnd(audio.currentTime); }
        else if (k === 'ArrowLeft' && !e.shiftKey) {
            e.preventDefault();
            if (selectedOnsetId !== null) {
                var ci = onsetIds.indexOf(parseInt(selectedOnsetId));
                if (ci > 0) selectOnset(String(onsetIds[ci-1]));
            }
        }
        else if (k === 'ArrowRight' && !e.shiftKey) {
            e.preventDefault();
            if (selectedOnsetId !== null) {
                var ci2 = onsetIds.indexOf(parseInt(selectedOnsetId));
                if (ci2 < onsetIds.length-1) selectOnset(String(onsetIds[ci2+1]));
            }
        }
        else if (k === 'ArrowLeft' && e.shiftKey) {
            e.preventDefault(); if(audio){audio.currentTime=Math.max(0,audio.currentTime-0.25);updateCursor();}
        }
        else if (k === 'ArrowRight' && e.shiftKey) {
            e.preventDefault(); if(audio){audio.currentTime=Math.min(duration,audio.currentTime+0.25);updateCursor();}
        }
        else if (k === 'Delete' || k === 'Backspace') { e.preventDefault(); clearAnnotation(); }
        else if (k >= '1' && k <= '8') {
            e.preventDefault(); assignLabel(LABEL_KEYS[parseInt(k)-1]);
        }
    });

    /* ── Grid calibration ─────────────────────────────────────────────── */
    var calMenu = document.getElementById('grid-calibration-menu');
    var anchorDisp = document.getElementById('grid-anchor-display');
    var calClickTime = 0;
    var gridLines = document.querySelectorAll('.grid-line');
    var gridLabels = document.querySelectorAll('.grid-label');

    function updateAnchorDisplay() {
        if (!anchorDisp) return;
        var rel = -beatZero;
        if (rel < 0) rel += Math.ceil(-rel / (BPM_INT * beatS)) * BPM_INT * beatS;
        var beatIdx = Math.round(rel / beatS);
        var m = Math.floor(beatIdx / BPM_INT) + 1;
        var b = (beatIdx % BPM_INT) + 1;
        anchorDisp.innerHTML = 'Grid anchor: <b>Measure ' + m + ' Beat ' + b +
            '</b> at <b>' + beatZero.toFixed(4) + 's</b>';
        var bzIn = document.getElementById('beat-zero-input');
        if (bzIn) bzIn.value = beatZero.toFixed(4);
    }

    function redrawGrid() {
        for (var g = 0; g < gridLines.length; g++) {
            var gl = gridLines[g];
            var t = parseFloat(gl.dataset.gridTime);
            var kind = gl.dataset.gridKind;
            var beatInMeasure = parseInt(gl.dataset.gridBeat);
            var newRel = t - beatZero;
            var measS = BPM_INT * beatS;
            var gridBeatIdx = Math.round(newRel / beatS);
            var newBeat = ((gridBeatIdx % BPM_INT) + BPM_INT) % BPM_INT;
            var newMeasure = Math.floor(gridBeatIdx / BPM_INT) + 1;
            var newT;
            if (kind === 'subdivision') {
                var parentBeatT = beatZero + gridBeatIdx * beatS;
                newT = parentBeatT + SHUF * beatS;
            } else {
                newT = beatZero + gridBeatIdx * beatS;
            }
            var x = plotLeft + ((newT - winStart) / duration) * plotWidth;
            gl.setAttribute('x1', x);
            gl.setAttribute('x2', x);
            if (x < plotLeft || x > plotLeft + plotWidth) {
                gl.setAttribute('opacity', '0');
            } else {
                gl.setAttribute('opacity', '0.8');
            }
        }
        for (var lb = 0; lb < gridLabels.length; lb++) {
            var lbl = gridLabels[lb];
            var lt = parseFloat(lbl.dataset.gridTime);
            var lRel = lt - beatZero;
            var lBeatIdx = Math.round(lRel / beatS);
            var lMeasure = Math.floor(lBeatIdx / BPM_INT) + 1;
            var lNewT = beatZero + lBeatIdx * beatS;
            var lx = plotLeft + ((lNewT - winStart) / duration) * plotWidth;
            lbl.setAttribute('x', lx);
            lbl.textContent = 'M' + lMeasure;
            if (lx < plotLeft || lx > plotLeft + plotWidth) {
                lbl.setAttribute('opacity', '0');
            } else {
                lbl.setAttribute('opacity', '1');
            }
        }
        updateAnchorDisplay();
    }

    function setGridAnchor(time, beatNumber) {
        beatZero = time - (beatNumber - 1) * beatS;
        redrawGrid();
    }

    function showCalMenu(x, y, time) {
        if (!calMenu) return;
        calClickTime = time;
        calMenu.style.left = x + 'px';
        calMenu.style.top = y + 'px';
        calMenu.style.display = 'block';
    }

    if (calMenu) {
        var items = calMenu.querySelectorAll('.cal-item');
        for (var ci = 0; ci < items.length; ci++) {
            (function(item) {
                item.addEventListener('click', function() {
                    var bn = parseInt(item.dataset.beat);
                    setGridAnchor(calClickTime, bn);
                    calMenu.style.display = 'none';
                });
            })(items[ci]);
        }
        document.addEventListener('click', function() {
            calMenu.style.display = 'none';
        });
    }

    for (var sv = 0; sv < svgs.length; sv++) {
        (function(svg) {
            svg.addEventListener('contextmenu', function(e) {
                e.preventDefault();
                var pt = svg.createSVGPoint();
                pt.x = e.clientX; pt.y = e.clientY;
                var svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
                var t = xToTime(svgPt.x) + winStart;
                showCalMenu(e.clientX, e.clientY, t);
            });
        })(svgs[sv]);
    }

    var calFromOnsetBtn = document.getElementById('cal-from-selected');
    if (calFromOnsetBtn) {
        calFromOnsetBtn.addEventListener('click', function() {
            if (selectedOnsetId === null) return;
            var el = document.querySelector('.onset-marker[data-id="'+selectedOnsetId+'"]');
            if (!el) return;
            var t = parseFloat(el.dataset.time) + winStart;
            showCalMenu(
                calFromOnsetBtn.getBoundingClientRect().left,
                calFromOnsetBtn.getBoundingClientRect().bottom + 2, t);
        });
    }

    updateAnchorDisplay();
    updateSelStatus();
})();
"""


# ── Report assembly ──────────────────────────────────────────────────────────


def render_report(
    *,
    wav_path: str,
    bass_audio: np.ndarray,
    sr: int,
    bpm: float,
    beats_per_measure: int,
    shuffle_fraction: float,
    start: float,
    duration: float,
    onset_times: np.ndarray,
    classifications: List[OnsetClassification],
    env_times: np.ndarray,
    env_values: np.ndarray,
    grid: List[GridLine],
    audio_src: str = "",
    grid_source: GridSource | None = None,
    onset_strengths: np.ndarray | None = None,
    beat_zero_s: float = 0.0,
    storage_key: str = "annotations",
    audio_diag_source: dict | None = None,
    audio_diag_export: dict | None = None,
    audio_export_mode: str = "stereo",
    sidecar_srcs: dict[str, str] | None = None,
    overview: dict | None = None,
    prev_href: str | None = None,
    next_href: str | None = None,
    total_windows: int = 1,
    window_index: int = 0,
) -> str:
    """Render the Beat Microscope HTML report as a string."""

    actual_duration = len(bass_audio) / sr if sr > 0 else 0.0
    if onset_strengths is None:
        onset_strengths = np.zeros(len(onset_times))
    if grid_source is None:
        grid_source = GRID_SOURCE_FIXED_BPM

    if sidecar_srcs is None:
        sidecar_srcs = {}

    # ── Audio player with transport controls ─────────────────────────────
    speed_buttons = "".join(
        f'<button class="tbtn speed-btn{" active" if s == "1.0" else ""}" '
        f'data-speed="{s}">{s}x</button>'
        for s in ["1.0", "0.75", "0.5", "0.33", "0.25"]
    )

    channel_options = ""
    if len(sidecar_srcs) > 1:
        for label, src in sidecar_srcs.items():
            sel = ' selected' if label == audio_export_mode else ''
            channel_options += (
                f'<option value="{html.escape(src)}"{sel}>'
                f'{html.escape(label)}</option>'
            )

    channel_html = (
        f'<label>Channel:</label>'
        f'<select id="channel-select">{channel_options}</select>'
    ) if channel_options else ""

    loop_html = (
        '<button class="tbtn" id="loop-toggle" title="L">Loop</button>'
        '<button class="tbtn" id="loop-set-start" title="[">[ set</button>'
        '<input type="number" id="loop-start-in" step="0.1" min="0">'
        '<input type="number" id="loop-end-in" step="0.1" min="0">'
        '<button class="tbtn" id="loop-set-end" title="]">] set</button>'
        '<button class="tbtn" id="loop-reset">Reset</button>'
        '<button class="tbtn" id="loop-restart" title="R">Restart</button>'
    )

    if audio_src:
        audio_html = (
            '<div class="audio-bar">'
            f'<audio id="audio-player" controls preload="auto" '
            f'src="{html.escape(audio_src)}"></audio>'
            '<div class="transport-row">'
            '<label>Speed:</label>'
            f'<span id="speed-label">1.0x</span>{speed_buttons}'
            f'{channel_html}'
            '</div>'
            f'<div class="transport-row">{loop_html}</div>'
            '</div>'
        )
    else:
        audio_html = ""

    # ── Overview + navigation ───────────────────────────────────────────
    if overview and overview.get("total_duration_s", 0) > 0:
        overview_html = (
            '<h2>Song Overview</h2>'
            + _svg_overview(overview, start, actual_duration)
        )
        prev_link = (
            f'<a href="{html.escape(prev_href)}">&#9664; Previous window</a>'
            if prev_href else '<span class="disabled">&#9664; Previous</span>'
        )
        next_link = (
            f'<a href="{html.escape(next_href)}">Next window &#9654;</a>'
            if next_href else '<span class="disabled">Next &#9654;</span>'
        )
        overview_html += (
            '<div class="nav-bar">'
            f'{prev_link}'
            f'<span class="win-label">Window {window_index + 1}/{total_windows} '
            f'({start:.1f}–{start + actual_duration:.1f}s '
            f'of {overview["total_duration_s"]:.1f}s)</span>'
            f'{next_link}'
            '</div>'
        )
    else:
        overview_html = ""

    # ── File summary ─────────────────────────────────────────────────────
    beat_zero_row = f"<tr><td>Beat zero (M1 B1)</td><td>{beat_zero_s:.4f}s</td></tr>"

    def _dbfs(val: float | None) -> str:
        return f"{val:.1f} dBFS" if val is not None else "—"

    diag_rows = ""
    diag_rows += f"<tr><td>Audio export mode</td><td>{html.escape(audio_export_mode)}</td></tr>"
    if audio_diag_source:
        diag_rows += (
            f"<tr><td>Source segment</td>"
            f"<td>{audio_diag_source['channels']}ch, "
            f"RMS {_dbfs(audio_diag_source['rms_dbfs'])}, "
            f"peak {_dbfs(audio_diag_source['peak_dbfs'])}</td></tr>"
        )
    if audio_diag_export:
        diag_rows += (
            f"<tr><td>Exported excerpt</td>"
            f"<td>{audio_diag_export['channels']}ch, "
            f"RMS {_dbfs(audio_diag_export['rms_dbfs'])}, "
            f"peak {_dbfs(audio_diag_export['peak_dbfs'])}</td></tr>"
        )

    summary_html = (
        "<h2>File Summary</h2>"
        "<table>"
        f"<tr><td>File</td><td>{html.escape(wav_path)}</td></tr>"
        f"<tr><td>Sample rate</td><td>{sr} Hz</td></tr>"
        f"<tr><td>BPM</td><td>{bpm}</td></tr>"
        f"<tr><td>Beats per measure</td><td>{beats_per_measure}</td></tr>"
        f"<tr><td>Shuffle fraction</td><td>{shuffle_fraction:.3f}</td></tr>"
        + beat_zero_row
        + f"<tr><td>Segment</td><td>{start:.2f}s – {start + duration:.2f}s "
        f"({actual_duration:.2f}s actual)</td></tr>"
        f"<tr><td>Onsets detected</td><td>{len(onset_times)}</td></tr>"
        + diag_rows
        + "</table>"
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_html = (
        '<div class="legend-box">'
        '<div class="legend-item">'
        '<span style="background:#ff9f1c;width:20px;height:3px;display:inline-block;"></span>'
        '<span>Measure line</span></div>'
        '<div class="legend-item">'
        '<span style="background:#4895ef;width:20px;height:3px;display:inline-block;"></span>'
        '<span>Beat line</span></div>'
        '<div class="legend-item">'
        '<span style="border-top:2px dashed #888;width:20px;display:inline-block;"></span>'
        '<span>Shuffle subdivision</span></div>'
        '<div class="legend-item">'
        '<span style="background:#ff6b6b;width:10px;height:10px;border-radius:50%;'
        'display:inline-block;"></span>'
        '<span>Detected bass onset</span></div>'
        '<div class="legend-item">'
        '<span style="background:#4cc9f0;width:20px;height:3px;display:inline-block;"></span>'
        '<span>Waveform / envelope</span></div>'
        '<div class="legend-item">'
        '<span style="background:#fff;width:1.5px;height:14px;display:inline-block;"></span>'
        '<span>Playback cursor</span></div>'
        '</div>'
    )

    # ── Plots ────────────────────────────────────────────────────────────
    waveform_html = (
        "<h2>Bass Waveform</h2>"
        + _svg_waveform(bass_audio, sr, actual_duration)
    )
    envelope_html = (
        "<h2>Onset-Strength Envelope</h2>"
        + _svg_envelope(
            env_times, env_values, onset_times,
            classifications, onset_strengths, actual_duration,
        )
    )
    timeline_html = (
        "<h2>Event Timeline</h2>"
        + _svg_timeline(
            grid, onset_times, classifications,
            onset_strengths, actual_duration,
        )
    )

    # ── Explanation ──────────────────────────────────────────────────────
    explanation_html = (
        '<div class="explanation">'
        "<p><b>Interpreting offsets:</b> A positive offset means the onset was "
        "played later than the nearest grid point. A negative offset means it "
        "was played earlier.</p>"
        "<p><b>Labels</b> are grid-relative and do not yet infer musical intent. "
        'An onset labeled "late" is late relative to the beat grid, not '
        "necessarily musically wrong.</p>"
        "<p><b>Workflow:</b> For annotation, work in short windows (2–6 seconds). "
        "Loop a region, listen at reduced speed, and label each onset.</p>"
        "</div>"
    )

    # ── Selected onset status ────────────────────────────────────────────
    sel_status_html = '<div id="selected-status">No onset selected</div>'

    # ── Beat Grid / Calibration ─────────────────────────────────────────
    cal_items = "".join(
        f'<div class="cal-item" data-beat="{b}">Make this Beat {b}</div>'
        for b in range(1, beats_per_measure + 1)
    )

    beat_s_val = 60.0 / bpm if bpm > 0 else 0.5
    measure_s_val = beat_s_val * beats_per_measure
    shifts = {
        "-1 beat": beat_zero_s - beat_s_val,
        "+1 beat": beat_zero_s + beat_s_val,
        "+½ beat": beat_zero_s + beat_s_val / 2,
        "-1 measure": beat_zero_s - measure_s_val,
        "+1 measure": beat_zero_s + measure_s_val,
    }
    shift_btns = "".join(
        f'<span class="tbtn" style="cursor:default;" '
        f'title="--beat-zero-s {v:.4f}">{lbl}: {v:.4f}s</span>'
        for lbl, v in shifts.items()
    )

    grid_settings_json = (
        '{"bpm":' + f'{bpm}'
        + ',"beats_per_measure":' + f'{beats_per_measure}'
        + ',"shuffle_fraction":' + f'{shuffle_fraction}'
        + ',"beat_zero_s":' + f'{beat_zero_s}'
        + ',"source_file":"'
        + html.escape(Path(wav_path).name).replace('"', '\\"')
        + '","notes":""}'
    )

    grid_source_html = (
        '<h2>Beat Grid</h2>'
        '<div id="grid-anchor-display"></div>'
        '<div class="transport-row" style="margin:0.3em 0;">'
        '<button class="tbtn" id="cal-from-selected">'
        'Calibrate from selected onset</button>'
        '<span class="explanation" style="margin:0;padding:0 8px;">'
        'or right-click any plot</span>'
        '</div>'
        f'<div id="grid-calibration-menu">{cal_items}</div>'
        '<details><summary>Advanced grid settings</summary>'
        '<div class="grid-source-box">'
        f'<p>Method: <span class="method">{html.escape(grid_source.method)}</span></p>'
        f'<p>{html.escape(grid_source.description)}</p>'
        '</div>'
        '<div class="transport-row" style="margin:0.5em 0;">'
        '<label>Shift grid:</label>' + shift_btns
        + '</div>'
        '<div class="transport-row" style="margin:0.3em 0;">'
        '<label>Beat zero:</label>'
        f'<input type="number" id="beat-zero-input" step="0.001" '
        f'value="{beat_zero_s:.4f}" style="width:90px;">'
        f'<button class="tbtn" id="export-grid-settings">'
        f'Export grid settings JSON</button>'
        '</div></details>'
        f'<script>document.getElementById("export-grid-settings")'
        f'.addEventListener("click",function(){{'
        f'var bz=document.getElementById("beat-zero-input").value;'
        f'var d=JSON.parse(\'{grid_settings_json}\');'
        f'd.beat_zero_s=parseFloat(bz);'
        f'var b=new Blob([JSON.stringify(d,null,2)],{{type:"application/json"}});'
        f'var u=URL.createObjectURL(b);'
        f'var a=document.createElement("a");a.href=u;'
        f'a.download="{html.escape(Path(wav_path).stem)}_grid_settings.json";'
        f'document.body.appendChild(a);a.click();document.body.removeChild(a);'
        f'URL.revokeObjectURL(u);'
        f'}});</script>'
    )

    # ── Annotation panel ────────────────────────────────────────────────
    ann_btns_html = "".join(
        f'<button class="ann-btn" data-label="{lab}">'
        f'{lab.replace("_", " ")}</button>'
        for lab in ANNOTATION_LABELS
    )
    annotation_html = (
        '<h2>Annotations</h2>'
        '<div id="annotation-panel" style="display:none;">'
        '<p id="ann-info"></p>'
        f'<div class="ann-buttons">{ann_btns_html}</div>'
        '<input type="text" id="ann-note" placeholder="Optional note...">'
        '<div class="ann-actions"><button id="ann-cancel">Close</button></div>'
        '</div>'
        '<div class="annotation-io">'
        '<button id="ann-export">Export annotations JSON</button>'
        '<label class="ann-import-label">Import annotations'
        '<input type="file" id="ann-import" accept=".json" style="display:none;">'
        '</label>'
        '<span id="ann-status"></span>'
        '</div>'
    )

    # ── Onset table (collapsed) ──────────────────────────────────────────
    rows: list[str] = []
    for i, c in enumerate(classifications):
        strength = float(onset_strengths[i]) if i < len(onset_strengths) else 0.0
        rows.append(
            f"<tr><td>{c.onset_time:.4f}</td><td>{c.nearest_measure}</td>"
            f"<td>{c.nearest_beat}</td><td>{c.offset_ms:+.1f}</td>"
            f"<td>{c.beat_fraction:.3f}</td><td>{html.escape(c.label)}</td>"
            f"<td>{strength:.2f}</td></tr>"
        )
    table_html = (
        f'<details><summary>Detected Onsets ({len(onset_times)})</summary>'
        "<table><tr><th>Time (s)</th><th>Measure</th><th>Beat</th>"
        "<th>Offset (ms)</th><th>Beat Fraction</th><th>Label</th>"
        "<th>Strength</th></tr>"
        + "\n".join(rows)
        + "</table></details>"
    )

    # ── JavaScript ───────────────────────────────────────────────────────
    source_file = Path(wav_path).name
    js = (
        _JS_TEMPLATE
        .replace("%DURATION%", f"{actual_duration:.6f}")
        .replace("%PLOT_LEFT%", str(_PLOT_L))
        .replace("%PLOT_WIDTH%", str(_PLOT_W))
        .replace("%STORAGE_KEY%", storage_key)
        .replace("%SOURCE_FILE%", source_file)
        .replace("%BPM%", f"{bpm}")
        .replace("%BEATS_PER_MEASURE%", str(beats_per_measure))
        .replace("%SHUFFLE_FRACTION%", f"{shuffle_fraction}")
        .replace("%BEAT_ZERO%", f"{beat_zero_s}")
        .replace("%WINDOW_START%", f"{start}")
    )

    # ── Keyboard shortcuts panel ────────────────────────────────────────
    shortcuts_html = (
        '<details><summary>Keyboard Shortcuts</summary>'
        '<table class="shortcut-table">'
        '<tr><td>Space</td><td>Play / pause</td></tr>'
        '<tr><td>R</td><td>Restart from loop start (or segment start)</td></tr>'
        '<tr><td>L</td><td>Toggle loop on/off</td></tr>'
        '<tr><td>[</td><td>Set loop start at cursor</td></tr>'
        '<tr><td>]</td><td>Set loop end at cursor</td></tr>'
        '<tr><td>← →</td><td>Select previous / next onset</td></tr>'
        '<tr><td>Shift+← →</td><td>Seek backward / forward 0.25 s</td></tr>'
        '<tr><td>1–8</td><td>Label selected onset '
        '(1 true_attack, 2 string_noise, 3 passing_note, '
        '4 downbeat, 5 beat3, 6 fill, 7 ignore, 8 uncertain)</td></tr>'
        '<tr><td>Delete</td><td>Clear selected onset annotation</td></tr>'
        '</table></details>'
    )

    # ── Assemble ─────────────────────────────────────────────────────────
    return (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>Beat Microscope</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        "<h1>Beat Microscope</h1>"
        + audio_html + overview_html + summary_html + legend_html
        + waveform_html + envelope_html + timeline_html
        + sel_status_html
        + explanation_html + grid_source_html
        + annotation_html + shortcuts_html + table_html
        + '<div id="tooltip"></div>'
        + f"<script>{js}</script>"
        + "</body></html>"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beat Microscope — inspect bass timing against a shuffle-aware beat grid.",
    )
    p.add_argument("wav", help="Stereo WAV file to analyze")
    p.add_argument("--bass-channel", type=int, default=0, dest="bass_channel")
    p.add_argument("--song-channel", type=int, default=1, dest="song_channel")
    p.add_argument("--bpm", type=float, default=146)
    p.add_argument("--beats-per-measure", type=int, default=4, dest="beats_per_measure")
    p.add_argument("--shuffle-fraction", type=float, default=0.667, dest="shuffle_fraction")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--output-dir", default="diagnostics", dest="output_dir")
    p.add_argument(
        "--delta", type=float, default=0.07,
        help="Librosa onset-strength threshold (default 0.07)",
    )
    p.add_argument(
        "--audio-export-mode", default="stereo", dest="audio_export_mode",
        choices=["stereo", "bass", "song", "mix"],
        help="Which audio to export as the sidecar WAV (default: stereo)",
    )
    p.add_argument(
        "--beat-zero-s", type=float, default=None, dest="beat_zero_s",
        help="Absolute song time of measure 1, beat 1 (overrides auto-phase)",
    )
    p.add_argument(
        "--grid-settings", type=str, default=None, dest="grid_settings",
        metavar="PATH",
        help="Load grid settings from JSON (overrides CLI bpm/meter/shuffle/beat-zero)",
    )
    p.add_argument(
        "--bass-anchor-beats", type=str, default=None, dest="bass_anchor_beats",
        metavar="BEATS",
        help="Comma-separated beat numbers for suggested phase alignment (e.g. 1,3)",
    )
    p.add_argument(
        "--auto-phase-from-bass-anchors", action="store_true", dest="auto_phase",
        help="Suggest beat_zero_s from bass anchor beats (overridden by --beat-zero-s)",
    )
    p.add_argument(
        "--use-annotations-for-phase", action="store_true",
        dest="use_annotations_for_phase",
        help="Filter/weight onsets by annotation labels during phase estimation",
    )
    p.add_argument(
        "--generate-all-windows", action="store_true", dest="generate_all_windows",
        help="Generate reports for every window across the full song",
    )
    p.add_argument(
        "--window-step", type=float, default=None, dest="window_step",
        help="Step between windows in seconds (default: same as --duration)",
    )
    return p.parse_args(argv)


def _report_filename(stem: str, start: float, multi: bool) -> str:
    if multi:
        return f"{stem}_microscope_{window_tag(start)}.html"
    return f"{stem}_microscope.html"


def _generate_window(
    *,
    wav_path: Path,
    raw_audio: np.ndarray,
    bass_full: np.ndarray,
    sr: int,
    is_stereo: bool,
    args: argparse.Namespace,
    win_start: float,
    win_duration: float,
    overview: dict,
    beat_zero: float,
    grid_source: GridSource,
    out_dir: Path,
    multi: bool,
    prev_href: str | None,
    next_href: str | None,
    total_windows: int,
    window_index: int,
) -> Path:
    """Generate a single-window report and its audio files. Returns HTML path."""
    import soundfile as sf

    bass_segment = segment_audio(bass_full, sr, win_start, win_duration)
    actual_duration = len(bass_segment) / sr

    onset_times = detect_onsets(bass_segment, sr, delta=args.delta)
    env_times, env_values = novelty_envelope(bass_segment, sr)

    if len(onset_times) > 0 and len(env_times) > 0:
        onset_strengths = np.interp(onset_times, env_times, env_values)
        peak_env = np.max(env_values)
        if peak_env > 0:
            onset_strengths = onset_strengths / peak_env
    else:
        onset_strengths = np.zeros(len(onset_times))

    beat_s = 60.0 / args.bpm
    n_measures = max(1, int(np.ceil(actual_duration / (beat_s * args.beats_per_measure))))

    grid = make_grid(
        bpm=args.bpm,
        beats_per_measure=args.beats_per_measure,
        n_measures=n_measures,
        shuffle_fraction=args.shuffle_fraction,
        offset=beat_zero,
    )

    classifications = [
        classify_onset_against_grid(
            t, args.bpm, args.beats_per_measure, args.shuffle_fraction,
            offset=beat_zero,
        )
        for t in onset_times
    ]

    # ── Audio export ─────────────────────────────────────────────────────
    wtag = window_tag(win_start) if multi else ""
    sfx = f"_{wtag}" if wtag else ""
    mode = args.audio_export_mode

    if is_stereo:
        source_segment = segment_audio(raw_audio, sr, win_start, win_duration)
        if mode == "stereo":
            export_audio = source_segment
        elif mode == "bass":
            export_audio = source_segment[:, args.bass_channel]
        elif mode == "song":
            export_audio = source_segment[:, args.song_channel]
        else:
            export_audio = np.mean(source_segment, axis=1)
    else:
        source_segment = bass_segment
        export_audio = bass_segment

    diag_source = audio_diagnostics(source_segment, "source segment")
    diag_export = audio_diagnostics(export_audio, "exported excerpt")

    audio_filename = f"{wav_path.stem}_excerpt{sfx}.wav"
    sf.write(str(out_dir / audio_filename), export_audio, sr, subtype="PCM_16")

    sidecar_srcs: dict[str, str] = {"stereo": audio_filename}
    if is_stereo:
        for label, data in [
            ("bass", source_segment[:, args.bass_channel]),
            ("song", source_segment[:, args.song_channel]),
        ]:
            fn = f"{wav_path.stem}_excerpt{sfx}_{label}.wav"
            sf.write(str(out_dir / fn), data, sr, subtype="PCM_16")
            sidecar_srcs[label] = fn

    storage_key = f"{wav_path.stem}_ann_s{window_tag(win_start)}_d{int(win_duration)}"

    report = render_report(
        wav_path=str(wav_path),
        bass_audio=bass_segment,
        sr=sr,
        bpm=args.bpm,
        beats_per_measure=args.beats_per_measure,
        shuffle_fraction=args.shuffle_fraction,
        start=win_start,
        duration=win_duration,
        onset_times=onset_times,
        classifications=classifications,
        env_times=env_times,
        env_values=env_values,
        grid=grid,
        audio_src=audio_filename,
        grid_source=grid_source,
        onset_strengths=onset_strengths,
        beat_zero_s=beat_zero,
        storage_key=storage_key,
        audio_diag_source=diag_source,
        audio_diag_export=diag_export,
        audio_export_mode=mode,
        sidecar_srcs=sidecar_srcs,
        overview=overview,
        prev_href=prev_href,
        next_href=next_href,
        total_windows=total_windows,
        window_index=window_index,
    )

    out_path = out_dir / _report_filename(wav_path.stem, win_start, multi)
    out_path.write_text(report, encoding="utf-8")
    return out_path


def main() -> None:
    args = _parse_args()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    if args.duration > 10:
        print("Note: Long segments may be visually dense; "
              "use shorter windows (2–6 s) for annotation.", file=sys.stderr)

    # ── Grid settings resolution (most-specific source wins) ─────────────
    # Priority: --grid-settings > --beat-zero-s > --auto-phase > defaults
    if args.grid_settings:
        gs = load_grid_settings(args.grid_settings)
        args.bpm = gs.get("bpm", args.bpm)
        args.beats_per_measure = gs.get("beats_per_measure", args.beats_per_measure)
        args.shuffle_fraction = gs.get("shuffle_fraction", args.shuffle_fraction)
        if args.beat_zero_s is None:
            args.beat_zero_s = gs.get("beat_zero_s", 0.0)
        print(f"Grid settings loaded from {args.grid_settings}")

    raw_audio, sr = load_audio(wav_path)
    is_stereo = raw_audio.ndim == 2
    bass_full = raw_audio[:, args.bass_channel] if is_stereo else raw_audio

    overview = compute_overview(bass_full, sr)
    total_dur = overview["total_duration_s"]

    # ── Beat-zero resolution ─────────────────────────────────────────────
    anchor_beats: list[int] = []
    if args.bass_anchor_beats:
        anchor_beats = [int(x.strip()) for x in args.bass_anchor_beats.split(",")]

    beat_zero = 0.0
    grid_source = GRID_SOURCE_FIXED_BPM

    if args.beat_zero_s is not None:
        beat_zero = args.beat_zero_s
        grid_source = GridSource(
            method="manual_beat_zero",
            description=(
                f"beat_zero_s = {beat_zero:.4f}s set manually via CLI or grid "
                f"settings file. BPM, meter, and shuffle fraction are user-supplied."
            ),
        )
        print(f"Manual beat_zero_s = {beat_zero:.4f}s")

    elif args.auto_phase and anchor_beats:
        full_onsets = detect_onsets(bass_full, sr, delta=args.delta)
        full_env_t, full_env_v = novelty_envelope(bass_full, sr)
        if len(full_onsets) > 0 and len(full_env_t) > 0:
            full_strengths = np.interp(full_onsets, full_env_t, full_env_v)
            pk = np.max(full_env_v)
            if pk > 0:
                full_strengths /= pk
        else:
            full_strengths = np.ones(len(full_onsets))

        phase_times, phase_weights = full_onsets, full_strengths
        used_annotations = False
        n_excluded = n_boosted = 0

        if args.use_annotations_for_phase:
            import json as _json
            ann_path = Path(args.output_dir) / f"{wav_path.stem}_annotations.json"
            loaded_anns: dict = {}
            if ann_path.exists():
                try:
                    data = _json.loads(ann_path.read_text(encoding="utf-8"))
                    for a in data.get("annotations", []):
                        loaded_anns[str(a["detected_onset_id"])] = a
                except (_json.JSONDecodeError, KeyError):
                    pass
            if loaded_anns:
                n_before = len(phase_times)
                phase_times, phase_weights = filter_onsets_for_phase(
                    full_onsets, full_strengths, loaded_anns)
                n_excluded = n_before - len(phase_times)
                n_boosted = sum(
                    1 for a in loaded_anns.values()
                    if a.get("label") in _PHASE_BOOST_LABELS
                    and int(a["detected_onset_id"]) < n_before)
                used_annotations = True

        beat_zero = estimate_beat_zero(
            phase_times, args.bpm, args.beats_per_measure,
            anchor_beats, onset_strengths=phase_weights)

        n_used = len(phase_times)
        ann_note = (
            f" Annotations applied: {n_excluded} excluded, "
            f"{n_boosted} boosted, {n_used} onsets used."
        ) if used_annotations else (
            f" {n_used} raw detected onsets used."
        )

        grid_source = GridSource(
            method="suggested_phase_from_bass_anchors",
            description=(
                f"Suggested beat_zero_s = {beat_zero:.4f}s from bass onsets "
                f"aligned to anchor beats {anchor_beats}. "
                f"This is a suggested alignment and may confuse "
                f"beats/subdivisions or beats 1/3 vs 2/4. "
                f"BPM, meter, and shuffle fraction are user-supplied."
                + ann_note),
        )
        print(f"Suggested phase: beat_zero_s = {beat_zero:.4f}s "
              f"(anchors={anchor_beats}, onsets_used={n_used})")

    # ── Export grid settings ─────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_grid_settings(
        out_dir / f"{wav_path.stem}_grid_settings.json",
        bpm=args.bpm, beats_per_measure=args.beats_per_measure,
        shuffle_fraction=args.shuffle_fraction, beat_zero_s=beat_zero,
        source_file=wav_path.name,
    )

    # ── Determine windows ────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win_dur = args.duration
    win_step = args.window_step if args.window_step is not None else win_dur

    if args.generate_all_windows:
        starts = list(np.arange(0, total_dur, win_step))
    else:
        starts = [args.start]

    multi = len(starts) > 1

    for idx, ws in enumerate(starts):
        prev_href = (
            _report_filename(wav_path.stem, starts[idx - 1], True)
            if idx > 0 else None
        )
        next_href = (
            _report_filename(wav_path.stem, starts[idx + 1], True)
            if idx < len(starts) - 1 else None
        )

        out_path = _generate_window(
            wav_path=wav_path, raw_audio=raw_audio, bass_full=bass_full,
            sr=sr, is_stereo=is_stereo, args=args,
            win_start=ws, win_duration=win_dur,
            overview=overview, beat_zero=beat_zero, grid_source=grid_source,
            out_dir=out_dir, multi=multi,
            prev_href=prev_href, next_href=next_href,
            total_windows=len(starts), window_index=idx,
        )
        print(f"[{idx + 1}/{len(starts)}] {out_path.name}  "
              f"({ws:.1f}–{ws + win_dur:.1f}s)")


if __name__ == "__main__":
    main()
