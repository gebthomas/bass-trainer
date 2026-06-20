"""SVG rendering helpers for Pocket Lab reports."""

from __future__ import annotations

import html
from typing import List

import numpy as np

from pocket_lab.grid import GridLine, OnsetClassification

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


def svg_waveform(
    audio: np.ndarray,
    sr: int,
    duration: float,
    svg_id: str = "svg-waveform",
) -> str:
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
        f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="plot-svg" id="{svg_id}" '
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


def svg_envelope(
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


def svg_timeline(
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


def svg_overview(
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
