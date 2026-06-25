#!/usr/bin/env python3
"""Phase Profile — discover where bass onsets cluster within the drum beat.

Instead of assigning onsets to a pre-chosen grid and measuring error,
this script estimates the distribution of beat-relative phases and finds
peaks. The peaks reveal which subdivisions the bassist actually uses.

Usage
-----
    python scripts/phase_profile_analysis.py "music-library/Mr. Brightside"
"""

from __future__ import annotations

import csv
import html as html_mod
import sys
from pathlib import Path

import librosa
import numpy as np


# ── Audio helpers ───────────────────────────────────────────────────────────

def scalar_tempo(x):
    return float(x[0]) if hasattr(x, "__len__") else float(x)


def detect_beats(drum_file: Path):
    y, sr = librosa.load(drum_file, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return scalar_tempo(tempo), beat_times


def detect_bass_onsets(bass_file: Path):
    y, sr = librosa.load(bass_file, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units="frames", backtrack=True,
        pre_max=3, post_max=3, pre_avg=8, post_avg=8,
        delta=0.2, wait=3,
    )
    return librosa.frames_to_time(onset_frames, sr=sr)


# ── Phase computation ───────────────────────────────────────────────────────

def compute_phases(bass_onsets, beat_times):
    """Compute beat-relative phase for each bass onset.

    Returns arrays: phases [0,1), beat_durations_ms, song_times.
    Only includes onsets that fall within a valid beat interval.
    """
    phases = []
    beat_durations_ms = []
    song_times = []

    for t in bass_onsets:
        i = np.searchsorted(beat_times, t) - 1
        if i < 0 or i >= len(beat_times) - 1:
            continue
        b0 = beat_times[i]
        b1 = beat_times[i + 1]
        beat_len = b1 - b0
        if beat_len <= 0:
            continue
        phase = (t - b0) / beat_len
        if phase < 0.0 or phase >= 1.0:
            continue
        phases.append(phase)
        beat_durations_ms.append(beat_len * 1000.0)
        song_times.append(t)

    return np.array(phases), np.array(beat_durations_ms), np.array(song_times)


# ── Circular histogram + smoothing ─────────────────────────────────────────

def phase_histogram(phases, n_bins=64):
    counts, edges = np.histogram(phases, bins=n_bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return counts, centers, edges


def circular_smooth(counts, sigma_bins=1.5):
    """Gaussian smooth a histogram with circular (wrap-around) boundary."""
    n = len(counts)
    kernel_half = int(4 * sigma_bins)
    kernel_x = np.arange(-kernel_half, kernel_half + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel /= kernel.sum()

    padded = np.concatenate([counts[-kernel_half:], counts, counts[:kernel_half]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[kernel_half : kernel_half + n]


# ── Peak detection ──────────────────────────────────────────────────────────

MUSICAL_LABELS = [
    (0.000, "beat"),
    (0.250, "e"),
    (0.333, "triplet-e"),
    (0.500, "&"),
    (0.667, "triplet-a"),
    (0.750, "a"),
]


def nearest_label(phase):
    best_dist = 1.0
    best_label = "?"
    best_pos = 0.0
    for pos, label in MUSICAL_LABELS:
        d = min(abs(phase - pos), abs(phase - pos - 1.0), abs(phase - pos + 1.0))
        if d < best_dist:
            best_dist = d
            best_label = label
            best_pos = pos
    return best_label, best_pos, best_dist


def detect_peaks(smoothed, bin_centers, min_height_frac=0.08,
                 merge_distance=0.10):
    """Find local maxima in the smoothed histogram, merging across wrap.

    min_height_frac: peak must be at least this fraction of the global max.
    merge_distance: peaks closer than this (circular) are merged into the
        taller one; the shorter peak's counts are absorbed.
    Returns list of (bin_index, center_phase, height).
    """
    n = len(smoothed)
    threshold = smoothed.max() * min_height_frac
    raw_peaks = []
    for i in range(n):
        left = (i - 1) % n
        right = (i + 1) % n
        if smoothed[i] > smoothed[left] and smoothed[i] > smoothed[right]:
            if smoothed[i] >= threshold:
                raw_peaks.append((i, bin_centers[i], smoothed[i]))

    raw_peaks.sort(key=lambda p: -p[2])
    merged = []
    used = set()
    for i, (bi, ci, hi) in enumerate(raw_peaks):
        if i in used:
            continue
        for j, (bj, cj, hj) in enumerate(raw_peaks):
            if j <= i or j in used:
                continue
            d = abs(ci - cj)
            d = min(d, 1.0 - d)
            if d < merge_distance:
                used.add(j)
        merged.append((bi, ci, hi))
    return merged


def refine_peak_center(phases, rough_center, radius=0.12):
    """Re-estimate peak center using circular mean of nearby onsets."""
    diffs = phases - rough_center
    diffs = diffs - np.round(diffs)
    mask = np.abs(diffs) < radius
    if mask.sum() == 0:
        return rough_center
    local = diffs[mask]
    refined = rough_center + float(np.mean(local))
    return refined % 1.0


def peak_spread_ms(phases, peak_center, beat_durations_ms, capture_radius=0.12):
    """Compute spread (std dev in ms) of onsets near a peak.

    Uses circular distance so clusters straddling phase=0 are handled.
    """
    diffs = phases - peak_center
    diffs = diffs - np.round(diffs)  # wrap to [-0.5, 0.5)
    mask = np.abs(diffs) < capture_radius
    if mask.sum() < 2:
        return 0.0, 0.0, int(mask.sum())
    local_phases = diffs[mask]
    local_beat_ms = beat_durations_ms[mask]
    deviations_ms = local_phases * local_beat_ms
    spread = float(np.std(deviations_ms))
    bias = float(np.mean(deviations_ms))
    return spread, bias, int(mask.sum())


# ── Interpretation ──────────────────────────────────────────────────────────

def interpret_profile(peak_info, total_onsets):
    """Generate human-readable interpretation from peak analysis."""
    lines = []

    dominant = [p for p in peak_info if p["occupancy_pct"] >= 15.0]
    minor = [p for p in peak_info if 5.0 <= p["occupancy_pct"] < 15.0]

    if not dominant:
        lines.append("No dominant phase clusters found — onset distribution "
                      "appears diffuse or noisy.")
        return "\n".join(lines)

    labels = [p["nearest_label"] for p in dominant]

    if set(labels) <= {"beat", "&"}:
        lines.append("Dominant clusters at beat and &: primarily an "
                      "eighth-note bass line.")
    elif set(labels) <= {"beat", "e", "&", "a"}:
        lines.append("Dominant clusters at standard sixteenth-note positions: "
                      "primarily a sixteenth-note bass line.")
    elif "triplet-a" in labels or "triplet-e" in labels:
        lines.append("Dominant clusters include triplet positions: "
                      "suggests shuffle or triplet feel.")
    else:
        label_str = ", ".join(labels)
        lines.append(f"Dominant clusters at: {label_str}.")

    dominant_count = sum(p["occupancy_n"] for p in dominant)
    dominant_pct = 100.0 * dominant_count / total_onsets if total_onsets > 0 else 0.0

    if dominant_pct < 60:
        lines.append(f"Warning: only {dominant_pct:.0f}% of onsets fall within "
                      "dominant clusters. Significant off-grid activity or "
                      "onset-detection noise may be present.")

    if minor:
        minor_labels = [f"{p['nearest_label']} ({p['occupancy_pct']:.0f}%)"
                        for p in minor]
        lines.append(f"Minor clusters: {', '.join(minor_labels)}.")

    for p in dominant:
        bias = p["bias_ms"]
        if abs(bias) > 5.0:
            direction = "behind" if bias > 0 else "ahead of"
            lines.append(
                f"  {p['nearest_label']}: systematic {abs(bias):.1f} ms "
                f"{direction} the grid ({p['spread_ms']:.1f} ms spread).")

    return "\n".join(lines)


# ── CSV output ──────────────────────────────────────────────────────────────

def write_summary_csv(path, song_name, tempo, peak_info):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song", "tempo_bpm", "peak_phase", "nearest_label",
                     "label_phase", "occupancy_n", "occupancy_pct",
                     "spread_ms", "bias_ms"])
        for p in peak_info:
            w.writerow([
                song_name, f"{tempo:.2f}", f"{p['peak_phase']:.4f}",
                p["nearest_label"], f"{p['label_phase']:.4f}",
                p["occupancy_n"], f"{p['occupancy_pct']:.1f}",
                f"{p['spread_ms']:.1f}", f"{p['bias_ms']:.1f}",
            ])


def write_observations_csv(path, phases, beat_durations_ms, song_times):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song_time_s", "phase", "beat_duration_ms"])
        for i in range(len(phases)):
            w.writerow([
                f"{song_times[i]:.4f}",
                f"{phases[i]:.6f}",
                f"{beat_durations_ms[i]:.1f}",
            ])


# ── HTML report ─────────────────────────────────────────────────────────────

_CSS = """\
body {
    font-family: 'Menlo','Consolas',monospace;
    background: #0f0f23; color: #ccc;
    padding: 20px; max-width: 960px; margin: auto;
}
h1 { color: #4cc9f0; margin-bottom: 0.3em; }
h2 { color: #ff9f1c; margin-top: 1.5em; margin-bottom: 0.4em; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
th, td { border: 1px solid #333; padding: 5px 8px; text-align: left; }
th { background: #1a1a2e; color: #4cc9f0; }
tr:nth-child(even) { background: #16162a; }
.interpretation {
    background: #1a1a2e; border-left: 3px solid #ff9f1c;
    padding: 12px 16px; margin: 1em 0; border-radius: 0 4px 4px 0;
    font-size: 13px; line-height: 1.7; white-space: pre-line;
}
.meta { color: #888; font-size: 12px; margin: 0.3em 0; }
"""

# SVG layout constants
_SVG_W = 900
_HIST_H = 300
_STRIP_H = 400
_PLOT_L = 60
_PLOT_R = 20
_PLOT_T = 25
_PLOT_B = 40
_PLOT_W = _SVG_W - _PLOT_L - _PLOT_R

_LABEL_COLORS = {
    "beat": "#ff9f1c",
    "e": "#4895ef",
    "&": "#ff6b6b",
    "a": "#06d6a0",
    "triplet-e": "#9b5de5",
    "triplet-a": "#9b5de5",
}

_REF_LINES = [
    (0.000, "beat", "#ff9f1c"),
    (0.250, "e", "#4895ef"),
    (0.500, "&", "#ff6b6b"),
    (0.750, "a", "#06d6a0"),
    (2 / 3, "trip", "#9b5de5"),
]


def _phase_to_x(phase):
    return _PLOT_L + phase * _PLOT_W


def _svg_histogram(counts, centers, smoothed, peaks, peak_info, total_onsets):
    """Render the phase histogram with smoothed overlay and peak markers."""
    plot_h = _HIST_H - _PLOT_T - _PLOT_B
    max_count = max(counts.max(), 1)
    max_smooth = max(smoothed.max(), 1e-9)
    bin_width = _PLOT_W / len(counts)

    parts = []

    # background
    parts.append(
        f'<rect x="{_PLOT_L}" y="{_PLOT_T}" width="{_PLOT_W}" '
        f'height="{plot_h}" fill="#1a1a2e" />')

    # reference lines
    for phase, label, color in _REF_LINES:
        x = _phase_to_x(phase)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_PLOT_T}" x2="{x:.1f}" '
            f'y2="{_PLOT_T + plot_h}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="4,3" opacity="0.5" />')
        parts.append(
            f'<text x="{x:.1f}" y="{_PLOT_T - 4}" fill="{color}" '
            f'font-size="9" text-anchor="middle">{label}</text>')

    # histogram bars
    for i, c in enumerate(counts):
        x = _PLOT_L + i * bin_width
        bar_h = (c / max_count) * (plot_h - 4)
        y = _PLOT_T + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bin_width:.1f}" '
            f'height="{bar_h:.1f}" fill="#4cc9f0" opacity="0.6" />')

    # smoothed density overlay
    scale_y = (plot_h - 4) / max_smooth
    points = []
    for i, s in enumerate(smoothed):
        x = _phase_to_x(centers[i])
        y = _PLOT_T + plot_h - s * scale_y
        points.append(f"{x:.1f},{y:.1f}")
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="#fff" stroke-width="2" opacity="0.9" />')

    # peak markers
    for pi, (bin_idx, center, height) in enumerate(peaks):
        x = _phase_to_x(center)
        y = _PLOT_T + plot_h - (height / max_smooth) * (plot_h - 4)
        info = peak_info[pi] if pi < len(peak_info) else {}
        label = info.get("nearest_label", "?")
        pct = info.get("occupancy_pct", 0)
        color = _LABEL_COLORS.get(label, "#fff")
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
            f'stroke="#fff" stroke-width="1.5" />')
        parts.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" fill="{color}" '
            f'font-size="11" text-anchor="middle" font-weight="bold">'
            f'{label} {pct:.0f}%</text>')

    # x-axis
    for tick in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
        x = _phase_to_x(tick)
        y_base = _PLOT_T + plot_h
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_base}" x2="{x:.1f}" '
            f'y2="{y_base + 4}" stroke="#888" stroke-width="0.5" />')
        parts.append(
            f'<text x="{x:.1f}" y="{y_base + 16}" fill="#888" '
            f'font-size="9" text-anchor="middle">{tick:.3g}</text>')

    # axis labels
    parts.append(
        f'<text x="{_PLOT_L + _PLOT_W / 2}" y="{_HIST_H - 2}" fill="#888" '
        f'font-size="11" text-anchor="middle">Phase within beat</text>')
    parts.append(
        f'<text x="14" y="{_PLOT_T + plot_h / 2 + 4}" fill="#888" '
        f'font-size="10" text-anchor="middle" '
        f'transform="rotate(-90,14,{_PLOT_T + plot_h / 2})">Count</text>')

    return (
        f'<svg viewBox="0 0 {_SVG_W} {_HIST_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#0f0f23;width:100%;display:block;margin:0.5em 0;">'
        + "\n".join(parts)
        + '</svg>')


def _svg_strip_plot(phases, song_times, peak_info):
    """Phase strip plot: x=phase, y=time through song."""
    plot_h = _STRIP_H - _PLOT_T - _PLOT_B
    if len(song_times) == 0:
        return "<p>No onsets to plot.</p>"

    t_min = song_times.min()
    t_max = song_times.max()
    t_range = max(t_max - t_min, 0.1)

    parts = []

    parts.append(
        f'<rect x="{_PLOT_L}" y="{_PLOT_T}" width="{_PLOT_W}" '
        f'height="{plot_h}" fill="#1a1a2e" />')

    # reference lines
    for phase, label, color in _REF_LINES:
        x = _phase_to_x(phase)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_PLOT_T}" x2="{x:.1f}" '
            f'y2="{_PLOT_T + plot_h}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="4,3" opacity="0.3" />')

    # cluster bands (light background shading for detected peaks)
    for p in peak_info:
        cx = _phase_to_x(p["peak_phase"])
        band_w = _PLOT_W * 0.16  # capture_radius = 0.08 on each side
        color = _LABEL_COLORS.get(p["nearest_label"], "#fff")
        parts.append(
            f'<rect x="{cx - band_w / 2:.1f}" y="{_PLOT_T}" '
            f'width="{band_w:.1f}" height="{plot_h}" '
            f'fill="{color}" opacity="0.06" />')

    # onset dots
    for i in range(len(phases)):
        x = _phase_to_x(phases[i])
        y = _PLOT_T + ((song_times[i] - t_min) / t_range) * plot_h
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
            f'fill="#4cc9f0" opacity="0.6" />')

    # x-axis (phase)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = _phase_to_x(tick)
        y_base = _PLOT_T + plot_h
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_base}" x2="{x:.1f}" '
            f'y2="{y_base + 4}" stroke="#888" stroke-width="0.5" />')
        parts.append(
            f'<text x="{x:.1f}" y="{y_base + 16}" fill="#888" '
            f'font-size="9" text-anchor="middle">{tick:.2g}</text>')

    # y-axis (song time)
    n_ticks = 6
    for j in range(n_ticks + 1):
        t = t_min + j * t_range / n_ticks
        y = _PLOT_T + j * plot_h / n_ticks
        parts.append(
            f'<line x1="{_PLOT_L - 4}" y1="{y:.1f}" x2="{_PLOT_L}" '
            f'y2="{y:.1f}" stroke="#888" stroke-width="0.5" />')
        parts.append(
            f'<text x="{_PLOT_L - 6}" y="{y + 3:.1f}" fill="#888" '
            f'font-size="9" text-anchor="end">{t:.0f}s</text>')

    parts.append(
        f'<text x="{_PLOT_L + _PLOT_W / 2}" y="{_STRIP_H - 2}" fill="#888" '
        f'font-size="11" text-anchor="middle">Phase within beat</text>')
    parts.append(
        f'<text x="14" y="{_PLOT_T + plot_h / 2 + 4}" fill="#888" '
        f'font-size="10" text-anchor="middle" '
        f'transform="rotate(-90,14,{_PLOT_T + plot_h / 2})">Song time</text>')

    return (
        f'<svg viewBox="0 0 {_SVG_W} {_STRIP_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#0f0f23;width:100%;display:block;margin:0.5em 0;">'
        + "\n".join(parts)
        + '</svg>')


def _peak_table_html(peak_info):
    rows = []
    for p in peak_info:
        color = _LABEL_COLORS.get(p["nearest_label"], "#ccc")
        rows.append(
            f'<tr>'
            f'<td style="color:{color};font-weight:bold">'
            f'{html_mod.escape(p["nearest_label"])}</td>'
            f'<td>{p["peak_phase"]:.4f}</td>'
            f'<td>{p["label_phase"]:.4f}</td>'
            f'<td>{p["occupancy_n"]}</td>'
            f'<td>{p["occupancy_pct"]:.1f}%</td>'
            f'<td>{p["spread_ms"]:.1f}</td>'
            f'<td>{p["bias_ms"]:+.1f}</td>'
            f'</tr>')
    return (
        '<table>'
        '<tr><th>Label</th><th>Peak phase</th><th>Label phase</th>'
        '<th>Count</th><th>Pct</th><th>Spread (ms)</th><th>Bias (ms)</th></tr>'
        + "\n".join(rows)
        + '</table>')


def render_html(song_name, tempo, n_beats, n_onsets, n_phases,
                counts, centers, smoothed, peaks, peak_info,
                phases, song_times, interpretation):
    hist_svg = _svg_histogram(counts, centers, smoothed, peaks, peak_info,
                              n_phases)
    strip_svg = _svg_strip_plot(phases, song_times, peak_info)
    peak_table = _peak_table_html(peak_info)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase Profile — {html_mod.escape(song_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Phase Profile — {html_mod.escape(song_name)}</h1>
<p class="meta">
    Tempo: {tempo:.2f} BPM &nbsp;|&nbsp;
    Beats: {n_beats} &nbsp;|&nbsp;
    Bass onsets: {n_onsets} &nbsp;|&nbsp;
    Onsets in beat range: {n_phases}
</p>

<h2>Phase Histogram</h2>
<p class="meta">
    Bar chart shows raw onset counts per phase bin.
    White curve is the smoothed density.
    Dashed lines mark standard subdivision positions.
    Circles mark detected peaks.
</p>
{hist_svg}

<h2>Phase Clusters</h2>
{peak_table}

<h2>Phase Strip Plot</h2>
<p class="meta">
    Each dot is one bass onset. X = phase within beat, Y = time through song.
    Vertical bands highlight detected clusters.
    Stable vertical columns indicate consistent subdivision placement.
</p>
{strip_svg}

<h2>Interpretation</h2>
<div class="interpretation">{html_mod.escape(interpretation)}</div>

</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python scripts/phase_profile_analysis.py '
            '"music-library/Mr. Brightside"')

    song_dir = Path(sys.argv[1])
    drum_file = song_dir / "drums.wav"
    bass_file = song_dir / "bass.wav"

    for f in [drum_file, bass_file]:
        if not f.exists():
            raise SystemExit(f"Error: {f} not found")

    # 1-4: detect beats and bass onsets
    print(f"Loading {drum_file} ...")
    tempo, beat_times = detect_beats(drum_file)
    print(f"Loading {bass_file} ...")
    bass_onsets = detect_bass_onsets(bass_file)

    print(f"Song: {song_dir.name}")
    print(f"Tempo: {tempo:.2f} BPM")
    print(f"Beats: {len(beat_times)}")
    print(f"Bass onsets: {len(bass_onsets)}")

    if len(beat_times) < 4:
        raise SystemExit("Too few beats detected — cannot build phase profile.")
    if len(bass_onsets) < 4:
        raise SystemExit("Too few bass onsets detected.")

    # 5-7: compute phases
    phases, beat_durations_ms, song_times = compute_phases(
        bass_onsets, beat_times)
    print(f"Onsets in beat range: {len(phases)}")

    if len(phases) < 4:
        raise SystemExit("Too few onsets fall within beat intervals.")

    # 8-9: histogram + smoothing
    n_bins = 64
    counts, centers, edges = phase_histogram(phases, n_bins=n_bins)
    smoothed = circular_smooth(counts, sigma_bins=1.5)

    # 10-11: peak detection and characterization
    peaks = detect_peaks(smoothed, centers, min_height_frac=0.08)
    peaks.sort(key=lambda p: -p[2])  # sort by height descending

    peak_info = []
    for bin_idx, center, height in peaks:
        center = refine_peak_center(phases, center)
        label, label_phase, label_dist = nearest_label(center)
        spread, bias, n_in_cluster = peak_spread_ms(
            phases, center, beat_durations_ms, capture_radius=0.12)
        # bias_ms: offset from nearest theoretical label (not from peak center)
        label_offset = center - label_phase
        if abs(label_offset) > 0.5:
            label_offset -= np.sign(label_offset)
        mean_beat_ms = float(np.mean(beat_durations_ms)) if len(beat_durations_ms) > 0 else 400.0
        bias_from_label = label_offset * mean_beat_ms

        peak_info.append({
            "peak_phase": center,
            "nearest_label": label,
            "label_phase": label_phase,
            "occupancy_n": n_in_cluster,
            "occupancy_pct": 100.0 * n_in_cluster / len(phases) if len(phases) > 0 else 0.0,
            "spread_ms": spread,
            "bias_ms": bias_from_label,
        })

    # Print summary
    print()
    print("Phase clusters:")
    print(f"{'Label':12s} {'Phase':>8s} {'Ref':>8s} {'Count':>6s} "
          f"{'Pct':>6s} {'Spread':>8s} {'Bias':>8s}")
    for p in peak_info:
        print(f"{p['nearest_label']:12s} {p['peak_phase']:8.4f} "
              f"{p['label_phase']:8.4f} {p['occupancy_n']:6d} "
              f"{p['occupancy_pct']:5.1f}% "
              f"{p['spread_ms']:7.1f}ms {p['bias_ms']:+7.1f}ms")

    interpretation = interpret_profile(peak_info, len(phases))
    print()
    print("Interpretation:")
    print(interpretation)

    # 12: write outputs
    write_summary_csv(song_dir / "phase_profile_summary.csv",
                      song_dir.name, tempo, peak_info)
    write_observations_csv(song_dir / "phase_observations.csv",
                           phases, beat_durations_ms, song_times)

    html_path = song_dir / "phase_profile_report.html"
    html_content = render_html(
        song_name=song_dir.name,
        tempo=tempo,
        n_beats=len(beat_times),
        n_onsets=len(bass_onsets),
        n_phases=len(phases),
        counts=counts,
        centers=centers,
        smoothed=smoothed,
        peaks=peaks,
        peak_info=peak_info,
        phases=phases,
        song_times=song_times,
        interpretation=interpretation,
    )
    html_path.write_text(html_content, encoding="utf-8")

    print()
    print(f"Wrote {song_dir / 'phase_profile_summary.csv'}")
    print(f"Wrote {song_dir / 'phase_observations.csv'}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
