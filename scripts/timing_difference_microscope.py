#!/usr/bin/env python3
"""Timing Difference Microscope — HTML report for inspecting matched onset pairs.

Reuses the onset detection and matching logic from bass_onset_compare.py.
Generates per-event audio excerpts and an interactive HTML inspector.
"""

from __future__ import annotations

import html as html_mod
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent
BASS_REF = PROJECT / "music-library" / "Mr. Brightside" / "bass.wav"
MASTER = PROJECT / "music-library" / "Mr. Brightside" / "master.wav"
TAKE_A = PROJECT / "personal recordings" / "June-22-2026" / "Mr Brightside memory.wav"
TAKE_B = PROJECT / "personal recordings" / "June-22-2026" / "Mr Brightside memory 2.wav"

OFFSET_A = 1.4307
OFFSET_B = 0.0834

ANALYSIS_START = 45.0
ANALYSIS_END = 200.0
MATCH_WINDOW_MS = 150.0

WINDOW_BEFORE = 1.0
WINDOW_AFTER = 1.5
WINDOW_DUR = WINDOW_BEFORE + WINDOW_AFTER

TOP_N = 10

OUT_DIR = PROJECT / "diagnostics" / "microscope"


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def load_left(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    return data[:, 0], sr


def detect_onsets(audio: np.ndarray, sr: int, delta: float = 0.07) -> np.ndarray:
    return librosa.onset.onset_detect(
        y=audio.astype(np.float32), sr=sr,
        backtrack=True, delta=delta, units="time",
    )


def filter_to_window(onsets: np.ndarray, start: float, end: float) -> np.ndarray:
    return onsets[(onsets >= start) & (onsets <= end)]


def match_onsets(
    onsets_a: np.ndarray, onsets_b: np.ndarray, max_window_s: float,
) -> list[tuple[float, float, float]]:
    used_b = set()
    rows = []
    for i, ta in enumerate(onsets_a):
        best_j, best_dist = None, max_window_s + 1
        for j, tb in enumerate(onsets_b):
            if j in used_b:
                continue
            dist = abs(ta - tb)
            if dist <= max_window_s and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            used_b.add(best_j)
            rows.append((ta, onsets_b[best_j], (onsets_b[best_j] - ta) * 1000.0))
    return rows


def segment(audio: np.ndarray, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    out_len = int(dur_s * sr)
    out = np.zeros(out_len, dtype=audio.dtype)
    s0 = int(start_s * sr)
    read_start = max(0, s0)
    write_offset = read_start - s0
    read_end = min(len(audio), s0 + out_len)
    length = read_end - read_start
    if length > 0 and write_offset >= 0:
        actual_len = min(length, out_len - write_offset)
        out[write_offset:write_offset + actual_len] = audio[read_start:read_start + actual_len]
    return out


def export_event_audio(
    event_id: str,
    ref_time: float,
    audio_tracks: dict[str, tuple[np.ndarray, int, float]],
) -> dict[str, str]:
    """Export audio segments for one event. Returns {label: filename}."""
    win_start = ref_time - WINDOW_BEFORE
    filenames = {}
    for label, (audio, sr, offset) in audio_tracks.items():
        native_start = win_start - offset
        seg = segment(audio, sr, native_start, WINDOW_DUR)
        fname = f"{event_id}_{label.replace(' ', '_').lower()}.wav"
        sf.write(str(OUT_DIR / fname), seg, sr, subtype="PCM_16")
        filenames[label] = fname
    return filenames


def svg_waveform(audio: np.ndarray, sr: int, color: str, label: str, svg_id: str) -> str:
    SVG_W, SVG_H = 830, 100
    PLOT_L, PLOT_R, PLOT_T, PLOT_B = 55, 15, 5, 18
    PLOT_W = SVG_W - PLOT_L - PLOT_R
    PLOT_H = SVG_H - PLOT_T - PLOT_B

    n = len(audio)
    if n == 0:
        return "<p>No audio.</p>"

    bucket = max(1, n // PLOT_W)
    nb = n // bucket
    trimmed = audio[:nb * bucket].reshape(nb, bucket)
    maxes = np.max(trimmed, axis=1)
    mins = np.min(trimmed, axis=1)
    peak = max(np.max(np.abs(maxes)), np.max(np.abs(mins)), 1e-9)
    mid_y = PLOT_T + PLOT_H / 2

    pts_top, pts_bot = [], []
    for i in range(nb):
        x = PLOT_L + i * PLOT_W / nb
        yt = mid_y - (maxes[i] / peak) * (PLOT_H / 2 - 2)
        yb = mid_y - (mins[i] / peak) * (PLOT_H / 2 - 2)
        pts_top.append(f"{x:.1f},{yt:.1f}")
        pts_bot.append(f"{x:.1f},{yb:.1f}")
    pts_bot.reverse()
    poly = " ".join(pts_top + pts_bot)

    duration = n / sr
    ticks = []
    for t_rel in np.arange(0, duration + 0.1, 0.25):
        tx = PLOT_L + (t_rel / duration) * PLOT_W
        if t_rel == round(t_rel):
            ticks.append(
                f'<text x="{tx:.1f}" y="{SVG_H - 2}" fill="#888" font-size="8" '
                f'text-anchor="middle">{t_rel - WINDOW_BEFORE:+.2f}s</text>'
            )
        ticks.append(
            f'<line x1="{tx:.1f}" y1="{SVG_H - PLOT_B}" x2="{tx:.1f}" '
            f'y2="{SVG_H - PLOT_B + 3}" stroke="#666" stroke-width="0.5" />'
        )

    zero_x = PLOT_L + (WINDOW_BEFORE / duration) * PLOT_W
    zero_line = (
        f'<line x1="{zero_x:.1f}" y1="{PLOT_T}" x2="{zero_x:.1f}" '
        f'y2="{PLOT_T + PLOT_H}" stroke="#ff6b6b" stroke-width="1.5" '
        f'stroke-dasharray="4,3" opacity="0.7" />'
    )

    y_label = (
        f'<text x="12" y="{mid_y + 4}" fill="#888" font-size="9" '
        f'transform="rotate(-90,12,{mid_y})" text-anchor="middle">'
        f'{html_mod.escape(label)}</text>'
    )

    cursor_line = (
        f'<line class="cursor" x1="{PLOT_L}" y1="{PLOT_T}" '
        f'x2="{PLOT_L}" y2="{PLOT_T + PLOT_H}" '
        f'stroke="#fff" stroke-width="1.5" opacity="0" />'
    )

    return (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" class="plot-svg" id="{svg_id}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;">'
        f'<line x1="{PLOT_L}" y1="{mid_y}" x2="{PLOT_L + PLOT_W}" y2="{mid_y}" '
        f'stroke="#444" stroke-width="0.5" />'
        f'<polygon points="{poly}" fill="{color}" opacity="0.8" />'
        + zero_line + y_label + "\n".join(ticks) + cursor_line
        + "</svg>"
    )


def build_event_html(
    event: dict,
    audio_tracks: dict[str, tuple[np.ndarray, int, float]],
) -> str:
    eid = event["id"]
    ref_time = event["ref_time"]
    win_start = ref_time - WINDOW_BEFORE

    filenames = export_event_audio(eid, ref_time, audio_tracks)

    colors = ["#4cc9f0", "#2ecc71", "#f1c40f"]
    waveforms = []
    channel_options = []
    first_src = None
    for idx, (label, (audio, sr, offset)) in enumerate(audio_tracks.items()):
        native_start = win_start - offset
        seg = segment(audio, sr, native_start, WINDOW_DUR)
        svg_id = f"svg-{eid}-{idx}"
        waveforms.append(svg_waveform(seg, sr, colors[idx % len(colors)], label, svg_id))
        src = filenames[label]
        if first_src is None:
            first_src = src
        sel = " selected" if idx == 0 else ""
        channel_options.append(
            f'<option value="{html_mod.escape(src)}"{sel}>'
            f'{html_mod.escape(label)}</option>'
        )

    speed_buttons = "".join(
        f'<button class="tbtn speed-btn{" active" if s == "1.0" else ""}" '
        f'data-speed="{s}" data-player="{eid}">{s}x</button>'
        for s in ["1.0", "0.75", "0.5", "0.33", "0.25"]
    )

    diff = event["diff_ms"]
    sign = "+" if diff >= 0 else ""
    diff_color = "#2ecc71" if abs(diff) < 30 else "#f1c40f" if abs(diff) < 80 else "#ff6b6b"

    section = f"""
    <div class="event-card" id="{eid}">
      <div class="event-header">
        <span class="event-rank">#{event['rank']}</span>
        <span class="event-category">{html_mod.escape(event['category'])}</span>
        <span class="event-diff" style="color:{diff_color}">{sign}{diff:.1f} ms</span>
        <span class="event-time">{ref_time:.3f} s</span>
      </div>
      <div class="audio-bar">
        <audio id="audio-{eid}" controls preload="none"
               src="{html_mod.escape(first_src)}"></audio>
        <div class="transport-row">
          <label>Speed:</label>{speed_buttons}
          <label>Track:</label>
          <select class="channel-select" data-player="{eid}">
            {"".join(channel_options)}
          </select>
        </div>
      </div>
      {"".join(waveforms)}
    </div>
    """
    return section


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
td a { color: #4cc9f0; text-decoration: none; }
td a:hover { text-decoration: underline; }
svg.plot-svg {
    display: block; width: 100%; margin: 0.2em 0;
    border-radius: 4px; cursor: pointer;
}
.audio-bar audio { width: 100%; }
.transport-row {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    padding: 4px 0; font-size: 12px;
}
.transport-row label { color: #888; }
.transport-row select {
    background: #1a1a2e; color: #ccc; border: 1px solid #333;
    padding: 2px 6px; border-radius: 3px; font-family: inherit;
    font-size: 11px;
}
.tbtn {
    background: #1a1a2e; color: #ccc; border: 1px solid #333;
    padding: 3px 8px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit;
}
.tbtn:hover { border-color: #4cc9f0; color: #4cc9f0; }
.tbtn.active { background: #4cc9f0; color: #0f0f23; border-color: #4cc9f0; }
.event-card {
    border: 1px solid #333; border-radius: 6px;
    padding: 12px 16px; margin: 1em 0;
    background: #12122a;
}
.event-header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 8px; font-size: 14px;
}
.event-rank { color: #4cc9f0; font-weight: bold; font-size: 18px; }
.event-category { color: #888; }
.event-diff { font-weight: bold; font-size: 16px; }
.event-time { color: #888; margin-left: auto; }
.comparison-section { margin-top: 2em; }
.comparison-header {
    color: #ff9f1c; font-size: 1.3em; font-weight: bold;
    border-bottom: 2px solid #ff9f1c; padding-bottom: 4px;
    margin-bottom: 0.5em;
}
.category-header {
    color: #ccc; font-size: 1.1em; margin-top: 1.5em; margin-bottom: 0.3em;
}
"""

_JS = """\
(function() {
    var SVG_W = 830, PLOT_L = 55, PLOT_R = 15;
    var PLOT_W = SVG_W - PLOT_L - PLOT_R;
    var PLOT_T = 5, PLOT_H = 77;
    var DURATION = %DURATION%;

    function timeToX(t) { return PLOT_L + (t / DURATION) * PLOT_W; }
    function xToTime(svg, e) {
        var pt = svg.createSVGPoint();
        pt.x = e.clientX; pt.y = e.clientY;
        var svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
        return Math.max(0, Math.min(((svgPt.x - PLOT_L) / PLOT_W) * DURATION, DURATION));
    }

    /* Speed buttons */
    document.querySelectorAll('.speed-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var pid = btn.dataset.player;
            var audio = document.getElementById('audio-' + pid);
            if (!audio) return;
            audio.playbackRate = parseFloat(btn.dataset.speed);
            document.querySelectorAll('.speed-btn[data-player="'+pid+'"]').forEach(function(b) {
                b.classList.remove('active');
            });
            btn.classList.add('active');
        });
    });

    /* Channel switching */
    document.querySelectorAll('.channel-select').forEach(function(sel) {
        sel.addEventListener('change', function() {
            var pid = sel.dataset.player;
            var audio = document.getElementById('audio-' + pid);
            if (!audio) return;
            var cur = audio.currentTime;
            var playing = !audio.paused;
            audio.src = sel.value;
            audio.currentTime = cur;
            if (playing) audio.play();
        });
    });

    /* Click-to-seek on waveforms */
    document.querySelectorAll('.plot-svg').forEach(function(svg) {
        var card = svg.closest('.event-card');
        if (!card) return;
        var pid = card.id;
        svg.addEventListener('click', function(e) {
            var audio = document.getElementById('audio-' + pid);
            if (!audio) return;
            audio.currentTime = xToTime(svg, e);
        });
    });

    /* Cursor sync */
    document.querySelectorAll('.event-card').forEach(function(card) {
        var pid = card.id;
        var audio = document.getElementById('audio-' + pid);
        if (!audio) return;
        var cursors = card.querySelectorAll('.cursor');
        function update() {
            var x = timeToX(audio.currentTime);
            cursors.forEach(function(c) {
                c.setAttribute('x1', x); c.setAttribute('x2', x);
                c.setAttribute('opacity', '0.8');
            });
            if (!audio.paused) requestAnimationFrame(update);
        }
        audio.addEventListener('play', function() { requestAnimationFrame(update); });
        audio.addEventListener('seeked', update);
        audio.addEventListener('pause', update);
        audio.addEventListener('timeupdate', update);
    });
})();
"""


def main():
    print("Loading audio...")
    bass_ref, sr_ref = load_mono(BASS_REF)
    master, sr_master = load_mono(MASTER)
    take_a_raw, sr_a = sf.read(TAKE_A, dtype="float64")
    take_b_raw, sr_b = sf.read(TAKE_B, dtype="float64")
    take_a_bass = take_a_raw[:, 0]
    take_b_bass = take_b_raw[:, 0]

    print("Detecting onsets...")
    ref_onsets = detect_onsets(bass_ref, sr_ref)
    a_onsets_raw = detect_onsets(take_a_bass, sr_a)
    b_onsets_raw = detect_onsets(take_b_bass, sr_b)

    a_onsets_master = a_onsets_raw + OFFSET_A
    b_onsets_master = b_onsets_raw + OFFSET_B

    ref = filter_to_window(ref_onsets, ANALYSIS_START, ANALYSIS_END)
    a_m = filter_to_window(a_onsets_master, ANALYSIS_START, ANALYSIS_END)
    b_m = filter_to_window(b_onsets_master, ANALYSIS_START, ANALYSIS_END)

    max_w = MATCH_WINDOW_MS / 1000.0

    comparisons = [
        {
            "name": "Reference vs Take A",
            "pairs": match_onsets(ref, a_m, max_w),
            "tracks": {
                "Ref bass": (bass_ref, sr_ref, 0.0),
                "Take A bass": (take_a_bass, sr_a, OFFSET_A),
                "Master song": (master, sr_master, 0.0),
            },
        },
        {
            "name": "Reference vs Take B",
            "pairs": match_onsets(ref, b_m, max_w),
            "tracks": {
                "Ref bass": (bass_ref, sr_ref, 0.0),
                "Take B bass": (take_b_bass, sr_b, OFFSET_B),
                "Master song": (master, sr_master, 0.0),
            },
        },
        {
            "name": "Take A vs Take B",
            "pairs": match_onsets(a_m, b_m, max_w),
            "tracks": {
                "Take A bass": (take_a_bass, sr_a, OFFSET_A),
                "Take B bass": (take_b_bass, sr_b, OFFSET_B),
                "Master song": (master, sr_master, 0.0),
            },
        },
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_events = []
    sections_html = []

    for comp in comparisons:
        cname = comp["name"]
        pairs = comp["pairs"]
        tracks = comp["tracks"]

        sorted_pos = sorted(pairs, key=lambda r: r[2], reverse=True)[:TOP_N]
        sorted_neg = sorted(pairs, key=lambda r: r[2])[:TOP_N]
        sorted_close = sorted(pairs, key=lambda r: abs(r[2]))[:TOP_N]

        categories = [
            ("Largest positive differences", sorted_pos),
            ("Largest negative differences", sorted_neg),
            ("Closest matches", sorted_close),
        ]

        comp_sections = []
        for cat_name, selected in categories:
            event_cards = []
            for rank, (t_a, t_b, diff_ms) in enumerate(selected, 1):
                eid = (
                    f"{cname.replace(' ', '_').lower()}_"
                    f"{cat_name.split()[1][:3]}_{rank}"
                )
                event = {
                    "id": eid,
                    "comparison": cname,
                    "category": cat_name,
                    "rank": rank,
                    "ref_time": t_a,
                    "second_time": t_b,
                    "diff_ms": diff_ms,
                }
                all_events.append(event)
                event_cards.append(build_event_html(event, tracks))
                print(f"  {cname} / {cat_name} #{rank}: "
                      f"{t_a:.3f}s, diff={diff_ms:+.1f}ms")

            comp_sections.append(
                f'<div class="category-header">{html_mod.escape(cat_name)}</div>'
                + "".join(event_cards)
            )

        sections_html.append(
            f'<div class="comparison-section">'
            f'<div class="comparison-header">{html_mod.escape(cname)}</div>'
            + "".join(comp_sections)
            + "</div>"
        )

    # Build overview table
    table_rows = []
    for ev in all_events:
        diff = ev["diff_ms"]
        sign = "+" if diff >= 0 else ""
        diff_color = "#2ecc71" if abs(diff) < 30 else "#f1c40f" if abs(diff) < 80 else "#ff6b6b"
        table_rows.append(
            f'<tr>'
            f'<td>{html_mod.escape(ev["comparison"])}</td>'
            f'<td>{html_mod.escape(ev["category"])}</td>'
            f'<td><a href="#{ev["id"]}">#{ev["rank"]}</a></td>'
            f'<td>{ev["ref_time"]:.3f} s</td>'
            f'<td style="color:{diff_color}">{sign}{diff:.1f}</td>'
            f'</tr>'
        )

    overview_table = (
        "<h2>Overview</h2>"
        "<table>"
        "<tr><th>Comparison</th><th>Category</th><th>Rank</th>"
        "<th>Time</th><th>Diff (ms)</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )

    js = _JS.replace("%DURATION%", f"{WINDOW_DUR:.6f}")

    report = (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>Timing Difference Microscope</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        "<h1>Timing Difference Microscope</h1>"
        f"<p style='color:#888;'>Analysis window: "
        f"{ANALYSIS_START:.0f}–{ANALYSIS_END:.0f} s &nbsp;|&nbsp; "
        f"Match tolerance: ±{MATCH_WINDOW_MS:.0f} ms &nbsp;|&nbsp; "
        f"Top {TOP_N} per category</p>"
        + overview_table
        + "".join(sections_html)
        + f"<script>{js}</script>"
        + "</body></html>"
    )

    out_path = OUT_DIR / "timing_difference_microscope.html"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {out_path}")
    print(f"Audio files: {OUT_DIR}/")


if __name__ == "__main__":
    main()
