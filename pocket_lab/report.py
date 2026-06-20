"""Beat Microscope HTML report assembly."""

from __future__ import annotations

import html
from pathlib import Path
from typing import List

import numpy as np

from pocket_lab.css import _CSS
from pocket_lab.grid import GridLine, OnsetClassification
from pocket_lab.grid_settings import (
    ANNOTATION_LABELS,
    GRID_SOURCE_FIXED_BPM,
    GridSource,
)
from pocket_lab.js import _JS_TEMPLATE
from pocket_lab.svg import (
    _PLOT_L,
    _PLOT_W,
    svg_envelope,
    svg_overview,
    svg_timeline,
    svg_waveform,
)


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
            + svg_overview(overview, start, actual_duration)
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
        + svg_waveform(bass_audio, sr, actual_duration)
    )
    envelope_html = (
        "<h2>Onset-Strength Envelope</h2>"
        + svg_envelope(
            env_times, env_values, onset_times,
            classifications, onset_strengths, actual_duration,
        )
    )
    timeline_html = (
        "<h2>Event Timeline</h2>"
        + svg_timeline(
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
