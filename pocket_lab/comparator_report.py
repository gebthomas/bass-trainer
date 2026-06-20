"""Take Comparator HTML report assembly."""

from __future__ import annotations

import html

import numpy as np

from pocket_lab.comparator_css import _COMPARATOR_CSS
from pocket_lab.comparator_js import _COMPARATOR_JS
from pocket_lab.css import _CSS
from pocket_lab.match_record import ComparisonResult, MatchCategory, MatchRecord
from pocket_lab.svg import _PLOT_L, _PLOT_W, _SVG_W, svg_overview, svg_waveform


def _category_class(cat: MatchCategory) -> str:
    return cat.value.replace("_", "-")


def _svg_comparison_timeline(
    matches: list[MatchRecord],
    duration: float,
) -> str:
    """Render comparison timeline: A above, B below, connecting lines for matches."""
    svg_h = 220
    mid_y = svg_h / 2
    row_a_y = mid_y - 30
    row_b_y = mid_y + 30
    pt = 10
    ph = svg_h - 20

    elements: list[str] = []

    elements.append(
        f'<line x1="{_PLOT_L}" y1="{mid_y}" x2="{_PLOT_L + _PLOT_W}" y2="{mid_y}" '
        f'stroke="#333" stroke-width="1" />'
    )
    elements.append(
        f'<text x="{_PLOT_L - 8}" y="{row_a_y + 4}" fill="#e74c3c" font-size="10" '
        f'text-anchor="end">A</text>'
    )
    elements.append(
        f'<text x="{_PLOT_L - 8}" y="{row_b_y + 4}" fill="#3498db" font-size="10" '
        f'text-anchor="end">B</text>'
    )

    for m in matches:
        if m.category == MatchCategory.MATCHED and m.onset_a and m.onset_b:
            xa = _PLOT_L + (m.onset_a.time_s / duration) * _PLOT_W if duration > 0 else _PLOT_L
            xb = _PLOT_L + (m.onset_b.time_s / duration) * _PLOT_W if duration > 0 else _PLOT_L
            diff = abs(m.timing_diff_ms) if m.timing_diff_ms else 0
            if diff < 10:
                color = "#2ecc71"
            elif diff < 25:
                color = "#f39c12"
            else:
                color = "#e74c3c"
            elements.append(
                f'<line x1="{xa:.1f}" y1="{row_a_y}" x2="{xb:.1f}" y2="{row_b_y}" '
                f'stroke="{color}" stroke-width="1" opacity="0.6" />'
            )
            elements.append(
                f'<circle class="onset-marker" data-take="A" data-id="{m.onset_a.onset_index}" '
                f'cx="{xa:.1f}" cy="{row_a_y}" r="4" fill="#e74c3c" stroke="#fff" '
                f'stroke-width="0.5" data-time="{m.onset_a.time_s:.4f}" '
                f'data-strength="{m.onset_a.strength:.2f}" '
                f'data-amp="{m.onset_a.amplitude_db:.1f}" />'
            )
            elements.append(
                f'<circle class="onset-marker" data-take="B" data-id="{m.onset_b.onset_index}" '
                f'cx="{xb:.1f}" cy="{row_b_y}" r="4" fill="#3498db" stroke="#fff" '
                f'stroke-width="0.5" data-time="{m.onset_b.time_s:.4f}" '
                f'data-strength="{m.onset_b.strength:.2f}" '
                f'data-amp="{m.onset_b.amplitude_db:.1f}" />'
            )

        elif m.category == MatchCategory.A_ONLY and m.onset_a:
            x = _PLOT_L + (m.onset_a.time_s / duration) * _PLOT_W if duration > 0 else _PLOT_L
            elements.append(
                f'<polygon points="{x:.1f},{row_a_y - 5} {x + 4:.1f},{row_a_y} '
                f'{x:.1f},{row_a_y + 5} {x - 4:.1f},{row_a_y}" '
                f'fill="#e74c3c" stroke="#fff" stroke-width="0.5" />'
            )

        elif m.category == MatchCategory.B_ONLY and m.onset_b:
            x = _PLOT_L + (m.onset_b.time_s / duration) * _PLOT_W if duration > 0 else _PLOT_L
            elements.append(
                f'<polygon points="{x:.1f},{row_b_y - 5} {x + 4:.1f},{row_b_y} '
                f'{x:.1f},{row_b_y + 5} {x - 4:.1f},{row_b_y}" '
                f'fill="#3498db" stroke="#fff" stroke-width="0.5" />'
            )

        elif m.category == MatchCategory.AMBIGUOUS and m.onset_a:
            x = _PLOT_L + (m.onset_a.time_s / duration) * _PLOT_W if duration > 0 else _PLOT_L
            elements.append(
                f'<rect x="{x - 4:.1f}" y="{row_a_y - 5}" width="8" height="10" '
                f'fill="#f39c12" stroke="#fff" stroke-width="0.5" opacity="0.8" />'
            )

        elif m.category == MatchCategory.NOISE:
            onset = m.onset_a or m.onset_b
            if onset:
                x = _PLOT_L + (onset.time_s / duration) * _PLOT_W if duration > 0 else _PLOT_L
                y = row_a_y if onset.take_label == "A" else row_b_y
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y}" r="3" fill="#555" '
                    f'stroke="#888" stroke-width="0.5" opacity="0.5" />'
                )

    from pocket_lab.svg import _cursor_line, _time_axis_svg
    return (
        f'<svg viewBox="0 0 {_SVG_W} {svg_h}" class="plot-svg" id="svg-comparison" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;">'
        + "\n".join(elements)
        + _time_axis_svg(duration, svg_h)
        + f'<rect class="loop-region" x="{_PLOT_L}" y="{pt}" '
        + f'width="{_PLOT_W}" height="{ph}" opacity="0" />'
        + _cursor_line(svg_h, pt, ph)
        + '</svg>'
    )


def _disagreement_card(m: MatchRecord, idx: int) -> str:
    cat_cls = _category_class(m.category)
    cat_label = m.category.value.replace("_", " ").upper()
    t = m.time_s

    if m.category == MatchCategory.MATCHED:
        diff = m.timing_diff_ms or 0.0
        amp = m.amplitude_diff_db or 0.0
        detail = f"Timing: {diff:+.1f}ms, Amplitude: {amp:+.1f}dB"
    elif m.category == MatchCategory.A_ONLY:
        s = m.onset_a.strength if m.onset_a else 0
        detail = f"Note in Take A only. Strength: {s:.2f}"
    elif m.category == MatchCategory.B_ONLY:
        s = m.onset_b.strength if m.onset_b else 0
        detail = f"Note in Take B only. Strength: {s:.2f}"
    elif m.category == MatchCategory.AMBIGUOUS:
        n = len(m.candidates_b)
        detail = f"{n} candidate matches, cannot resolve"
    else:
        onset = m.onset_a or m.onset_b
        s = onset.strength if onset else 0
        detail = f"Likely artifact. Strength: {s:.2f}"

    return (
        f'<div class="disagreement-card" data-match-idx="{idx}" '
        f'data-time="{t:.4f}" data-category="{m.category.value}">'
        f'<span class="category-badge {cat_cls}">{cat_label}</span>'
        f'<span class="time">{t:.3f}s</span>'
        f'<span class="detail">{html.escape(detail)}</span>'
        f'</div>'
    )


def _sweep_table_html(sweep_rows: list[dict], used_threshold: float) -> str:
    if not sweep_rows:
        return ""
    rows = ""
    for r in sweep_rows:
        sel = ' class="selected-row"' if r["threshold"] == used_threshold else ""
        rows += (
            f'<tr{sel}><td>{r["threshold"]:.2f}</td>'
            f'<td>{r["matched"]}</td><td>{r["a_only"]}</td>'
            f'<td>{r["b_only"]}</td><td>{r["ambiguous"]}</td>'
            f'<td>{r["noise"]}</td></tr>'
        )
    return (
        '<details><summary>Noise Threshold Sweep</summary>'
        '<table class="sweep-table">'
        '<tr><th>Threshold</th><th>Matched</th><th>A-only</th>'
        '<th>B-only</th><th>Ambiguous</th><th>Noise</th></tr>'
        + rows
        + '</table>'
        '<p class="explanation">Highlighted row is the threshold used for this report. '
        'Use <code>--noise-threshold</code> to override.</p>'
        '</details>'
    )


def _onset_diagnostic_html(onset_diags: dict | None) -> str:
    if not onset_diags:
        return ""
    parts = ['<details><summary>Onset Detection Diagnostics</summary>']

    for key in ["a", "b"]:
        diag = onset_diags[key]
        label = diag["label"]
        sp = diag["spacing"]
        st = diag["strength"]
        q = diag["quantization"]

        parts.append(f'<h3>{html.escape(label)}</h3>')

        parts.append(
            '<div class="sync-diag">'
            f'<b>Quantization:</b> {q["frame_period_ms"]:.2f}ms frame period '
            f'(hop={q["hop_length"]}, sr={q["sr"]}). '
            f'All onset times snap to frame boundaries.<br>'
            f'<b>Onsets:</b> {sp["count"]} detected<br>'
            f'<b>Spacing:</b> min={sp["min_spacing_ms"]:.1f}ms, '
            f'median={sp["median_spacing_ms"]:.1f}ms, '
            f'mean={sp["mean_spacing_ms"]:.1f}ms<br>'
            f'<b>Strength:</b> min={st["min"]:.4f}, max={st["max"]:.4f}, '
            f'mean={st["mean"]:.4f}, median={st["median"]:.4f}<br>'
            f'<b>Weak onsets:</b> '
            f'{st["below_001"]} below 0.01, '
            f'{st["below_01"]} below 0.1, '
            f'{st["below_05"]} below 0.5'
            '</div>'
        )

        parts.append(
            '<table class="sweep-table" style="width:auto;margin:0.5em 0;">'
            '<tr><th>Neighbor distance</th><th>Onsets in close pair</th><th>Fraction</th></tr>'
        )
        for cp in diag["close_pairs"]:
            parts.append(
                f'<tr><td>&lt;{cp["threshold_ms"]:.0f}ms</td>'
                f'<td>{cp["count"]}</td>'
                f'<td>{cp["fraction"]:.1%}</td></tr>'
            )
        parts.append('</table>')

        parts.append(
            '<table class="sweep-table" style="width:auto;margin:0.5em 0;">'
            '<tr><th>Spacing range</th><th>Count</th><th></th></tr>'
        )
        max_count = max((h["count"] for h in diag["histogram"]), default=1)
        for h in diag["histogram"]:
            bar_w = int(h["count"] / max(max_count, 1) * 150)
            parts.append(
                f'<tr><td>{html.escape(h["label"])}</td>'
                f'<td style="text-align:right">{h["count"]}</td>'
                f'<td><span style="display:inline-block;height:10px;width:{bar_w}px;'
                f'background:#4cc9f0;border-radius:2px;"></span></td></tr>'
            )
        parts.append('</table>')

    parts.append('</details>')
    return "\n".join(parts)


def _sync_diagnostic_html(sync_info: dict | None) -> str:
    if not sync_info:
        return ""
    return (
        '<details open><summary>Sync Diagnostic</summary>'
        '<div class="sync-diag">'
        f'Alignment offset: <b>{sync_info["alignment_offset_s"]:+.4f}s</b> '
        f'(confidence: {sync_info["alignment_confidence"]:.3f})<br>'
        f'Window start in A: <b>{sync_info["window_start_a"]:.2f}s</b> · '
        f'Window start in B (native): <b>{sync_info["window_start_b"]:.2f}s</b><br>'
        f'Sidecar A bass: <b>{sync_info["bass_a_duration"]:.3f}s</b> '
        f'({sync_info["bass_a_samples"]} samples) · '
        f'Sidecar B bass: <b>{sync_info["bass_b_duration"]:.3f}s</b> '
        f'({sync_info["bass_b_samples"]} samples) · '
        f'Song: <b>{sync_info["song_duration"]:.3f}s</b>'
        '</div></details>'
    )


def _hidden_audio(element_id: str, src: str) -> str:
    if not src:
        return ""
    return f'<audio id="{element_id}" preload="auto" src="{html.escape(src)}"></audio>'


def render_comparator_report(
    *,
    result: ComparisonResult,
    bass_audio_a: np.ndarray,
    bass_audio_b: np.ndarray,
    sr: int,
    duration: float,
    audio_src_a: str = "",
    audio_src_b: str = "",
    audio_src_song: str = "",
    audio_src_song_b: str = "",
    audio_src_stereo_a: str = "",
    audio_src_stereo_b: str = "",
    sweep_rows: list[dict] | None = None,
    noise_threshold: float = 0.0,
    onset_diags: dict | None = None,
    sync_info: dict | None = None,
    overview: dict | None = None,
    window_start: float = 0.0,
    prev_href: str | None = None,
    next_href: str | None = None,
    total_windows: int = 1,
    window_index: int = 0,
) -> str:
    """Render the Take Comparator HTML report."""
    actual_dur = max(
        len(bass_audio_a) / sr if sr > 0 else 0.0,
        len(bass_audio_b) / sr if sr > 0 else 0.0,
    )
    if actual_dur == 0:
        actual_dur = duration

    # ── Audio bar ────────────────────────────────────────────────────────
    speed_buttons = "".join(
        f'<button class="tbtn speed-btn{" active" if s == "1.0" else ""}" '
        f'data-speed="{s}">{s}x</button>'
        for s in ["1.0", "0.75", "0.5", "0.33", "0.25"]
    )

    loop_html = (
        '<button class="tbtn" id="loop-toggle" title="L">Loop</button>'
        '<button class="tbtn" id="loop-set-start" title="[">[ set</button>'
        '<input type="number" id="loop-start-in" step="0.1" min="0">'
        '<input type="number" id="loop-end-in" step="0.1" min="0">'
        '<button class="tbtn" id="loop-set-end" title="]">] set</button>'
        '<button class="tbtn" id="loop-reset">Reset</button>'
        '<button class="tbtn" id="loop-restart" title="R">Restart</button>'
    )

    has_audio = audio_src_a or audio_src_b
    audio_html = ""
    if has_audio:
        audio_html = (
            '<div class="audio-bar">'
            + _hidden_audio("audio-bass-a", audio_src_a)
            + _hidden_audio("audio-bass-b", audio_src_b)
            + _hidden_audio("audio-song", audio_src_song)
            + _hidden_audio("audio-song-b", audio_src_song_b)
            + _hidden_audio("audio-stereo-a", audio_src_stereo_a)
            + _hidden_audio("audio-stereo-b", audio_src_stereo_b)
            + '<div class="channel-row">'
            '<label>Channels:</label>'
            '<button class="ch-toggle active" id="ch-toggle-bass-a" data-ch="bass_a">A Bass (1)</button>'
            '<button class="ch-toggle active" id="ch-toggle-bass-b" data-ch="bass_b">B Bass (2)</button>'
            '<button class="ch-toggle" id="ch-toggle-song" data-ch="song">Song (3)</button>'
        )
        if audio_src_stereo_a:
            audio_html += '<button class="ch-toggle" data-ch="stereo_a">A Original</button>'
        if audio_src_stereo_b:
            audio_html += '<button class="ch-toggle" data-ch="stereo_b">B Original</button>'
        if audio_src_song_b:
            audio_html += '<button class="ch-toggle" data-ch="song_b">Song B</button>'
        audio_html += (
            '</div>'
            '<div class="channel-row">'
            '<label>Presets:</label>'
        )
        if audio_src_stereo_a:
            audio_html += '<button class="preset-btn" id="preset-stereo-a">A original</button>'
        if audio_src_stereo_b:
            audio_html += '<button class="preset-btn" id="preset-stereo-b">B original</button>'
        audio_html += (
            '<button class="preset-btn" id="preset-bass-a">A bass</button>'
            '<button class="preset-btn" id="preset-bass-b">B bass</button>'
            '<button class="preset-btn active" id="preset-ab">A+B bass</button>'
            '<button class="preset-btn" id="preset-abs">A+B+song</button>'
            '<button class="preset-btn" id="preset-song">Song</button>'
        )
        if audio_src_song_b:
            audio_html += '<button class="preset-btn" id="preset-sync-check">Sync check</button>'
        audio_html += (
            '</div>'
            '<div class="transport-row">'
            '<button class="tbtn" id="play-btn">Play / Pause (Space)</button>'
            '<label>Speed:</label>'
            f'<span id="speed-label">1.0x</span>{speed_buttons}'
            '</div>'
            f'<div class="transport-row">{loop_html}</div>'
            '</div>'
        )

    # ── Overview + navigation ───────────────────────────────────────────
    overview_html = ""
    if overview and overview.get("total_duration_s", 0) > 0:
        overview_html = (
            '<h2>Song Overview</h2>'
            + svg_overview(overview, window_start, actual_dur)
        )
        prev_link = (
            f'<a href="{html.escape(prev_href)}">&#9664; Previous</a>'
            if prev_href else '<span class="disabled">&#9664; Previous</span>'
        )
        next_link = (
            f'<a href="{html.escape(next_href)}">Next &#9654;</a>'
            if next_href else '<span class="disabled">Next &#9654;</span>'
        )
        overview_html += (
            '<div class="nav-bar">'
            f'{prev_link}'
            f'<span class="win-label">Window {window_index + 1}/{total_windows} '
            f'({window_start:.1f}–{window_start + actual_dur:.1f}s '
            f'of {overview["total_duration_s"]:.1f}s)</span>'
            f'{next_link}'
            '</div>'
        )

    # ── Alignment + sync ───────────────────────────────────────────────
    align_html = _sync_diagnostic_html(sync_info)

    # ── Summary statistics ──────────────────────────────────────────────
    timing = result.timing_diffs_ms
    amps = result.amplitude_diffs_db
    mean_t = np.mean(timing) if timing else 0.0
    std_t = np.std(timing) if timing else 0.0
    mean_a = np.mean(amps) if amps else 0.0

    summary_html = (
        '<h2>Summary</h2>'
        '<div class="stat-row">'
        f'<span class="stat-item"><span class="stat-value">{result.matched_count}</span>'
        f'<span class="stat-label">matched</span></span>'
        f'<span class="stat-item"><span class="stat-value">{result.a_only_count}</span>'
        f'<span class="stat-label">A-only</span></span>'
        f'<span class="stat-item"><span class="stat-value">{result.b_only_count}</span>'
        f'<span class="stat-label">B-only</span></span>'
        f'<span class="stat-item"><span class="stat-value">{result.ambiguous_count}</span>'
        f'<span class="stat-label">ambiguous</span></span>'
        f'<span class="stat-item"><span class="stat-value">{result.noise_count}</span>'
        f'<span class="stat-label">noise</span></span>'
        '</div>'
        '<div class="stat-row">'
        f'<span class="stat-item"><span class="stat-value">{mean_t:+.1f}ms</span>'
        f'<span class="stat-label">mean timing diff</span></span>'
        f'<span class="stat-item"><span class="stat-value">{std_t:.1f}ms</span>'
        f'<span class="stat-label">timing std</span></span>'
        f'<span class="stat-item"><span class="stat-value">{mean_a:+.1f}dB</span>'
        f'<span class="stat-label">mean amplitude diff</span></span>'
        '</div>'
    )

    # ── Threshold sweep ─────────────────────────────────────────────────
    sweep_html = _sweep_table_html(sweep_rows or [], noise_threshold)
    onset_diag_html = _onset_diagnostic_html(onset_diags)

    # ── Waveforms ────────────────────────────────────────────────────────
    waveform_a_html = (
        "<h2>Take A: Bass Waveform</h2>"
        + svg_waveform(bass_audio_a, sr, actual_dur, svg_id="svg-waveform-a")
    )
    waveform_b_html = (
        "<h2>Take B: Bass Waveform</h2>"
        + svg_waveform(bass_audio_b, sr, actual_dur, svg_id="svg-waveform-b")
    )

    # ── Comparison timeline ─────────────────────────────────────────────
    timeline_html = (
        "<h2>Comparison Timeline</h2>"
        + _svg_comparison_timeline(result.matches, actual_dur)
    )

    # ── Legend ───────────────────────────────────────────────────────────
    legend_html = (
        '<div class="legend-box">'
        '<div class="legend-item">'
        '<span style="background:#2ecc71;width:12px;height:3px;display:inline-block;"></span>'
        '<span>Tight match (&lt;10ms)</span></div>'
        '<div class="legend-item">'
        '<span style="background:#f39c12;width:12px;height:3px;display:inline-block;"></span>'
        '<span>Moderate (10-25ms)</span></div>'
        '<div class="legend-item">'
        '<span style="background:#e74c3c;width:12px;height:3px;display:inline-block;"></span>'
        '<span>Large (&gt;25ms) / A onset</span></div>'
        '<div class="legend-item">'
        '<span style="background:#3498db;width:12px;height:3px;display:inline-block;"></span>'
        '<span>B onset</span></div>'
        '<div class="legend-item">'
        '<span style="background:#555;width:8px;height:8px;border-radius:50%;'
        'display:inline-block;"></span>'
        '<span>Noise</span></div>'
        '</div>'
    )

    # ── Event filter controls ───────────────────────────────────────────
    filter_html = (
        '<div class="filter-row">'
        '<label style="color:#888;">Show:</label>'
        '<button class="filter-btn active" data-category="matched">Matched</button>'
        '<button class="filter-btn active" data-category="a_only">A-only</button>'
        '<button class="filter-btn active" data-category="b_only">B-only</button>'
        '<button class="filter-btn active" data-category="ambiguous">Ambiguous</button>'
        '<button class="filter-btn" data-category="noise">Noise</button>'
        '<span class="sep">|</span>'
        '<button class="qf-btn" id="qf-all">All</button>'
        '<button class="qf-btn" id="qf-disagreements">Disagreements</button>'
        '<button class="qf-btn" id="qf-matched">Matched only</button>'
        '<span class="sep">|</span>'
        '<button class="zoom-btn" id="zoom-event-btn" '
        'title="Loop ±2s around selected event (Z)">Zoom to event (Z)</button>'
        '</div>'
    )

    # ── Disagreement cards ──────────────────────────────────────────────
    cards_html = '<h2>Events</h2>' + filter_html + '<div id="disagreement-list">'
    for i, m in enumerate(result.matches):
        cards_html += _disagreement_card(m, i)
    cards_html += '</div>'

    # ── Matched detail table ────────────────────────────────────────────
    matched = [m for m in result.matches if m.category == MatchCategory.MATCHED]
    rows: list[str] = []
    for m in matched:
        ta = m.onset_a.time_s if m.onset_a else 0
        tb = m.onset_b.time_s if m.onset_b else 0
        dt = m.timing_diff_ms or 0
        aa = m.onset_a.amplitude_db if m.onset_a else 0
        ab = m.onset_b.amplitude_db if m.onset_b else 0
        da = m.amplitude_diff_db or 0
        sa = m.onset_a.strength if m.onset_a else 0
        sb = m.onset_b.strength if m.onset_b else 0
        rows.append(
            f"<tr><td>{ta:.4f}</td><td>{tb:.4f}</td><td>{dt:+.1f}</td>"
            f"<td>{aa:.1f}</td><td>{ab:.1f}</td><td>{da:+.1f}</td>"
            f"<td>{sa:.2f}</td><td>{sb:.2f}</td></tr>"
        )
    detail_html = (
        f'<details><summary>Matched Notes ({len(matched)})</summary>'
        "<table><tr><th>Time A</th><th>Time B</th><th>Diff (ms)</th>"
        "<th>Amp A</th><th>Amp B</th><th>Amp Diff</th>"
        "<th>Str A</th><th>Str B</th></tr>"
        + "\n".join(rows)
        + "</table></details>"
    )

    # ── Keyboard shortcuts ──────────────────────────────────────────────
    shortcuts_html = (
        '<details><summary>Keyboard Shortcuts</summary>'
        '<table class="shortcut-table">'
        '<tr><td>Space</td><td>Play / pause</td></tr>'
        '<tr><td>1</td><td>Toggle A bass on/off</td></tr>'
        '<tr><td>2</td><td>Toggle B bass on/off</td></tr>'
        '<tr><td>3</td><td>Toggle song on/off</td></tr>'
        '<tr><td>Z</td><td>Zoom/loop ±2s around selected event</td></tr>'
        '<tr><td>R</td><td>Restart</td></tr>'
        '<tr><td>L</td><td>Toggle loop</td></tr>'
        '<tr><td>[</td><td>Set loop start</td></tr>'
        '<tr><td>]</td><td>Set loop end</td></tr>'
        '<tr><td>Up / Down</td><td>Navigate events</td></tr>'
        '<tr><td>Shift+Left/Right</td><td>Seek 0.25s</td></tr>'
        '</table></details>'
    )

    # ── JavaScript ──────────────────────────────────────────────────────
    js = (
        _COMPARATOR_JS
        .replace("%DURATION%", f"{actual_dur:.6f}")
        .replace("%PLOT_LEFT%", str(_PLOT_L))
        .replace("%PLOT_WIDTH%", str(_PLOT_W))
    )

    # ── Assemble ────────────────────────────────────────────────────────
    return (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>Take Comparator</title>"
        f"<style>{_CSS}\n{_COMPARATOR_CSS}</style>"
        "</head><body>"
        "<h1>Take Comparator</h1>"
        + audio_html + overview_html + align_html + summary_html
        + sweep_html + onset_diag_html + legend_html
        + waveform_a_html + waveform_b_html + timeline_html
        + cards_html + detail_html + shortcuts_html
        + '<div id="tooltip"></div>'
        + f"<script>{js}</script>"
        + "</body></html>"
    )
