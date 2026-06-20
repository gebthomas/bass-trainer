"""CSS additions for the Take Comparator report."""

_COMPARATOR_CSS = """\
.category-badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 10px; font-weight: bold; text-transform: uppercase;
    margin-right: 8px;
}
.category-badge.matched { background: #2ecc71; color: #0f0f23; }
.category-badge.a-only { background: #e74c3c; color: #fff; }
.category-badge.b-only { background: #3498db; color: #fff; }
.category-badge.ambiguous { background: #f39c12; color: #0f0f23; }
.category-badge.noise { background: #555; color: #ccc; }
.disagreement-card {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px; margin: 3px 0; border-radius: 4px;
    background: #1a1a2e; border: 1px solid #333; cursor: pointer;
    font-size: 12px;
}
.disagreement-card:hover { border-color: #4cc9f0; }
.disagreement-card.selected { border-color: #4cc9f0; background: #1f1f3a; }
.disagreement-card .time { color: #4cc9f0; min-width: 60px; }
.disagreement-card .detail { color: #999; flex: 1; }
.filter-row {
    display: flex; flex-wrap: wrap; gap: 6px; padding: 6px 0;
    font-size: 12px; align-items: center;
}
.filter-btn, .qf-btn {
    background: #1a1a2e; color: #ccc; border: 1px solid #555;
    padding: 3px 10px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit;
}
.filter-btn:hover, .qf-btn:hover { border-color: #4cc9f0; }
.filter-btn.active { background: #4cc9f0; color: #0f0f23; border-color: #4cc9f0; }
.qf-btn:hover { color: #4cc9f0; }
.stat-row {
    display: flex; flex-wrap: wrap; gap: 20px; padding: 8px 14px;
    background: #1a1a2e; border-radius: 4px; margin: 0.5em 0;
    font-size: 13px;
}
.stat-item .stat-value { color: #4cc9f0; font-weight: bold; }
.stat-item .stat-label { color: #888; margin-left: 4px; }
.channel-row {
    display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
    padding: 4px 0; font-size: 12px;
}
.channel-row label { color: #888; }
.ch-toggle {
    padding: 3px 10px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit; border: 1px solid #555;
    background: #0f0f23; color: #888;
}
.ch-toggle:hover { border-color: #4cc9f0; }
.ch-toggle.active { font-weight: bold; }
.ch-toggle.active[data-ch="bass_a"] { background: #e74c3c; color: #fff; border-color: #e74c3c; }
.ch-toggle.active[data-ch="bass_b"] { background: #3498db; color: #fff; border-color: #3498db; }
.ch-toggle.active[data-ch="song"] { background: #2ecc71; color: #fff; border-color: #2ecc71; }
.ch-toggle.active[data-ch="song_b"] { background: #27ae60; color: #fff; border-color: #27ae60; }
.ch-toggle.active[data-ch="stereo_a"] { background: #c0392b; color: #fff; border-color: #c0392b; }
.ch-toggle.active[data-ch="stereo_b"] { background: #2980b9; color: #fff; border-color: #2980b9; }
.preset-btn {
    background: #1a1a2e; color: #ccc; border: 1px solid #333;
    padding: 3px 8px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit;
}
.preset-btn:hover { border-color: #4cc9f0; color: #4cc9f0; }
.preset-btn.active { background: #4cc9f0; color: #0f0f23; border-color: #4cc9f0; }
.sweep-table td { text-align: right; }
.sweep-table td:first-child { text-align: left; }
.sweep-table tr.selected-row { background: #1f2f1f; }
.sync-diag {
    background: #1a1a2e; border-left: 3px solid #4cc9f0;
    padding: 10px 14px; margin: 0.5em 0; border-radius: 0 4px 4px 0;
    font-size: 12px; line-height: 1.7;
}
.sync-diag b { color: #4cc9f0; }
.zoom-btn {
    background: #1a1a2e; color: #f39c12; border: 1px solid #f39c12;
    padding: 3px 10px; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: inherit;
}
.zoom-btn:hover { background: #f39c12; color: #0f0f23; }
.sep { color: #555; margin: 0 4px; }
"""
