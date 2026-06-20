"""Shared CSS for Pocket Lab HTML reports."""

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
