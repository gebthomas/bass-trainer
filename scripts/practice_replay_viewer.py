#!/usr/bin/env python3
"""Generate a self-contained interactive HTML replay page for a practice session.

Reads a saved .session.json file (and optionally the recorded WAV audio) and
writes a single HTML file that can be opened directly in a browser — no server
required.

Usage
-----
    python scripts/practice_replay_viewer.py \\
        --session-log sessions/foo.session.json \\
        --audio sessions/foo.wav \\
        --out replay.html \\
        [--window-ms 300]

The HTML file embeds the audio as base64 and all session timing data as JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_log import (
    EXTRA_ONSET,
    TARGET_HIT,
    TARGET_MISS,
    SessionLog,
    load_session_log_file,
)


# ── Pure data extraction ──────────────────────────────────────────────────────

def extract_replay_data(log: SessionLog) -> dict:
    """Extract all timing arrays needed for the HTML replay from a SessionLog.

    Uses session metadata (bpm, beats, count_in, latency_ms) to build the
    nominal target schedule, then maps log events onto that schedule by
    target_index.  Beats with no matching event are treated as misses.

    Returns
    -------
    dict with keys:
        bpm, beats, count_in, latency_ms, beat_s, count_in_s,
        total_duration, target_times_s, onset_data, raw_onset_times_s,
        extra_onset_times_s, click_times_s.

    onset_data is a list of dicts (one per beat), each with:
        beat (int), target_s (float), onset_s (float|None),
        err_ms (float|None), is_miss (bool).

    raw_onset_times_s is empty when latency_ms == 0.
    """
    bpm        = float(log.metadata.get("bpm",        "120"))
    beats      = int(log.metadata.get("beats",        "4"))
    count_in   = int(log.metadata.get("count_in",     "2"))
    latency_ms = float(log.metadata.get("latency_ms", "0"))

    beat_s     = 60.0 / bpm
    count_in_s = count_in * beat_s

    target_times_s = [count_in_s + i * beat_s for i in range(beats)]

    # Collect per-target data from log events
    hit_errors:  dict[int, float] = {}  # target_index → timing_error_s
    miss_indices: set[int]        = set()
    extra_onsets: list[float]     = []

    for ev in log.events:
        if (ev.event_type == TARGET_HIT
                and ev.target_index is not None
                and ev.value is not None):
            hit_errors[ev.target_index] = float(ev.value)
        elif ev.event_type == TARGET_MISS and ev.target_index is not None:
            miss_indices.add(ev.target_index)
        elif ev.event_type == EXTRA_ONSET:
            extra_onsets.append(float(ev.time_sec))

    onset_data: list[dict] = []
    for i, target_s in enumerate(target_times_s):
        if i in hit_errors:
            err_s  = hit_errors[i]
            err_ms = err_s * 1000.0
            onset_data.append({
                "beat":     i,
                "target_s": target_s,
                "onset_s":  target_s + err_s,
                "err_ms":   err_ms,
                "is_miss":  False,
            })
        else:
            onset_data.append({
                "beat":     i,
                "target_s": target_s,
                "onset_s":  None,
                "err_ms":   None,
                "is_miss":  True,
            })

    # Pre-correction onset positions (add latency back to undo compensation)
    raw_onset_times_s: list[float] = []
    if latency_ms != 0.0:
        offset_s = latency_ms / 1000.0
        for d in onset_data:
            if d["onset_s"] is not None:
                raw_onset_times_s.append(d["onset_s"] + offset_s)

    # Click schedule: count-in beats then exercise beats
    click_times_s: list[float] = [i * beat_s for i in range(count_in)]
    for i in range(beats):
        click_times_s.append(count_in_s + i * beat_s)

    # One extra beat of padding so the last beat isn't at the right edge
    total_duration = count_in_s + beats * beat_s + beat_s

    return {
        "bpm":                 bpm,
        "beats":               beats,
        "count_in":            count_in,
        "latency_ms":          latency_ms,
        "beat_s":              beat_s,
        "count_in_s":          count_in_s,
        "total_duration":      total_duration,
        "target_times_s":      target_times_s,
        "onset_data":          onset_data,
        "raw_onset_times_s":   raw_onset_times_s,
        "extra_onset_times_s": extra_onsets,
        "click_times_s":       click_times_s,
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bass Practice Replay</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #121220;
  color: #d0d0e8;
  font-family: 'Segoe UI', system-ui, sans-serif;
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}
h2 { font-size: 1.2rem; font-weight: 600; margin-bottom: 4px; color: #e8e8ff; }
.meta { font-size: 0.8rem; color: #888; margin-bottom: 16px; }
.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
audio { flex-shrink: 0; height: 36px; }
button {
  background: #2a2a40;
  color: #c0c0e0;
  border: 1px solid #444;
  border-radius: 4px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: background 0.1s;
}
button:hover { background: #3a3a58; }
button:disabled { opacity: 0.4; cursor: default; }
button.active { background: #3d2d70; border-color: #7c4dbd; color: #e0d0ff; }
button.miss { color: #e74c3c; border-color: rgba(231,76,60,0.5); }
button.miss.active { background: #3d1515; border-color: #e74c3c; }
.beat-row { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 10px; }
#time-display {
  font-size: 0.85rem;
  font-family: monospace;
  color: #aaa;
  min-width: 80px;
}
canvas {
  display: block;
  border: 1px solid #2a2a3a;
  background: #0a0a14;
  cursor: crosshair;
  width: 100%;
  height: 220px;
}
.hint { font-size: 0.72rem; color: #444; margin-top: 5px; }
.legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 0.72rem;
  color: #888;
  align-items: center;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.sw-solid  { width: 20px; height: 3px; border-radius: 1px; flex-shrink: 0; }
.sw-dashed { width: 20px; height: 0; border-top-width: 2px; border-top-style: dashed; flex-shrink: 0; }
</style>
</head>
<body>
<h2 id="page-title">Bass Practice Replay</h2>
<div class="meta" id="page-meta"></div>

<div class="controls">
  <audio id="player" controls></audio>
  <button id="play-region-btn">&#9654; Play Region</button>
  <button id="zoom-out-btn">Zoom Out</button>
  <span id="time-display">0.000 s</span>
</div>

<div class="beat-row" id="beat-row"></div>
<canvas id="waveform"></canvas>
<div class="hint">Click waveform to seek &nbsp;&middot;&nbsp; Beat buttons zoom to &plusmn;window &nbsp;&middot;&nbsp; Zoom Out to see full session</div>
<div class="legend" id="legend"></div>

<script>
/* jshint esversion: 6 */
const DATA      = INJECT_DATA_JSON;
const AUDIO_B64 = "INJECT_AUDIO_B64";
const WINDOW_MS = INJECT_WINDOW_MS;

// ── Audio element ─────────────────────────────────────────────────────────
const player = document.getElementById('player');
if (AUDIO_B64) {
  player.src = 'data:audio/wav;base64,' + AUDIO_B64;
} else {
  player.style.display = 'none';
  document.getElementById('play-region-btn').disabled = true;
}

// ── Decode audio buffer for waveform rendering ────────────────────────────
let audioBuffer = null;
if (AUDIO_B64) {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const bin = atob(AUDIO_B64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    audioCtx.decodeAudioData(arr.buffer.slice(0)).then(buf => {
      audioBuffer = buf;
      draw();
    }).catch(e => console.warn('Audio decode error:', e));
  } catch(e) {
    console.warn('AudioContext unavailable:', e);
  }
}

// ── View state ────────────────────────────────────────────────────────────
const totalDur    = DATA.total_duration;
let viewStart     = 0;
let viewEnd       = totalDur;
let selectedBeat  = null;
let animId        = null;
let regionEnd     = null;

// ── Page header ───────────────────────────────────────────────────────────
const hitCount  = DATA.onset_data.filter(d => !d.is_miss).length;
const missCount = DATA.onset_data.filter(d =>  d.is_miss).length;
document.getElementById('page-title').textContent =
  `Bass Practice Replay — ${DATA.bpm.toFixed(0)} BPM`;
document.getElementById('page-meta').textContent =
  `${DATA.beats} beats · ${DATA.count_in} count-in · ` +
  `latency: ${DATA.latency_ms.toFixed(0)} ms · ` +
  `hits: ${hitCount}/${DATA.beats} · misses: ${missCount}`;

// ── Beat buttons ──────────────────────────────────────────────────────────
const beatRow = document.getElementById('beat-row');
DATA.onset_data.forEach((d, i) => {
  const btn = document.createElement('button');
  const errStr = (!d.is_miss && d.err_ms !== null)
    ? ` (${d.err_ms > 0 ? '+' : ''}${d.err_ms.toFixed(0)}ms)` : '';
  btn.textContent = `${i + 1}` + (d.is_miss ? ' ✗' : errStr);
  btn.title = d.is_miss
    ? `Beat ${i + 1}: MISS`
    : `Beat ${i + 1}: ${d.err_ms.toFixed(0)} ms`;
  if (d.is_miss) btn.classList.add('miss');
  btn.addEventListener('click', () => selectBeat(i));
  beatRow.appendChild(btn);
});

function selectBeat(i) {
  selectedBeat = i;
  const target  = DATA.target_times_s[i];
  const halfWin = WINDOW_MS / 1000.0;
  viewStart = Math.max(0.0, target - halfWin);
  viewEnd   = Math.min(totalDur, target + halfWin);
  updateBeatButtons();
  draw();
}

function updateBeatButtons() {
  beatRow.querySelectorAll('button').forEach((b, i) => {
    b.classList.toggle('active', i === selectedBeat);
  });
}

// ── Zoom out ──────────────────────────────────────────────────────────────
document.getElementById('zoom-out-btn').addEventListener('click', () => {
  selectedBeat = null;
  viewStart = 0;
  viewEnd   = totalDur;
  updateBeatButtons();
  draw();
});

// ── Play region ───────────────────────────────────────────────────────────
document.getElementById('play-region-btn').addEventListener('click', () => {
  regionEnd = viewEnd;
  player.currentTime = viewStart;
  player.play().catch(() => {});
});

// ── Playhead animation ────────────────────────────────────────────────────
player.addEventListener('play',  () => startPlayhead());
player.addEventListener('pause', () => {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  draw();
});

function startPlayhead() {
  if (animId) cancelAnimationFrame(animId);
  const loop = () => {
    if (regionEnd !== null && player.currentTime >= regionEnd) {
      player.pause();
      regionEnd = null;
      return;
    }
    draw();
    animId = requestAnimationFrame(loop);
  };
  animId = requestAnimationFrame(loop);
}

// ── Canvas ────────────────────────────────────────────────────────────────
const canvas = document.getElementById('waveform');
const ctx2d  = canvas.getContext('2d');

function resizeCanvas() {
  const rect   = canvas.getBoundingClientRect();
  canvas.width  = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  draw();
}
window.addEventListener('resize', resizeCanvas);

canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  const t    = viewStart + frac * (viewEnd - viewStart);
  player.currentTime = Math.max(0, Math.min(t, player.duration || 0));
  draw();
});

// ── Drawing ───────────────────────────────────────────────────────────────
function tToX(t) {
  return (t - viewStart) / (viewEnd - viewStart) * canvas.width;
}

function draw() {
  const W = canvas.width;
  const H = canvas.height;
  if (!W || !H) return;

  ctx2d.clearRect(0, 0, W, H);

  // Count-in shading
  if (DATA.count_in_s > viewStart) {
    const x0 = Math.max(0, tToX(0));
    const x1 = tToX(Math.min(DATA.count_in_s, viewEnd));
    ctx2d.fillStyle = 'rgba(100,149,237,0.07)';
    ctx2d.fillRect(x0, 0, x1 - x0, H);
  }

  // Waveform
  if (audioBuffer) {
    const ch   = audioBuffer.getChannelData(0);
    const sr   = audioBuffer.sampleRate;
    const mid  = H * 0.5;
    const span = viewEnd - viewStart;
    ctx2d.strokeStyle = '#505068';
    ctx2d.lineWidth   = 0.8;
    ctx2d.beginPath();
    for (let px = 0; px < W; px++) {
      const tA = viewStart + px       / W * span;
      const tB = viewStart + (px + 1) / W * span;
      const sA = Math.max(0,          Math.floor(tA * sr));
      const sB = Math.min(ch.length,  Math.ceil(tB  * sr));
      let lo = 0, hi = 0;
      for (let s = sA; s < sB; s++) {
        if (ch[s] > hi) hi = ch[s];
        if (ch[s] < lo) lo = ch[s];
      }
      ctx2d.moveTo(px, mid - hi * mid * 0.88);
      ctx2d.lineTo(px, mid - lo * mid * 0.88);
    }
    ctx2d.stroke();
  }

  // Click tick marks at top 6 %
  const tickH = H * 0.06;
  DATA.click_times_s.forEach(t => {
    if (t < viewStart || t > viewEnd) return;
    ctx2d.strokeStyle = t < DATA.count_in_s
      ? 'rgba(100,149,237,0.9)' : 'rgba(124,77,189,0.9)';
    ctx2d.lineWidth = 2.5;
    ctx2d.lineCap   = 'butt';
    const x = tToX(t);
    ctx2d.beginPath();
    ctx2d.moveTo(x, 0);
    ctx2d.lineTo(x, tickH);
    ctx2d.stroke();
  });

  // Target beat lines + labels
  ctx2d.font = '10px monospace';
  DATA.target_times_s.forEach((t, i) => {
    if (t < viewStart - 0.01 || t > viewEnd + 0.01) return;
    const d      = DATA.onset_data[i];
    const isMiss = d.is_miss;
    const x      = tToX(t);
    ctx2d.strokeStyle = isMiss ? 'rgba(231,76,60,0.65)' : 'rgba(90,90,110,0.5)';
    ctx2d.lineWidth   = 1;
    ctx2d.setLineDash([4, 4]);
    ctx2d.beginPath();
    ctx2d.moveTo(x, 0);
    ctx2d.lineTo(x, H);
    ctx2d.stroke();
    ctx2d.setLineDash([]);
    ctx2d.fillStyle = isMiss ? '#e74c3c' : '#666';
    ctx2d.fillText(String(i + 1) + (isMiss ? ' MISS' : ''), x + 3, 16);
  });

  // Raw onset lines (only when latency correction was applied)
  if (DATA.latency_ms !== 0) {
    ctx2d.strokeStyle = 'rgba(100,149,237,0.28)';
    ctx2d.lineWidth   = 0.8;
    ctx2d.setLineDash([2, 4]);
    DATA.raw_onset_times_s.forEach(t => {
      if (t < viewStart || t > viewEnd) return;
      const x = tToX(t);
      ctx2d.beginPath();
      ctx2d.moveTo(x, 0);
      ctx2d.lineTo(x, H);
      ctx2d.stroke();
    });
    ctx2d.setLineDash([]);
  }

  // Compensated onset lines + error labels
  ctx2d.font = '10px monospace';
  DATA.onset_data.forEach(d => {
    if (d.is_miss || d.onset_s === null) return;
    const t = d.onset_s;
    if (t < viewStart - 0.01 || t > viewEnd + 0.01) return;
    const abs   = Math.abs(d.err_ms);
    const color = abs <= 50 ? '#2ecc71' : abs <= 120 ? '#e67e22' : '#e74c3c';
    const x     = tToX(t);
    ctx2d.strokeStyle = color;
    ctx2d.lineWidth   = 1.8;
    ctx2d.setLineDash([]);
    ctx2d.beginPath();
    ctx2d.moveTo(x, 0);
    ctx2d.lineTo(x, H);
    ctx2d.stroke();
    const sign = d.err_ms > 0 ? '+' : '';
    ctx2d.fillStyle = color;
    ctx2d.fillText(sign + d.err_ms.toFixed(0) + 'ms', x + 3, H - 6);
  });

  // Playhead
  const ct = player.currentTime || 0;
  document.getElementById('time-display').textContent = ct.toFixed(3) + ' s';
  if (ct >= viewStart && ct <= viewEnd) {
    const xph = tToX(ct);
    ctx2d.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx2d.lineWidth   = 1.5;
    ctx2d.setLineDash([]);
    ctx2d.beginPath();
    ctx2d.moveTo(xph, 0);
    ctx2d.lineTo(xph, H);
    ctx2d.stroke();
    ctx2d.fillStyle = 'rgba(255,255,255,0.75)';
    ctx2d.beginPath();
    ctx2d.moveTo(xph - 5, 0);
    ctx2d.lineTo(xph + 5, 0);
    ctx2d.lineTo(xph, 9);
    ctx2d.closePath();
    ctx2d.fill();
  }
}

// ── Legend ────────────────────────────────────────────────────────────────
(function buildLegend() {
  const items = [
    { color: '#505068',               style: 'solid',  label: 'Waveform' },
    { color: 'rgba(90,90,110,0.5)',   style: 'dashed', label: 'Target beat' },
    { color: '#2ecc71',               style: 'solid',  label: 'Onset ≤50 ms (good)' },
    { color: '#e67e22',               style: 'solid',  label: 'Onset ≤120 ms (warn)' },
    { color: '#e74c3c',               style: 'solid',  label: 'Onset >120 ms / miss' },
    { color: 'rgba(100,149,237,0.9)', style: 'solid',  label: 'Count-in click' },
    { color: 'rgba(124,77,189,0.9)',  style: 'solid',  label: 'Exercise click' },
  ];
  if (DATA.latency_ms !== 0) {
    items.push({
      color: 'rgba(100,149,237,0.3)',
      style: 'dashed',
      label: `Raw onset (+${DATA.latency_ms.toFixed(0)} ms)`,
    });
  }
  const leg = document.getElementById('legend');
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'legend-item';
    const sw  = document.createElement('div');
    if (item.style === 'dashed') {
      sw.className = 'sw-dashed';
      sw.style.borderTopColor = item.color;
    } else {
      sw.className = 'sw-solid';
      sw.style.background = item.color;
    }
    const lbl = document.createElement('span');
    lbl.textContent = item.label;
    div.appendChild(sw);
    div.appendChild(lbl);
    leg.appendChild(div);
  });
})();

// ── Initial layout ────────────────────────────────────────────────────────
requestAnimationFrame(() => { resizeCanvas(); });
</script>
</body>
</html>
"""

# Sentinel strings that cannot appear in valid JSON or base64, so replacements
# are unambiguous and safe against nested occurrences.
_SENTINEL_DATA   = "INJECT_DATA_JSON"
_SENTINEL_AUDIO  = "INJECT_AUDIO_B64"
_SENTINEL_WINDOW = "INJECT_WINDOW_MS"


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(data: dict, audio_b64: str, window_ms: float) -> str:
    """Render the HTML template with session data, audio, and window size."""
    # Replace </  with <\/ so the JSON cannot accidentally close a <script> tag.
    safe_json = json.dumps(data).replace("</", "<\\/")
    html = _HTML_TEMPLATE
    html = html.replace(_SENTINEL_DATA,   safe_json)
    html = html.replace(_SENTINEL_AUDIO,  audio_b64)
    html = html.replace(_SENTINEL_WINDOW, str(window_ms))
    return html


# ── Audio loader ──────────────────────────────────────────────────────────────

def _read_audio_b64(audio_path: str | None) -> str:
    """Base64-encode the WAV file at *audio_path*, or return '' if absent."""
    if not audio_path:
        return ""
    p = Path(audio_path)
    if not p.exists():
        print(f"Warning: audio file not found: {audio_path}", file=sys.stderr)
        return ""
    return base64.b64encode(p.read_bytes()).decode("ascii")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a self-contained HTML replay page for a practice session."
    )
    p.add_argument("--session-log", required=True, metavar="PATH", dest="session_log",
                   help="Path to the .session.json file")
    p.add_argument("--audio", default=None, metavar="PATH",
                   help="Path to the WAV audio file to embed (optional)")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="Output HTML file path")
    p.add_argument("--window-ms", type=float, default=300.0, dest="window_ms",
                   metavar="MS",
                   help="Half-window in ms for per-beat zoom (default: 300)")
    return p.parse_args()


def main() -> None:
    args      = _parse_args()
    log       = load_session_log_file(args.session_log)
    data      = extract_replay_data(log)
    audio_b64 = _read_audio_b64(args.audio)
    html      = build_html(data, audio_b64, args.window_ms)
    out       = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
