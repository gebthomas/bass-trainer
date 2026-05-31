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

Time base
---------
All time values are in seconds from the start of the captured audio recording.

    count_in_s = count_in_beats × 60 / bpm          (first exercise beat = DOWNBEAT)
    target[i]  = count_in_s + i × beat_s
    corrected onset = event.time_sec                 (latency already subtracted)
    raw onset        = event.time_sec + latency_ms/1000  (physical WAV position)
    click[k]   = k × beat_s (count-in), count_in_s + i × beat_s (exercise)
"""

from __future__ import annotations

import argparse
import base64
import json
import math
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

    Time-base reference (all times in seconds from session/WAV start):

        t_raw       = raw detected onset = sample_pos / sample_rate at capture time
                      (physical position in the WAV file; aligns with visible attack)
        t_comp      = t_raw - latency_ms/1000
                      (performance time; latency already removed by compensate_onset_times
                      before it reaches SessionEngine → SessionLog.event.time_sec)
        target[i]   = count_in_s + i * beat_s
                      (when player *should* play; equals t_comp for perfect timing)

        event.time_sec  = t_comp  (compensated, NOT raw audio time)
        event.value     = timing_error_s = t_comp - target_s  (seconds; neg=early)

        viewer.onset_s  = target_s + event.value = t_comp    (line drawn in session time)
        viewer.raw      = onset_s + latency_ms/1000 = t_raw  (should align with WAV attack)

    No double latency: t_comp is latency-subtracted exactly once (in demo's main loop via
    compensate_onset_times). The viewer adds latency back to recover t_raw for the raw line.

    Returns a dict with keys: bpm, beats, count_in, latency_ms, beat_s,
    count_in_s, total_duration, target_times_s, onset_data,
    raw_onset_times_s, extra_onset_times_s, click_times_s.

    onset_data is a list of dicts (one per beat): beat, target_s, onset_s,
    err_ms, is_miss.  raw_onset_times_s is empty when latency_ms == 0.
    """
    bpm        = float(log.metadata.get("bpm",        "120"))
    beats      = int(log.metadata.get("beats",        "4"))
    count_in   = int(log.metadata.get("count_in",     "2"))
    latency_ms = float(log.metadata.get("latency_ms", "0"))

    beat_s     = 60.0 / bpm
    count_in_s = count_in * beat_s

    target_times_s = [count_in_s + i * beat_s for i in range(beats)]

    hit_errors:  dict[int, float] = {}
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
                "onset_s":  target_s + err_s,   # = event.time_sec (compensated)
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

    # Raw = compensated + latency (physical WAV position before latency correction)
    raw_onset_times_s: list[float] = []
    if latency_ms != 0.0:
        offset_s = latency_ms / 1000.0
        for d in onset_data:
            if d["onset_s"] is not None:
                raw_onset_times_s.append(d["onset_s"] + offset_s)

    # Click schedule in waveform time
    click_times_s: list[float] = [i * beat_s for i in range(count_in)]
    for i in range(beats):
        click_times_s.append(count_in_s + i * beat_s)

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


# ── Inspect-mode helpers (pure, tested) ───────────────────────────────────────

def compute_inspect_view(
    center_s: float,
    window_ms: float,
    total_duration: float,
) -> tuple[float, float]:
    """Return (view_start_s, view_end_s) for a window centered on center_s.

    Clamps both ends to [0, total_duration].
    """
    half = window_ms / 1000.0
    return max(0.0, center_s - half), min(total_duration, center_s + half)


def ruler_step_ms(window_ms: float) -> float:
    """Return ruler tick spacing (ms) appropriate for the given half-window.

    ≤ 25 ms  →  10 ms steps  (fine scale)
    >  25 ms  →  25 ms steps  (coarse scale)
    """
    return 10.0 if window_ms <= 25.0 else 25.0


def sample_index_to_s(sample_index: int, sample_rate: int) -> float:
    """Convert a sample index to seconds: sample_index / sample_rate.

    This is the fundamental waveform x-axis time base.  WAV sample 0 = t=0.
    A sample at index N occurred at N / sample_rate seconds from recording start.
    """
    return sample_index / sample_rate


def effective_view_duration(planned_s: float, audio_s: float | None) -> float:
    """Return the duration to use for the waveform x-axis view.

    Uses the actual audio duration when it is a positive finite number; this
    is the physically recorded length and may differ from the planned session
    length (e.g. the session was stopped early).  Falls back to planned_s
    when no audio is available (no WAV embedded).
    """
    if audio_s is not None and audio_s > 0:
        return audio_s
    return planned_s


def ruler_ticks(
    view_start_s: float,
    view_end_s: float,
    center_s: float,
    step_ms: float,
) -> list[tuple[float, str]]:
    """Return (time_s, label) pairs for ruler ticks within [view_start_s, view_end_s].

    Labels are relative to center_s in ms: '-50 ms', '0', '+25 ms', etc.
    Returns an empty list if step_ms <= 0.
    """
    step_s = step_ms / 1000.0
    if step_s <= 0:
        return []
    n_start = math.ceil( (view_start_s - center_s) / step_s)
    n_end   = math.floor((view_end_s   - center_s) / step_s)
    ticks: list[tuple[float, str]] = []
    for n in range(n_start, n_end + 1):
        t      = center_s + n * step_s
        rel_ms = round(n * step_ms)
        if rel_ms == 0:
            label = "0"
        elif rel_ms > 0:
            label = f"+{rel_ms} ms"
        else:
            label = f"{rel_ms} ms"
        ticks.append((t, label))
    return ticks


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
  background: #121220; color: #d0d0e8;
  font-family: 'Segoe UI', system-ui, sans-serif;
  padding: 20px; max-width: 1440px; margin: 0 auto;
}
h2    { font-size: 1.2rem; font-weight: 600; margin-bottom: 4px; color: #e8e8ff; }
.meta { font-size: 0.8rem; color: #777; margin-bottom: 14px; }

.ctrl-row {
  display: flex; align-items: center; gap: 7px;
  margin-bottom: 8px; flex-wrap: wrap;
}
.ctrl-label { font-size: 0.72rem; color: #555; flex-shrink: 0; user-select: none; }
.vdiv { width: 1px; height: 22px; background: #2c2c3e; flex-shrink: 0; }

button {
  background: #1e1e32; color: #a8a8c8;
  border: 1px solid #35354c; border-radius: 4px;
  padding: 4px 9px; cursor: pointer; font-size: 0.8rem; line-height: 1.5;
  transition: background 0.1s, border-color 0.1s; white-space: nowrap;
}
button:hover    { background: #2a2a45; border-color: #50506a; }
button:disabled { opacity: 0.3; cursor: default; }
button.active   { background: #2e1f68; border-color: #7c4dbd; color: #d8c8ff; }
button.miss     { color: #c0443a; border-color: rgba(192,68,58,0.4); }
button.miss.active { background: #3a1010; border-color: #c0443a; color: #ff8880; }
button.spd      { font-family: monospace; }
button.spd.cur  { background: #0f2535; border-color: #4488bb; color: #88ccff; }
button.loop-on  { background: #0e2e20; border-color: #2ecc71; color: #70e8a0; }
button.insp-on  { background: #1e2d10; border-color: #88cc44; color: #bbee88; }
button.warn     { color: #c07820; border-color: rgba(192,120,32,0.4); }

audio  { height: 34px; flex-shrink: 0; }

.beat-row { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 10px; }
.beat-row button.downbeat { border-color: rgba(255,210,50,0.55); color: #d4aa30; }
.beat-row button.downbeat.active { background: #2a2000; border-color: #ffd032; color: #ffe060; }

canvas {
  display: block; border: 1px solid #202030;
  background: #07070f; cursor: crosshair; width: 100%; height: 250px;
}

#nudge-val {
  font-family: monospace; font-size: 0.82rem;
  min-width: 62px; text-align: center; padding: 3px 7px;
  border: 1px solid #35354c; border-radius: 4px;
  background: #111122; color: #888; user-select: none;
}
#nudge-val.live { color: #e07820; border-color: rgba(224,120,32,0.55); }

#time-disp  { font-family: monospace; font-size: 0.8rem;  color: #666; min-width: 70px; }
#speed-disp { font-family: monospace; font-size: 0.76rem; color: #4488bb; min-width: 32px; }

.hint { font-size: 0.68rem; color: #333; margin-top: 5px; line-height: 2.0; }
kbd {
  font-family: monospace; font-size: 0.68rem;
  background: #1c1c2e; border: 1px solid #3a3a50;
  border-radius: 3px; padding: 1px 4px;
}
.tb-note { font-size: 0.68rem; color: #383850; margin-top: 4px; line-height: 1.6; }
.legend {
  display: flex; gap: 14px; flex-wrap: wrap;
  margin-top: 8px; font-size: 0.7rem; color: #555; align-items: center;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.sw-s { width: 18px; height: 3px; border-radius: 1px; flex-shrink: 0; }
.sw-d { width: 18px; height: 0; border-top-width: 2px; border-top-style: dashed; flex-shrink: 0; }
button.gain.cur { background: #0f2535; border-color: #4488bb; color: #88ccff; }
button.vw.cur   { background: #0d2535; border-color: #3ab0d0; color: #70d8f0; }
#loop-info {
  font-family: monospace; font-size: 0.72rem; color: #2ecc71;
  margin-top: 4px; min-height: 1.1em;
}
#debug-panel {
  font-family: monospace; font-size: 0.67rem; color: #44445a;
  margin-top: 5px; display: flex; flex-wrap: wrap; gap: 16px; line-height: 2.0;
}
#debug-panel span::before { content: attr(data-lbl) ": "; color: #333348; }
</style>
</head>
<body>
<div id="error-banner" style="display:none;background:#3a0a0a;color:#ff6060;border:1px solid #c03030;border-radius:4px;padding:10px 14px;margin-bottom:10px;font-family:monospace;font-size:0.85rem"></div>
<h2 id="pg-title">Bass Practice Replay</h2>
<div class="meta" id="pg-meta"></div>

<!-- Row 1: player + speed + time -->
<div class="ctrl-row">
  <audio id="player" controls></audio>
  <div class="vdiv"></div>
  <span class="ctrl-label">Speed</span>
  <button class="spd" data-rate="0.25">0.25&times;</button>
  <button class="spd" data-rate="0.5">0.5&times;</button>
  <button class="spd" data-rate="0.75">0.75&times;</button>
  <button class="spd" data-rate="1">1&times;</button>
  <span id="speed-disp">1&times;</span>
  <div class="vdiv"></div>
  <span id="time-disp">0.000 s</span>
</div>

<!-- Row 2: playback + loop + navigation -->
<div class="ctrl-row">
  <button id="btn-prev">&#9664; Prev</button>
  <button id="btn-play-region">&#9654; Play Window</button>
  <button id="btn-loop">&#8635; Loop</button>
  <button id="btn-next">Next &#9654;</button>
  <div class="vdiv"></div>
  <button id="btn-zoom-out">Zoom Out</button>
  <div class="vdiv" id="vwdiv" style="display:none"></div>
  <span class="ctrl-label" id="vwlbl" style="display:none">View</span>
  <button class="vw" id="btn-vw-perf" data-mode="performance" style="display:none"
    title="Waveform shifted left by latency_ms: corrected onset aligns with waveform attack">Performance</button>
  <button class="vw" id="btn-vw-raw"  data-mode="raw"         style="display:none"
    title="Waveform in raw audio time: raw audio onset aligns with waveform attack">Raw Audio</button>
</div>

<!-- Row 3: view center + window + gain + nudge -->
<div class="ctrl-row">
  <span class="ctrl-label">Center on</span>
  <button id="btn-ctr-target">Target</button>
  <button id="btn-ctr-corr">Corrected</button>
  <button id="btn-ctr-raw" style="display:none">Raw</button>
  <div class="vdiv"></div>
  <span class="ctrl-label">&plusmn;</span>
  <button class="rwin" data-ms="25">25 ms</button>
  <button class="rwin" data-ms="50">50 ms</button>
  <button class="rwin" data-ms="100">100 ms</button>
  <button class="rwin" data-ms="250">250 ms</button>
  <button class="rwin" data-ms="1000">1 s</button>
  <div class="vdiv"></div>
  <span class="ctrl-label">Gain</span>
  <button class="gain" data-g="1">1&times;</button>
  <button class="gain" data-g="2">2&times;</button>
  <button class="gain" data-g="4">4&times;</button>
  <button class="gain" data-g="8">8&times;</button>
  <div class="vdiv"></div>
  <span class="ctrl-label">Nudge</span>
  <button id="btn-nudge-m">&#8722;10 ms</button>
  <div id="nudge-val">0 ms</div>
  <button id="btn-nudge-p">+10 ms</button>
  <button id="btn-nudge-r" class="warn">Reset</button>
</div>

<!-- Row 4: inspect mode -->
<div class="ctrl-row" id="inspect-row">
  <span class="ctrl-label">Inspect</span>
  <button id="btn-insp-target">Target</button>
  <button id="btn-insp-corr">Corrected</button>
  <button id="btn-insp-raw">Raw Onset</button>
  <div class="vdiv"></div>
  <span class="ctrl-label">&plusmn;</span>
  <button class="iwin" data-ms="10">10 ms</button>
  <button class="iwin" data-ms="25">25 ms</button>
  <button class="iwin" data-ms="50">50 ms</button>
  <button class="iwin" data-ms="100">100 ms</button>
  <div class="vdiv" id="insp-extra-div" style="display:none"></div>
  <button id="btn-play-insp"  style="display:none">&#9654; Play Inspected</button>
  <button id="btn-exit-insp"  style="display:none">&#215; Exit Inspect</button>
</div>

<div class="beat-row" id="beat-row"></div>
<canvas id="waveform"></canvas>
<div id="loop-info"></div>
<div id="debug-panel">
  <span id="dbg-view"    data-lbl="view"></span>
  <span id="dbg-dur"     data-lbl="duration"></span>
  <span id="dbg-target"  data-lbl="target"></span>
  <span id="dbg-corr"    data-lbl="corrected (t_comp)"></span>
  <span id="dbg-raw"     data-lbl="raw (t_raw)"></span>
  <span id="dbg-diff"    data-lbl="raw−corr"></span>
  <span id="dbg-head"    data-lbl="playhead"></span>
  <span id="dbg-buf"     data-lbl="sr"></span>
  <span id="dbg-warn"    style="color:#e07820"></span>
</div>

<div class="hint">
  <kbd>Space</kbd> play/pause &nbsp;
  <kbd>&larr;</kbd><kbd>&rarr;</kbd> prev/next beat &nbsp;
  <kbd>[</kbd><kbd>]</kbd> speed &nbsp;
  <kbd>L</kbd> loop &nbsp;
  <kbd>T</kbd> inspect target &nbsp;
  <kbd>C</kbd> inspect corrected &nbsp;
  <kbd>R</kbd> inspect raw &nbsp;
  <kbd>1</kbd>&ndash;<kbd>4</kbd> inspect window &nbsp;&nbsp;
  Click waveform to seek &nbsp;&middot;&nbsp;
  Center on / &plusmn; buttons reposition view &nbsp;&middot;&nbsp;
  Gain 2&times;/4&times;/8&times; amplifies waveform (clipping is visible)
</div>
<div class="tb-note">
  <b>Time bases</b> &nbsp;&middot;&nbsp;
  Waveform x&#x2011;axis = sample&#x2011;index / sample&#x2011;rate = t<sub>raw</sub> (raw audio time from recording start) &nbsp;&middot;&nbsp;
  Target = count&#x2011;in + beat position (session/performance time) &nbsp;&middot;&nbsp;
  Corrected onset = t<sub>raw</sub> &minus; latency (already subtracted once in SessionLog; drawn in session time) &nbsp;&middot;&nbsp;
  Raw audio onset = corrected + latency = t<sub>raw</sub> &mdash; <em>this line should sit on the waveform attack</em> &nbsp;&middot;&nbsp;
  Corrected onset is latency&#x2011;ms before the attack; target aligns with corrected onset for on&#x2011;time playing
</div>
<div class="legend" id="legend"></div>

<script>
/* ── Injected session data (no DOM access) ────────────────────────────── */
const DATA      = INJECT_DATA_JSON;
const AUDIO_B64 = "INJECT_AUDIO_B64";
const WINDOW_MS = INJECT_WINDOW_MS;

/* ── Visible error banner (callable before _init completes) ───────────── */
function showError(msg) {
  const b = document.getElementById('error-banner');
  if (b) { b.textContent = '⚠ Init error: ' + msg; b.style.display = 'block'; }
  console.error('Bass Replay init error:', msg);
}

window.addEventListener('DOMContentLoaded', function() {
  try { _init(); } catch(e) { showError(e.message || String(e)); }
});

function _init() {

/* ── DOM lookups — must precede any draw() call ──────────────────────── */
const player = document.getElementById('player');
const canvas = document.getElementById('waveform');
if (!canvas) { showError('canvas#waveform not found'); return; }
const ctx2d  = canvas.getContext('2d');
if (!ctx2d)  { showError('2d context unavailable'); return; }

/* ── Audio element ─────────────────────────────────────────────────────── */
if (AUDIO_B64) {
  player.src = 'data:audio/wav;base64,' + AUDIO_B64;
} else {
  player.style.display = 'none';
  ['btn-play-region','btn-loop','btn-play-insp'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  });
}

/* ── Decode audio for waveform rendering ─────────────────────────────────
   Use atob+Uint8Array rather than fetch() to stay fully offline.          */
let audioBuffer = null;
if (AUDIO_B64) {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const bin = atob(AUDIO_B64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    audioCtx.decodeAudioData(arr.buffer.slice(0)).then(buf => {
      audioBuffer = buf;
      // Switch x-axis to actual audio length; planned duration may be longer
      // (e.g. session stopped early) or shorter (recording started before session).
      audioDur = buf.duration > 0 ? buf.duration : totalDur;
      if (viewEnd > audioDur) viewEnd = audioDur;
      // One-beat mode: centre view on the target instead of showing full duration.
      if (DATA.beats === 1 && selectedBeat === null) selectBeat(0);
      else draw();
    }).catch(e => console.warn('Audio decode error:', e));
  } catch(e) { console.warn('AudioContext unavailable:', e); }
}

/* ── State ───────────────────────────────────────────────────────────────
   windowMs     : half-window for beat-view zoom (ms)
   viewStart/End: currently visible time range (seconds from recording start)
   selectedBeat : index of selected beat button (null = none)
   animId       : requestAnimationFrame handle
   isLooping    : loop viewStart→viewEnd on Play Region/Play Inspected
   loopEnd      : null = free play; number = stop/loop boundary
   nudgeMs      : visual-only shift for compensated onset lines
   inspectMode  : null | 'target' | 'corrected' | 'raw'
   inspectCenter: inspected time in seconds (null when no inspect mode)
   inspectLabel : text label shown on center marker
   inspectWinMs : half-window for inspect zoom (ms)               */
const totalDur    = DATA.total_duration;  // planned duration from session metadata
let audioDur      = totalDur;             // actual audio duration; updated after decode
let windowMs      = WINDOW_MS;
let viewStart     = 0;
let viewEnd       = totalDur;             // clamped to audioDur once audio loads
let selectedBeat  = null;
let animId        = null;
let isLooping     = false;
let loopEnd       = null;
let nudgeMs       = 0;
let inspectMode   = null;
let inspectCenter = null;
let inspectLabel  = null;
let inspectWinMs  = 50;
let waveGain      = 1;
// 'performance': waveform shifted left by latency_ms (corrected onset aligns with attack)
// 'raw'        : waveform in raw audio time          (raw audio onset aligns with attack)
let viewMode      = DATA.latency_ms !== 0 ? 'performance' : 'raw';
const latencyS    = DATA.latency_ms / 1000.0;

/* ── View-mode waveform offset ───────────────────────────────────────────
   In Performance view the waveform is fetched latency_ms later in the WAV
   so the transient attack appears at the corrected-onset display position. */
function curWaveOff() {
  return viewMode === 'performance' ? latencyS : 0.0;
}

/* ── Speed ───────────────────────────────────────────────────────────────*/
const SPEEDS = [0.25, 0.5, 0.75, 1.0];
let spdIdx   = SPEEDS.indexOf(1.0);

function setSpeed(rate) {
  const i = SPEEDS.indexOf(rate);
  if (i === -1) return;
  spdIdx = i;
  player.playbackRate = rate;
  document.querySelectorAll('.spd').forEach(b =>
    b.classList.toggle('cur', parseFloat(b.dataset.rate) === rate)
  );
  document.getElementById('speed-disp').textContent = rate + '×';
}
function adjustSpeed(dir) {
  const n = spdIdx + dir;
  if (n >= 0 && n < SPEEDS.length) setSpeed(SPEEDS[n]);
}
document.querySelectorAll('.spd').forEach(b =>
  b.addEventListener('click', () => setSpeed(parseFloat(b.dataset.rate)))
);

/* ── Page header ─────────────────────────────────────────────────────────*/
const hitCount  = DATA.onset_data.filter(d => !d.is_miss).length;
const missCount = DATA.onset_data.filter(d =>  d.is_miss).length;
document.getElementById('pg-title').textContent =
  'Bass Practice Replay — ' + DATA.bpm.toFixed(0) + ' BPM';
document.getElementById('pg-meta').textContent =
  DATA.beats + ' beats · ' + DATA.count_in + ' count-in · ' +
  'latency: ' + DATA.latency_ms.toFixed(0) + ' ms · ' +
  'hits: ' + hitCount + '/' + DATA.beats + ' · misses: ' + missCount;

/* ── Beat buttons ────────────────────────────────────────────────────────*/
const beatRow = document.getElementById('beat-row');
DATA.onset_data.forEach((d, i) => {
  const btn = document.createElement('button');
  const errStr = (!d.is_miss && d.err_ms !== null)
    ? ' (' + (d.err_ms > 0 ? '+' : '') + d.err_ms.toFixed(0) + 'ms)' : '';
  btn.textContent = (i + 1) + (d.is_miss ? ' ✗' : errStr);
  btn.title = d.is_miss ? 'Beat ' + (i+1) + ': MISS'
                        : 'Beat ' + (i+1) + ': ' + d.err_ms.toFixed(0) + ' ms';
  if (d.is_miss)  btn.classList.add('miss');
  if (i === 0)    btn.classList.add('downbeat');
  btn.addEventListener('click', () => selectBeat(i));
  beatRow.appendChild(btn);
});

function selectBeat(i) {
  selectedBeat = i;
  if (inspectMode !== null) {
    startInspect(inspectMode, i);  // re-centre on new beat
    return;
  }
  const halfWin = windowMs / 1000.0;
  const target  = DATA.target_times_s[i];
  viewStart = Math.max(0.0,      target - halfWin);
  viewEnd   = Math.min(audioDur, target + halfWin);
  updateBeatButtons();
  draw();
}
function updateBeatButtons() {
  beatRow.querySelectorAll('button').forEach((b, i) =>
    b.classList.toggle('active', i === selectedBeat)
  );
}

/* ── View window buttons ─────────────────────────────────────────────────*/
document.querySelectorAll('.rwin').forEach(b => {
  b.addEventListener('click', () => {
    windowMs = parseInt(b.dataset.ms, 10);
    document.querySelectorAll('.rwin').forEach(x => x.classList.toggle('active', x === b));
    if (selectedBeat !== null && inspectMode === null) selectBeat(selectedBeat);
  });
});
(function initRwinBtn() {
  let matched = false;
  document.querySelectorAll('.rwin').forEach(b => {
    const ms = parseInt(b.dataset.ms, 10);
    if (ms === windowMs) { b.classList.add('active'); matched = true; }
  });
  if (!matched) {
    const fb = document.querySelector('.rwin[data-ms="250"]');
    if (fb) { fb.classList.add('active'); windowMs = 250; }
  }
})();

/* ── Center-on buttons ───────────────────────────────────────────────────
   Snap viewStart/viewEnd to ±windowMs around a specific time point.
   Raw button hidden when latency == 0 (no raw onset available).          */
function centerOn(timeSec) {
  if (timeSec === null || timeSec === undefined) return;
  const half = windowMs / 1000.0;
  viewStart = Math.max(0.0,      timeSec - half);
  viewEnd   = Math.min(audioDur, timeSec + half);
  draw();
}
document.getElementById('btn-ctr-target').addEventListener('click', () => {
  if (selectedBeat === null) return;
  centerOn(DATA.target_times_s[selectedBeat]);
});
document.getElementById('btn-ctr-corr').addEventListener('click', () => {
  if (selectedBeat === null) return;
  const d = DATA.onset_data[selectedBeat];
  if (d && d.onset_s !== null) centerOn(d.onset_s + nudgeMs / 1000.0);
});
document.getElementById('btn-ctr-raw').addEventListener('click', () => {
  if (selectedBeat === null) return;
  const d = DATA.onset_data[selectedBeat];
  if (d && d.onset_s !== null) centerOn(d.onset_s + DATA.latency_ms / 1000.0);
});
if (DATA.latency_ms !== 0) {
  const el = document.getElementById('btn-ctr-raw');
  if (el) el.style.display = '';
}

/* ── Gain buttons ────────────────────────────────────────────────────────
   Multiply waveform vertical amplitude by waveGain.  Canvas clips values
   outside [0, H] naturally, so clipping is visible rather than silent.  */
function setGain(g) {
  waveGain = g;
  document.querySelectorAll('.gain').forEach(b =>
    b.classList.toggle('cur', parseFloat(b.dataset.g) === g)
  );
  draw();
}
document.querySelectorAll('.gain').forEach(b =>
  b.addEventListener('click', () => setGain(parseFloat(b.dataset.g)))
);

/* ── Zoom out ────────────────────────────────────────────────────────────*/
document.getElementById('btn-zoom-out').addEventListener('click', () => {
  inspectMode = null; inspectCenter = null; inspectLabel = null;
  updateInspectButtons();
  selectedBeat = null; viewStart = 0; viewEnd = audioDur; loopEnd = null;
  updateBeatButtons(); draw();
});

/* ── View-mode toggle ────────────────────────────────────────────────────
   Performance : waveform shifted left by latency_ms; corrected onset sits
                 on the transient — natural for timing analysis.
   Raw Audio   : waveform in sample/sr time; raw-onset marker sits on the
                 transient — useful for debugging time-base alignment.      */
function setViewMode(mode) {
  viewMode = mode;
  document.querySelectorAll('.vw').forEach(b =>
    b.classList.toggle('cur', b.dataset.mode === mode)
  );
  draw();
}
document.querySelectorAll('.vw').forEach(b =>
  b.addEventListener('click', () => setViewMode(b.dataset.mode))
);
if (DATA.latency_ms !== 0) {
  ['vwdiv', 'vwlbl', 'btn-vw-perf', 'btn-vw-raw'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = '';
  });
}

/* ── Prev / Next beat ────────────────────────────────────────────────────*/
document.getElementById('btn-prev').addEventListener('click', () => {
  if (selectedBeat === null) selectBeat(DATA.beats - 1);
  else if (selectedBeat > 0) selectBeat(selectedBeat - 1);
});
document.getElementById('btn-next').addEventListener('click', () => {
  if (selectedBeat === null)              selectBeat(0);
  else if (selectedBeat < DATA.beats - 1) selectBeat(selectedBeat + 1);
});

/* ── Loop ────────────────────────────────────────────────────────────────*/
const loopBtn = document.getElementById('btn-loop');
function toggleLoop() {
  isLooping = !isLooping;
  loopBtn.classList.toggle('loop-on', isLooping);
  loopBtn.textContent = isLooping ? '虵 Loop ON' : '虵 Loop';
}
loopBtn.addEventListener('click', toggleLoop);

/* ── Shared play-region function ─────────────────────────────────────────*/
function playRegion() {
  const off = curWaveOff();          // display time → WAV time offset
  loopEnd = viewEnd + off;
  player.currentTime = viewStart + off;
  player.play().catch(() => {});
}
document.getElementById('btn-play-region').addEventListener('click', playRegion);
document.getElementById('btn-play-insp').addEventListener('click',   playRegion);

/* ── Nudge ───────────────────────────────────────────────────────────────*/
function applyNudge(ms) {
  nudgeMs = ms;
  const el = document.getElementById('nudge-val');
  el.textContent = (ms > 0 ? '+' : '') + ms.toFixed(0) + ' ms';
  el.classList.toggle('live', ms !== 0);
  draw();
}
document.getElementById('btn-nudge-m').addEventListener('click', () => applyNudge(nudgeMs - 10));
document.getElementById('btn-nudge-p').addEventListener('click', () => applyNudge(nudgeMs + 10));
document.getElementById('btn-nudge-r').addEventListener('click', () => applyNudge(0));

/* ── Inspect mode ────────────────────────────────────────────────────────
   Each Inspect button re-centres the view on the inspected time point
   and displays a ruler showing ms relative to that centre.

   inspectMode   : null | 'target' | 'corrected' | 'raw'
   inspectCenter : time in seconds (same coordinate as WAV x-axis)         */

function inspectColor() {
  if (inspectMode === 'target') return '#ffe060';
  if (inspectMode === 'raw')    return '#78a8ff';
  if (inspectMode === 'corrected' && selectedBeat !== null) {
    const d = DATA.onset_data[selectedBeat];
    if (d && d.err_ms !== null) {
      const abs = Math.abs(d.err_ms + nudgeMs);
      return abs <= 50 ? '#2ecc71' : abs <= 120 ? '#e67e22' : '#e74c3c';
    }
  }
  return '#aaaaff';
}

function startInspect(mode, beatIdx) {
  if (beatIdx === null) return;
  const d = DATA.onset_data[beatIdx];
  let center = null;

  if (mode === 'target') {
    center = DATA.target_times_s[beatIdx];
    inspectLabel = 'TARGET';
  } else if (mode === 'corrected') {
    if (d.is_miss || d.onset_s === null) return;
    center = d.onset_s + nudgeMs / 1000.0;
    inspectLabel = 'CORRECTED ONSET';
  } else if (mode === 'raw') {
    if (DATA.latency_ms === 0 || d.onset_s === null) return;
    center = d.onset_s + DATA.latency_ms / 1000.0;
    inspectLabel = 'RAW ONSET';
  }
  if (center === null) return;

  inspectMode   = mode;
  inspectCenter = center;
  selectedBeat  = beatIdx;

  const half = inspectWinMs / 1000.0;
  viewStart = Math.max(0.0,      center - half);
  viewEnd   = Math.min(audioDur, center + half);

  updateBeatButtons();
  updateInspectButtons();
  draw();
}

function exitInspect() {
  inspectMode = null; inspectCenter = null; inspectLabel = null;
  updateInspectButtons();
  if (selectedBeat !== null) {
    const halfWin = windowMs / 1000.0;
    const target  = DATA.target_times_s[selectedBeat];
    viewStart = Math.max(0.0,      target - halfWin);
    viewEnd   = Math.min(audioDur, target + halfWin);
  } else {
    viewStart = 0; viewEnd = audioDur;
  }
  draw();
}

function updateInspectButtons() {
  ['target','corr','raw'].forEach(k => {
    const el = document.getElementById('btn-insp-' + k);
    if (el) el.classList.toggle('insp-on',
      (k === 'corr' ? 'corrected' : k) === inspectMode);
  });
  const inMode = inspectMode !== null;
  const divEl  = document.getElementById('insp-extra-div');
  ['btn-play-insp','btn-exit-insp'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = inMode ? '' : 'none';
  });
  if (divEl) divEl.style.display = inMode ? '' : 'none';
}

function setInspectWin(ms) {
  inspectWinMs = ms;
  document.querySelectorAll('.iwin').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.ms, 10) === ms)
  );
  if (inspectMode !== null && selectedBeat !== null) startInspect(inspectMode, selectedBeat);
}

// Wire inspect buttons
document.getElementById('btn-insp-target').addEventListener('click', () => {
  if (selectedBeat !== null) startInspect('target', selectedBeat);
});
document.getElementById('btn-insp-corr').addEventListener('click', () => {
  if (selectedBeat !== null) startInspect('corrected', selectedBeat);
});
(function setupRawBtn() {
  const el = document.getElementById('btn-insp-raw');
  if (DATA.latency_ms === 0) { el.style.display = 'none'; return; }
  el.addEventListener('click', () => {
    if (selectedBeat !== null) startInspect('raw', selectedBeat);
  });
})();
document.getElementById('btn-exit-insp').addEventListener('click', exitInspect);
document.querySelectorAll('.iwin').forEach(b =>
  b.addEventListener('click', () => setInspectWin(parseInt(b.dataset.ms, 10)))
);
// Initialise iwin active state
document.querySelectorAll('.iwin').forEach(b =>
  b.classList.toggle('active', parseInt(b.dataset.ms, 10) === inspectWinMs)
);

/* ── Playhead animation ──────────────────────────────────────────────────
   requestAnimationFrame reads player.currentTime (the authoritative media
   clock) on every frame, so the playhead stays accurate at any playback
   rate without wall-clock drift.                                          */
player.addEventListener('play',  () => startPlayhead());
player.addEventListener('pause', () => {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  loopEnd = null;
  draw();
});

function startPlayhead() {
  if (animId) cancelAnimationFrame(animId);
  const tick = () => {
    const ct = player.currentTime || 0;
    if (loopEnd !== null && ct >= loopEnd) {
      if (isLooping) {
        player.currentTime = viewStart + curWaveOff(); // display → WAV time
        // Continue — do NOT return; next frame redraws
      } else {
        player.pause();
        return;
      }
    }
    draw();
    animId = requestAnimationFrame(tick);
  };
  animId = requestAnimationFrame(tick);
}

/* ── Canvas ──────────────────────────────────────────────────────────────*/
function resizeCanvas() {
  const r      = canvas.getBoundingClientRect();
  canvas.width  = Math.round(r.width);
  canvas.height = Math.round(r.height);
  draw();
}
window.addEventListener('resize', resizeCanvas);

canvas.addEventListener('click', e => {
  const r    = canvas.getBoundingClientRect();
  const frac = (e.clientX - r.left) / r.width;
  const t    = viewStart + frac * (viewEnd - viewStart);  // display time
  player.currentTime = Math.max(0, Math.min(t + curWaveOff(), player.duration || 0));
  draw();
});

/* ── Drawing helpers ─────────────────────────────────────────────────────*/
function tToX(t) {
  return (t - viewStart) / (viewEnd - viewStart) * canvas.width;
}

function rulerStepMs() {
  return inspectWinMs <= 25 ? 10 : 25;
}

function drawLoopBracket(y, W) {
  ctx2d.strokeStyle = 'rgba(46,204,113,0.55)';
  ctx2d.lineWidth   = 1.5;
  ctx2d.lineCap     = 'square';
  ctx2d.setLineDash([]);
  ctx2d.beginPath();
  ctx2d.moveTo(2, y + 5); ctx2d.lineTo(2, y);
  ctx2d.lineTo(W - 2, y); ctx2d.lineTo(W - 2, y + 5);
  ctx2d.stroke();
  ctx2d.lineCap = 'butt';
}

function drawCenterMarker(wH, W) {
  const t = inspectCenter;
  if (t < viewStart || t > viewEnd) return;
  const x     = tToX(t);
  const tickH = wH * 0.06;
  const color = inspectColor();

  // Thick centre line (covers full waveform height at reduced opacity)
  ctx2d.strokeStyle = color;
  ctx2d.lineWidth   = 3;
  ctx2d.globalAlpha = 0.75;
  ctx2d.setLineDash([]);
  ctx2d.beginPath();
  ctx2d.moveTo(x, tickH + 2);
  ctx2d.lineTo(x, wH - 2);
  ctx2d.stroke();
  ctx2d.globalAlpha = 1.0;

  // Label box just below click ticks
  ctx2d.font = 'bold 9px sans-serif';
  ctx2d.textAlign    = 'center';
  ctx2d.textBaseline = 'top';
  const tw = ctx2d.measureText(inspectLabel).width;
  const lx = Math.max(tw / 2 + 4, Math.min(W - tw / 2 - 4, x));
  const ly = tickH + 4;
  ctx2d.fillStyle = 'rgba(6,6,18,0.88)';
  ctx2d.fillRect(lx - tw / 2 - 3, ly, tw + 6, 13);
  ctx2d.fillStyle = color;
  ctx2d.fillText(inspectLabel, lx, ly + 1);
  ctx2d.textAlign    = 'left';
  ctx2d.textBaseline = 'alphabetic';
}

function drawRuler(wH, W, H) {
  const stepMs = rulerStepMs();
  const stepS  = stepMs / 1000.0;
  const center = inspectCenter;

  // Ruler background
  ctx2d.fillStyle = '#0c0c1e';
  ctx2d.fillRect(0, wH, W, H - wH);

  // Separator line
  ctx2d.strokeStyle = '#25253a';
  ctx2d.lineWidth   = 1;
  ctx2d.setLineDash([]);
  ctx2d.beginPath(); ctx2d.moveTo(0, wH); ctx2d.lineTo(W, wH); ctx2d.stroke();

  if (stepS <= 0) return;
  const n0 = Math.ceil( (viewStart - center) / stepS);
  const n1 = Math.floor((viewEnd   - center) / stepS);

  ctx2d.font = '9px monospace';
  ctx2d.textBaseline = 'top';
  ctx2d.textAlign    = 'center';

  for (let n = n0; n <= n1; n++) {
    const t      = center + n * stepS;
    const x      = tToX(t);
    const isZero = (n === 0);
    const relMs  = Math.round(n * stepMs);
    const lbl    = relMs === 0 ? '0'
                 : relMs  > 0 ? '+' + relMs + 'ms'
                 :              relMs + 'ms';

    ctx2d.strokeStyle = isZero ? '#7070cc' : '#333348';
    ctx2d.lineWidth   = isZero ? 1.5 : 1;
    ctx2d.beginPath();
    ctx2d.moveTo(x, wH + 1); ctx2d.lineTo(x, wH + (isZero ? 10 : 6)); ctx2d.stroke();

    ctx2d.fillStyle = isZero ? '#8888ee' : '#404058';
    const cx = Math.max(14, Math.min(W - 14, x));
    ctx2d.fillText(lbl, cx, wH + 11);
  }

  ctx2d.textAlign    = 'left';
  ctx2d.textBaseline = 'alphabetic';
}

/* ── Main draw ───────────────────────────────────────────────────────────*/
function draw() {
  const W = canvas.width;
  const H = canvas.height;
  if (!W || !H) return;

  // Reserve bottom 32px for ruler when inspect mode is active
  const rulerH = (inspectMode !== null) ? 32 : 0;
  const wH     = H - rulerH;

  ctx2d.clearRect(0, 0, W, H);

  // ── Count-in shading ────────────────────────────────────────────────
  if (DATA.count_in_s > viewStart) {
    const x0 = Math.max(0, tToX(0));
    const x1 = tToX(Math.min(DATA.count_in_s, viewEnd));
    ctx2d.fillStyle = 'rgba(100,149,237,0.06)';
    ctx2d.fillRect(x0, 0, x1 - x0, wH);
  }

  // ── Waveform (min-max per pixel) ─────────────────────────────────────
  if (audioBuffer) {
    const ch   = audioBuffer.getChannelData(0);
    const sr   = audioBuffer.sampleRate;
    const mid  = wH * 0.5;
    const span = viewEnd - viewStart;
    ctx2d.strokeStyle = '#505068';
    ctx2d.lineWidth   = 0.8;
    const waveOff = curWaveOff(); // >0 in Performance view: fetch samples later in WAV
    ctx2d.beginPath();
    for (let px = 0; px < W; px++) {
      const tA = viewStart + px       / W * span;
      const tB = viewStart + (px + 1) / W * span;
      const sA = Math.max(0,         Math.floor((tA + waveOff) * sr));
      const sB = Math.min(ch.length, Math.ceil( (tB + waveOff) * sr));
      let lo = 0, hi = 0;
      for (let s = sA; s < sB; s++) {
        if (ch[s] > hi) hi = ch[s];
        if (ch[s] < lo) lo = ch[s];
      }
      ctx2d.moveTo(px, mid - hi * mid * 0.88 * waveGain);
      ctx2d.lineTo(px, mid - lo * mid * 0.88 * waveGain);
    }
    ctx2d.stroke();
  }

  // ── Loop region shading (behind everything except count-in) ─────────
  if (isLooping) {
    ctx2d.fillStyle = 'rgba(46,204,113,0.07)';
    ctx2d.fillRect(0, 0, W, wH);
  }

  // ── Click tick marks (top 6 % of waveform area) ───────────────────────
  const tickH = wH * 0.06;
  DATA.click_times_s.forEach(t => {
    if (t < viewStart || t > viewEnd) return;
    ctx2d.strokeStyle = t < DATA.count_in_s
      ? 'rgba(100,149,237,0.9)' : 'rgba(124,77,189,0.9)';
    ctx2d.lineWidth = 2.5; ctx2d.lineCap = 'butt';
    const x = tToX(t);
    ctx2d.beginPath(); ctx2d.moveTo(x, 0); ctx2d.lineTo(x, tickH); ctx2d.stroke();
  });

  // ── Loop region bracket (just below click ticks) ─────────────────────
  if (isLooping) drawLoopBracket(tickH + 3, W);

  // ── Target beat lines + labels ───────────────────────────────────────
  // Beat 1 (downbeat, i=0): gold, solid, slightly thicker
  // Selected beat: purple highlight
  // Other hits: dim grey dashed; Misses: red dashed
  ctx2d.font = '10px monospace';
  DATA.target_times_s.forEach((t, i) => {
    if (t < viewStart - 0.01 || t > viewEnd + 0.01) return;
    const d          = DATA.onset_data[i];
    const isMiss     = d.is_miss;
    const isDownbeat = (i === 0);
    const isSelected = (i === selectedBeat) && (inspectMode === null);
    const x          = tToX(t);

    let lineColor, lineW, dash;
    if (isMiss)       { lineColor='rgba(192,68,58,0.65)'; lineW=1;   dash=[4,4]; }
    else if (isDownbeat) { lineColor='rgba(255,210,50,0.65)'; lineW=2;   dash=[]; }
    else if (isSelected) { lineColor='rgba(180,140,255,0.6)'; lineW=1.5; dash=[4,4]; }
    else               { lineColor='rgba(90,90,110,0.45)';   lineW=1;   dash=[4,4]; }

    ctx2d.strokeStyle=lineColor; ctx2d.lineWidth=lineW; ctx2d.setLineDash(dash);
    ctx2d.beginPath(); ctx2d.moveTo(x, 0); ctx2d.lineTo(x, wH); ctx2d.stroke();
    ctx2d.setLineDash([]);

    const lbl = (isDownbeat ? '▼' : '') + (i + 1) + (isMiss ? ' MISS' : '');
    ctx2d.fillStyle = isMiss ? '#c0443a' : (isDownbeat ? '#ffd032' : (isSelected ? '#b090ff' : '#555'));
    ctx2d.fillText(lbl, x + 3, 16);
  });

  // ── Raw onset lines (only in Raw Audio view) ─────────────────────────────
  // In Performance view the waveform is already shifted so the transient sits
  // on the corrected-onset marker; the raw line would overlap it exactly.
  if (DATA.latency_ms !== 0 && viewMode === 'raw') {
    ctx2d.strokeStyle = 'rgba(120,180,255,0.70)';
    ctx2d.lineWidth   = 1.5;
    ctx2d.setLineDash([4, 3]);
    ctx2d.font = '9px monospace';
    DATA.raw_onset_times_s.forEach(t => {
      if (t < viewStart || t > viewEnd) return;
      const x = tToX(t);
      ctx2d.beginPath(); ctx2d.moveTo(x, 0); ctx2d.lineTo(x, wH); ctx2d.stroke();
      ctx2d.fillStyle = 'rgba(120,180,255,0.80)';
      ctx2d.fillText('RAW', x + 3, wH * 0.3);
    });
    ctx2d.setLineDash([]);
  }

  // ── Compensated onset lines (nudge-shifted for visual preview) ────────
  const nudgeS = nudgeMs / 1000.0;
  ctx2d.font = '10px monospace';
  DATA.onset_data.forEach(d => {
    if (d.is_miss || d.onset_s === null) return;
    const t = d.onset_s + nudgeS;
    if (t < viewStart - 0.01 || t > viewEnd + 0.01) return;
    const abs   = Math.abs(d.err_ms);
    const color = abs <= 50 ? '#2ecc71' : abs <= 120 ? '#e67e22' : '#e74c3c';
    const x     = tToX(t);
    ctx2d.strokeStyle = color; ctx2d.lineWidth = 2.0; ctx2d.setLineDash([]);
    ctx2d.beginPath(); ctx2d.moveTo(x, 0); ctx2d.lineTo(x, wH); ctx2d.stroke();
    const sign     = d.err_ms > 0 ? '+' : '';
    const nudgeSfx = nudgeMs !== 0 ? ' [n' + (nudgeMs > 0 ? '+' : '') + nudgeMs.toFixed(0) + ']' : '';
    ctx2d.fillStyle = color;
    ctx2d.fillText(sign + d.err_ms.toFixed(0) + 'ms' + nudgeSfx, x + 3, wH - 6);
  });

  // ── Inspect centre marker (on top of everything else) ─────────────────
  if (inspectMode !== null && inspectCenter !== null) {
    drawCenterMarker(wH, W);
  }

  // ── Loop indicator bar (bottom edge of waveform area) ─────────────────
  if (isLooping) {
    ctx2d.fillStyle = 'rgba(46,204,113,0.14)';
    ctx2d.fillRect(0, wH - 3, W, 3);
  }

  // ── Playhead ──────────────────────────────────────────────────────────
  const ct     = player.currentTime || 0;
  const ctDisp = ct - curWaveOff();   // WAV time → display (session) time
  document.getElementById('time-disp').textContent = ct.toFixed(3) + ' s';
  if (ctDisp >= viewStart && ctDisp <= viewEnd) {
    const xph = tToX(ctDisp);
    ctx2d.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx2d.lineWidth   = 1.5; ctx2d.setLineDash([]);
    ctx2d.beginPath(); ctx2d.moveTo(xph, 0); ctx2d.lineTo(xph, wH); ctx2d.stroke();
    ctx2d.fillStyle = 'rgba(255,255,255,0.75)';
    ctx2d.beginPath();
    ctx2d.moveTo(xph - 5, 0); ctx2d.lineTo(xph + 5, 0); ctx2d.lineTo(xph, 9);
    ctx2d.closePath(); ctx2d.fill();
  }

  // ── View-mode label (top-right corner of waveform) ───────────────────
  if (DATA.latency_ms !== 0) {
    const modeLabel = viewMode === 'performance'
      ? 'Performance View  (−' + DATA.latency_ms.toFixed(0) + ' ms)'
      : 'Raw Audio View';
    ctx2d.font         = '9px monospace';
    ctx2d.fillStyle    = viewMode === 'performance' ? 'rgba(58,176,208,0.85)' : 'rgba(140,140,180,0.70)';
    ctx2d.textAlign    = 'right';
    ctx2d.textBaseline = 'top';
    ctx2d.fillText(modeLabel, W - 4, 4);
    ctx2d.textAlign    = 'left';
    ctx2d.textBaseline = 'alphabetic';
  }

  // ── Time ruler (inspect mode only) ────────────────────────────────────
  if (inspectMode !== null && inspectCenter !== null) {
    drawRuler(wH, W, H);
  }

  // ── Loop info text (below canvas) ────────────────────────────────────
  const loopInfoEl = document.getElementById('loop-info');
  if (loopInfoEl) {
    loopInfoEl.textContent = isLooping
      ? 'Looping  ' + viewStart.toFixed(3) + ' s → ' + viewEnd.toFixed(3) + ' s'
      : '';
  }

  // ── Debug panel ───────────────────────────────────────────────────────
  // Shows raw time values so waveform/line alignment can be diagnosed.
  (function updateDebug() {
    const d = selectedBeat !== null ? DATA.onset_data[selectedBeat] : null;
    const s = (v, digits) => v !== null && v !== undefined ? v.toFixed(digits) + ' s' : '—';
    const tgt  = d ? s(DATA.target_times_s[selectedBeat], 4) : '—';
    const corr = (d && d.onset_s !== null) ? s(d.onset_s, 4) : '—';
    const raw  = (d && d.onset_s !== null && DATA.latency_ms !== 0)
               ? s(d.onset_s + DATA.latency_ms / 1000.0, 4) : '—';
    // raw − corr = latency_ms (confirms no double-correction: raw IS t_comp + latency)
    const diff = (d && d.onset_s !== null && DATA.latency_ms !== 0)
               ? DATA.latency_ms.toFixed(1) + ' ms  (= configured latency)' : '—';
    const head = s(player.currentTime || 0, 4);
    const sr   = audioBuffer ? audioBuffer.sampleRate + ' Hz' : 'no audio';
    const durTxt = 'planned ' + totalDur.toFixed(3) + ' s / audio ' +
                   (audioBuffer ? audioDur.toFixed(3) + ' s' : '—');
    // Warning: any marker falls outside the actual audio
    const maxMarker = Math.max(
      ...DATA.target_times_s,
      ...DATA.onset_data.filter(x => x.onset_s !== null).map(x => x.onset_s),
      ...(DATA.latency_ms !== 0 ? DATA.raw_onset_times_s : [0])
    );
    const warn = (audioBuffer && maxMarker > audioDur)
               ? '⚠ markers extend past audio (' + maxMarker.toFixed(3) + 's > ' + audioDur.toFixed(3) + 's)'
               : '';
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    set('dbg-view',   viewStart.toFixed(3) + '–' + viewEnd.toFixed(3) + ' s');
    set('dbg-dur',    durTxt);
    set('dbg-target', tgt);
    set('dbg-corr',   corr);
    set('dbg-raw',    raw);
    set('dbg-diff',   diff);
    set('dbg-head',   head);
    set('dbg-buf',    sr);
    set('dbg-warn',   warn);
  })();
}

/* ── Legend ──────────────────────────────────────────────────────────────*/
(function buildLegend() {
  const items = [
    { color: '#505068',               cls: 'sw-s', label: 'Waveform (raw audio)' },
    { color: 'rgba(255,210,50,0.65)', cls: 'sw-s', label: '▼ Beat 1 target (downbeat)' },
    { color: 'rgba(90,90,110,0.45)',  cls: 'sw-d', label: 'Target beat (session time)' },
    { color: '#2ecc71',               cls: 'sw-s', label: 'Corrected onset ≤50 ms (good)' },
    { color: '#e67e22',               cls: 'sw-s', label: 'Corrected onset ≤120 ms (warn)' },
    { color: '#e74c3c',               cls: 'sw-s', label: 'Corrected onset >120 ms / miss' },
    { color: 'rgba(100,149,237,0.9)', cls: 'sw-s', label: 'Count-in click' },
    { color: 'rgba(124,77,189,0.9)',  cls: 'sw-s', label: 'Exercise click' },
  ];
  if (DATA.latency_ms !== 0) items.push({
    color: 'rgba(120,180,255,0.70)',
    cls:   'sw-d',
    label: 'Raw audio onset (+' + DATA.latency_ms.toFixed(0) + ' ms) — should align with waveform attack',
  });
  const leg = document.getElementById('legend');
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'legend-item';
    const sw = document.createElement('div');
    sw.className = item.cls;
    if (item.cls === 'sw-d') sw.style.borderTopColor = item.color;
    else sw.style.background = item.color;
    const lbl = document.createElement('span');
    lbl.textContent = item.label;
    div.appendChild(sw); div.appendChild(lbl);
    leg.appendChild(div);
  });
})();

/* ── Keyboard shortcuts ──────────────────────────────────────────────────
   Space  play/pause           [ / ]    speed down/up
   ← / →  prev/next beat       L        loop toggle
   T      inspect target        C        inspect corrected
   R      inspect raw           1-4      inspect window size (10/25/50/100ms)  */
document.addEventListener('keydown', e => {
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  switch (e.key) {
    case ' ':
      if (tag === 'BUTTON') return;
      e.preventDefault();
      player.paused ? player.play().catch(() => {}) : player.pause();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      if (selectedBeat === null)    selectBeat(0);
      else if (selectedBeat > 0)   selectBeat(selectedBeat - 1);
      break;
    case 'ArrowRight':
      e.preventDefault();
      if (selectedBeat === null)                  selectBeat(0);
      else if (selectedBeat < DATA.beats - 1)     selectBeat(selectedBeat + 1);
      break;
    case '[': adjustSpeed(-1); break;
    case ']': adjustSpeed(+1); break;
    case 'l': case 'L': toggleLoop(); break;
    case 't': case 'T':
      if (selectedBeat !== null) startInspect('target', selectedBeat);
      break;
    case 'c': case 'C':
      if (selectedBeat !== null) startInspect('corrected', selectedBeat);
      break;
    case 'r': case 'R':
      if (selectedBeat !== null && DATA.latency_ms !== 0) startInspect('raw', selectedBeat);
      break;
    case '1': setInspectWin(10);  break;
    case '2': setInspectWin(25);  break;
    case '3': setInspectWin(50);  break;
    case '4': setInspectWin(100); break;
  }
});

/* ── Initial kicks (canvas declared; all functions defined) ──────────────*/
setGain(1);
setSpeed(1.0);
setViewMode(viewMode);
requestAnimationFrame(() => { resizeCanvas(); });

} // end _init()
</script>
</body>
</html>
"""

# Sentinels — must not appear in JSON, base64, or numbers
_SENTINEL_DATA   = "INJECT_DATA_JSON"
_SENTINEL_AUDIO  = "INJECT_AUDIO_B64"
_SENTINEL_WINDOW = "INJECT_WINDOW_MS"


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(data: dict, audio_b64: str, window_ms: float) -> str:
    """Render the HTML template with session data, audio, and window size."""
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
