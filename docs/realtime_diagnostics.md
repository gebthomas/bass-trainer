# Timing-Feedback Diagnostic Architecture

This document covers the offline diagnostic pipeline: the modules that read a
saved `.session.json` file and produce timing visualisations and numeric
summaries.  For the live-session evaluation paths (pull/push) see
[`realtime_architecture.md`](realtime_architecture.md).

---

## Overview

A live practice session produces a `SessionLog` file.  The diagnostic pipeline
reads that file and re-evaluates timing performance with configurable windows,
independent of the tolerance that was used during the live session.

```
Live practice
─────────────
  audio  ─▶  OnsetAdapter / live_pipeline  ─▶  SessionEngine / feedback_events
                                                         │
                                                   save_session_log_file()
                                                         │
                                               example.session.json   ◀── archive

Offline diagnostics
───────────────────
  example.session.json
         │
         ▼
  scenario_from_session_file()       ← extracts [target_times_s, onset_times_s]
         │
         ▼
  evaluate_targets()                 ← greedy nearest-match (configurable tolerance)
         │
         ├─▶  build_figure() ─▶ save_figure()   →  example.timeline.png
         │
         └─▶  summarize_evaluations()
                    │
                    └─▶  format_summary_text()  →  example.summary.txt
```

---

## Module Reference

### `core/window_analyzer.py`

**Role:** Pure audio-window energy and onset analysis.

**Used by:** `core/live_pipeline.py` (pull path only).
**Not used by** the diagnostic scripts.

`evaluate_window(audio, sample_rate, ...)` returns a dict with:

| Key | Type | Meaning |
|-----|------|---------|
| `detected` | `bool` | `rms >= min_rms` |
| `rms` | `float` | Root-mean-square of the window |
| `peak` | `float` | Maximum absolute amplitude |
| `onset_found` | `bool` | Amplitude rose above dynamic threshold |
| `onset_sample` | `int\|None` | First sample crossing the threshold |
| `onset_time_s` | `float\|None` | `onset_sample / sample_rate` |

Onset detection uses a local baseline (first 15 ms of the window) and a
dynamic threshold of `max(onset_threshold, baseline × rise_ratio)`.  This
rejects sustained resonance without rejecting new attacks.

---

### `core/live_feedback.py`

**Role:** Target-readiness timing helper for the pull path.

**Used by:** `core/live_pipeline.py`.
**Not used by** the diagnostic scripts.

`target_state(targets, idx, bpm, ..., current_sample, evaluated)` returns
`"pending"`, `"ready"`, or `"evaluated"`.  `"ready"` fires when
`current_time >= window_end + margin_s`, i.e. the analysis window for a target
has closed.

`ready_targets(targets, bpm, ..., current_sample, evaluated_indices)` returns
the list of indices that are ready but not yet evaluated, in ascending order.

This module is narrowly scoped to *when* to trigger evaluation, not *what* the
evaluation result is.

---

### `core/live_pipeline.py`

**Role:** Pull-path orchestrator.  Ties together window extraction, energy
analysis, feedback scoring, and optional `TempoTracker` adaptation.

`process_realtime_audio(audio, current_sample, session, tempo_tracker=None)`
calls `session.update(current_sample)` to get newly-ready targets, extracts
each target's audio window via `core.audio_windows.extract_target_window()`,
runs `window_analyzer.evaluate_window()`, and wraps the result in a
`feedback_event` dict.

When a `TempoTracker` is provided the extraction window is shifted toward the
tracker's adjusted target time, and `tracker.observe(nominal, actual)` is
called for each matched onset.

**Timing error convention (pull path):** `timing_error_s = (abs_onset - target_sample) / sample_rate`.
Positive = late; negative = early.  This is relative to the *nominal* grid even
in adaptive mode so that the number reported to the player has stable musical
meaning.

---

### `core/realtime_evaluator.py`

**Role:** Offline/diagnostic target matching engine.  No audio, no hardware.

**Used by:** `scripts/plot_session_timeline.py` and
`scripts/session_diagnostic_report.py`.  **Not used** in the live path.

```python
from core.realtime_evaluator import TargetEvaluation, evaluate_targets

evals: list[TargetEvaluation] = evaluate_targets(
    target_times_s,       # list[float] — expected beat times
    onset_times_s,        # list[float] — detected onset times (any order)
    tolerance_s=0.08,     # ±80 ms acceptance window (inclusive)
    on_time_threshold_s=0.03,  # ±30 ms on-time band (inclusive)
)
```

Returns one frozen `TargetEvaluation` per target.  Key fields:

| Field | Type | Meaning |
|-------|------|---------|
| `target_index` | `int` | Position in `target_times_s` |
| `expected_time_s` | `float` | Nominal target time |
| `actual_time_s` | `float\|None` | Matched onset time; `None` for a miss |
| `signed_error_ms` | `float\|None` | `(actual − expected) × 1000`; `None` for a miss |
| `classification` | `str` | `"on_time"`, `"early"`, `"late"`, or `"miss"` |
| `matched_onset_index` | `int\|None` | Index into the original `onset_times_s` list |

**Note on tolerance:** The live session uses `match_window_s(bpm)` from
`core/timing_policy.py` (approximately half a beat, 250 ms at 120 BPM).  The
diagnostic default is a flat 80 ms.  These windows produce different
classifications for the same data.  This is intentional — see design decisions.

---

### `core/session_log.py`

**Role:** Versioned telemetry schema.  Pure data model; no audio, no hardware.

Key types:

```python
SessionEvent(
    time_sec:     float,
    event_type:   str,          # "target_hit" | "target_miss" | "extra_onset"
    target_index: int | None,
    value:        float | None, # for target_hit: signed timing error (onset − expected), seconds
    ...
)

SessionLog(
    schema_version: int,        # must be 1
    started_at:     str,        # ISO 8601
    events:         list[SessionEvent],
    metrics:        dict[str, float],
    metadata:       dict[str, str],  # "bpm", "device", "beats", etc.
)
```

Serialisation: `save_session_log_file(log, path)` / `load_session_log_file(path)`.

**Event encoding used by the diagnostic extractor:**

| Event type | `time_sec` | `value` |
|------------|------------|---------|
| `target_hit` | onset time (s) | `onset − expected` (signed error, s) |
| `target_miss` | expected target time (s) | `None` |
| `extra_onset` | onset time (s) | `None` |

The diagnostic extractor recovers `expected_time = time_sec − value` for hits,
and `expected_time = time_sec` for misses.

---

### `scripts/plot_session_timeline.py`

**Role:** Diagnostic visualisation.  Produces a PNG with one row per scenario,
each row containing a two-lane timeline, signed error bars, and a numeric
summary panel.

**Key public API:**

```python
from plot_session_timeline import (
    TimingScenario,             # NamedTuple: target_times_s, onset_times_s, title, desc
    EvaluationSummary,          # NamedTuple: n_targets, n_on_time, ..., mean_signed_error_ms
    scenario_from_session_log,  # SessionLog → TimingScenario
    scenario_from_session_file, # path → TimingScenario
    session_tolerance_s,        # SessionLog → float | None  (BPM-derived tolerance)
    build_figure,               # list[TimingScenario] → plt.Figure
    save_figure,                # (fig, path) → resolved Path
    summarize_evaluations,      # (evals, onset_times_s) → EvaluationSummary
)
```

`session_tolerance_s(log)` reads `log.metadata["bpm"]` and returns
`match_window_s(bpm)` when the value is a valid positive number; returns
`None` for a missing key, non-numeric string, zero, or negative BPM.

`build_figure()` calls `evaluate_targets()` internally.  Callers supply raw
arrays; they do not supply pre-computed evaluations.

**Backend:** `matplotlib.use("Agg")` is set unconditionally at module load.
The module never opens a display window.  All output is via `save_figure()`.

**Built-in demo scenarios:** `perfect`, `gradual_drift`, `mixed`, `misses`,
`double_hit` — accessed via `_ALL_SCENARIOS` dict.

---

### `scripts/session_diagnostic_report.py`

**Role:** One-stop report generator.  Given a session file, writes both output
files and prints their paths.

```python
from session_diagnostic_report import (
    derive_output_paths,    # (session_path, out_dir) → (png_path, txt_path)
    format_summary_text,    # (path, scenario, summary, tol_ms, on_time_ms, tolerance_source="") → str
    generate_report,        # (session_path, *, out_dir, tolerance_ms=None, on_time_ms) → (Path, Path)
)
```

`generate_report()` calls `load_session_log_file()` → `scenario_from_session_log()`
→ evaluates tolerance using the resolution order below → `evaluate_targets()`
→ `summarize_evaluations()` → `build_figure()` → `save_figure()` →
`format_summary_text()`.  It does not contain any evaluation logic of its own.

**Tolerance resolution in `generate_report()` (first valid source wins):**

| Priority | Condition | Effective tolerance | `tolerance_source` label |
|----------|-----------|--------------------|-----------------------|
| 1 | `tolerance_ms` explicitly given | that value | `"explicit"` |
| 2 | `metadata["match_window_s"]` valid and positive | that value | `"session metadata match_window_s"` |
| 3 | `metadata["bpm"]` valid and positive | `match_window_s(bpm)` | `"session BPM (BPM)"` |
| 4 | none of the above | 80 ms flat | `"fallback default"` |

`session_tolerance_s(log)` in `plot_session_timeline.py` implements the same
priority (tiers 2–3) for use by the CLI `main()` in session-file mode.

`format_summary_text()` includes a `Tolerance source:` line in the report when
`tolerance_source` is non-empty.

---

## Data Flow Detail

### 1. Extraction (`scenario_from_session_file`)

```
.session.json
  │
  └▶ load_session_log_file()
        │
        └▶ scenario_from_session_log(log)
              │
              ├─ for each target_hit:
              │     expected = event.time_sec − event.value
              │     onset    = event.time_sec
              │
              ├─ for each target_miss:
              │     expected = event.time_sec
              │     (no onset)
              │
              └─ for each extra_onset:
                    onset = event.time_sec
                    (no target)
              │
              └▶ TimingScenario(
                   target_times_s = sorted by target_index,
                   onset_times_s  = sorted ascending,
                   title          = "Session YYYY-MM-DD",
                   description    = "N targets · BPM · device",
                 )
```

### 2. Evaluation (`evaluate_targets`)

Onsets are sorted by time internally.  Targets are processed left-to-right.
For each target, all unconsumed onsets within `±tolerance_s` are gathered; the
nearest one (smallest `|onset − target|`) is selected and marked consumed.
Ties break by earlier onset time.  Unmatched targets become `"miss"`.
Unmatched onsets are silently ignored (they appear as grey circles in the
timeline plot; their count is reported in the summary).

### 3. Summary (`summarize_evaluations`)

Counts per classification and error statistics over matched (non-miss) targets
only.  `n_unmatched_onsets = len(onset_times_s) − len({consumed indices})`.
Error fields are `None` when all targets are misses.

### 4. Output

`build_figure()` produces an `n×3` subplot grid (timeline | error bars |
summary text) for `n` scenarios.  `save_figure(fig, path)` creates missing
parent directories, writes the PNG at `dpi=150`, and returns the resolved
absolute path.

`format_summary_text()` returns a fixed-width plain-text string.  The output
is deterministic: same inputs always produce identical text.

---

## Design Decisions

### Pure seconds-based evaluation

`evaluate_targets()` works entirely in seconds.  It has no concept of BPM,
beat positions, measures, or subdivisions.  Callers are responsible for
converting beat-grid targets to absolute times before calling it.

This keeps the evaluator simple and fully reusable: it works equally for
metronome-exercise targets, recording-aligned targets, or any other
seconds-based sequence.

### Greedy target-ordered nearest-match

Targets are processed in time order.  The nearest available onset wins.
Earlier targets have priority over later targets when both are in range of the
same onset.  Each onset can be consumed at most once.

This policy is deterministic and O(T × N) where T = number of targets and
N = number of onsets.  It is identical in spirit to the policy used in
`core/session_runner.py` (offline) and `core/session_engine.py` (live push
path).

### Frozen `TargetEvaluation` records

`TargetEvaluation` is a `@dataclass(frozen=True)`.  Results are immutable value
objects.  This makes it safe to pass them through multiple layers (plotting,
summarisation, formatting) without defensive copying.

### Persisted match window and tolerance defaulting

**Where `match_window_s` is written**

Two session-creation sites now store the exact match window used for event
classification as `SessionLog.metadata["match_window_s"]` (a string-encoded
float in seconds):

| Site | Source of the value |
|------|---------------------|
| `core/session_runner.run_session_bundle()` | `_default_match_window(bundle)` — `match_window_s(exercise.bpm)` or `match_window_s(estimated_bpm(alignment))` |
| `scripts/live_feedback_demo.py` | `match_window_s(bpm)` using the BPM passed to the session |

This means every newly created log carries the window it was evaluated with,
so future diagnostics can reproduce the original policy exactly without relying
on a `match_window_s()` re-computation that might return a different value if
`timing_policy.py` ever changes.

**Diagnostic tolerance resolution order**

When `--tolerance-ms` is not passed explicitly, `session_tolerance_s(log)` in
`plot_session_timeline.py` and the inline resolution in `generate_report()` both
use the following priority (first valid value wins):

1. `metadata["match_window_s"]` — exact window persisted at creation time
2. `match_window_s(metadata["bpm"])` — BPM-derived fallback for older logs that
   pre-date the `match_window_s` metadata key
3. `None` / 80 ms flat fallback (when neither key is present or valid)

An explicit `--tolerance-ms` override always wins over all three tiers.

`generate_report()` labels each source in the `Tolerance source:` line of the
plain-text summary so it is always clear which policy was applied.

Previously the diagnostic default was a flat 80 ms regardless of session tempo,
a 3× gap vs. the live engine's ≈ 250 ms at 120 BPM that caused live hits to
appear as diagnostic misses.

### Agg/save-only plotting

`matplotlib.use("Agg")` is called at module load in `plot_session_timeline.py`
before any `pyplot` import.  This locks the backend to the non-interactive
rasteriser.  There is no interactive display mode.  All output requires an
explicit `save_figure()` call or `--output` / `--out-dir` flag.

---

## Known Limitations

**Greedy matching can be suboptimal in dense passages.**  If two targets are
close together and a single onset falls between them, the first target claims
the onset and the second becomes a miss, even if the onset was metrically
closer to the second target.  Global optimal assignment (e.g. minimum-cost
matching) would handle this but is not implemented.

**No pitch or tone integration.**  The diagnostic pipeline operates on timing
alone.  Pitch error, note identity, and dynamic level are not captured in the
diagnostic output, even though `SessionEvent` has a `value` field that could
carry pitch data and `TargetEvaluation` could be extended with pitch fields.

**Diagnostic tolerance axis differs from live-session axis.**  The live session
is onset-first (each onset claims the nearest unclaimed target within the window)
while `evaluate_targets()` is target-first (each target claims the nearest
unconsumed onset).  For typical sessions the outcomes are identical; they diverge
only when a single onset falls equidistant between two close targets.  The
default tolerance now matches the live engine's `match_window_s(bpm)`, so the
*tolerance window* no longer causes false misses, but the axis difference remains
a minor theoretical discrepancy.

**Save-only; no interactive plot.**  `plot_session_timeline.py` cannot display
an interactive window.  Switching to an interactive backend would require
removing `matplotlib.use("Agg")` and calling `plt.show()`, which would break
the save-only path.

---

## Common Commands

```bash
# Demo mode: plot all built-in scenarios
python scripts/plot_session_timeline.py

# Demo mode: specific scenarios only
python scripts/plot_session_timeline.py --scenario perfect gradual_drift

# Demo mode: custom output path
python scripts/plot_session_timeline.py --output figs/demo.png

# Session file: default output path (same directory as session file)
python scripts/plot_session_timeline.py diagnostics/example.session.json

# Session file: custom output path
python scripts/plot_session_timeline.py diagnostics/example.session.json \
    --output diagnostics/example.timeline.png

# Session file: widen tolerance to match live session at 120 BPM
python scripts/plot_session_timeline.py diagnostics/example.session.json \
    --tolerance-ms 250

# Session file: tighten on-time window to ±20 ms
python scripts/plot_session_timeline.py diagnostics/example.session.json \
    --on-time-ms 20

# Full diagnostic report: PNG + plain-text summary, next to session file
python scripts/session_diagnostic_report.py diagnostics/example.session.json

# Full diagnostic report: custom output directory
python scripts/session_diagnostic_report.py diagnostics/example.session.json \
    --out-dir reports/

# Full diagnostic report: match original live tolerance and tighten on-time
python scripts/session_diagnostic_report.py diagnostics/example.session.json \
    --tolerance-ms 250 --on-time-ms 20 --out-dir reports/
```

Default output paths when `--output` / `--out-dir` are omitted:

| Script | Default output |
|--------|---------------|
| `plot_session_timeline.py` | `./diagnostic_timeline.png` |
| `session_diagnostic_report.py` | `<session_dir>/<stem>.timeline.png` and `<session_dir>/<stem>.summary.txt` |

---

## Extension Points

The following extensions are explicitly called out in `docs/future_directions.md`
or already partially scaffolded in the codebase.

**Tempo curve overlay.**  `TempoTracker` already tracks `tempo_ratio` and
`phase_offset` per observation.  A `plot_tempo_curve(ax, tracker_history)`
function could occupy a fourth subplot column in `build_figure()`, showing how
the player's tempo drifted over the session.

**Rolling timing error.**  Compute a windowed mean of `signed_error_ms` across
consecutive targets (e.g. a 4-target rolling window) and plot it as a line
overlay on the error-bar panel.  This would reveal systematic drift that is
invisible in per-target bars.

**Pitch evaluation.**  `SessionEvent.value` currently holds the timing error.
A richer schema (e.g. a separate `pitch_cents` field, or a second event type)
could extend `scenario_from_session_log` to also extract pitch errors.
`TargetEvaluation` would need a `pitch_error_cents` field.  The `feedback_event`
dict already defines `pitch_error_cents`; it is `None` today because
`OnsetAdapter` does not do pitch analysis.

**Groove-aware or adaptive tolerance.**  Replace the flat `tolerance_s` in
`evaluate_targets()` with a per-target tolerance derived from the local beat
density or time signature.  The function signature already accepts a scalar;
it could accept a `Sequence[float]` of per-target windows without changing
the matching algorithm.

**Global-assignment matching.**  Replace the greedy left-to-right policy in
`evaluate_targets()` with a minimum-cost assignment (e.g. the Hungarian
algorithm on an N×M cost matrix).  This would handle dense passages more
accurately.  The rest of the pipeline — summarisation, plotting, reporting —
is unchanged because it only sees the `list[TargetEvaluation]` output.
