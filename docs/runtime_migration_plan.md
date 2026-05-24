# Runtime Migration Plan

This document describes how the current live/runtime system should migrate
toward the canonical `SessionLog` architecture without breaking the working
live audio path.

No code changes are described here.  This is the design record that precedes
implementation.

---

## 1. Current Split

### Old live/runtime tier

| Module | Role | Key output |
|---|---|---|
| `matching.py` (`TargetMatcher`) | Candidate-accumulation-based matching; pitch-aware; finalises targets on timeout | `ResultsLogger.append_hit/miss/extra()` + `recently_finalized` dicts for live display |
| `results.py` (`ResultsLogger`) | Mutable row accumulator; saves per-event rows to CSV (with numpy means) | `session_*.csv`, `print_summary()` |
| `feedback_events.py` | Pure functions: `feedback_event()` produces a severity-annotated dict per target; `summarize_feedback()` aggregates | Runtime display dicts with `severity`, `messages`, `timing_error_s`, `pitch_error_cents` |
| `live_pipeline.py` (`process_realtime_audio`) | Buffer-pull DSP path; extracts audio windows, runs `evaluate_window`, calls `feedback_event()` | Rich event dicts with adaptive timing fields |
| `session_engine.py` (`SessionEngine`) | Onset-push path; greedy nearest-target matching; calls `feedback_event()` | Feedback event dicts; already uses `timing_policy.match_window_s()` |
| `session_replay.py` | Drives `SessionEngine` offline with captured onset streams for regression testing | Feedback event dicts + stats |
| `practice_log.py` | Reads from `ResultsLogger.results`; appends one aggregate row per session to a running CSV | Longitudinal CSV log (one row per session) |

### New canonical tier

| Module | Role |
|---|---|
| `core/exercise.py` | Target definitions in beats |
| `core/alignment.py` | Beat-to-audio-time mapping |
| `core/practice_mode.py` | Session configuration record |
| `core/session_bundle.py` | Resolved in-memory union of all session assets |
| `core/session_log.py` (`SessionLog`, `SessionEvent`) | Canonical persistent event representation |
| `core/session_runner.py` | Deterministic offline session execution → `SessionLog` |
| `core/severity_policy.py` | Derived severity classification from `SessionEvent.value` |
| `core/log_metrics.py` (`LogMetrics`) | Structured metrics computed from `SessionLog` |
| `core/session_store.py` | Managed persistence with protected-log retention |
| `core/timing_policy.py` | Canonical match-window policy: `clamp(30/bpm, 0.10, 0.35)` |

---

## 2. Migration Principles

**Do not break the working live audio path.**  The old tier is the only path
that currently handles real audio hardware, onset detection, DSP evaluation,
and pitch analysis.  No migration step should touch that path until an
equivalent new path is tested and verified.

**`SessionEvent` is canonical for persistence.**  Once a session ends, its
events should be represented as `SessionLog` / `SessionEvent` records, not as
`ResultsLogger` rows or feedback event dicts.  The live dict format is a
transient computation artifact, not a storage format.

**`feedback_event()` dicts remain temporary runtime presentation events.**
They carry severity, human-readable messages, and live display fields that have
no home in `SessionEvent` yet (pitch details, adaptive timing debug fields).
Do not replace them during live feedback.  Replace them at the point where data
is persisted, not where it is displayed.

**Prefer adapters over rewrites.**  The migration path with the lowest risk is
to add an adapter layer that converts old-tier output to new-tier structures at
session end.  Rewrites of live matching logic come later, after the adapter
path is verified.

**Do not let timing policies diverge further.**  Three match-window formulas
currently coexist (see Risk §6.3).  The migration should freeze the old-tier
formula where it is and not add new call sites for it.  All new-tier code
must use `timing_policy.match_window_s()`.

---

## 3. Proposed Migration Sequence

### Phase 1 — Session capture (add; don't remove)

Convert the output of a completed session into a `SessionLog` at session-end
time, using an adapter function.  Save it via `session_store.save_session_log()`.

Inputs available at session end:
- `ResultsLogger.results` — list of row dicts with `event_type`, `target_time`,
  `played_time`, `corrected_timing_error_ms`, `pitch_ok`, etc.
- Session metadata: `started_at`, `ended_at`, `exercise_path`, `alignment_path`

The adapter maps each row to a `SessionEvent` (see §4 mapping table).
`TARGET_HIT` events get `value = corrected_timing_error_ms / 1000.0`.
`TARGET_MISS` and `EXTRA_ONSET` events get `value = None`.

The `ResultsLogger.save_csv()` path continues to run in parallel during this
phase.  Both outputs coexist.  This gives a verification window: compare
`LogMetrics` against `ResultsLogger.print_summary()` for the same session.

### Phase 2 — Metrics from SessionLog

Replace `ResultsLogger.print_summary()` output with `compute_log_metrics()` on
the newly written `SessionLog`.  During this phase both numbers are produced
and can be compared.  Fix discrepancies in the adapter before proceeding.

Also replace `practice_log.append_practice_log()` with the aggregate metrics
written into `SessionLog.metrics` (already stored at session end by the
adapter).  The running CSV log becomes redundant once this is verified.

### Phase 3 — Stop writing CSV outputs

Once the `SessionLog` path is verified to produce correct numbers:
- Stop calling `ResultsLogger.save_csv()`.
- Stop calling `practice_log.append_practice_log()`.
- `session_store.save_session_log()` becomes the sole persistence path.

`ResultsLogger` continues to exist in memory during a live session for the
`recently_finalized` dicts consumed by the live display.

### Phase 4 — Route SessionEngine toward SessionEvent directly

`SessionEngine.on_onset()` and `update_time()` currently produce feedback event
dicts.  In this phase, add a mode where they emit `SessionEvent` records
directly into a `SessionLog` (via `append_event()`), bypassing the adapter
layer.  `feedback_event()` continues to be called for the live display path in
parallel.

This removes the round-trip: old dict → adapter → `SessionEvent`.  It also
eliminates the risk of information loss in the adapter.

### Phase 5 — Migrate live display to derived severity

`feedback_event()` currently computes severity inline.  Once the direct
`SessionEvent` path is active, switch the live display to read severity from
`event_timing_severity()` rather than from the embedded `severity` field in
the feedback dict.  This removes the last place severity is computed outside
`severity_policy.py`.

### Phase 6 — Retire old-tier pieces

See §7 (exit criteria) for the specific conditions under which each module can
be safely deleted.

---

## 4. Concept Mapping

| Old concept | New concept | Notes |
|---|---|---|
| `target_idx` (int, 0-based) | `SessionEvent.target_index` | Identical semantics |
| `target["time"]` (beat position, float) | `Target.time` in `Exercise` | Same unit; old uses raw dict, new uses dataclass |
| `corrected_timing_error_ms` | `SessionEvent.value` × 1000 | Old is ms; new is seconds. Divide by 1000 in adapter |
| `raw_timing_error_ms` | Not directly represented | Carry in `SessionEvent.metadata["raw_timing_error_ms"]` if needed |
| `feedback_event()["severity"]` | `event_timing_severity(event)` from `severity_policy` | Numerically identical thresholds today (see §6.2) |
| `event_type = "hit"` | `TARGET_HIT = "target_hit"` | String rename |
| `event_type = "missed"` | `TARGET_MISS = "target_miss"` | String rename |
| `event_type = "extra"` | `EXTRA_ONSET = "extra_onset"` | String rename |
| `ResultsLogger` | `SessionLog` + `log_metrics.compute_log_metrics()` | `ResultsLogger` is stateful; `SessionLog` is a dataclass |
| `ResultsLogger.save_csv()` | `session_store.save_session_log()` | CSV is lossy (aggregate only); JSON preserves per-event data |
| `practice_log.append_practice_log()` | `SessionLog.metrics` + `session_store` | Running CSV → per-session JSON with aggregate metrics embedded |
| `summarize_feedback(events)` | `compute_log_metrics(log)` | Both produce aggregate counts and means |
| `session_replay.replay_session_data()` | `session_runner.run_session_bundle()` | Old: dict-based; new: typed `SessionBundle` |
| `get_match_window()` in `matching.py` | `timing_policy.match_window_s()` | Different formulas; see §6.3 |
| `SessionEngine.match_window_s` | `timing_policy.match_window_s()` | `SessionEngine` already uses canonical policy |
| `pitch_ok`, `cents_error` | Not yet in `SessionEvent` schema | Defer; carry in `SessionEvent.metadata` as strings during migration |

---

## 5. Explicit Non-Goals

The following are not part of this migration plan and must not be introduced
as side effects:

- **No broad rewrite of matching logic.**  `TargetMatcher` and `ResultsLogger`
  are not being replaced in this document.  They continue to operate during
  the transition.

- **No deletion of `matching.py`, `results.py`, `live_pipeline.py`, or
  `feedback_events.py`** during Phases 1–4.  Deletion is gated on verified
  equivalent coverage in the new tier.

- **No pitch schema expansion.**  The question of how to represent
  `pitch_ok`, `cents_error`, and `pitch_stability_cents` in `SessionEvent` is
  open.  Typed fields versus metadata is a separate decision.  During migration,
  pitch data may be carried in `SessionEvent.metadata` as strings if needed,
  but no new typed fields are added until the pitch schema question is resolved.

- **No playback transport or DSP work.**  This migration covers the data
  plumbing between live matching and persistent telemetry.  The audio stack,
  onset detection, and playback transport remain outside scope.

- **No severity storage in `SessionEvent`.**  Severity stays derived.  Nothing
  in this migration should add a `severity` field to `SessionEvent`.

---

## 6. Risks

### 6.1 Double event systems

During Phases 1–2, the same session is written in both formats.  If the adapter
has a bug, the two outputs diverge silently.  The mitigation is to run both
paths in parallel and assert agreement in `compute_log_metrics()` versus
`ResultsLogger.print_summary()` before proceeding to Phase 3.  Do not retire
the CSV path until this comparison has been run on real sessions.

### 6.2 Timing threshold alignment

`feedback_events.py` embeds `_TIMING_GOOD = 0.05` and `_TIMING_WARN = 0.12`
directly in the module.  These happen to match `severity_policy`'s
`DEFAULT_GOOD_THRESHOLD_S` and `DEFAULT_WARN_THRESHOLD_S` exactly today.  This
alignment is **incidental and unprotected** — there is no shared constant,
no test, and no import coupling between the two files.  If either is changed
independently, severity labels diverge between the live display path and the
post-session analytics path without any warning.

Mitigation: before Phase 5, add a test that asserts the two sets of thresholds
are equal.  During Phase 5 (live display migrated to `severity_policy`), remove
the local constants from `feedback_events.py` entirely.

### 6.3 Match-window formula drift

Three distinct match-window formulas currently coexist:

| Site | Formula | Range |
|---|---|---|
| `matching.py` (`get_match_window`) | `min(prev_gap, next_gap) × 0.40` | [0.12, 0.40] s |
| `session_engine.py` | `timing_policy.match_window_s()` = `clamp(30/bpm, 0.10, 0.35)` | [0.10, 0.35] s |
| `session_runner.py` | same as `session_engine.py` | [0.10, 0.35] s |

A session evaluated live by `TargetMatcher` may produce different hit/miss
outcomes than the same onset stream evaluated offline by `session_runner`.
The numbers in a `SessionLog` produced via the adapter (Phase 1) will reflect
the `TargetMatcher` decisions; the numbers in a `SessionLog` produced by
`session_runner` directly will reflect the canonical policy.

This divergence should be documented in any adapter code.  Do not silently
normalise it; surface it so it can be resolved intentionally when `TargetMatcher`
is eventually migrated or retired.

### 6.4 Adaptive offset not represented

`TargetMatcher` accepts a `timing_offset_ms` correction that shifts onset times
before matching.  `ResultsLogger` records both `raw_timing_error_ms` and
`corrected_timing_error_ms`.  The new tier has no equivalent correction field
in `SessionEvent`.  An adapter that writes `value = corrected_timing_error_ms / 1000`
discards the offset provenance; an adapter that writes `value = raw_timing_error_ms / 1000`
discards the correction.

During Phase 1, write `corrected_timing_error_ms / 1000` as `value` (this is
what the player actually experienced) and carry `raw_timing_error_ms` and
`adaptive_offset_ms` in `SessionEvent.metadata` as strings for auditability.
This is a temporary decision that the pitch-schema discussion can revisit.

### 6.5 Loss of pitch and constraint data

`ResultsLogger` carries per-hit pitch fields (`detected_freq_hz`,
`cents_error`, `pitch_stability_cents`, `pitch_match_ratio`, `voiced_ratio`)
and constraint classification fields that have no typed home in `SessionEvent`.
A Phase 1 adapter that ignores these fields produces a `SessionLog` that is
less rich than the CSV for pitch-intensive workflows.

The safe path is: carry any field that matters in `SessionEvent.metadata` as
strings, at the cost of losing type safety.  Reject any adapter design that
silently drops data without documenting where it went.

---

## 7. Exit Criteria for Retiring Old-Tier Pieces

Each module may be retired only when all of the following conditions for it
are satisfied:

### `practice_log.py`

- `session_store.save_session_log()` is called for every session that currently
  calls `append_practice_log()`.
- The aggregate metrics in `SessionLog.metrics` cover `n_hits`, `n_misses`,
  `n_extras`, `mean_timing_error_ms`, `mean_abs_timing_error_ms`, and any
  pitch-accuracy fields that were being written to the CSV.
- A transition note in the commit message records the date of last CSV write
  and where the historical CSVs are kept.

### `results.py` (`ResultsLogger`)

- `SessionLog` + `compute_log_metrics()` produces equivalent counts and means
  for at least 10 real sessions, verified by comparison.
- The live display path no longer relies on `recently_finalized` dicts from
  `TargetMatcher`, OR those dicts are produced independently of `ResultsLogger`.
- No production code path calls `ResultsLogger.save_csv()`.
- Pitch and constraint data has a documented home (typed fields or metadata)
  in `SessionEvent`.

### `feedback_events.py`

- `event_timing_severity()` from `severity_policy` is used everywhere severity
  is computed, including the live display path.
- `summarize_feedback()` is unused (replaced by `compute_log_metrics()`).
- The timing thresholds in `feedback_events.py` have been removed (no longer
  the authority).

### `matching.py` (`TargetMatcher`)

- No production or near-production path calls `TargetMatcher.process_onset_against_targets()`.
- The `get_match_window()` local formula has been replaced by `timing_policy.match_window_s()`
  in every active call site.
- The behavioral difference between the two window policies has been explicitly
  accepted or corrected (not silently ignored).

### `live_pipeline.py`

- DSP evaluation output is written directly into `SessionEvent` records.
- The buffer-extraction and onset-detection logic has been moved or superseded.
- No production path calls `process_realtime_audio()`.

### `session_replay.py`

- `session_runner.run_session_bundle()` with typed `SessionBundle` fixtures
  covers all regression test cases currently using `replay_session_data()`.
- The JSON fixture format used by `session_replay` has been either migrated to
  `SessionBundle`-compatible fixtures or archived.
