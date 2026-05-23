# Session Architecture

This document describes the resolved-session layer: the set of data models that
together represent a fully configured practice session, how they relate to each
other, and what each layer is responsible for.

---

## Layers at a Glance

```
Exercise           what to play, at what tempo
BeatAlignment      when each beat falls in an audio recording
PracticeMode       which mode, which files to load
SessionLog         what happened during a session
SessionBundle      resolved union of all of the above
```

Each layer is a pure data model with no audio hardware, no playback, and no UI
dependencies.

---

## Exercise

**File:** `core/exercise.py`

**Owns:**
- The musical target definition: what notes to play and at which beat positions.
- Nominal tempo (`bpm`) and count-in length (`count_in_beats`).
- Per-target fields: `time` (beat position), `label`, `expected_pitch`,
  `duration_beats`, `metadata`.
- Versioned JSON serialisation.

**Does not own:**
- Any audio recording or recording timeline.
- Absolute audio timestamps — Exercise targets are in beats, not seconds.
- Playback, metronome clicks, or any hardware interaction.
- Session history or telemetry.

**Beat positions** are zero-based, relative to the end of the count-in.  A
target with `time=0` is the first beat after the count-in; `time=1` is the
second beat, and so on.  Fractional times (e.g. `1.5`) represent subdivisions.

---

## BeatAlignment

**File:** `core/alignment.py`

**Owns:**
- The mapping from musical beat indices to absolute audio time in seconds.
- Two anchor points (`first_beat_time_sec`, `last_beat_time_sec`) and a beat
  count, from which a uniform grid is derived.
- Time-signature bookkeeping (`beats_per_bar`) for display and bar counting.
- Alignment provenance: `alignment_method` (`manual_tap`, `player_onsets`,
  `imported`), `confirmed_by_user`, `confidence`.
- Versioned JSON serialisation.

**Does not own:**
- The audio file itself — only a path reference (`audio_file`).
- Any exercise or target note definitions.
- Playback or transport control.

### Current limitation — constant tempo

The current model assumes **constant tempo** throughout the recording.  A
single uniform beat grid is interpolated between `first_beat_time_sec` and
`last_beat_time_sec`.  All inter-beat intervals are equal.

This is a valid approximation for recordings made to a click track or drum
machine, and for live performances with small, consistent drift.

It does not handle intentional tempo changes (rubato, ritardando, accelerando),
section-level BPM shifts, or human performances with significant beat-to-beat
variation.

### Future path — variable tempo via anchor points

When constant tempo is insufficient, the plan is to replace the single
`(first, last)` anchor pair with a list of timed anchor points:

```json
[
  {"beat_index": 0,  "audio_time_sec": 2.00},
  {"beat_index": 8,  "audio_time_sec": 6.10},
  {"beat_index": 16, "audio_time_sec": 9.80}
]
```

`beat_index_to_audio_time()` would locate the surrounding segment and
interpolate linearly (or with a spline) within it.  The `schema_version` field
already exists so that a future multi-anchor format can be detected and
deserialised differently from the current single-grid format.  This extension
is intentionally deferred.

---

## PracticeMode

**File:** `core/practice_mode.py`

**Owns:**
- The session mode declaration: `metronome_exercise`,
  `recording_aligned_exercise`, or `play_to_align`.
- File path references to the exercise (`exercise_path`), alignment
  (`alignment_path`), and audio file (`audio_file`).
- A human-readable `description` and free-form `metadata`.
- Versioned JSON serialisation.
- Mode-specific asset requirement rules (which paths are required vs. optional).

**Does not own:**
- The loaded Exercise or BeatAlignment objects — those are resolved by
  `SessionBundle`.
- Any session history or telemetry.
- Playback or transport logic.

PracticeMode is a configuration record, not an executor.  It records intent;
`SessionBundle` resolves that intent into loaded objects.

---

## SessionLog

**File:** `core/session_log.py`

**Owns:**
- A timestamped list of `SessionEvent` records accumulated during a session.
- Per-event fields: `time_sec`, `event_type` (e.g. `"hit"`, `"miss"`,
  `"extra"`), optional `target_index`, `message`, `value`, `metadata`.
- Aggregate `metrics` computed at session end (e.g. `hit_rate`,
  `mean_timing_error_ms`).
- Session-level timestamps (`started_at`, `ended_at`) and optional back-
  references to the asset files used (`practice_mode_path`, `exercise_path`,
  `alignment_path`).
- Versioned JSON serialisation.

**Does not own:**
- Any live audio or playback state.
- Target definitions — those come from Exercise.
- The logic that produces events — that comes from the session engine or live
  pipeline.

A `SessionLog` can be built incrementally during a live session via
`append_event()`, or loaded from a previously saved file for replay and
analytics.  It is deliberately separate from the asset layer so that the same
Exercise can be logged across many sessions without coupling the log format to
the exercise schema.

---

## SessionBundle

**File:** `core/session_bundle.py`

**Owns:**
- The resolved, in-memory union of all session assets:
  `practice_mode`, `exercise`, `alignment`, `session_log`.
- Mode-appropriate validation rules (which assets must be present or absent).
- The `load_session_bundle()` entry point that reads a PracticeMode file,
  resolves and loads the referenced assets, and returns a validated bundle.
- The `bundle_target_audio_times()` helper that derives absolute audio times
  for each exercise target.

**Does not own:**
- File serialisation of its own state — the bundle is a transient in-memory
  object; its constituent parts each serialise independently.
- Transport, playback, or live audio.
- Session event collection — the `session_log` field is populated externally
  after or during a session; `load_session_bundle()` never creates one
  automatically.

### Path resolution rule

`load_session_bundle()` resolves relative paths in the PracticeMode relative
to the **directory containing the PracticeMode file**.  Absolute paths are
used as-is.

```
practice_modes/my_session.json   ← practice mode file
  exercise_path: "../exercises/walking_bass.json"
                                 ← resolved to:
  practice_modes/../exercises/walking_bass.json
  = exercises/walking_bass.json  ✓
```

This means a practice mode file and its referenced assets can be moved
together as a self-contained directory tree without breaking path references,
as long as the relative layout is preserved.

---

## How Target Times Are Resolved

### `metronome_exercise`

No recording is involved.  Target times are derived purely from the exercise:

```
beat_s       = 60.0 / exercise.bpm
count_in_s   = exercise.count_in_beats × beat_s
target_time  = count_in_s + target.time × beat_s
```

Example: `bpm=120`, `count_in_beats=2`, targets at beats `[0, 1, 2, 3]`:

```
beat_s     = 0.5 s
count_in_s = 1.0 s
times      = [1.0, 1.5, 2.0, 2.5] s
```

The exercise `bpm` is the sole authority on tempo.  There is no recording
timeline.

### `recording_aligned_exercise`

The exercise beat positions are mapped onto the recording's audio timeline
using the alignment grid:

```
period      = (last_beat_time_sec − first_beat_time_sec) / (beat_count − 1)
target_time = first_beat_time_sec + target.time × period
```

Example: alignment with `first=2.0 s`, `last=5.0 s`, `beat_count=4`
(period = 1.0 s), targets at beats `[0, 1, 2, 3]`:

```
times = [2.0, 3.0, 4.0, 5.0] s
```

In this mode the exercise `bpm` field is **not used** for target placement.
The recording's measured beat grid governs timing.  The exercise `bpm` remains
useful as a nominal reference for display, but the alignment is authoritative.

### `play_to_align`

`bundle_target_audio_times()` returns an empty list.  No exercise or alignment
exists yet for this session.  The intent of `play_to_align` is for the player
to perform against the recording so that an alignment can be generated from
their onsets.  Until that alignment is produced, there are no known beat times
to evaluate against.

---

## Future Layers

### Transport / playback layer

The current architecture stops at resolved target times.  A future transport
layer will sit above `SessionBundle` and own:

- The session clock: a monotonically advancing time source (system clock or
  audio callback counter).
- Count-in gating: suppressing evaluation until the count-in completes.
- Looping and section control: repeating a range of bars or targets.
- Audio file playback scheduling: when to start the backing track relative to
  the session clock.

The transport layer will consume a `SessionBundle` but will not modify it.

### Live audio integration

Above the transport layer, a live audio integration layer will own:

- The audio hardware callback and sample-rate negotiation.
- Onset detection and pitch analysis, reading from the audio buffer.
- Routing detected onsets and pitch results into `SessionEngine` or the live
  pipeline evaluator.
- Writing `SessionEvent` records into a `SessionLog` as events are produced.

This layer is intentionally kept separate from the asset and transport layers
so that:

- The data models remain testable without hardware.
- The transport clock can be driven by a simulated time source in tests.
- The onset detector can be swapped (RMS/peak → spectral flux → learned model)
  without changing the session or bundle API.

### Full resolved-session picture

```
Exercise
  + BeatAlignment          (recording-aligned mode only)
  + PracticeMode           (mode declaration and asset references)
  = SessionBundle          (resolved, validated, in-memory)
      ↓
  Transport layer          (session clock, count-in, looping)
      ↓
  Live audio layer         (onset detection, pitch, hardware I/O)
      ↓
  SessionLog               (events accumulated during the session)
      ↓
  Metrics / analytics      (post-session summary and progress history)
```

Each boundary in this stack is a clean interface: the layer above drives the
layer below through a defined API, and neither layer reaches across the
boundary to touch the other's internals.
