# Practice Modes

This document describes the three-layer model used to configure practice sessions:
**Exercise**, **BeatAlignment**, and **PracticeMode**.

---

## Layer 1 — Exercise

An `Exercise` is the musical target definition.  It describes *what* to play:
the tempo, the count-in, and a list of target notes with their beat positions.

```json
{
  "schema_version": 1,
  "name": "Basic Four Beats",
  "bpm": 120.0,
  "count_in_beats": 2,
  "description": "Four quarter-note targets. Play one note on each beat.",
  "tags": ["timing", "beginner", "quarter-notes"],
  "targets": [
    {"time": 0},
    {"time": 1},
    {"time": 2},
    {"time": 3}
  ]
}
```

Target `time` values are beat positions relative to the start of the exercise
(after the count-in).  Optional fields on each target include `label`,
`expected_pitch`, `duration_beats`, and `metadata`.

The `Exercise` model is independent of any recording or playback device.  Its
`bpm` field is the *nominal* tempo used by the synthetic metronome clock.  In
recording-aligned sessions the alignment governs timing instead, and the
exercise `bpm` field is not used for target placement.

---

## Layer 2 — BeatAlignment

A `BeatAlignment` maps a recording's beat indices to absolute audio times in
seconds.  It answers the question: *at what point in this audio file does beat
N occur?*

```json
{
  "schema_version": 1,
  "audio_file": "tracks/basic_four_beats.wav",
  "alignment_method": "manual_tap",
  "first_beat_time_sec": 2.0,
  "last_beat_time_sec": 5.0,
  "beat_count": 4,
  "beats_per_bar": 4,
  "confirmed_by_user": true,
  "confidence": "high"
}
```

Given this alignment the beat grid is:

| Beat index | Audio time |
|:---:|:---:|
| 0 | 2.0 s |
| 1 | 3.0 s |
| 2 | 4.0 s |
| 3 | 5.0 s |

### Alignment methods

| Value | Meaning |
|---|---|
| `"manual_tap"` | User tapped beats in real time against the recording |
| `"player_onsets"` | Derived from detected note onsets in a practice pass |
| `"imported"` | Loaded from an external source (DAW, annotation file, etc.) |

### Time signatures

`beats_per_bar` is stored for display and bar-counting purposes.  It does **not**
affect beat timing: `beat_time()` and `estimated_bpm()` are independent of
`beats_per_bar`.  3/4, 4/4, and 5/4 are all valid; the alignment grid is
identical as long as the anchor times and beat count are the same.

### Current limitation — constant tempo

The current model assumes **constant tempo** throughout the recording.  A
single uniform grid is interpolated between `first_beat_time_sec` and
`last_beat_time_sec`.  All inter-beat intervals are equal.

This works well for:
- Recordings made to a click track or drum machine.
- Performances with small, consistent tempo drift that the uniform grid
  absorbs without meaningful error.

It does **not** handle:
- Intentional tempo changes (rubato, ritardando, accelerando).
- Section-level BPM shifts.
- Human performances with significant beat-to-beat drift.

### Future path — variable tempo via anchor points

When constant tempo is insufficient, the plan is to replace the single
`(first, last)` anchor pair with a list of timed anchor points:

```
[
  {"beat_index": 0,  "audio_time_sec": 2.00},
  {"beat_index": 8,  "audio_time_sec": 6.10},
  {"beat_index": 16, "audio_time_sec": 9.80}
]
```

`beat_time()` would locate the surrounding segment and interpolate linearly
(or with a spline) within it.  The `schema_version` field already exists so a
future multi-anchor format can be detected and deserialized differently from
the current single-grid format.

---

## Layer 3 — PracticeMode

A `PracticeMode` is the composition layer.  It does not contain musical or
timing data itself; it points to the assets (exercise file, alignment file,
audio file) and declares which session mode to use.

```json
{
  "schema_version": 1,
  "mode": "recording_aligned_exercise",
  "exercise_path": "exercises/basic_four_beats.json",
  "alignment_path": "alignments/basic_four_beats_track.json",
  "description": "Four quarter-note exercise aligned to a backing track."
}
```

### Mode: `metronome_exercise`

**What it is:** The session runs against a synthetic metronome clock generated
from the exercise's `bpm` and `count_in_beats`.  No backing recording is
involved.

**Required assets:** `exercise_path`

**Use when:**
- Practising a pattern in isolation without a backing track.
- Working on pure timing accuracy against a click.
- The relevant exercise exists and no recording alignment is available yet.

**Example:**

```json
{
  "schema_version": 1,
  "mode": "metronome_exercise",
  "exercise_path": "exercises/basic_four_beats.json",
  "description": "Four quarter-note exercise with a synthetic metronome clock."
}
```

---

### Mode: `recording_aligned_exercise`

**What it is:** Target beat positions from the exercise are mapped onto a
backing audio recording via the beat alignment.  The recording's timeline
governs when each target is expected — not the exercise's `bpm` field.

**Required assets:** `exercise_path`, `alignment_path`

**Use when:**
- Practising along with a specific recorded track or backing loop.
- The exercise describes the musical targets and the alignment describes where
  those beats fall in the recording.

**How target times are computed:**

```
audio_time = alignment.first_beat_time_sec
           + target.time × beat_period_sec(alignment)
```

For example, with `first_beat_time_sec = 2.0` and `beat_period_sec = 1.0 s`,
an exercise target at beat 2 maps to audio time **4.0 s**.

**Example:**

```json
{
  "schema_version": 1,
  "mode": "recording_aligned_exercise",
  "exercise_path": "exercises/basic_four_beats.json",
  "alignment_path": "alignments/basic_four_beats_track.json",
  "description": "Four quarter-note exercise aligned to a backing track."
}
```

**Current limitation:** Because `BeatAlignment` assumes constant tempo, this
mode inherits that constraint.  If the backing track has significant tempo
variation the computed target times will drift from the true beat positions.

---

### Mode: `play_to_align`

**What it is:** The user has a recording but no alignment yet.  They play
simple beat-aligned notes against the recording; the system uses the detected
onset times to generate a `BeatAlignment` for later use.

**Required assets:** `audio_file`

**Use when:**
- Starting work on a new recording that has not been aligned yet.
- Generating an alignment by ear rather than by manual entry.

**Example:**

```json
{
  "schema_version": 1,
  "mode": "play_to_align",
  "audio_file": "tracks/my_track.wav",
  "description": "Play along with a recording to generate a beat alignment."
}
```

**Note:** Alignment generation from onset detection is not yet implemented.
This mode currently functions as a configuration placeholder that records the
user's intent.  The alignment will be written to `alignment_path` once the
generation step exists.

---

## Asset requirements summary

| Mode | `exercise_path` | `alignment_path` | `audio_file` |
|---|:---:|:---:|:---:|
| `metronome_exercise` | required | — | — |
| `recording_aligned_exercise` | required | required | optional |
| `play_to_align` | — | — | required |

`—` = not required (field may be absent or null).

---

## File locations

By convention the project stores fixtures under:

```
exercises/          Exercise JSON files
alignments/         BeatAlignment JSON files
practice_modes/     PracticeMode JSON files
tracks/             Audio recordings (not version-controlled)
```

Paths stored in `PracticeMode` are relative to the project root.

---

## Relationship diagram

```
PracticeMode
├── exercise_path  ──►  Exercise         (what to play, at what tempo)
├── alignment_path ──►  BeatAlignment    (when in the recording each beat falls)
└── audio_file     ──►  audio recording  (the backing track itself)
```

`Exercise` and `BeatAlignment` are independent of each other and can be
authored separately.  `PracticeMode` is what binds them together for a
specific session configuration.
