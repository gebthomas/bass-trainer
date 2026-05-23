# Match Window Policy

This document defines the canonical timing-window policy for onset-to-target
matching: what window formula the system should use, why, and how to migrate
there from the three implementations that currently coexist.

---

## 1. Current Implementations

### 1a. `matching.py` — local-gap formula (old tier)

```python
def get_match_window(target_index, targets):
    # neighbours
    prev_gap = targets[i]["time"] - targets[i-1]["time"]   # seconds
    next_gap = targets[i+1]["time"] - targets[i]["time"]   # seconds
    local_gap = min(prev_gap, next_gap)                    # tighter neighbour
    window = local_gap * 0.40
    return max(0.12, min(0.40, window))                    # clamped [0.12, 0.40] s
```

The targets list passed to `TargetMatcher` has already had beat positions
converted to audio seconds by the old live pipeline before this function is
called, so the gaps are in seconds.

Characteristics:
- Window scales with local note density, not with tempo.
- Hard minimum 120 ms, hard maximum 400 ms.
- Example outputs at 120 bpm:

  | Rhythm       | Gap (s) | Raw (40%) | Clamped   |
  |---|---|---|---|
  | quarter note | 0.50    | 0.200 s   | 0.200 s   |
  | eighth note  | 0.25    | 0.100 s   | 0.120 s   |
  | sixteenth    | 0.125   | 0.050 s   | 0.120 s   |
  | half note    | 1.00    | 0.400 s   | 0.400 s   |
  | whole note   | 2.00    | 0.800 s   | 0.400 s   |

- Adjacent windows never overlap (40% < 50% of gap), so no onset can be
  geometrically equidistant between two targets unless there is a long gap on
  one side and a short gap on the other, in which case the per-target windows
  differ in width.

Used by: `TargetMatcher` in the old live pipeline.  Not used by `SessionEngine`
or `session_runner`.

### 1b. `SessionEngine` — half-beat formula (new tier, live)

```python
self.match_window_s = 30.0 / self.bpm   # half a beat in seconds
```

Set once at construction time; constant for the entire session.  Overridable
via the `match_window_s` constructor argument.

Characteristics:
- Window is exactly half a beat regardless of how dense or sparse the rhythm
  is at a given moment.
- Scales with tempo: faster tempo → tighter window.
- Example outputs:

  | BPM | Window   |
  |---|---|
  |  60 | 0.500 s  |
  |  90 | 0.333 s  |
  | 120 | 0.250 s  |
  | 160 | 0.188 s  |
  | 200 | 0.150 s  |

- At 120 bpm, adjacent quarter-note targets (gap = 0.5 s) each have a
  ±0.25 s window, so their windows just touch.  Adjacent eighth-note targets
  (gap = 0.25 s) have overlapping windows: any onset between two targets
  can match either one.  The greedy nearest-target algorithm resolves this
  correctly, but an onset exactly halfway between two targets takes the
  lower-indexed one.

Used by: `SessionEngine`.  The tracker-adjusted reference time means the
window tracks the player's actual beat under tempo drift, but the width of
the window itself is fixed.

### 1c. `session_runner` — half-beat formula (new tier, batch)

```python
# metronome_exercise
return 30.0 / bundle.exercise.bpm

# recording_aligned_exercise
return 30.0 / estimated_bpm(bundle.alignment)

# fallback (unreachable in validated bundles)
return 0.25
```

Identical in philosophy to `SessionEngine` but computed per-run from the
bundle rather than stored on an object.  For recording-aligned mode, the BPM
is estimated from the alignment's `(first_beat_time_sec, last_beat_time_sec,
beat_count)` rather than taken from the exercise's nominal BPM field.

Used by: `run_session_bundle()` in the new batch execution path.

---

## 2. Musical Trade-offs

### Fixed absolute window

A single constant (e.g. 100 ms) applied everywhere.

- **Pro:** Maximally simple; one number to tune; independent of tempo.
- **Con:** Does not reflect musical reality.  At 60 bpm a quarter note lasts
  one second; 100 ms is a small fraction of that.  At 200 bpm a quarter note
  lasts 300 ms; 100 ms is a third of it.  The same absolute error means very
  different things musically at different tempos.

Not recommended for a tempo-aware training tool.

### Beat-relative window

Window proportional to one beat at the session tempo (the current policy for
`SessionEngine` and `session_runner`).

- **Pro:** Musically coherent — the tolerance is always the same fraction of
  a beat, so the feel of "how forgiving the grader is" stays constant as
  tempo changes.  Half a beat is easy to communicate: "play within the
  second half of the previous beat or the first half of the next beat."
- **Pro:** Single formula, one parameter (the fraction — currently 1/2).
- **Con:** Does not adapt to local rhythm density.  At 120 bpm with eighth
  notes (gap = 0.25 s), the half-beat window (0.25 s) spans the entire
  inter-note gap.  Adjacent windows fully overlap; the greedy algorithm still
  resolves matches correctly, but the zone of ambiguity is large.
- **Con:** At very slow tempos the window can grow wide enough to include
  non-adjacent targets.  At very fast tempos it can shrink to less than a
  typical onset-detector jitter window.

### Local inter-target-gap window

Window as a fraction (e.g. 40%) of the distance to the nearest neighbour
target.

- **Pro:** Adjacent windows never overlap when the fraction is < 50%.  Dense
  passages are handled safely without special-casing.
- **Pro:** Automatically scales with rhythm density — sparse passages get
  wider tolerance, dense passages get tighter.
- **Con:** Window is not musically constant: the same 50 ms error is accepted
  for a dense passage but rejected for a sparse one, even at the same tempo.
  This is the reverse of what most players expect.
- **Con:** Requires inspecting neighboring targets; not a simple one-variable
  formula.
- **Con:** The clamping bounds ([0.12, 0.40] s in `matching.py`) are
  empirically chosen without a clear musical rationale.
- **Con:** Does not scale with tempo at all.  A very slow, evenly-spaced
  exercise would give large windows; a very fast, evenly-spaced exercise
  would clamp at the minimum.

### Adaptive window

Window that shrinks as the player's measured accuracy improves over sessions.

- **Pro:** Provides a progression path — a beginner gets generous windows,
  an advanced player is held to tighter standards.
- **Con:** Non-deterministic across sessions; reproducibility of scoring
  breaks.  A saved log is ambiguous unless the window used at record time is
  stored alongside.
- **Con:** Adds history management overhead.
- **Con:** Not yet feasible without per-session performance tracking.

Deferred.  Revisit after per-session metrics are stable.

### Asymmetric window

Different tolerances for early onsets vs late onsets.

- **Pro:** Musically motivated for bass: a note played slightly late against
  a click may still be perceived as in the pocket, while a note played early
  disrupts the groove.  Rushing tends to be more audible than dragging.
- **Pro:** A wider late window would reduce false misses for players with
  heavier attack or deeper note-onset points on the string.
- **Con:** More parameters to explain and tune.
- **Con:** Complicates the matching rule: the symmetry assumption (`delta <=
  window`) breaks, and the error sign must be tested at candidate-selection
  time.

Worth considering in a future difficulty or style setting, but not yet.

---

## 3. Requirements for This Project

The following requirements are taken from the project's stated philosophy:

**Groove over precision.**  The system trains consistent feel and time, not
metronome-click accuracy.  A player consistently 40 ms late is demonstrating
pocket; a player alternating 80 ms early and 80 ms late has a timing
problem.  The window should be wide enough that a player in the groove is
never penalised for a natural, consistent offset.

**Sparse and dense rhythms.**  A walking bass line at 140 bpm has quarter
notes every 0.43 s.  A funk line at 90 bpm may include 16th-note triplets
with gaps under 0.10 s.  The policy must handle both without misrouting onsets.

**Tempo drift tolerance.**  Live practice against a recording with slight
human feel, or against a click at an unfamiliar tempo, may involve modest
drift.  The window should not close so tightly that a player slightly behind
the grid gets chronic misses.

**User legibility.**  Players should be able to predict whether a given
performance will be scored as a hit.  A formula the teacher can state in one
sentence beats a formula that requires looking up the neighbour targets.

**Determinism.**  Given the same onset stream and the same session bundle,
the system should produce identical results every time.  Window width must
not depend on runtime state that is not part of the input.

---

## 4. Recommended Canonical Policy

**Use a beat-relative half-beat window, clamped to [0.10, 0.35] seconds.**

```
window_s = clamp(30.0 / bpm, lo=0.10, hi=0.35)
```

### Rationale

The half-beat formula (`30.0 / bpm`) is already used by the two newest
modules, is the most musically legible choice ("you have half a beat"), and
scales naturally with tempo.  The clamping addresses the two failure modes:

- **Lower clamp 0.10 s** (100 ms): Prevents the window from shrinking below
  a level that onset detectors can reliably distinguish.  At 300 bpm (the
  edge of playable bass tempos), the unclamped window would be 0.10 s, which
  is the clamp exactly — so the clamp is inactive below 300 bpm.  At 180 bpm
  (the realistic upper end of bass playing) the unclamped window is 0.167 s,
  which is also above the clamp.  The lower bound is therefore a safety net
  for unrealistic inputs rather than an active shaping factor.

- **Upper clamp 0.35 s** (350 ms): Prevents the window from expanding so
  wide at slow tempos that adjacent targets' windows overlap for common
  rhythms.  At 60 bpm quarter notes (gap = 1.0 s), the unclamped window
  would be 0.50 s, which covers 100% of the half-gap — i.e., windows just
  touch.  With the 0.35 s ceiling the window covers 70% of the half-gap,
  leaving a gap between adjacent acceptance zones.  This matters most for
  sparse, slow ballads where a stray bass note from a fill should not
  accidentally claim a target it is nowhere near.

### Window overlap analysis at canonical policy

At 120 bpm (window = 0.25 s):

| Rhythm         | Gap  | Window | % of gap | Overlap? |
|---|---|---|---|---|
| Whole note     | 2.00 | 0.25 s | 12.5%    | No       |
| Half note      | 1.00 | 0.25 s | 25.0%    | No       |
| Quarter note   | 0.50 | 0.25 s | 50.0%    | Touch    |
| Eighth note    | 0.25 | 0.25 s | 100.0%   | Yes      |
| Sixteenth note | 0.125| 0.25 s | 200.0%   | Yes      |

Eighth-note and sixteenth-note windows overlap at 120 bpm.  This is
acceptable given the groove-first requirement: the greedy nearest-target
algorithm resolves these cases correctly for all but pathologically
ambiguous onset positions.  An onset equidistant between two eighth-note
targets is resolved to the lower-indexed target, which is consistent and
deterministic.  If window overlap becomes a real source of mis-scoring in
practice, narrowing to 40% of a beat (`24.0 / bpm`) would eliminate overlap
for eighth notes without tightening significantly for quarter notes.

### Stated formula

```python
def match_window_s(bpm: float) -> float:
    """Half-beat match window in seconds, clamped to a safe range."""
    return max(0.10, min(0.35, 30.0 / bpm))
```

This single function is the entire canonical policy.  It takes one number
(BPM) and returns one number (seconds).  No target list needed, no history
needed.

For `recording_aligned_exercise`, pass `estimated_bpm(alignment)` as the
`bpm` argument.  The alignment's derived BPM is the authoritative tempo for
that mode, not the exercise's nominal BPM field.

---

## 5. Module Ownership

The canonical policy function should live in a new file: **`core/timing_policy.py`**.

```
core/timing_policy.py          ← canonical policy (new file)
core/session_engine.py         ← import and delegate
core/session_runner.py         ← import and delegate
core/matching.py               ← leave as-is (old tier, do not migrate)
```

`timing_policy.py` should export only `match_window_s(bpm)`.  It must have
no imports from other `core` modules.  Any module can depend on it safely.

This placement was already identified in `docs/event_schema_direction.md`
§5b as the correct home for the canonical window definition.

---

## 6. Migration Path

### What to migrate

| Module | Current formula | Action |
|---|---|---|
| `session_runner._default_match_window()` | `30.0 / bpm` or `30.0 / estimated_bpm` | Replace with `match_window_s(bpm)` after adding clamping. |
| `SessionEngine.__post_init__` | `30.0 / self.bpm` | Replace with `match_window_s(self.bpm)` after adding clamping. |

Both of these already use the half-beat formula; migration is a one-line
change per site plus the addition of `core/timing_policy.py`.

The default `match_window_s` argument on `SessionEngine` should remain
overridable via the constructor parameter so that tests can inject specific
values without having to manipulate BPM.

### What to leave alone

`matching.py` and `TargetMatcher` are old-tier code.  Their local-gap
formula should not be migrated to the canonical policy.  Doing so would
change evaluation behavior for any live sessions that still use the old
pipeline without a clear benefit — the old pipeline is already marked for
eventual retirement.  When `matching.py` is retired, the formula disappears
with it.

### Migration order

1. Create `core/timing_policy.py` with the `match_window_s(bpm)` function
   and unit tests in `tests/test_timing_policy.py`.
2. Migrate `session_runner._default_match_window()` to call
   `timing_policy.match_window_s`.
3. Migrate `SessionEngine.__post_init__` to call `timing_policy.match_window_s`.
4. Update existing tests that hard-code window values to derive them via
   `timing_policy.match_window_s(bpm)` so they remain correct if the formula
   is ever adjusted.

Steps 2 and 3 can be done in either order; neither depends on the other.

---

## 7. Open Questions

### Should tolerance depend on difficulty?

One approach: a difficulty setting scales the window fraction (e.g. easy =
60% of a beat, normal = 50%, hard = 40%).  The formula becomes
`(difficulty_fraction * 60.0) / bpm`.

This is the cleanest path to difficulty scaling.  It does not require
history and remains deterministic.  The difficulty parameter would live in
`PracticeMode` or as a session-level override.

**Decision deferred.**  A difficulty field does not yet exist in the data
model.  When it is added, `timing_policy.match_window_s` should accept an
optional `fraction: float = 0.5` parameter rather than adding a separate
function.

### Should recording-aligned sessions use a different window?

The argument for a different window: in recording-aligned mode, the player
is matching against a human recording that may have subtle feel imperfections.
A slightly wider window might reduce false misses caused by the recording's
own groove variation rather than the player's error.

The argument against: the window is already half a beat.  A skilled player
hitting a slightly-swung note in an aligned recording should be within half a
beat of the grid target.  Widening further conflates the recording's feel
with the player's accuracy.

**Current recommendation: no separate window for recording-aligned mode.**
Use the same formula with `estimated_bpm(alignment)` as the BPM argument.
Revisit if false-miss rates are noticeably higher in aligned mode than in
metronome mode for the same player at the same tempo.

### Should windows scale with player skill?

Adaptive narrowing based on measured accuracy over sessions would create a
progression path.  As noted in §2, this breaks determinism: the window at
record time must be stored alongside the event to allow replay.

`SessionLog.metrics` does not yet capture what window was used.  If adaptive
windows are introduced, the window value (and the formula version) must be
stored in `SessionLog` or `SessionEvent.metadata` so that a saved log can be
re-evaluated consistently.

**Deferred until per-session accuracy metrics are stable and the `SessionLog`
schema has a clear place to record evaluation parameters.**  The canonical
`match_window_s(bpm)` function should be treated as the v1 policy; a future
`match_window_s(bpm, skill_level)` signature would be v2.
