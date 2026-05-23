# Event Schema Direction

This document records the architectural decision made after the systems-level
review: **`SessionEvent` / `SessionLog` is the canonical persistent event
representation.** `feedback_event()` dicts are legacy runtime events that
remain in place without modification for now.

---

## 1. Why `SessionEvent` is canonical

`SessionEvent` is a validated, versioned, serialisable dataclass.  It can be
written to disk, loaded back, and re-inspected after the session ends.  It has
no dependency on audio hardware, numpy, or any runtime context.

`feedback_event()` dicts are plain Python dicts with no schema enforcement, no
versioning, and no file I/O.  They were designed to carry per-note feedback to
a live display during a session — not to persist across sessions or serve as
the basis for long-term analytics.  Their field set (`detected_note`,
`severity`, `messages`, `pitch_error_cents`) reflects live presentation
concerns, not storage concerns.

The distinction matters because persistence and presentation have different
stability requirements.  Persistent format changes are migrations.  Presentation
format changes are refactors.  Treating the dict format as canonical conflates
the two.

`SessionLog` also has a versioned JSON round-trip, a validated schema, and a
clear ownership model — it is the record of what happened during a session.
`feedback_event()` dicts have none of these properties.

---

## 2. Why `feedback_event()` dicts are not removed yet

The old dict path is still the only path wired into the live audio pipeline.
`session_engine.py`, `live_pipeline.py`, `session_replay.py`, and
`metrics.py` all consume or produce the old dict format.  These modules
implement features — tempo tracking, adaptive window shifting, rolling metrics,
good/warn/miss severity — that the new `SessionEvent` path does not yet cover.

Removing or rewriting the old path before the new path provides equivalent
coverage would break functioning workflows and tests.  The two paths coexist
deliberately: the old path remains authoritative for live audio evaluation; the
new path becomes authoritative for persistence and batch analysis.

The old path will be retired only after:
- the canonical event vocabulary is stable,
- severity classification exists on the new path,
- `SessionMetrics` can be derived from `SessionLog`, and
- at minimum the live replay and metrics workflows have been re-implemented
  against `SessionEvent`.

Until then, modifying the old dict format or `metrics.py` is out of scope.

---

## 3. Current overlap between the two schemas

These are known conflicts to be resolved during the cleanup sequence (§5).
They are documented here so they are not re-introduced or papered over in the
meantime.

### 3a. `target_idx` vs `target_index`

`feedback_event()` dicts use the key `target_idx` (no `e`).  `SessionEvent`
uses the field name `target_index`.  These refer to the same concept.  Any
bridge code written before the vocabulary is unified must handle this rename
explicitly.

### 3b. Event type vocabulary

| Concept | Old dict (`event_type` / `detected_note`) | New `SessionEvent.event_type` |
|---|---|---|
| player hit the target | `detected_note` is not None | `"target_hit"` |
| target with no onset detected | `detected_note` is None | `"target_miss"` |
| onset that matched no target | `event_type == "extra"` (in `results.py`) | `"extra_onset"` |

The old dict path encodes hit/miss as a None check on `detected_note`, not as
a separate `event_type` field.  `results.py` uses `event_type == "missed"` and
`event_type == "extra"`.  `session_runner.py` uses `"target_hit"`,
`"target_miss"`, and `"extra_onset"`.

The new vocabulary (`target_hit`, `target_miss`, `extra_onset`) is the
intended canonical form.  It is explicit, symmetric, and does not require a
secondary field to determine event category.

### 3c. `timing_error_s` vs `value`

`feedback_event()` dicts carry timing error as `timing_error_s` (signed float,
seconds, positive = late).  `SessionEvent` carries it as `value` (optional
float, same sign convention).

`value` is a general-purpose numeric payload.  This is appropriate for now,
while the full field set of a `SessionEvent` is still being decided.  Once
the permanent field set is settled, timing error may get a dedicated typed
field rather than sharing `value` with any numeric result.

---

## 4. Current non-goals

These actions are explicitly deferred and should not be started until the
cleanup sequence in §5 is complete.

**Do not build a bridge yet.**  A translation layer from `SessionEvent` lists
to old-style `feedback_event()` dicts (or the reverse) would couple two
schemas that are still in flux.  Building a bridge now would fix both sides in
place prematurely and make the eventual cleanup harder.

**Do not add severity to `SessionEvent` yet.**  The `good`/`warn`/`miss`
thresholds live in `feedback_events.py` (`_TIMING_GOOD = 0.05 s`,
`_TIMING_WARN = 0.12 s`).  Adding severity to `SessionEvent` before the
canonical threshold home is settled would introduce a second copy of the same
constants.  Severity will be added to the new path as part of step (d) in the
cleanup sequence.

**Do not rewrite `metrics.py` yet.**  `compute_session_metrics()` works
correctly against the old dict format.  Rewriting it against `SessionEvent`
before the event schema is stable means rewriting it twice.

---

## 5. Near-term cleanup sequence

These steps are ordered by dependency.  Each step should be completed and
tested before the next begins.

### Step a — Define canonical event vocabulary

Decide and document the exact set of `event_type` strings that `SessionEvent`
will use, and what fields each event type carries.  The current candidates are
`"target_hit"`, `"target_miss"`, and `"extra_onset"`.  The open questions in
§6 must be resolved before this step closes.

Once the vocabulary is fixed, add it as a named constant set or validator in
`session_log.py` so that `append_event()` can reject unknown event types.

### Step b — Define canonical match-window policy

There are currently three match-window calculations in the codebase:

| Site | Formula |
|---|---|
| `matching.py` `get_match_window()` | 40% of local inter-target gap, clamped to 0.12–0.40 s |
| `session_engine.py` default | `30.0 / bpm` (half a beat) |
| `session_runner._default_match_window()` | `30.0 / bpm` or `30.0 / estimated_bpm(alignment)` |

These produce different windows for the same session.  The canonical window
should be defined once, documented with its musical rationale, and used by
all matching sites.  This belongs in `session_runner.py` or a new
`core/timing_policy.py` — not duplicated across modules.

### Step c — Add event translation only if still needed

After steps (a) and (b), evaluate whether a translation layer between old and
new event schemas is actually required or whether the old dict path can simply
be left in place until it is retired entirely.  If a bridge is needed — for
example, to run `compute_session_metrics()` against a `SessionLog` — build it
as a narrow adapter function, not as a rewrite of either schema.

### Step d — Add metrics on top of canonical `SessionEvent`

Once vocabulary and severity are stable, add a `compute_log_metrics()` function
(or equivalent) that accepts a `SessionLog` and returns structured metrics.
This becomes the new-path counterpart to `compute_session_metrics()`.  The two
can coexist during transition; `compute_session_metrics()` is not removed until
the live pipeline is also migrated.

---

## 6. Open questions

These questions must be resolved before step (a) in the cleanup sequence can
close.  They are recorded here to avoid decisions being made implicitly.

**Should `target_hit.value` remain the signed timing error?**

Currently `value` is the only numeric payload on a `SessionEvent`, and
`session_runner.py` uses it for the signed timing error.  This works for
timing-only sessions.  If a session also captures pitch error, `value` cannot
hold both.  Options: keep `value` as timing error only and add a second field
for pitch, rename `value` to `timing_error_s` and add `pitch_error_cents`, or
keep `value` as a generic float and move per-dimension results into
`SessionEvent.metadata` until the field set is settled.  The generic `value`
approach defers the decision at the cost of type safety.

**Should severity belong in `SessionEvent` or be derived later?**

One design has severity stored on each event at write time (the `feedback_event()`
dict approach — severity is baked in).  The other derives severity from the
raw numeric fields at read time, so threshold changes retroactively recompute
history.  Stored severity is simpler to query but freezes the threshold at the
moment of recording.  Derived severity requires keeping the raw numbers but
allows re-evaluation with different thresholds.  Given that this system is
focused on self-improvement over time, derived severity is more consistent with
the goal: a timing error of 55 ms is always 55 ms, and what counts as "warn"
may evolve as the player improves.

**Should pitch results belong in `SessionEvent.metadata` or typed fields?**

Pitch detection does not yet produce results in the new event path.  When it
does, the pitch error in cents, confidence, and detected note name are natural
candidates for dedicated `SessionEvent` fields — but adding them prematurely
creates fields that are always `None` for sessions without pitch analysis.
`metadata` is the short-term holding area until pitch is reliable enough to
promote to typed fields.  The line between "experimental" (metadata) and
"stable" (typed field) should be crossing confidence and in-tune rate, not
just detection of a frequency.
