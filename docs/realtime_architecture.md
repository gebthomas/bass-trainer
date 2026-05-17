# Real-Time Feedback Architecture

## Purpose

This project is evolving toward a real-time bass practice system that can listen to a player, detect note attacks, compare them with expected musical targets, and return immediate timing feedback.

The current architecture has two parallel paths:

1. The original pull/window-buffer path, which analyzes audio windows after target times have passed.
2. The newer push/onset-event path, which receives detected onsets as events and routes them through a session engine.

Both paths currently coexist. The push path is the likely long-term architecture for interactive real-time feedback.

---

## Pull Path: Window-Based Evaluation

The existing live pipeline is centered around:

- `core/practice_session.py`
- `core/live_feedback.py`
- `core/live_pipeline.py`
- `core/feedback_events.py`
- `core/tempo_tracker.py`
- `scripts/live_feedback_demo.py`

In this path, the system maintains a growing audio buffer. As playback time advances, `PracticeSession.update(current_sample)` asks which targets are ready for evaluation. For each ready target, `live_pipeline.process_realtime_audio()` extracts the corresponding audio window, analyzes it, and emits feedback events.

This approach is useful because it works directly from buffered audio and does not require a separate onset stream. It also allows the evaluation window to shift using `TempoTracker.adjusted_target_time()`, which helps when the player gradually speeds up or slows down.

However, the pull path only updates the tracker when an onset is found inside a target evaluation window. It does not naturally consume every detected onset, including count-in beats, extra notes, or free-time events. That limits its usefulness for a fully interactive practice system.

---

## Push Path: Onset-Event Evaluation

The newer architecture is centered around:

- `core/session_engine.py`
- `core/onset_adapter.py`
- `core/tempo_tracker.py`
- `core/feedback_events.py`
- `scripts/session_engine_demo.py`

In this path, detected note attacks are pushed into the system as events:

```text
audio block
  ↓
OnsetAdapter.process_block()
  ↓
SessionEngine.on_onset(onset_time_s)
  ↓
TempoTracker.observe()
  ↓
feedback_event()
  ↓
console / future UI
```

### OnsetAdapter

`OnsetAdapter` converts raw audio blocks into session-time onset timestamps. For each block it computes RMS and peak amplitude. An onset fires when `rms >= min_rms OR peak >= min_peak` — the OR condition means it catches both sustained notes (energy spread across a block) and sharp transients (energy concentrated in a few samples, as with a plucked or slapped bass). A refractory period (default 150 ms) suppresses double-triggers within a single attack. The onset time is pinned to the sample of maximum absolute amplitude within the block, and reported as `onset_sample / sample_rate`.

`OnsetAdapter` is pure NumPy — no audio hardware dependency. The audio callback or read loop supplies the blocks.

### SessionEngine

`SessionEngine` routes each onset to a pending target. For every unevaluated target it computes a reference time — the nominal grid time when no tracker anchor exists yet, or `TempoTracker.adjusted_target_time(nominal)` once the tracker has an anchor. The onset is matched to the closest pending target whose reference time is within `match_window_s` (default: half a beat).

When a match is found:
- `TempoTracker.observe(nominal, onset_time_s)` is called so the tracker can update its tempo and phase estimates.
- A `feedback_event` dict is emitted. `timing_error_s` is always `onset_time_s - nominal_time_s` — relative to the nominal grid, not the adjusted one — so the error reported to the player has musical meaning regardless of tracker drift.

`SessionEngine.update_time(current_time_s)` should be called once per block. It emits miss events for targets whose acceptance windows have closed without a matching onset.

### TempoTracker

`TempoTracker` maintains two degrees of freedom relative to a nominal beat grid:

- `tempo_ratio`: the ratio of the player's actual beat duration to the nominal. Greater than 1 means the player is playing slower than nominal; less than 1 means faster.
- `phase_offset`: a residual constant offset in seconds after tempo correction. Positive means the player is consistently late.

The adjusted time for any nominal beat position is:

```text
adjusted(nominal_t) = anchor_actual
                    + (nominal_t - anchor_nominal) * tempo_ratio
                    + phase_offset
```

The anchor is set from the first accepted observation. Both `phase_offset` and `tempo_ratio` are updated with slow EMAs so that isolated timing mistakes do not retrain the grid. Observations whose prediction error exceeds the outlier limit are silently discarded.

To handle sustained acceleration or deceleration, the tracker widens its outlier threshold when the last several accepted errors all share the same sign and the smallest of them exceeds a magnitude floor. This lets the tracker follow a player who is gradually speeding up or slowing down without stalling at the original tempo.

`TempoTracker.confidence()` returns a 0–1 score based on the mean squared residual over a sliding window of recent accepted observations.

### Feedback Events

`feedback_events.feedback_event()` classifies each evaluated target into `good`, `warn`, or `miss` based on:

- Timing: `|error| <= 50 ms` → good, `<= 120 ms` → warn, beyond → miss
- Pitch: `|error| <= 25 cents` → good, `<= 50 cents` → warn, beyond → miss
- No detected note → miss
- Confidence below 0.5 → at least warn

Overall severity is the worst of all applicable dimensions. In the push path, pitch and confidence fields are `None` because `OnsetAdapter` does not perform pitch analysis; only timing severity is active.

---

## Time Reference

Both paths use the audio sample clock as the sole time reference. Session time in seconds is `sample_index / sample_rate`. Wall-clock time is not used. This keeps onset times, target nominal times, and tracker predictions in a single consistent coordinate system and avoids drift between the audio hardware clock and the system clock.

---

## Comparison

| Aspect | Pull path | Push path |
|---|---|---|
| Primary trigger | Elapsed sample time | Detected onset event |
| Audio analysis | Full window (pitch + onset) | Onset only (no pitch yet) |
| Tracker update | Only when onset found in window | Every matched onset |
| Miss detection | Window expiry | `update_time()` per block |
| Count-in / extra notes | Not consumed | Received but unmatched (ignored) |
| Hardware dependency | Buffer + pitch detector | `OnsetAdapter` only |

The pull path is better suited to pitch-verified feedback where the full audio window is needed. The push path is better suited to timing-only feedback and to architectures where onset detection is a separate upstream stage.

---

## Demo Scripts

`scripts/session_engine_demo.py` exercises the full push path end-to-end: it plays an audible count-in via `sounddevice`, opens an `InputStream`, feeds blocks through `OnsetAdapter`, routes onsets through `SessionEngine` and `TempoTracker`, and prints formatted timing feedback to the console. It uses a blocking read loop rather than a callback, with a 1024-frame block size (~21 ms at 48 kHz).

`scripts/live_feedback_demo.py` exercises the pull path with the original buffer-based pipeline.
