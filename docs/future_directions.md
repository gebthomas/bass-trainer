# Future Directions

This document describes possible future directions for the project.

Some items discussed here are intentionally deferred in the current phase. See `not_now.md` for explicit deferrals and scope constraints.

---

## 1. Musical Feedback

The current system evaluates timing only. Richer musical feedback is the most direct path toward a useful practice tool.

**Likely near-term next steps**
- **Pitch detection** — identify the note the player actually played and compare it to the expected note. The analysis window already exists in the pull path; a pitch estimator (e.g. YIN or autocorrelation) can slot in without changing the event schema (`pitch_error_cents` is already a field in feedback events, currently `None`).
- **Note-name validation** — once pitch is available, flag wrong notes explicitly ("played A, expected E") rather than just reporting cents deviation.
- **Rhythmic subdivision support** — extend target definitions to include eighth notes, triplets, and sixteenth notes. The engine's beat-time model generalises to any subdivision; targets need a `subdivision` or fractional `time` field.

**Interesting future ideas**
- **Articulation detection** — distinguish plucked vs slapped vs popped attacks from the onset waveform shape or spectral profile. Useful for style exercises.
- **Dynamics / velocity tracking** — track RMS at the onset as a proxy for dynamic level. Flag notes that are too quiet or unevenly matched across the bar.
- **Sustain length** — measure how long the note rings before it decays below a threshold. Relevant for legato phrasing exercises.
- **Groove / feel analysis** — after collecting a session's timing errors, identify systematic micro-timing patterns (e.g. consistently pushing or pulling the beat on specific subdivisions). Requires more events than a four-note exercise.
- **Swing detection** — compare even-eighth vs swung-eighth ratios across a session. Distinct from timing accuracy: a player can be consistently swung without being "wrong."
- **Tone / noise quality metrics** — spectral flatness or noise floor ratio as a rough proxy for clean vs buzzy tone. Instrument-dependent; needs calibration.
- **Muting / string-noise detection** — detect high-frequency transients between beats that indicate string noise. Low priority until tone metrics are validated.

---

## 2. Practice UX

The current interface is a console read-out. UX improvements matter most after the core detection pipeline is reliable.

**Likely near-term next steps**
- **Session summaries** — the metrics layer (`core/metrics.py`) already produces rich data; the near-term step is surfacing it more clearly after each session and optionally saving it.
- **Exercise libraries** — a directory of `Exercise` JSON files covering common patterns (quarter notes, walking bass, arpeggios, ii-V-I). The `Exercise` model is already in place.
- **Visual timing display** — a minimal terminal bar or simple GUI window showing the timing error per beat, so the player gets spatial feedback rather than just a number.

**Interesting future ideas**
- **Scrolling notation or tab** — display the exercise target ahead of time so the player can read along, with a playhead advancing through the bar.
- **Live streak indicators** — show the current good/miss streak in real time, not just at session end. `StreakStats.current_good_streak` already tracks this.
- **Adaptive metronome** — a click track that can slow down when the player is struggling and gradually return to target tempo as accuracy improves.
- **Progressive difficulty** — automatically tighten the match window or increase the tempo after a threshold of consistently good sessions.
- **"Problem measure" repetition** — after a session, automatically loop the subset of targets where the player consistently missed or scored warn, until they improve.
- **Loop practice** — let the exercise repeat continuously without restarting the script, accumulating metrics across repetitions.
- **Audio cues vs visual cues** — allow the count-in and feedback to be audio tones, visual flashes, or both, depending on the player's preference and setup.
- **Hands-free operation** — pedal or voice trigger to start/stop sessions without touching a keyboard. Especially relevant for upright bass players.

---

## 3. Adaptive / Intelligent Features

These features require more session data to be meaningful. They are longer-term.

**Likely near-term next steps**
- **Automatic threshold tuning** — given a calibration pass (player plucks open strings at known dynamics), auto-set `MIN_RMS` and `MIN_PEAK` for the current instrument and interface. This removes the manual threshold tuning step from the road test plan.
- **Confidence scoring** — `TempoTracker.confidence()` is already computed; surface it in session summaries and use it to qualify metrics (a low-confidence session means the tracker did not have enough data to adapt).

**Interesting future ideas**
- **Personalised match windows** — learn the player's typical reaction time offset from their first few sessions and pre-adjust the match window center, rather than always centering on the nominal grid.
- **Tempo adaptation** — track long-term BPM trends across sessions to identify consistent drift patterns (does this player always slow down in bar 3?).
- **Fatigue detection** — monitor rolling metrics within a session for deteriorating hit rate or widening timing std, and surface a "take a break" prompt.
- **Identifying consistently weak transitions** — across multiple sessions, correlate missed or warn events with specific interval transitions (e.g. E→A) to suggest targeted exercises.
- **Recommending exercises** — given a history of session metrics, suggest which exercise file to load next using rule-based pattern matching. ML-based recommendation systems are out of scope for now (see `not_now.md`).
- **Automatic difficulty adjustment** — tighten the severity thresholds or add more targets as the player's baseline improves.
- **Comparing sessions over time** — trend lines for hit rate, mean timing error, and consistency across sessions. Requires persistence first.

*Note on recovery training: training recovery from disruption — described in `design_principles.md` §6 as a core value — will likely require architectural extensions beyond the current session-engine / target-matching model. It is not a near-term feature of the existing architecture.*

---

## 4. Persistence and Data

Nothing is saved across sessions yet. The replay harness is the only durable artifact.

**Likely near-term next steps**
- **Save session history** — after each live session, write a structured JSON file (onset timestamps + metrics) to a local directory. The replay fixture format already defines a suitable schema.
- **Replay captured sessions** — the replay harness (`core/session_replay.py`) already exists; wiring the live demo to write a capture file is a small step.

**Interesting future ideas**
- **Export metrics** — CSV or JSON export of per-session aggregates for external analysis in a spreadsheet or notebook.
- **Exercise packs** — curated sets of `Exercise` JSON files, organised by style, difficulty, or technique, distributed as a simple directory.
- **Cloud sync** — optional sync of session history to a cloud store. Low priority; local-only is the right default.
- **Local-only privacy mode** — explicit setting to ensure no session audio or data leaves the device. The current architecture is already local-only; this would just be a documented guarantee.

---

## 5. Architecture Considerations

The current architecture works well for the four-quarter-note use case. Scaling to richer musical content will require some structural changes.

**Likely near-term next steps**
- **Pitch pipeline separation** — add pitch detection as a separate analysis step that reads from the same audio buffer as the pull path, without coupling it to onset detection. The pull path's analysis window is already the natural place for this.
- **GUI boundary separation** — establish a clear interface between the session engine (produces events) and any display layer (consumes events), so the engine never calls UI code directly. The event dict schema is already a natural seam.

**Interesting future ideas**
- **Separating audio capture from analysis** — move audio I/O onto a dedicated thread or process, and pass blocks to the analysis side via a queue. The current single-thread read-loop works at 48 kHz / 1024-frame blocks; pitch analysis may require longer windows that make the loop latency-sensitive.
- **Multi-onset-per-block detection** — the current `OnsetAdapter` detects at most one onset per block. Fast passages (sixteenth notes above ~120 BPM) with small block sizes may need sub-block onset scanning.
- **FFT or spectral onset detection** — replace the current RMS/peak detector with a spectral flux or phase-deviation detector for better sensitivity on low-energy or high-frequency attacks. This is a drop-in replacement for `OnsetAdapter` if the interface is kept stable.
- **Realtime visualisation pipeline** — a lightweight event bus that feeds a UI loop independently of the audio callback, avoiding dropped frames or added latency in the audio path.
- **Latency calibration** — measure and compensate for the round-trip latency of the audio interface so that timing errors are reported relative to the actual note onset, not the detected sample. Interface latency varies by device and buffer size.
- **Device profiles** — save per-device threshold settings (min_rms, min_peak, refractory_s) so the player does not re-tune after switching instruments or interfaces.
- **Plugin architecture for detectors / analysers** — define a minimal interface for onset detectors and pitch analysers so alternative implementations can be swapped in without touching the engine. Premature until at least two real implementations exist.

---

## 6. Explicit Non-Goals (for now)

These are out of scope until the core practice loop is validated and stable.

- **DAW replacement** — the system is a feedback tool for a live player, not a recording or production environment. No MIDI output, no track management, no mixing.
- **Full transcription** — automatically transcribing what the player played is a much harder problem than comparing against a known target. Not needed for the feedback loop.
- **Studio-grade audio processing** — the onset detector is intentionally simple. High-fidelity spectral analysis, room correction, and noise reduction belong in a different kind of tool.
- **Multiplayer or network features** — no latency-tolerant networking, no remote sessions, no shared score views.
- **Machine-learning-heavy architecture before simpler methods are validated** — learned onset detectors, neural pitch estimators, and RL-based difficulty adaptation are interesting but introduce opacity and data dependencies. The current signal-processing approach should be pushed as far as it can go first.
- **Mobile** — the system targets a desktop Python environment with a low-latency audio interface. Mobile audio I/O constraints are a separate engineering problem.
