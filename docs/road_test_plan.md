# Road Test Plan: Session Engine Live Demo

## Goal

Verify the live push-path timing system with real bass input.

Test path:

```text
bass/interface input
→ sounddevice read-loop
→ OnsetAdapter
→ SessionEngine
→ TempoTracker
→ console timing feedback
```

---

## Before Testing

Run the test suite:

```
python3.11 -m pytest
```

Expected: all tests pass, 0 failed. (The exact count grows as modules are added; the important signal is zero failures.)

Then confirm the demo script loads:

```
python scripts/session_engine_demo.py --help
```

---

## Step 1: Identify Input Device

```
python scripts/session_engine_demo.py --debug
```

If the wrong input is selected, specify the device by name fragment or index:

```
python scripts/session_engine_demo.py --device MiniMe --debug
python scripts/session_engine_demo.py --device <device_index> --debug
```

Success signs:
- The interface appears in the printed device list
- RMS changes when bass is played
- No buffer overflow warnings

---

## Step 2: Confirm Onset Detection

Run with debug output and a relaxed tempo:

```
python scripts/session_engine_demo.py --device MiniMe --debug --bpm 100
```

Ignore timing feedback initially. Pluck several notes and check:
- Each clear pluck produces exactly one `[onset]` line
- Soft plucks are not missed
- One pluck does not create multiple onset lines
- RMS and/or peak levels are visibly above threshold in the debug readout

Likely fixes:

| Symptom | Likely cause | Try |
|---|---|---|
| No onsets | Threshold too high or gain too low | Lower `MIN_PEAK`, raise interface gain |
| Too many onsets | Threshold too low or note decay retriggers | Raise `MIN_PEAK`, increase `REFRACTORY_S` |
| Wrong device or silence | Wrong input device or channel | Use `--device` with the correct name or index |

---

## Step 3: Four Quarter-Note Timing Test

```
python scripts/session_engine_demo.py --device MiniMe --bpm 100
```

Play four quarter notes after the count-in.

Expected:
- Four feedback events emitted
- Mostly `good` or `warn` severity
- No unexplained misses
- Tracker BPM stays near the target tempo

Repeat at a faster tempo:

```
python scripts/session_engine_demo.py --device MiniMe --bpm 120
```

---

## Step 4: Threshold Tuning

Tune one parameter at a time. Starting defaults in `scripts/session_engine_demo.py`:

```python
MIN_RMS      = 0.018
MIN_PEAK     = 0.15
REFRACTORY_S = 0.150
```

| Symptom | Likely cause | Try |
|---|---|---|
| No onsets | Threshold too high, gain too low, wrong channel | Lower `MIN_PEAK`, raise gain |
| Double triggers | Threshold too low or note decay retriggers | Raise `MIN_PEAK`, raise `REFRACTORY_S` |
| Every other fast note missed | Refractory period too long | Lower `REFRACTORY_S` |
| Timing consistently shifted | Count-in / input alignment issue | Record console output before changing engine logic |
| Many misses despite onsets firing | Match window too narrow or onset routing issue | Inspect onset timestamps vs target nominals |

---

## Step 5: Save Console Output Before Changing Anything

For any failure, capture and save:
- Command used
- Device selected
- BPM
- Full debug output
- Onset timestamps
- Feedback events printed
- Session summary

Paste this output into a conversation with Claude before changing any code.

---

## Do Not Change Yet

Avoid modifying these until live behavior is fully understood:

- `core/session_engine.py`
- `core/tempo_tracker.py`
- `core/feedback_events.py`
- Replay fixture format
- Metrics definitions

The first live failures are most likely device, channel, gain, threshold, or refractory issues — not engine logic.

---

## Success Criteria

The first live milestone is:

- Correct input device selected
- Clear bass plucks produce exactly one onset each
- Four quarter-note exercise produces four evaluated targets
- No buffer overflow warnings
- Feedback timing roughly matches what was played

Only after this works should we test:

- Eighth notes and faster patterns
- Higher tempos
- Pitch detection
- GUI or visual feedback
- Persistent session logging
