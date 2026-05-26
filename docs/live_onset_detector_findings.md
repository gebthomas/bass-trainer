# Live Onset Detector Findings

## 1. Original −30 ms Artifact

### Symptom

Live bass onsets reported by the fixed-grid pipeline clustered around **−30 ms** even
when timing intentionally varied across beats.  The bias was not random noise — it was
a floor: nearly every detected onset landed at exactly −30 ms, regardless of when the
note was actually played.

### Root cause

`evaluate_window` (pre-fix) used a simple absolute-amplitude threshold:

```python
crossings = np.where(np.abs(mono) >= onset_threshold)[0]
onset_sample = int(crossings[0])   # first sample that exceeds 0.02
```

The extraction window was placed **30 ms before the nominal beat** (pre-roll = 0.03 s).
Bass strings sustain resonance from the previous note at amplitudes that routinely exceed
the 0.02 threshold.  That resonance was already present at sample 0 of every window, so
`onset_sample` became 0, and the reported timing error was always:

```
timing_error = (window_start − beat_target) = −30 ms
```

No amount of playing variation could move the reported onset off this floor because the
detector never reached the actual new attack — it stopped at the first sample of the window.

### Why it was not obvious

The debug output showed a signed error, not a raw onset time, so the −30 ms values looked
like plausible "slightly early" playing.  The clustering only became apparent after
collecting multiple beats and noticing the error distribution was a spike, not a spread.

---

## 2. Rise-Based Evaluator Fix

### Approach

Replace the absolute threshold with a **local-baseline dynamic threshold**.  Rather than
asking "does amplitude exceed a fixed floor?", ask "does amplitude rise meaningfully above
what it was at the start of this window?"

### Implementation (`core/realtime_evaluator.py`)

```python
env = np.abs(mono)

baseline_n    = max(1, round(sample_rate * baseline_window_s))   # default 15 ms
baseline_n    = min(baseline_n, len(env))
baseline      = float(np.mean(env[:baseline_n]))
dyn_threshold = max(onset_threshold, baseline * rise_ratio)       # default rise_ratio = 2.5

crossings = np.where(env >= dyn_threshold)[0]
```

**Key parameters (defaults):**

| Parameter | Default | Effect |
|---|---|---|
| `onset_threshold` | 0.02 | Absolute floor; prevents triggers in true silence |
| `baseline_window_s` | 0.015 s | Length of baseline measurement (first 15 ms of window) |
| `rise_ratio` | 2.5 | Signal must be 2.5× the local baseline to count as onset |

**Why it works:**

- Sustained resonance: baseline ≈ resonance level → dyn_threshold ≈ 2.5 × resonance.
  The sustained signal never rises above its own level, so no threshold crossing is found.
- New attack: baseline ≈ silence or prior resonance → attack amplitude exceeds 2.5× baseline.
  Threshold crossing is found at the moment of the new attack.

**What did not change:** `rms`, `peak`, function signature defaults, and all callers.
The only changed behaviour is for a constant signal at exactly `onset_threshold`:
previously reported as onset-found; now reported as no onset (sustained ≠ attack).
One existing test was renamed and its assertion inverted to reflect this.

---

## 3. Recorded Diagnostics Results

Tests were run by recording live bass playing into `diagnostics/bass_*bpm_*.wav` using
`scripts/live_feedback_demo.py --record-wav`, then comparing detectors offline with
`scripts/compare_onset_detectors.py`.

### 60 BPM — 16 beats, count-in 4

| Detector | matched | missed | extras | mean err | mean \|err\| |
|---|---|---|---|---|---|
| live (rise-based) | 15/16 | 1 | 0 | −14.9 ms | 27.6 ms |
| spectral flux | 12/16 | 4 | 15 | +96.0 ms | 155.6 ms |
| energy derivative | 14/16 | 2 | 58 | +57.0 ms | 81.7 ms |

### 90 BPM — 24 beats, count-in 4

| Detector | matched | missed | extras | mean err | mean \|err\| |
|---|---|---|---|---|---|
| live (rise-based) | 20/24 | 4 | 0 | −15 ms | ~25 ms |
| spectral flux | — | — | many | — | higher |
| energy derivative | — | — | many | — | higher |

### 120 BPM — 24 beats, count-in 4

| Detector | matched | missed | extras | mean err | mean \|err\| |
|---|---|---|---|---|---|
| live (rise-based) | 24/24 | 0 | 0 | −22 ms | ~26 ms |
| spectral flux | — | — | many | — | higher |
| energy derivative | — | — | many | — | higher |

**Pattern across tempos:** the live rise-based detector achieves a consistent mean signed
error in the −15 to −22 ms range, with mean absolute error around 25–28 ms.  Faster tempos
do not degrade accuracy, and the miss rate drops to zero at 120 BPM.

---

## 4. Detector Comparison

### Librosa spectral-flux (`spectral` column)

- Runs on the entire file; does not respect per-beat windows.
- Detects count-in clicks, inter-note transients, and pick noise as onsets.
- Produced **15 unmatched onsets** at 60 BPM, causing many beats to match to wrong onsets.
- High mean absolute error (>100 ms) makes it less useful for single-note timing feedback.
- Backtracking (`backtrack=True`) helps but cannot compensate for fundamentally wrong
  onset candidates at the global level.

### Frame-energy derivative (`energy` column)

- Numpy-only; computes first-order difference of per-frame peak amplitude.
- Bass sustain triggers repeated detections: **58 unmatched onsets at 60 BPM**.
- Every decay curve, fretting noise, and harmonic bloom registers as a new onset.
- The default threshold (0.02) is too sensitive for sustained bass; raising it loses real onsets.
- No windowing means count-in events contaminate the match pool.

### Live windowed detector (`live` column)

- Analyzes one extraction window per beat; no cross-beat contamination.
- Rise-based threshold suppresses sustained resonance within the window.
- Zero unmatched onsets (each window produces at most one onset report).
- Consistently lowest absolute error across all tested tempos.

---

## 5. Current Decision

**Keep the live rise-based detector.  Do not switch to librosa spectral-flux or the
energy-derivative detector for live feedback.**

Rationale:
- The per-window architecture ensures one detection attempt per beat with no cross-beat
  leakage.  Global detectors cannot replicate this without a separate windowing step that
  would make them equivalent to the live approach anyway.
- The rise-based threshold eliminates the −30 ms floor artifact that dominated the
  original detector.
- Absolute error of ~25–28 ms is within acceptable range for a bass practice tool, where
  the human perceptual JND for rhythm is roughly 20–30 ms.

---

## 6. Known Limitation: Residual Early Bias

After the fix, a **mean signed error of −15 to −22 ms** persists.  This is a systematic
early bias, not random noise.

**Likely causes (not yet confirmed):**

1. **Attack onset vs. perceived note center:** the detector fires at the first sample that
   crosses the dynamic threshold, which is the very beginning of the attack transient.
   The perceptual "beat" of a plucked bass note aligns closer to the energy peak (a few
   milliseconds later).  This gap appears as an early bias even when the player is on time.

2. **Threshold crossing during the attack ramp:** the threshold is crossed as amplitude is
   still rising.  A higher `rise_ratio` or a short confirmation delay could move the
   reported onset to a more representative point, but both changes risk false negatives on
   weaker attacks.

3. **Pre-roll interaction:** the 15 ms baseline window must fit entirely within the 30 ms
   pre-roll.  If the attack begins very close to the window start (e.g., early playing or
   a long attack transient), the baseline measurement picks up some of the attack energy,
   raising the threshold and delaying detection, but this effect is small and inconsistent.

The residual bias is **stable** — it does not increase with tempo and does not cause
misclassifications (no onsets reported as misses due to the bias alone).

---

## 7. Future Work

### Per-device / per-player calibration offset

A static latency offset could be measured once and subtracted from all reported errors.
This would shift the mean error toward 0 ms without changing detection logic.  Appropriate
if the bias proves consistent across multiple sessions and players.

Implementation sketch:
- Add `--latency-offset-ms` flag to `live_feedback_demo.py`.
- Apply offset when computing `timing_error_s` in `_evaluation_to_result`.
- Store measured offset in a per-device config file.

### Richer onset detector

Consider replacing or augmenting the rise-based detector only if production session logs
show:
- Persistent false misses on beats that the player confirmed they played.
- Mean absolute error consistently above 40 ms across multiple sessions.
- Systematic mis-scoring that correlates with playing style (e.g., slap vs. fingerstyle).

Candidates if a richer detector becomes necessary:
- Phase-deviation onset detection (more robust to slowly rising attacks).
- Per-window spectral flux (keeps the windowing isolation, adds frequency-domain
  sensitivity).
- Adaptive `rise_ratio` based on per-session noise floor estimation.

None of these are warranted until real session data shows the current detector is
insufficient for meaningful feedback.
