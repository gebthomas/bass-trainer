# Beat Microscope Implementation Notes — 2026-06-19

## Summary

Added `scripts/pocket_lab_inspect.py`, the first Pocket Lab tool. It loads a stereo WAV recording, extracts the bass channel, detects onsets in a selected time segment, and generates a self-contained HTML diagnostic report (the Beat Microscope).

## CLI Example

```
python scripts/pocket_lab_inspect.py recording.wav --bpm 146
python scripts/pocket_lab_inspect.py recording.wav --bpm 120 --beats-per-measure 3
python scripts/pocket_lab_inspect.py recording.wav --bpm 146 --start 4.0 --duration 16 --shuffle-fraction 0.667
```

Flags:

- `--bass-channel` (default 0)
- `--song-channel` (default 1)
- `--bpm` (default 146)
- `--beats-per-measure` (default 4)
- `--shuffle-fraction` (default 0.667)
- `--start` (default 0)
- `--duration` (default 8)
- `--output-dir` (default diagnostics)
- `--delta` (default 0.07, librosa onset-strength threshold)

## Helper Functions

All pure and independently testable:

- `load_audio` — load a WAV file via soundfile, return samples and sample rate.
- `segment_audio` — extract a time-bounded slice from a 1-D audio array.
- `novelty_envelope` — compute onset-strength envelope via librosa.
- `detect_onsets` — spectral-flux onset detection via librosa.
- `make_grid` — build a beat grid with measure, beat, and shuffle subdivision lines. Parameterized by `beats_per_measure`, not hardcoded to 4/4.
- `classify_onset_against_grid` — classify a single onset time against the grid, returning nearest measure/beat, signed offset in ms, beat fraction, and a preliminary label (on-beat, shuffle, early, late, between).
- `render_report` — assemble the full HTML report from all computed data.

## Report Contents

The generated HTML file includes:

1. File summary (path, sample rate, BPM, beats per measure, shuffle fraction, segment range, onset count).
2. Bass waveform plot (inline SVG).
3. Onset-strength envelope plot with onset markers (inline SVG).
4. Event timeline with measure lines, beat lines, shuffle subdivision lines, and detected onset markers (inline SVG).
5. Onset classification table (time, nearest measure, nearest beat, offset in ms, beat fraction, preliminary label).
6. Embedded audio excerpt (base64-encoded WAV in an `<audio>` element).

## Tests Added

`tests/test_pocket_lab_inspect.py` — 25 tests across two classes:

- `TestMakeGrid` (14 tests): line counts, beat spacing, shuffle subdivisions, measure boundaries, and offset handling for 4/4 at 146 BPM, 3/4 at 120 BPM, and shuffle_fraction 2/3.
- `TestClassifyOnsetAgainstGrid` (11 tests): on-beat, shuffle, early, late, and between labels at both tempos; offset_ms sign semantics; beat_fraction accuracy; shuffle_fraction 2/3 classification; offset parameter.

## Current Limitations

- Onset detection is preliminary — uses librosa spectral-flux with a fixed delta threshold, which works for plucked bass but may need tuning for other styles.
- No manual annotation yet — all onsets are algorithmically detected with no way to correct or add events by hand.
- Beat grid assumes a supplied BPM and fixed phase rather than performing dynamic beat tracking. Phase alignment and tempo drift are not handled.
- Embedded audio behavior (base64-encoded WAV in an `<audio>` element) should be tested across target browsers for compatibility and performance with longer excerpts.
- Classification is grid-relative and does not yet understand musical intent — a "late" label means late relative to the grid, not necessarily musically wrong.

## Next Implementation Targets

- Manual annotation layer for correcting and adding onset events.
- Relative timing analyzer for comparing bass and song timing.
- Controlled timing-shift examples (-60 ms to +60 ms) for perceptual calibration.
- Improved audio/visual synchronization in the HTML report (playback cursor or highlight linked to the timeline).
