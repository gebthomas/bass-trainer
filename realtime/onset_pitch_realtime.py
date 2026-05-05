import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import librosa
import math
import time
import threading
from collections import deque
from core.targets import load_targets, timing_error_ms, compare_note
from core.calibration import load_calibration, save_calibration, run_calibration_summary
from core.matching import TargetMatcher
from core.results import ResultsLogger
from core.pitch import estimate_pitch, note_to_hz, cents_between
from core.constraints import classify_note_against_chord, chord_at_time
from core.practice_log import append_practice_log
from realtime.metronome import Metronome
import sounddevice as sd

targets = load_targets(
    PROJECT_ROOT / "tests" / "targets" / "scales" / "major_C_60bpm_eighths.json"
)
DEVICE_ID = 1
SAMPLE_RATE = 48000
CHANNELS = 2
CHANNEL_INDEX = 0
BLOCK_SIZE = 512

MIN_RMS = 0.018
RISE_RATIO = 1.6
REFRACTORY_MS = 180
HISTORY_BLOCKS = 3

ONSET_DETECTOR_MODE = "smoothed_lockout"
SMOOTHING_BLOCKS = 4
SLOPE_THRESHOLD = 0.01
RELEASE_THRESHOLD = MIN_RMS * 1.2
FORCE_REARM_FRACTION_OF_BEAT = 0.45
MIN_SPACING_FRACTION_OF_BEAT = 0.5
SMOOTHED_LOCKOUT_SPACING_FRACTION_OF_BEAT = 0.25
USE_TEMPO_SPACING = True

DECAY_TAU_SEC = 0.8
ENERGY_BREAK_RATIO = 2.2
ATTACK_TRACK_SEC = 0.04

TARGET_WINDOW_MODE = False
TW_PRE_SEC = 0.10
TW_POST_SEC = 0.35
TW_PITCH_MATCH_CENTS = 50.0
TW_CONFIDENT_RATIO = 0.25    # min pitch_match_ratio to include in adaptive offset
TW_MIN_CONFIDENT_FOR_OFFSET = 4   # fall back to all_delays when fewer confident matches

PITCH_DELAY_SEC = 0.08
PITCH_WINDOW_SEC = 0.12
MATCH_WINDOW_FRACTION = 0.50
MIN_MATCH_WINDOW_SEC = 0.12
MAX_MATCH_WINDOW_SEC = 0.40

METRONOME_ENABLED = True
METRONOME_MODE = "count_in_and_click"  # options: count_in_and_click, count_in_only, silent
CALIBRATION_MODE = False
METRONOME_BPM = 60
METRONOME_VOLUME = 0.35
COUNT_IN_BEATS = 4
BEATS_PER_MEASURE = 4
COUNT_IN_FIRST_BEAT_FREQ = 1200
COUNT_IN_REGULAR_BEAT_FREQ = 800
PLAY_FIRST_BEAT_FREQ = 1200
PLAY_REGULAR_BEAT_FREQ = 800
TIMING_OFFSET_MS = -150
LIVE_FEEDBACK = True
AUTO_STOP_AFTER_TARGETS = True
POST_TARGET_BEATS = 2
CONSTRAINT_MODE = True
PROGRESSION_FILE = PROJECT_ROOT / "tests" / "progressions" / "ii_v_i_C.json"
CALIBRATION_CONFIG_PATH = PROJECT_ROOT / "config" / "calibration.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OFFLINE_MODE = False
OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "real_audio" / "fretless_finger" / "repeated_60_A1_fretless.wav"
DEFAULT_TARGET_FILE = PROJECT_ROOT / "tests" / "targets" / "scales" / "major_C_60bpm_eighths.json"
OFFLINE_TARGET_FILE = PROJECT_ROOT / "tests" / "targets" / "scales" / "major_C_60bpm_eighths.json"
if OFFLINE_MODE:
    targets = load_targets(OFFLINE_TARGET_FILE)
else:
    targets = load_targets(DEFAULT_TARGET_FILE)

APPLY_CALIBRATION_IN_OFFLINE_MODE = True
CALIBRATION_TARGETS = [{"time": float(i), "note": "D2"} for i in range(0, 8)]

progression = []
if CONSTRAINT_MODE:
    with open(PROGRESSION_FILE, encoding="utf-8") as _f:
        progression = json.load(_f)

start_time = None
last_onset_time = -999

energy_history = deque(maxlen=HISTORY_BLOCKS)
rms_history = deque(maxlen=SMOOTHING_BLOCKS)
previous_smoothed_rms = None
active_note = False
last_peak_energy = 0.0
last_peak_time = -999.0
attack_tracking = False
attack_start_time = 0.0
attack_peak_energy = 0.0

audio_buffer = deque(maxlen=int(SAMPLE_RATE * 2))
pending_onsets = []
constraint_counts = {"chord": 0, "scale": 0, "out": 0}
CONSTRAINT_WEIGHTS = {"chord": 2, "scale": 1, "out": -2}
constraint_score = 0

onset_lock = threading.Lock()
audio_buffer_lock = threading.Lock()

def cleanup_audio(metronome_instance=None):
    if METRONOME_ENABLED and metronome_instance is not None:
        try:
            metronome_instance.stop()
        except Exception:
            pass
    try:
        sd.stop()
    except Exception:
        pass
    time.sleep(0.2)
    print("Audio cleanup complete.")


def load_offline_audio():
    audio_data, sr = librosa.load(str(OFFLINE_AUDIO_FILE), sr=SAMPLE_RATE, mono=False)
    if audio_data.ndim > 1:
        audio_data = audio_data[CHANNEL_INDEX]
    return audio_data


def process_audio_chunk(audio, elapsed):
    global last_onset_time, active_note, previous_smoothed_rms, last_peak_energy, last_peak_time, attack_tracking, attack_start_time, attack_peak_energy

    with audio_buffer_lock:
        audio_buffer.extend(audio)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    beat_dur = 60.0 / METRONOME_BPM
    spacing_fraction = (SMOOTHED_LOCKOUT_SPACING_FRACTION_OF_BEAT
                        if ONSET_DETECTOR_MODE == "smoothed_lockout"
                        else MIN_SPACING_FRACTION_OF_BEAT)
    min_spacing_sec = max(REFRACTORY_MS / 1000.0,
                          spacing_fraction * beat_dur) if USE_TEMPO_SPACING else (REFRACTORY_MS / 1000.0)
    spaced = (elapsed - last_onset_time) > min_spacing_sec

    onset_detected = False

    if ONSET_DETECTOR_MODE == "rms_rise":
        recent_energy = np.mean(energy_history) if energy_history else rms
        strong_enough = rms > MIN_RMS
        rising_fast = rms > recent_energy * RISE_RATIO

        if strong_enough and rising_fast and spaced:
            onset_detected = True

    elif ONSET_DETECTOR_MODE == "smoothed_lockout":
        rms_history.append(rms)
        smoothed_rms = np.mean(rms_history)
        slope = 0.0 if previous_smoothed_rms is None else smoothed_rms - previous_smoothed_rms

        force_rearm_sec = FORCE_REARM_FRACTION_OF_BEAT * beat_dur
        rearmed = False
        if active_note:
            if smoothed_rms <= RELEASE_THRESHOLD:
                active_note = False
                rearmed = True
            elif (elapsed - last_onset_time) >= force_rearm_sec:
                active_note = False
                rearmed = True

        strong_enough = smoothed_rms >= MIN_RMS
        rising_fast = slope >= SLOPE_THRESHOLD

        if not active_note and strong_enough and rising_fast and spaced:
            onset_detected = True
            active_note = True

        previous_smoothed_rms = smoothed_rms

    elif ONSET_DETECTOR_MODE == "decay_break":
        rms_history.append(rms)
        smoothed_rms = np.mean(rms_history)
        slope = 0.0 if previous_smoothed_rms is None else smoothed_rms - previous_smoothed_rms

        if attack_tracking:
            attack_peak_energy = max(attack_peak_energy, smoothed_rms)
            if elapsed - attack_start_time >= ATTACK_TRACK_SEC:
                last_peak_energy = attack_peak_energy
                last_peak_time = attack_start_time
                attack_tracking = False
        else:
            strong_enough = smoothed_rms >= MIN_RMS
            rising = slope > 0

            if last_peak_energy == 0.0:
                if strong_enough and rising and spaced:
                    onset_detected = True
            else:
                expected_energy = last_peak_energy * math.exp(-(elapsed - last_peak_time) / DECAY_TAU_SEC)
                breaks_decay = smoothed_rms > expected_energy * ENERGY_BREAK_RATIO
                if strong_enough and rising and breaks_decay and spaced:
                    onset_detected = True

            if onset_detected:
                attack_tracking = True
                attack_start_time = elapsed
                attack_peak_energy = smoothed_rms

        previous_smoothed_rms = smoothed_rms

    else:
        raise ValueError(f"Unsupported ONSET_DETECTOR_MODE: {ONSET_DETECTOR_MODE}")

    if onset_detected:
        last_onset_time = elapsed
        with onset_lock:
            pending_onsets.append(elapsed)
        print(f"Onset at {elapsed:7.3f} s | peak={peak:.3f} rms={rms:.3f}")

    energy_history.append(rms)


def print_constraint_summary():
    if not CONSTRAINT_MODE:
        return
    total = sum(constraint_counts.values())
    if total == 0:
        return
    chord = constraint_counts["chord"]
    scale = constraint_counts["scale"]
    out   = constraint_counts["out"]
    spn   = constraint_score / total
    print("\nConstraint summary")
    print(f"Chord tones:  {chord} ({chord / total * 100:.1f}%)")
    print(f"Scale tones:  {scale} ({scale / total * 100:.1f}%)")
    print(f"Outside:      {out} ({out   / total * 100:.1f}%)")
    print(f"Score:        {constraint_score:+d}  ({spn:+.2f} per note)")


def get_current_chord(elapsed_time):
    return chord_at_time(progression, elapsed_time, loop=True)


def _print_live_feedback(event):
    if event["type"] == "miss":
        print(f"MISS {event['target_note']} @ {event['target_time']:.3f}s")
        return

    target  = event["target_note"]
    played  = event["played_note"]
    err     = event["corrected_error"]
    label   = event["timing_label"]
    cents   = event["cents_error"]
    stab    = event["stability_cents"]

    cents_str = f"{cents:+.0f}c" if cents != "" else "?c"
    stab_str  = f"{stab:.0f}c"  if stab  is not None else "?"

    if event["pitch_ok"]:
        print(f"HIT {target}  timing={err:+.0f}ms {label}  pitch={cents_str}  stable={stab_str}")
    else:
        print(f"WRONG {played} -> {target}  timing={err:+.0f}ms {label}  pitch={cents_str}")


def process_pending_onsets(elapsed, matcher):
    global constraint_score
    processed_onsets = []
    with onset_lock:
        pending_snapshot = list(pending_onsets)

    for onset_time in pending_snapshot:
        if elapsed >= onset_time + PITCH_DELAY_SEC + PITCH_WINDOW_SEC:
            with audio_buffer_lock:
                buffer_snapshot = list(audio_buffer)
            buffer_array = np.array(buffer_snapshot, dtype=np.float32)

            end_samples_back = int((elapsed - onset_time - PITCH_DELAY_SEC - PITCH_WINDOW_SEC) * SAMPLE_RATE)
            window_samples = int(PITCH_WINDOW_SEC * SAMPLE_RATE)

            end_index = len(buffer_array) - max(0, end_samples_back)
            start_index = end_index - window_samples

            if start_index >= 0:
                segment = buffer_array[start_index:end_index]
                freq, note, stability_cents = estimate_pitch(segment)
                if freq is not None:
                    if CONSTRAINT_MODE and not OFFLINE_MODE:
                        chord = get_current_chord(onset_time)
                        if chord:
                            classification = classify_note_against_chord(note, chord)
                            constraint_counts[classification] += 1
                            constraint_score += CONSTRAINT_WEIGHTS[classification]
                            matcher.results_logger.register_constraint(
                                onset_time, chord, classification, CONSTRAINT_WEIGHTS[classification]
                            )
                            print(f"{classification.upper():5s} {note} over {chord}")
                    handled = matcher.process_onset_against_targets(
                        onset_time,
                        note,
                        timing_error_ms,
                        compare_note,
                        detected_freq_hz=freq,
                        pitch_stability_cents=stability_cents,
                    )
                    if handled:
                        processed_onsets.append(onset_time)
                else:
                    print("    Pitch: no stable pitch detected")
                    processed_onsets.append(onset_time)

    if processed_onsets:
        with onset_lock:
            for onset_time in processed_onsets:
                if onset_time in pending_onsets:
                    pending_onsets.remove(onset_time)

    if LIVE_FEEDBACK and not OFFLINE_MODE:
        for event in matcher.recently_finalized:
            _print_live_feedback(event)
        matcher.recently_finalized.clear()


def run_target_window_mode(audio_data, targets, results_logger):
    """Pitch-window scoring: no onset detection; scan each target window with pyin."""
    print("Target-window mode: scanning pitch per target window.")

    f0, voiced_flag, _ = librosa.pyin(
        audio_data,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("G4"),
        sr=SAMPLE_RATE,
        frame_length=4096,
        hop_length=512,
    )
    frame_times = librosa.times_like(f0, sr=SAMPLE_RATE, hop_length=512)

    candidates = []   # list of (target, match_dict | None)

    for target in targets:
        t_target = target["time"]
        target_hz = note_to_hz(target["note"])

        window = (frame_times >= t_target - TW_PRE_SEC) & (frame_times <= t_target + TW_POST_SEC)
        win_times  = frame_times[window]
        win_f0     = f0[window]
        win_voiced = voiced_flag[window]

        n_frames = len(win_times)
        n_voiced = int(np.sum(win_voiced))
        voiced_ratio = n_voiced / n_frames if n_frames > 0 else 0.0

        usable      = win_voiced & ~np.isnan(win_f0) & (win_f0 > 0)
        usable_f0   = win_f0[usable]
        usable_times = win_times[usable]

        if not np.any(usable):
            candidates.append((target, None))
            continue

        cents_errors = 1200.0 * np.log2(usable_f0 / target_hz)
        match = np.abs(cents_errors) <= TW_PITCH_MATCH_CENTS
        pitch_match_ratio = float(np.sum(match)) / len(usable_f0)

        if not np.any(match):
            candidates.append((target, None))
            continue

        first_time    = float(usable_times[match][0])
        matched_f0    = usable_f0[match]
        matched_cents = cents_errors[match]
        median_freq   = float(np.median(matched_f0))
        median_cents  = float(np.median(matched_cents))
        stability     = (float(np.percentile(matched_cents, 75) - np.percentile(matched_cents, 25))
                         if len(matched_cents) > 1 else 0.0)

        candidates.append((target, {
            "first_match_time":  first_time,
            "delay_ms":          1000.0 * (first_time - t_target),
            "median_freq_hz":    median_freq,
            "median_cents_error": median_cents,
            "stability_cents":   stability,
            "pitch_match_ratio": pitch_match_ratio,
            "voiced_ratio":      voiced_ratio,
        }))

    # Adaptive offset — prefer confident matches; fall back to all matches when too few
    confident_delays = [
        c["delay_ms"] for _, c in candidates
        if c is not None and c["pitch_match_ratio"] >= TW_CONFIDENT_RATIO
    ]
    all_delays = [
        c["delay_ms"] for _, c in candidates
        if c is not None
    ]

    if len(confident_delays) >= TW_MIN_CONFIDENT_FOR_OFFSET:
        adaptive_offset_ms = float(np.median(confident_delays))
        offset_status = f"stable, from {len(confident_delays)} confident matches"
    elif all_delays:
        adaptive_offset_ms = float(np.median(all_delays))
        n_conf = len(confident_delays)
        offset_status = f"provisional, only {n_conf} confident match{'es' if n_conf != 1 else ''}"
    else:
        adaptive_offset_ms = 0.0
        offset_status = "none"

    print(f"Adaptive offset: {adaptive_offset_ms:+.0f} ms  ({offset_status})")

    for target, c in candidates:
        if c is None:
            results_logger.append_miss(
                target,
                target_window_status="missed",
                adaptive_offset_ms=adaptive_offset_ms,
            )
            print(f"  MISSED  {target['note']:>3} @ {target['time']:.3f}s")
            continue

        raw_delay_ms  = c["delay_ms"]
        adj_delay_ms  = raw_delay_ms - adaptive_offset_ms
        target_hz     = note_to_hz(target["note"])
        error_cents   = cents_between(c["median_freq_hz"], target_hz)
        tw_status     = "strong" if c["pitch_match_ratio"] >= TW_CONFIDENT_RATIO else "weak"

        if abs(adj_delay_ms) < 30:
            timing_label = "tight"
        elif abs(adj_delay_ms) < 80:
            timing_label = "ok"
        else:
            timing_label = "off"

        results_logger.append_hit(
            target,
            c["first_match_time"],
            raw_delay_ms,
            adj_delay_ms,
            True,
            timing_label,
            target["note"],
            detected_freq_hz=c["median_freq_hz"],
            target_freq_hz=target_hz,
            cents_error=error_cents,
            pitch_stability_cents=c["stability_cents"],
            pitch_match_ratio=c["pitch_match_ratio"],
            voiced_ratio=c["voiced_ratio"],
            target_window_status=tw_status,
            adaptive_offset_ms=adaptive_offset_ms,
        )
        print(
            f"  {tw_status.upper():6s} {target['note']:>3} @ {target['time']:.3f}s"
            f" | first={c['first_match_time']:.3f}s"
            f" raw={raw_delay_ms:+.0f}ms  adj={adj_delay_ms:+.0f}ms"
            f"  cents={error_cents:+.1f}  match={c['pitch_match_ratio']:.0%}"
        )


def callback(indata, frames, callback_time, status):
    global start_time, last_onset_time

    if status:
        print(status)

    audio = indata[:, CHANNEL_INDEX].copy()

    now = time.perf_counter()
    if start_time is None or now < start_time:
        return

    elapsed = now - start_time
    process_audio_chunk(audio, elapsed)


def run_offline_audio(matcher):
    audio_data = load_offline_audio()

    if TARGET_WINDOW_MODE:
        run_target_window_mode(audio_data, targets, matcher.results_logger)
        return

    num_samples = len(audio_data)
    sample_index = 0

    while sample_index < num_samples:
        chunk = audio_data[sample_index : sample_index + BLOCK_SIZE]
        elapsed = sample_index / SAMPLE_RATE
        process_audio_chunk(chunk, elapsed)
        process_pending_onsets(elapsed, matcher)

        if CALIBRATION_MODE and matcher.current_target_index >= len(targets):
            break

        sample_index += BLOCK_SIZE

    final_elapsed = num_samples / SAMPLE_RATE + PITCH_DELAY_SEC + PITCH_WINDOW_SEC
    process_pending_onsets(final_elapsed, matcher)


if not OFFLINE_MODE:
    input("Press Enter when ready...")

if OFFLINE_MODE:
    if APPLY_CALIBRATION_IN_OFFLINE_MODE:
        TIMING_OFFSET_MS = load_calibration(CALIBRATION_CONFIG_PATH, TIMING_OFFSET_MS)
        print(f"Loaded TIMING_OFFSET_MS = {TIMING_OFFSET_MS:+.1f} ms")
    else:
        TIMING_OFFSET_MS = 0
        print("Offline mode: calibration correction disabled. TIMING_OFFSET_MS = 0 ms")
else:
    TIMING_OFFSET_MS = load_calibration(CALIBRATION_CONFIG_PATH, TIMING_OFFSET_MS)
    print(f"Loaded TIMING_OFFSET_MS = {TIMING_OFFSET_MS:+.1f} ms")

if CALIBRATION_MODE:
    print("Calibration mode enabled: using 8-note 60 BPM target pattern.")
    targets = CALIBRATION_TARGETS

results_logger = ResultsLogger(RESULTS_DIR)
matcher = TargetMatcher(targets, TIMING_OFFSET_MS, results_logger)

if CONSTRAINT_MODE and progression:
    for _t in targets:
        _chord = chord_at_time(progression, _t["time"], loop=True)
        if _chord:
            results_logger._target_chord[_t["time"]] = _chord

pending_onsets.clear()

play_start_time = time.perf_counter()
if METRONOME_ENABLED and METRONOME_MODE != "silent":
    beat_duration = 60.0 / METRONOME_BPM
    count_in_duration = COUNT_IN_BEATS * beat_duration if METRONOME_MODE != "silent" else 0.0
    play_start_time = time.perf_counter() + count_in_duration

start_time = play_start_time
metronome = None
if METRONOME_ENABLED and not OFFLINE_MODE:
    metronome = Metronome(
        METRONOME_BPM,
        mode=METRONOME_MODE,
        count_in_beats=COUNT_IN_BEATS,
        volume=METRONOME_VOLUME,
        beats_per_measure=BEATS_PER_MEASURE,
    )
    metronome.start(play_start_time)

if OFFLINE_MODE:
    print("Offline mode: processing audio file immediately.")
else:
    print("Listening. Press Ctrl+C to stop.")
interrupted = False
try:
    if OFFLINE_MODE:
        run_offline_audio(matcher)
    else:
        with sd.InputStream(
            device=DEVICE_ID,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=callback,
        ):

            while True:
                now = time.perf_counter()
                if start_time is not None:
                    elapsed = now - start_time

                    process_pending_onsets(elapsed, matcher)

                    if CALIBRATION_MODE and matcher.current_target_index >= len(targets):
                        break

                    if AUTO_STOP_AFTER_TARGETS and not CALIBRATION_MODE and targets:
                        last_target_time = targets[-1]["time"]
                        beat_dur = 60.0 / METRONOME_BPM
                        if elapsed > last_target_time + POST_TARGET_BEATS * beat_dur:
                            with onset_lock:
                                no_pending = len(pending_onsets) == 0
                            if no_pending:
                                break

                time.sleep(0.02)

except KeyboardInterrupt:
    interrupted = True
else:
    interrupted = False
finally:
    if not OFFLINE_MODE:
        cleanup_audio(metronome)
    if not TARGET_WINDOW_MODE:
        matcher.finalize_remaining_targets()
        if LIVE_FEEDBACK and not OFFLINE_MODE:
            for event in matcher.recently_finalized:
                _print_live_feedback(event)
            matcher.recently_finalized.clear()

    if CALIBRATION_MODE and not interrupted:
        calibration_errors = [
            r["raw_timing_error_ms"] for r in results_logger.results if "raw_timing_error_ms" in r
        ]
        run_calibration_summary(calibration_errors, save_calibration, CALIBRATION_CONFIG_PATH)
    results_logger.print_summary()
    print_constraint_summary()
    if not CALIBRATION_MODE and results_logger.results:
        results_logger.save_csv()
        if not OFFLINE_MODE:
            append_practice_log(
                results_logger,
                PROJECT_ROOT / "data" / "practice_log.csv",
                {
                    "mode": "constraint" if CONSTRAINT_MODE else "target",
                    "bpm": METRONOME_BPM,
                    "exercise_name": PROGRESSION_FILE.name if CONSTRAINT_MODE else DEFAULT_TARGET_FILE.name,
                },
            )