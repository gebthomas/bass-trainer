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
from core.pitch import estimate_pitch
from realtime.metronome import Metronome
import sounddevice as sd

targets = load_targets("targets/targets.json")

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
SMOOTHING_BLOCKS = 8
SLOPE_THRESHOLD = 0.01
RELEASE_THRESHOLD = MIN_RMS * 1.2
FORCE_REARM_FRACTION_OF_BEAT = 0.30
MIN_SPACING_FRACTION_OF_BEAT = 0.5
SMOOTHED_LOCKOUT_SPACING_FRACTION_OF_BEAT = 0.25
USE_TEMPO_SPACING = True

PITCH_DELAY_SEC = 0.08
PITCH_WINDOW_SEC = 0.12
MATCH_WINDOW_FRACTION = 0.40
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
CALIBRATION_CONFIG_PATH = PROJECT_ROOT / "config" / "calibration.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OFFLINE_MODE = True
#OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "slow_perfect.wav"
#OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "slow_late_150ms.wav"
#OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "slow_missed_first.wav"
#OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "slow_extra_between.wav"
#OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "pentatonic_perfect.wav"
#OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "pentatonic_late_100ms.wav"
#OFFLINE_TARGET_FILE = PROJECT_ROOT / "tests" / "targets" / "slow_quarter.json"
#OFFLINE_TARGET_FILE = PROJECT_ROOT / "tests" / "targets" / "pentatonic_60.json"
OFFLINE_AUDIO_FILE = PROJECT_ROOT / "tests" / "real_audio" / "fretted_finger" / "slow_quarter_clean.wav"
OFFLINE_TARGET_FILE = PROJECT_ROOT / "tests" / "targets" / "slow_quarter.json"
APPLY_CALIBRATION_IN_OFFLINE_MODE = True
if OFFLINE_MODE:
    targets = load_targets(OFFLINE_TARGET_FILE)
CALIBRATION_TARGETS = [{"time": float(i), "note": "D2"} for i in range(0, 8)]

start_time = None
last_onset_time = -999

energy_history = deque(maxlen=HISTORY_BLOCKS)
rms_history = deque(maxlen=SMOOTHING_BLOCKS)
previous_smoothed_rms = None
active_note = False

audio_buffer = deque(maxlen=int(SAMPLE_RATE * 2))
pending_onsets = []

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
    global last_onset_time, active_note, previous_smoothed_rms

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

    else:
        raise ValueError(f"Unsupported ONSET_DETECTOR_MODE: {ONSET_DETECTOR_MODE}")

    if onset_detected:
        last_onset_time = elapsed
        with onset_lock:
            pending_onsets.append(elapsed)
        print(f"Onset at {elapsed:7.3f} s | peak={peak:.3f} rms={rms:.3f}")

    energy_history.append(rms)


def process_pending_onsets(elapsed, matcher):
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
                freq, note = estimate_pitch(segment)
                if freq is not None:
                    handled = matcher.process_onset_against_targets(
                        onset_time,
                        note,
                        timing_error_ms,
                        compare_note,
                    )
                    if handled:
                        processed_onsets.append(onset_time)
                else:
                    print("    Pitch: no stable pitch detected")

    if processed_onsets:
        with onset_lock:
            for onset_time in processed_onsets:
                if onset_time in pending_onsets:
                    pending_onsets.remove(onset_time)


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

                time.sleep(0.02)

except KeyboardInterrupt:
    interrupted = True
else:
    interrupted = False
finally:
    if not OFFLINE_MODE:
        cleanup_audio(metronome)
    matcher.finalize_remaining_targets()

    if CALIBRATION_MODE and not interrupted:
        calibration_errors = [
            r["raw_timing_error_ms"] for r in results_logger.results if "raw_timing_error_ms" in r
        ]
        run_calibration_summary(calibration_errors, save_calibration, CALIBRATION_CONFIG_PATH)
    results_logger.print_summary()
    if not CALIBRATION_MODE and results_logger.results:
        results_logger.save_csv()