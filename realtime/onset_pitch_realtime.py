import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import sounddevice as sd
import numpy as np
import librosa
import math
import time
from collections import deque
from core.targets import load_targets, timing_error_ms, compare_note

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

PITCH_DELAY_SEC = 0.08
PITCH_WINDOW_SEC = 0.12

current_target_index = 0

start_time = None
last_onset_time = -999

energy_history = deque(maxlen=HISTORY_BLOCKS)
audio_buffer = deque(maxlen=int(SAMPLE_RATE * 2))
pending_onsets = []
results = []

def play_click(frequency=1000, duration=0.04, volume=0.4):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    click = volume * np.sin(2 * np.pi * frequency * t)
    sd.play(click, SAMPLE_RATE)
    sd.wait()

def count_in(bpm=60, beats=4):
    beat_duration = 60.0 / bpm

    print(f"\nCount-in at {bpm} BPM")
    for i in range(beats, 0, -1):
        print(f"{i}...")
        play_click(frequency=1200 if i == 1 else 800)
        time.sleep(max(0, beat_duration - 0.04))

    print("PLAY\n")

def hz_to_note(freq):
    midi = round(69 + 12 * math.log2(freq / 440.0))
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}", midi

def print_summary(results):
    if not results:
        print("No notes evaluated.")
        return

    errors = [r["timing_error_ms"] for r in results]
    pitch_accuracy = sum(r["pitch_ok"] for r in results) / len(results) * 100

    print("\nSession summary")
    print(f"Notes evaluated: {len(results)}")
    print(f"Average timing error: {np.mean(errors):+.1f} ms")
    print(f"Average absolute timing error: {np.mean(np.abs(errors)):.1f} ms")
    print(f"Pitch accuracy: {pitch_accuracy:.1f}%")

def estimate_pitch(segment):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        segment,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("G4"),
        sr=SAMPLE_RATE,
        frame_length=4096,
        hop_length=512,
    )

    valid = f0[~np.isnan(f0)]

    if len(valid) == 0:
        return None, None

    freq = float(np.median(valid))
    note, midi = hz_to_note(freq)
    return freq, note


def callback(indata, frames, callback_time, status):
    global start_time, last_onset_time

    if status:
        print(status)

    if start_time is None:
        start_time = time.perf_counter()

    audio = indata[:, CHANNEL_INDEX].copy()
    audio_buffer.extend(audio)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    recent_energy = np.mean(energy_history) if energy_history else rms

    now = time.perf_counter()
    elapsed = now - start_time
    time_since_last = (elapsed - last_onset_time) * 1000

    strong_enough = rms > MIN_RMS
    rising_fast = rms > recent_energy * RISE_RATIO
    outside_refractory = time_since_last > REFRACTORY_MS

    if strong_enough and rising_fast and outside_refractory:
        last_onset_time = elapsed
        pending_onsets.append(elapsed)
        print(f"Onset at {elapsed:7.3f} s | peak={peak:.3f} rms={rms:.3f}")

    energy_history.append(rms)


input("Press Enter when ready...")
count_in(bpm=60, beats=4)

current_target_index = 0
start_time = None
pending_onsets.clear()

print("Listening. Press Ctrl+C to stop.")
try:
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

                for onset_time in pending_onsets[:]:
                    if elapsed >= onset_time + PITCH_DELAY_SEC + PITCH_WINDOW_SEC:
                        buffer_array = np.array(audio_buffer, dtype=np.float32)

                        end_samples_back = int((elapsed - onset_time - PITCH_DELAY_SEC - PITCH_WINDOW_SEC) * SAMPLE_RATE)
                        window_samples = int(PITCH_WINDOW_SEC * SAMPLE_RATE)

                        end_index = len(buffer_array) - max(0, end_samples_back)
                        start_index = end_index - window_samples

                        if start_index >= 0:
                            segment = buffer_array[start_index:end_index]
                            freq, note = estimate_pitch(segment)
                            if freq is not None:
                                if current_target_index >= len(targets):
                                    print(f"    Extra note at {onset_time:.3f}s: no remaining target")
                                    pending_onsets.remove(onset_time)
                                    continue
                                
                                target = targets[current_target_index]
                                current_target_index += 1

                                error = timing_error_ms(onset_time, target["time"])
                                pitch_ok = compare_note(note, target["note"])

                                if abs(error) < 30:
                                    timing_label = "tight"
                                elif abs(error) < 80:
                                    timing_label = "ok"
                                else:
                                    timing_label = "off"

                                print(
                                    f"    Target {target['note']} @ {target['time']:.3f}s | "
                                    f"Played {note} @ {onset_time:.3f}s | "
                                    f"{error:+d} ms | "
                                    f"pitch {'OK' if pitch_ok else 'WRONG'} | "
                                    f"timing {timing_label}"
                                )
                                results.append({
                                    "target_note": target["note"],
                                    "target_time": target["time"],
                                    "played_note": note,
                                    "played_time": onset_time,
                                    "timing_error_ms": error,
                                    "pitch_ok": pitch_ok,
                                    "timing_label": timing_label
                                })
                            else:
                                print("    Pitch: no stable pitch detected")

                        pending_onsets.remove(onset_time)

            time.sleep(0.02)

except KeyboardInterrupt:
    print_summary(results)