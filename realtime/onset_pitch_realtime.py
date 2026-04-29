import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import sounddevice as sd
import numpy as np
import librosa
import math
import time
import threading
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

METRONOME_ENABLED = True
METRONOME_MODE = "count_in_and_click"  # options: count_in_and_click, count_in_only, silent
METRONOME_BPM = 60
METRONOME_VOLUME = 0.35
COUNT_IN_BEATS = 4
BEATS_PER_MEASURE = 4
COUNT_IN_FIRST_BEAT_FREQ = 1200
COUNT_IN_REGULAR_BEAT_FREQ = 800
PLAY_FIRST_BEAT_FREQ = 1200
PLAY_REGULAR_BEAT_FREQ = 800

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

class Metronome:
    def __init__(self, bpm, mode="count_in_and_click", count_in_beats=4,
                 volume=0.35, beats_per_measure=4):
        self.bpm = bpm
        self.mode = mode
        self.count_in_beats = count_in_beats
        self.volume = volume
        self.beats_per_measure = beats_per_measure
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.play_start_time = None

    def start(self, play_start_time):
        self.play_start_time = play_start_time
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _run(self):
        if self.mode == "silent":
            return

        beat_duration = 60.0 / self.bpm
        count_in_duration = self.count_in_beats * beat_duration
        next_click_time = self.play_start_time - count_in_duration
        beat = 1
        click_phase = "count_in" if self.count_in_beats > 0 else "play"

        if self.count_in_beats == 0:
            print(f"\nPlay start at {self.play_start_time:.3f}s")
            print("PLAY\n")

        while not self.stop_event.is_set():
            now = time.perf_counter()

            if now >= next_click_time:
                if click_phase == "count_in":
                    if beat == 1:
                        print(f"\nCount-in at {self.bpm} BPM")
                    freq = COUNT_IN_FIRST_BEAT_FREQ if beat == 1 else COUNT_IN_REGULAR_BEAT_FREQ
                    play_click(frequency=freq, duration=0.035, volume=self.volume)

                    beat += 1
                    if beat > self.count_in_beats:
                        if self.mode == "count_in_and_click":
                            click_phase = "play"
                            beat = 1
                            print("PLAY\n")
                        else:
                            return
                else:
                    freq = PLAY_FIRST_BEAT_FREQ if beat == 1 else PLAY_REGULAR_BEAT_FREQ
                    play_click(frequency=freq, duration=0.035, volume=self.volume)
                    beat += 1
                    if beat > self.beats_per_measure:
                        beat = 1

                next_click_time += beat_duration
            else:
                time.sleep(0.001)

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

    audio = indata[:, CHANNEL_INDEX].copy()
    audio_buffer.extend(audio)

    now = time.perf_counter()
    if start_time is None or now < start_time:
        return

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    recent_energy = np.mean(energy_history) if energy_history else rms
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

current_target_index = 0
pending_onsets.clear()

play_start_time = time.perf_counter()
if METRONOME_ENABLED and METRONOME_MODE != "silent":
    beat_duration = 60.0 / METRONOME_BPM
    count_in_duration = COUNT_IN_BEATS * beat_duration if METRONOME_MODE != "silent" else 0.0
    play_start_time = time.perf_counter() + count_in_duration

start_time = play_start_time
metronome = None
if METRONOME_ENABLED:
    metronome = Metronome(
        METRONOME_BPM,
        mode=METRONOME_MODE,
        count_in_beats=COUNT_IN_BEATS,
        volume=METRONOME_VOLUME,
        beats_per_measure=BEATS_PER_MEASURE,
    )
    metronome.start(play_start_time)

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
    if METRONOME_ENABLED and metronome is not None:
        metronome.stop()
    print_summary(results)