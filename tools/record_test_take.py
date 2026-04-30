import argparse
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from realtime.metronome import Metronome

DEVICE_ID = 1
SAMPLE_RATE = 48000
CHANNELS = 1

METRONOME_ENABLED = True
METRONOME_MODE = "count_in_and_click"
METRONOME_BPM = 60
METRONOME_VOLUME = 0.35
COUNT_IN_BEATS = 4
BEATS_PER_MEASURE = 4

RECORD_DURATION_SEC = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a test take aligned to metronome PLAY time.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="record_test_take.wav",
        help="Path to save the recorded WAV file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=RECORD_DURATION_SEC,
        help="Recording duration in seconds.",
    )
    return parser.parse_args()


def compute_play_start_time() -> float:
    now = time.perf_counter()
    beat_duration = 60.0 / METRONOME_BPM
    count_in_duration = COUNT_IN_BEATS * beat_duration if METRONOME_MODE != "silent" else 0.0
    return now + count_in_duration


def timestamped_audio_callback(audio_chunks, lock, indata, frames, callback_time, status):
    if status:
        print(status)

    timestamp = time.perf_counter()

    chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
    with lock:
        audio_chunks.append((timestamp, chunk))


def assemble_recording(audio_chunks, play_start_time, duration_sec):
    required_samples = int(round(duration_sec * SAMPLE_RATE))
    output = np.zeros(required_samples, dtype=np.float32)

    for timestamp, chunk in audio_chunks:
        start_sample = int(round((timestamp - play_start_time) * SAMPLE_RATE))
        end_sample = start_sample + len(chunk)

        out_start = max(start_sample, 0)
        out_end = min(end_sample, required_samples)
        if out_end <= out_start:
            continue

        chunk_start = max(0, -start_sample)
        chunk_end = chunk_start + (out_end - out_start)
        output[out_start:out_end] = chunk[chunk_start:chunk_end]

    return output


def record_take(output_path: Path, duration: float) -> None:
    audio_chunks = []
    audio_lock = threading.Lock()
    metronome = None
    play_start_time = None

    def callback(indata, frames, callback_time, status):
        timestamped_audio_callback(audio_chunks, audio_lock, indata, frames, callback_time, status)

    stream = sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        callback=callback,
    )

    try:
        stream.start()
        play_start_time = compute_play_start_time()

        if METRONOME_ENABLED:
            metronome = Metronome(
                METRONOME_BPM,
                mode=METRONOME_MODE,
                count_in_beats=COUNT_IN_BEATS,
                volume=METRONOME_VOLUME,
                beats_per_measure=BEATS_PER_MEASURE,
            )
            metronome.start(play_start_time)

        delay = max(0.0, play_start_time - time.perf_counter())
        print(f"Recording will start in {delay:.3f} s (perf_counter={play_start_time:.3f})")
        end_time = play_start_time + duration
        while time.perf_counter() < end_time:
            time.sleep(0.01)

        with audio_lock:
            chunks = list(audio_chunks)
    finally:
        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if metronome is not None:
            try:
                metronome.stop()
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass

    recording = assemble_recording(chunks, play_start_time, duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), SAMPLE_RATE, recording)

    peak = float(np.max(np.abs(recording)))
    rms = float(np.sqrt(np.mean(recording ** 2)))

    print(f"Saved {output_path}")
    print(f"Peak: {peak:.6f}")
    print(f"RMS: {rms:.6f}")


if __name__ == "__main__":
    args = parse_args()
    output_file = Path(args.output_path)
    print("Press Enter to arm recording...")
    input()
    record_take(output_file, args.duration)
