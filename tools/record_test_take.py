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

RECORD_SECONDS = 36.0
PRE_ROLL_SECONDS = 1.0
OUTPUT_FILE = "record_test_take.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a test take aligned to metronome PLAY time.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=OUTPUT_FILE,
        help="Path to save the recorded WAV file.",
    )
    parser.add_argument("--bpm", type=float, default=METRONOME_BPM, help="Metronome BPM.")
    parser.add_argument("--duration", type=float, default=RECORD_SECONDS, help="Recording duration in seconds.")
    parser.add_argument("--count-in-beats", type=int, default=COUNT_IN_BEATS, help="Number of count-in beats.")
    parser.add_argument("--beats-per-measure", type=int, default=BEATS_PER_MEASURE, help="Beats per measure.")
    parser.add_argument("--metronome-mode", type=str, default=METRONOME_MODE, help="Metronome mode.")
    parser.add_argument("--metronome-volume", type=float, default=METRONOME_VOLUME, help="Metronome volume (0.0–1.0).")
    parser.add_argument("--pre-roll-seconds", type=float, default=PRE_ROLL_SECONDS, help="Seconds of audio to capture before the downbeat.")
    parser.add_argument("--allow-non-minime", action="store_true", help="Skip MiniMe device requirement check.")
    return parser.parse_args()


def check_input_device(allow_non_minime: bool) -> None:
    input_info = sd.query_devices(DEVICE_ID)
    input_name = input_info["name"]
    input_channels = input_info["max_input_channels"]
    sample_rate = input_info["default_samplerate"]

    default_output_id = sd.default.device[1]
    output_name = sd.query_devices(default_output_id)["name"] if default_output_id >= 0 else "(none)"

    print(f"Input device:     {input_name}")
    print(f"Input channels:   {input_channels}")
    print(f"Sample rate:      {int(sample_rate)} Hz")
    print(f"Output device:    {output_name}")

    if "MiniMe" not in input_name:
        print(f"WARNING: Input device '{input_name}' does not contain 'MiniMe'.")
        if not allow_non_minime:
            raise RuntimeError(
                f"Input device '{input_name}' is not the MiniMe. Use --allow-non-minime to override."
            )


def compute_play_start_time(bpm: float, count_in_beats: int, metronome_mode: str) -> float:
    now = time.perf_counter()
    beat_duration = 60.0 / bpm
    count_in_duration = count_in_beats * beat_duration if metronome_mode != "silent" else 0.0
    return now + count_in_duration


def record_take(
    output_path: Path,
    duration: float,
    bpm: float,
    count_in_beats: int,
    beats_per_measure: int,
    metronome_mode: str,
    metronome_volume: float,
    pre_roll_seconds: float,
) -> None:
    print(f"BPM:              {bpm}")
    print(f"Count-in beats:   {count_in_beats}")
    print(f"Beats per measure:{beats_per_measure}")
    print(f"Recording duration: {duration} seconds")
    print(f"Pre-roll seconds: {pre_roll_seconds}")
    print(f"Downbeat occurs at saved WAV time: {pre_roll_seconds} seconds")
    print(f"Output file: {output_path}")

    audio_chunks = []
    audio_lock = threading.Lock()
    recording_gate = threading.Event()
    callback_count = [0]
    metronome = None

    def callback(indata, _frames, _callback_time, status):
        if status:
            print(status)
        callback_count[0] += 1
        if recording_gate.is_set():
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            with audio_lock:
                audio_chunks.append(chunk)

    stream = sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        callback=callback,
    )

    try:
        stream.start()
        downbeat_time = compute_play_start_time(bpm, count_in_beats, metronome_mode)
        gate_open_time = downbeat_time - pre_roll_seconds

        if METRONOME_ENABLED:
            metronome = Metronome(
                bpm,
                mode=metronome_mode,
                count_in_beats=count_in_beats,
                volume=metronome_volume,
                beats_per_measure=beats_per_measure,
            )
            metronome.start(downbeat_time)

        delay = max(0.0, gate_open_time - time.perf_counter())
        print(f"Recording will start in {delay:.3f} s")

        # Spin until gate_open_time (downbeat - pre_roll), then open the recording gate
        while time.perf_counter() < gate_open_time:
            time.sleep(0.001)
        recording_gate.set()

        end_time = gate_open_time + duration
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

    required_samples = int(round(duration * SAMPLE_RATE))

    if chunks:
        raw = np.concatenate(chunks)
    else:
        raw = np.zeros(required_samples, dtype=np.float32)

    # Trim or zero-pad to exact duration
    if len(raw) >= required_samples:
        recording = raw[:required_samples]
    else:
        recording = np.zeros(required_samples, dtype=np.float32)
        recording[:len(raw)] = raw

    print(f"Callback count:   {callback_count[0]}")
    print(f"Total samples:    {len(raw)}")
    print(f"Expected samples: {required_samples}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), SAMPLE_RATE, recording)

    peak = float(np.max(np.abs(recording)))
    rms = float(np.sqrt(np.mean(recording ** 2)))

    print(f"Saved: {output_path}")
    print(f"Peak: {peak:.6f}")
    print(f"RMS: {rms:.6f}")


if __name__ == "__main__":
    args = parse_args()
    output_file = Path(args.output_path)
    check_input_device(args.allow_non_minime)
    print("Press Enter to arm recording...")
    input()
    record_take(
        output_file,
        duration=args.duration,
        bpm=args.bpm,
        count_in_beats=args.count_in_beats,
        beats_per_measure=args.beats_per_measure,
        metronome_mode=args.metronome_mode,
        metronome_volume=args.metronome_volume,
        pre_roll_seconds=args.pre_roll_seconds,
    )
