import json
import math
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 48000
AMPLITUDE = 0.45
DEFAULT_DURATION_SEC = 0.35

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS_DIR = PROJECT_ROOT / "tests" / "targets"
AUDIO_DIR = PROJECT_ROOT / "tests" / "audio"


def note_to_hz(note: str) -> float:
    names = {
        "C": 0, "C#": 1, "D": 2, "D#": 3,
        "E": 4, "F": 5, "F#": 6, "G": 7,
        "G#": 8, "A": 9, "A#": 10, "B": 11,
    }

    if len(note) == 2:
        name = note[0]
        octave = int(note[1])
    else:
        name = note[:2]
        octave = int(note[2])

    midi = 12 * (octave + 1) + names[name]
    return 440.0 * (2 ** ((midi - 69) / 12))


def envelope(n_samples: int) -> np.ndarray:
    env = np.ones(n_samples)

    attack = int(0.015 * SAMPLE_RATE)
    release = int(0.080 * SAMPLE_RATE)

    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)

    if release > 0 and release < n_samples:
        env[-release:] = np.linspace(1, 0, release)

    return env


def synth_note(freq: float, duration_sec: float) -> np.ndarray:
    n = int(duration_sec * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    # Slight harmonic content: more bass-like than pure sine
    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.35 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    )

    wave = wave / np.max(np.abs(wave))
    wave = wave * envelope(n)
    return (AMPLITUDE * wave).astype(np.float32)


def load_targets(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_audio(targets, output_path: Path, timing_offset_sec=0.0,
                   missing_indices=None, extra_notes=None):
    missing_indices = set(missing_indices or [])
    extra_notes = extra_notes or []

    events = []

    for i, target in enumerate(targets):
        if i in missing_indices:
            continue

        events.append({
            "time": target["time"] + timing_offset_sec,
            "note": target["note"],
            "duration": target.get("duration", DEFAULT_DURATION_SEC),
        })

    for extra in extra_notes:
        events.append({
            "time": extra["time"],
            "note": extra["note"],
            "duration": extra.get("duration", DEFAULT_DURATION_SEC),
        })

    events.sort(key=lambda e: e["time"])

    total_duration = max(e["time"] + e["duration"] for e in events) + 1.0
    audio = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)

    for event in events:
        freq = note_to_hz(event["note"])
        note_audio = synth_note(freq, event["duration"])
        start = int(event["time"] * SAMPLE_RATE)
        end = start + len(note_audio)

        audio[start:end] += note_audio[: len(audio[start:end])]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, SAMPLE_RATE, audio)
    print(f"Saved {output_path}")


def generate_all():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    slow = load_targets(TARGETS_DIR / "slow_quarter.json")
    pent = load_targets(TARGETS_DIR / "pentatonic_60.json")

    generate_audio(slow, AUDIO_DIR / "slow_perfect.wav")
    generate_audio(slow, AUDIO_DIR / "slow_late_150ms.wav", timing_offset_sec=0.150)
    generate_audio(slow, AUDIO_DIR / "slow_early_100ms.wav", timing_offset_sec=-0.100)
    generate_audio(slow, AUDIO_DIR / "slow_missed_first.wav", missing_indices=[0])
    generate_audio(
        slow,
        AUDIO_DIR / "slow_extra_between.wav",
        extra_notes=[{"time": 2.65, "note": "D2"}],
    )

    generate_audio(pent, AUDIO_DIR / "pentatonic_perfect.wav")
    generate_audio(pent, AUDIO_DIR / "pentatonic_late_100ms.wav", timing_offset_sec=0.100)


if __name__ == "__main__":
    generate_all()