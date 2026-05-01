import numpy as np
import librosa
import math


def hz_to_note(freq):
    midi = round(69 + 12 * math.log2(freq / 440.0))
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}", midi


def note_to_hz(note_name: str) -> float:
    return float(librosa.note_to_hz(note_name))


def cents_between(detected_hz: float, target_hz: float) -> float:
    return 1200.0 * math.log2(detected_hz / target_hz)


def estimate_pitch(segment):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        segment,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("G4"),
        sr=48000,
        frame_length=4096,
        hop_length=512,
    )

    valid = f0[~np.isnan(f0)]
    if len(valid) == 0:
        return None, None, None

    freq = float(np.median(valid))
    note, midi = hz_to_note(freq)

    if len(valid) > 1:
        cents = 1200.0 * np.log2(valid / freq)
        stability_cents = float(np.percentile(cents, 75) - np.percentile(cents, 25))
    else:
        stability_cents = 0.0

    return freq, note, stability_cents
