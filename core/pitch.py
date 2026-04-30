import numpy as np
import librosa
import math


def hz_to_note(freq):
    midi = round(69 + 12 * math.log2(freq / 440.0))
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}", midi


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
        return None, None

    freq = float(np.median(valid))
    note, midi = hz_to_note(freq)
    return freq, note
