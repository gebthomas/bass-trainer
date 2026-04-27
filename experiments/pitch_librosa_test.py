import librosa
import numpy as np
import math

FILENAME = "bass_ch1_test.wav"

def hz_to_note(freq):
    midi = round(69 + 12 * math.log2(freq / 440.0))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}", midi

print("Analyzing:", FILENAME)

audio, sr = librosa.load(FILENAME, sr=None, mono=True)

f0, voiced_flag, voiced_prob = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz("E1"),
    fmax=librosa.note_to_hz("G4"),
    sr=sr,
    frame_length=4096,
    hop_length=512,
)

times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

last_midi = None

for t, freq in zip(times, f0):
    if np.isnan(freq):
        continue

    note, midi = hz_to_note(freq)

    if midi != last_midi:
        print(f"{t:6.3f} s | {freq:7.2f} Hz | {note} | MIDI {midi}")
        last_midi = midi