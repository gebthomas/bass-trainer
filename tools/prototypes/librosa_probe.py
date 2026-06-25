import librosa
from pathlib import Path

audio_file = "music-library/Wagon Wheel/drums.wav"

y, sr = librosa.load(audio_file, sr=None)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

tempo = float(tempo[0])

beat_times = librosa.frames_to_time(
    beat_frames,
    sr=sr
)

print(f"Tempo: {tempo:.2f} BPM")
print(f"Beat count: {len(beat_times)}")
print(f"First beat: {beat_times[0]:.3f}")
print(f"Last beat: {beat_times[-1]:.3f}")

out = Path("music-library/Wagon Wheel/beat_times.csv")

with open(out, "w") as f:
    f.write("beat_index,beat_time_s\n")

    for i, t in enumerate(beat_times):
        f.write(f"{i},{t:.6f}\n")

print(f"Wrote {out}")