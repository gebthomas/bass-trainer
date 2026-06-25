from pathlib import Path
import sys
import csv
import numpy as np
import librosa

def scalar_tempo(x):
    return float(x[0]) if hasattr(x, "__len__") else float(x)

def detect_beats(drum_file: Path):
    y, sr = librosa.load(drum_file, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return scalar_tempo(tempo), beat_times

def detect_bass_onsets(bass_file: Path):
    y, sr = librosa.load(bass_file, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
        delta=0.2,
        wait=3,
    )
    return librosa.frames_to_time(onset_frames, sr=sr)

GRIDS = {
    "quarter": np.array([0.0]),
    "eighth": np.array([0.0, 0.5]),
    "sixteenth": np.array([0.0, 0.25, 0.5, 0.75]),
    "thirtysecond": np.arange(8) / 8.0,
    "triplet": np.array([0.0, 1/3, 2/3]),
    "shuffle": np.array([0.0, 2/3]),
}

def fit_grid(bass_onsets, beat_times, grid):
    errors_ms = []

    for t in bass_onsets:
        i = np.searchsorted(beat_times, t) - 1
        if i < 0 or i >= len(beat_times) - 1:
            continue

        b0 = beat_times[i]
        b1 = beat_times[i + 1]
        beat_len = b1 - b0

        if beat_len <= 0:
            continue

        phase = (t - b0) / beat_len

        nearest_slot = grid[np.argmin(np.abs(grid - phase))]
        slot_time = b0 + nearest_slot * beat_len

        errors_ms.append((t - slot_time) * 1000.0)

    return np.array(errors_ms)

def summarize(errors):
    if len(errors) == 0:
        return None

    return {
        "n": len(errors),
        "mean_ms": float(np.mean(errors)),
        "mean_abs_ms": float(np.mean(np.abs(errors))),
        "std_ms": float(np.std(errors)),
        "within_20_ms": float(np.mean(np.abs(errors) <= 20) * 100),
        "within_40_ms": float(np.mean(np.abs(errors) <= 40) * 100),
        "within_60_ms": float(np.mean(np.abs(errors) <= 60) * 100),
    }
def thirtysecond_occupancy(bass_onsets, beat_times):
    labels = {
        0: "beat",
        1: "off-16th",
        2: "e",
        3: "off-16th",
        4: "&",
        5: "off-16th",
        6: "a",
        7: "off-16th",
    }

    counts = {i: 0 for i in range(8)}
    errors_ms = {i: [] for i in range(8)}

    for t in bass_onsets:
        i = np.searchsorted(beat_times, t) - 1
        if i < 0 or i >= len(beat_times) - 1:
            continue

        b0 = beat_times[i]
        b1 = beat_times[i + 1]
        beat_len = b1 - b0

        if beat_len <= 0:
            continue

        phase = (t - b0) / beat_len

        raw_slot = int(np.round(phase * 8))

        if raw_slot == 8:
            slot = 0
            slot_time = b1
        else:
            slot = raw_slot
            slot_time = b0 + (slot / 8.0) * beat_len
        counts[slot] += 1
        errors_ms[slot].append((t - slot_time) * 1000.0)

    total = sum(counts.values())

    print()
    print("32nd-slot occupancy:")
    print("slot  label       count   pct     mean_err_ms")

    for slot in range(8):
        n = counts[slot]
        pct = 100 * n / total if total else 0
        mean_err = np.mean(errors_ms[slot]) if errors_ms[slot] else 0.0

        print(
            f"{slot}/8   {labels[slot]:10s} "
            f"{n:5d}  {pct:5.1f}%  {mean_err:10.1f}"
        )

    eighth_count = counts[0] + counts[4]
    sixteenth_count = counts[0] + counts[2] + counts[4] + counts[6]
    off_16th_count = counts[1] + counts[3] + counts[5] + counts[7]

    print()
    print(f"Eighth-grid occupancy:     {eighth_count:5d}  {100 * eighth_count / total:5.1f}%")
    print(f"Sixteenth-grid occupancy:  {sixteenth_count:5d}  {100 * sixteenth_count / total:5.1f}%")
    print(f"Off-16th 32nd occupancy:   {off_16th_count:5d}  {100 * off_16th_count / total:5.1f}%")

    return counts

def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/groove_grid_analysis.py "music-library/Mr Brightside"')

    song_dir = Path(sys.argv[1])
    drum_file = song_dir / "drums.wav"
    bass_file = song_dir / "bass.wav"

    tempo, beat_times = detect_beats(drum_file)
    bass_onsets = detect_bass_onsets(bass_file)

    print(f"Song: {song_dir.name}")
    print(f"Tempo: {tempo:.2f} BPM")
    print(f"Beats: {len(beat_times)}")
    print(f"Bass onsets: {len(bass_onsets)}")
    print()

    rows = []

    for name, grid in GRIDS.items():
        errors = fit_grid(bass_onsets, beat_times, grid)
        s = summarize(errors)
        if s is None:
            continue

        rows.append((name, s))
        print(
            f"{name:10s} "
            f"n={s['n']:4d}  "
            f"mean={s['mean_ms']:7.1f} ms  "
            f"mean_abs={s['mean_abs_ms']:6.1f} ms  "
            f"std={s['std_ms']:6.1f} ms  "
            f"±20={s['within_20_ms']:5.1f}%  "
            f"±40={s['within_40_ms']:5.1f}%  "
            f"±60={s['within_60_ms']:5.1f}%"
        )
        
    thirtysecond_occupancy(bass_onsets, beat_times)
    best = min(rows, key=lambda x: x[1]["mean_abs_ms"])
    print()
    print(f"Best grid by mean absolute error: {best[0]}")

    out = song_dir / "groove_grid_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid", "n", "mean_ms", "mean_abs_ms", "std_ms", "within_20_ms", "within_40_ms", "within_60_ms"])
        for name, s in rows:
            writer.writerow([
                name,
                s["n"],
                f"{s['mean_ms']:.3f}",
                f"{s['mean_abs_ms']:.3f}",
                f"{s['std_ms']:.3f}",
                f"{s['within_20_ms']:.1f}",
                f"{s['within_40_ms']:.1f}",
                f"{s['within_60_ms']:.1f}",
            ])

    print(f"Wrote {out}")

if __name__ == "__main__":
    main()