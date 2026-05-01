import sys
import json
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import math

# ---- Match your main program ----
SAMPLE_RATE = 48000
BLOCK_SIZE = 512

SMOOTHING_BLOCKS = 4
MIN_RMS = 0.018
REFRACTORY_MS = 180
METRONOME_BPM = 60
MIN_SPACING_FRACTION_OF_BEAT = 0.5
USE_TEMPO_SPACING = True

DECAY_TAU_SEC = 0.8
ENERGY_BREAK_RATIO = 2.2
ATTACK_TRACK_SEC = 0.04

# ---- Pitch diagnostic ----
PYIN_HOP_SIZE = BLOCK_SIZE
PYIN_FRAME_LENGTH = 4096          # ~85 ms at 48 kHz; covers 3+ cycles of low E1
PITCH_CHANGE_SEMITONES = 1.0
PITCH_STABLE_FRAMES = 2

# MIDI reference ticks for bass range (open strings + octave markers)
_PITCH_TICKS = [28, 33, 38, 43, 48, 52, 57]   # E1 A1 D2 G2 C3 E3 A3

def midi_to_note(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(m) % 12]}{int(m) // 12 - 1}"


def compute_pitch(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("B0"),
        fmax=librosa.note_to_hz("C5"),
        sr=SAMPLE_RATE,
        hop_length=PYIN_HOP_SIZE,
        frame_length=PYIN_FRAME_LENGTH,
    )
    pitch_times = librosa.times_like(f0, sr=SAMPLE_RATE, hop_length=PYIN_HOP_SIZE)
    safe_f0 = np.where((f0 is not None) & (f0 > 0), f0, 1.0)
    midi = np.where(voiced_flag & (f0 > 0), librosa.hz_to_midi(safe_f0), np.nan)
    return pitch_times, midi


def find_pitch_changes(pitch_times: np.ndarray, midi: np.ndarray) -> list[tuple]:
    """Return (time, from_midi, to_midi) for stable pitch transitions >= 1 semitone."""
    rounded = np.full_like(midi, np.nan)
    valid = ~np.isnan(midi)
    rounded[valid] = np.round(midi[valid])

    changes = []
    n = len(rounded)
    for i in range(1, n - PITCH_STABLE_FRAMES):
        if np.isnan(rounded[i - 1]) or np.isnan(rounded[i]):
            continue
        if abs(rounded[i] - rounded[i - 1]) < PITCH_CHANGE_SEMITONES:
            continue
        stable = all(
            i + j < n
            and not np.isnan(rounded[i + j])
            and abs(rounded[i + j] - rounded[i]) < PITCH_CHANGE_SEMITONES
            for j in range(PITCH_STABLE_FRAMES)
        )
        if stable:
            changes.append((pitch_times[i], int(rounded[i - 1]), int(rounded[i])))
    return changes


def load_target_times(path: Path) -> list[float]:
    with open(path) as f:
        data = json.load(f)
    return sorted(t["time"] for t in data)


def load_audio(path):
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y

def compute_rms(y):
    n = len(y)
    times = []
    rms = []

    for i in range(0, n, BLOCK_SIZE):
        block = y[i:i+BLOCK_SIZE]
        if len(block) == 0:
            continue
        val = np.sqrt(np.mean(block**2))
        rms.append(val)
        times.append(i / SAMPLE_RATE)

    return np.array(times), np.array(rms)

def smooth(rms):
    smoothed = []
    for i in range(len(rms)):
        start = max(0, i - SMOOTHING_BLOCKS + 1)
        smoothed.append(np.mean(rms[start:i+1]))
    return np.array(smoothed)

def simulate_decay(times, smoothed):
    # state mirrors decay_break globals in onset_pitch_realtime.py exactly
    previous_smoothed_rms = None
    last_peak_energy = 0.0
    last_peak_time = -999.0
    attack_tracking = False
    attack_start_time = 0.0
    attack_peak_energy = 0.0
    last_onset_time = -999.0

    beat_dur = 60.0 / METRONOME_BPM
    min_spacing_sec = (max(REFRACTORY_MS / 1000.0, MIN_SPACING_FRACTION_OF_BEAT * beat_dur)
                       if USE_TEMPO_SPACING else REFRACTORY_MS / 1000.0)

    predicted = []
    threshold = []
    slopes = []
    detected_onsets = []
    decay_pass_times = []
    slope_pass_times = []

    for i in range(len(times)):
        t = times[i]
        val = smoothed[i]

        slope = 0.0 if previous_smoothed_rms is None else val - previous_smoothed_rms

        # compute expected decay from committed peak (same as realtime)
        if last_peak_energy != 0.0:
            expected = last_peak_energy * math.exp(-(t - last_peak_time) / DECAY_TAU_SEC)
        else:
            expected = 0.0

        predicted.append(expected)
        threshold.append(expected * ENERGY_BREAK_RATIO)
        slopes.append(slope)

        # decay condition pass: track regardless of attack/spacing for visualization
        if last_peak_energy != 0.0 and val >= MIN_RMS and val > expected * ENERGY_BREAK_RATIO:
            decay_pass_times.append(t)
        if val >= MIN_RMS and slope > 0:
            slope_pass_times.append(t)

        spaced = (t - last_onset_time) > min_spacing_sec
        onset_detected = False

        if attack_tracking:
            attack_peak_energy = max(attack_peak_energy, val)
            if t - attack_start_time >= ATTACK_TRACK_SEC:
                last_peak_energy = attack_peak_energy
                last_peak_time = attack_start_time
                attack_tracking = False
        else:
            strong_enough = val >= MIN_RMS
            rising = slope > 0

            if last_peak_energy == 0.0:
                if strong_enough and rising and spaced:
                    onset_detected = True
            else:
                breaks_decay = val > expected * ENERGY_BREAK_RATIO
                if strong_enough and rising and breaks_decay and spaced:
                    onset_detected = True

            if onset_detected:
                attack_tracking = True
                attack_start_time = t
                attack_peak_energy = val

        if onset_detected:
            last_onset_time = t
            detected_onsets.append(t)

        previous_smoothed_rms = val

    return (np.array(predicted), np.array(threshold), np.array(slopes),
            detected_onsets, decay_pass_times, slope_pass_times)

def plot(times, smoothed, predicted, onsets, threshold, slopes, decay_pass_times, slope_pass_times,
         pitch_times=None, midi=None, pitch_changes=None, target_times=None):

    have_pitch = pitch_times is not None and midi is not None
    if have_pitch:
        _, (ax1, ax3) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        _, ax1 = plt.subplots(figsize=(12, 6))
        ax3 = None

    # --- energy panel (unchanged) ---
    ax1.plot(times, smoothed, label="Smoothed RMS", linewidth=2)
    ax1.plot(times, predicted, label="Expected Decay", linestyle="--")
    ax1.plot(times, threshold, label=f"Threshold (decay × {ENERGY_BREAK_RATIO})", linestyle=":", color="orange", linewidth=1.5)

    if decay_pass_times:
        decay_vals = np.interp(decay_pass_times, times, smoothed)
        ax1.scatter(decay_pass_times, decay_vals, marker="^", color="green",
                    s=40, label="Decay cond passes", zorder=5, alpha=0.7)
    if slope_pass_times:
        slope_vals = np.interp(slope_pass_times, times, smoothed)
        ax1.scatter(slope_pass_times, slope_vals, marker="v", color="purple",
                    s=20, label="Slope cond passes", zorder=4, alpha=0.4)

    for t in onsets:
        ax1.axvline(t, color="red", linestyle=":", linewidth=1.2)
    if onsets:
        onset_vals = np.interp(onsets, times, smoothed)
        ax1.scatter(onsets, onset_vals, marker="D", color="red", s=70,
                    label="Detected Onset", zorder=6)

    ax1.axhline(MIN_RMS, linestyle="--", color="gray", linewidth=1, label="MIN_RMS")

    ax2 = ax1.twinx()
    ax2.plot(times, slopes, color="gray", linewidth=0.8, alpha=0.45, label="Slope")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("Slope", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title("Decay Model vs Signal")
    ax1.set_ylabel("Energy")
    if ax3 is None:
        ax1.set_xlabel("Time (s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # --- pitch panel (experimental) ---
    if ax3 is not None:
        voiced = ~np.isnan(midi)
        ax3.plot(pitch_times[voiced], midi[voiced], ".", color="steelblue",
                 markersize=3, label="pyin pitch")

        if pitch_changes:
            ax3.scatter(
                [c[0] for c in pitch_changes],
                [c[2] for c in pitch_changes],
                marker="|", color="darkorange", s=100, linewidths=2,
                label="Pitch change", zorder=5,
            )

        for t in onsets:
            ax3.axvline(t, color="red", linestyle=":", linewidth=1.0)
        if onsets:
            ax3.axvline(np.nan, color="red", linestyle=":", linewidth=1.0, label="Detected onset")

        if target_times:
            for t in target_times:
                ax3.axvline(t, color="green", linestyle="--", linewidth=0.8)
            ax3.axvline(np.nan, color="green", linestyle="--", linewidth=0.8, label="Target")

        ax3.set_yticks(_PITCH_TICKS)
        ax3.set_yticklabels([midi_to_note(m) for m in _PITCH_TICKS], fontsize=7)
        ax3.set_ylim(24, 64)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pitch")
        ax3.set_title("Pitch — pyin (experimental)")
        ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_decay_model.py path/to/file.wav [path/to/targets.json]")
        return

    path = Path(sys.argv[1])
    target_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    y = load_audio(path)

    times, rms = compute_rms(y)
    smoothed = smooth(rms)
    pitch_times, midi = compute_pitch(y)

    predicted, threshold, slopes, onsets, decay_pass_times, slope_pass_times = simulate_decay(times, smoothed)
    pitch_changes = find_pitch_changes(pitch_times, midi)
    target_times = load_target_times(target_path) if target_path else None

    print("Detected onsets:")
    for t in onsets:
        print(f"  {t:.3f}s")

    print("\nPitch changes (>= 1 semitone, stable >= 2 frames):")
    for t, from_m, to_m in pitch_changes:
        print(f"  {t:.3f}s  {midi_to_note(from_m)} -> {midi_to_note(to_m)}")

    plot(times, smoothed, predicted, onsets, threshold, slopes, decay_pass_times, slope_pass_times,
         pitch_times=pitch_times, midi=midi, pitch_changes=pitch_changes, target_times=target_times)

if __name__ == "__main__":
    main()