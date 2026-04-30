import sys
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt

# ---- Match your main program settings ----
SAMPLE_RATE = 48000
BLOCK_SIZE = 512

MIN_RMS = 0.018
RISE_RATIO = 1.6
REFRACTORY_MS = 180

# Optional tempo-aware spacing (set BPM to your test value)
USE_TEMPO_SPACING = True
BPM = 60
MIN_SPACING_FRACTION_OF_BEAT = 0.5  # e.g., 0.5 * beat

def load_audio(path: Path):
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y, sr

def compute_rms_blocks(y):
    n = len(y)
    num_blocks = int(np.ceil(n / BLOCK_SIZE))
    rms = []
    times = []
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        end = min((i + 1) * BLOCK_SIZE, n)
        block = y[start:end]
        if len(block) == 0:
            continue
        val = np.sqrt(np.mean(block ** 2))
        rms.append(val)
        times.append(start / SAMPLE_RATE)
    return np.array(times), np.array(rms)

def detect_onsets(times, rms):
    onsets = []
    last_onset_t = -1e9

    # compute spacing constraint
    beat_dur = 60.0 / BPM
    min_spacing_sec = max(REFRACTORY_MS / 1000.0,
                          MIN_SPACING_FRACTION_OF_BEAT * beat_dur) if USE_TEMPO_SPACING else (REFRACTORY_MS / 1000.0)

    for i in range(len(rms)):
        t = times[i]
        val = rms[i]

        recent = np.mean(rms[max(0, i-3):i+1]) if i > 0 else val

        strong_enough = val > MIN_RMS
        rising_fast = val > recent * RISE_RATIO
        spaced = (t - last_onset_t) > min_spacing_sec

        if strong_enough and rising_fast and spaced:
            onsets.append(t)
            last_onset_t = t

    return onsets

def plot_all(y, times, rms, onsets):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # waveform
    t_wave = np.arange(len(y)) / SAMPLE_RATE
    axs[0].plot(t_wave, y, linewidth=0.8)
    axs[0].set_title("Waveform")

    # RMS
    axs[1].plot(times, rms, label="RMS")
    axs[1].axhline(MIN_RMS, linestyle="--", label="MIN_RMS")

    for t in onsets:
        axs[0].axvline(t, color="r", alpha=0.6)
        axs[1].axvline(t, color="r", alpha=0.6)

    axs[1].set_title("RMS + Detected Onsets")
    axs[1].legend()

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_onsets.py path/to/file.wav")
        return

    path = Path(sys.argv[1])
    y, _ = load_audio(path)

    times, rms = compute_rms_blocks(y)
    onsets = detect_onsets(times, rms)

    print(f"Detected {len(onsets)} onsets:")
    for t in onsets:
        print(f"{t:.3f}s")

    plot_all(y, times, rms, onsets)

if __name__ == "__main__":
    main()