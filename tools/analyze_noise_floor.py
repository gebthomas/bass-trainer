import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import librosa
from scipy.signal import find_peaks

PROJECT_SR = 48000   # target resample rate; matches project-wide SAMPLE_RATE

FFT_MIN_HZ = 20.0
FFT_MAX_HZ = 500.0
FFT_N_PEAKS = 12

# Frequency bands to call out explicitly (mains hum and harmonics)
HUM_BANDS = [
    ( 58.0,  60.0, " 58–60 Hz  (mains fundamental, 60 Hz region)"),
    (118.0, 120.0, "118–120 Hz (2nd harmonic)"),
    (176.0, 180.0, "176–180 Hz (3rd harmonic)"),
]


def load_audio(path):
    try:
        audio, sr = librosa.load(str(path), sr=PROJECT_SR, mono=False)
        print(f"Sample rate: {sr} Hz  (resampled to project rate)")
    except Exception:
        audio, sr = librosa.load(str(path), sr=None, mono=False)
        print(f"Sample rate: {sr} Hz  (native — project rate unavailable)")
    return audio, sr


def fft_analysis(audio, sr):
    """Return (freqs, rel_spectrum) for the full signal with Hann window."""
    n = len(audio)
    window   = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
    max_amp  = spectrum.max()
    rel      = spectrum / max_amp if max_amp > 0 else spectrum
    return freqs, spectrum, rel


def top_peaks(freqs, spectrum, rel, n=FFT_N_PEAKS, min_hz=FFT_MIN_HZ, max_hz=FFT_MAX_HZ):
    mask   = (freqs >= min_hz) & (freqs <= max_hz)
    mfreqs = freqs[mask]
    mspec  = spectrum[mask]
    mrel   = rel[mask]

    if mspec.max() == 0:
        return []

    peaks, _ = find_peaks(mspec, prominence=mspec.max() * 0.001)
    if len(peaks) == 0:
        return []

    top_idx = np.argsort(mspec[peaks])[::-1][:n]
    return sorted(
        [(float(mfreqs[p]), float(mrel[p])) for p in peaks[top_idx]],
        key=lambda x: x[0],
    )


def band_peak(freqs, rel, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return None, None
    band_rel  = rel[mask]
    band_freq = freqs[mask]
    idx = np.argmax(band_rel)
    return float(band_freq[idx]), float(band_rel[idx])


def analyze_channel(audio, sr, label):
    duration = len(audio) / sr
    peak_amp = float(np.max(np.abs(audio)))
    rms      = float(np.sqrt(np.mean(audio ** 2)))

    freqs, spectrum, rel = fft_analysis(audio, sr)
    freq_res = sr / len(audio)

    print(f"\n{'─' * 52}")
    print(f"  Channel: {label}")
    print(f"{'─' * 52}")
    print(f"  Duration:      {duration:.3f} s")
    print(f"  Peak:          {peak_amp:.6f}")
    print(f"  RMS:           {rms:.6f}")
    print(f"  FFT resolution: {freq_res:.4f} Hz/bin")

    peaks = top_peaks(freqs, spectrum, rel)
    print(f"\n  Top {FFT_N_PEAKS} FFT peaks ({FFT_MIN_HZ:.0f}–{FFT_MAX_HZ:.0f} Hz):")
    if peaks:
        print(f"  {'#':>3}  {'Freq (Hz)':>10}  {'Rel. mag':>10}")
        for i, (freq, r) in enumerate(peaks, 1):
            print(f"  {i:>3}  {freq:>10.2f}  {r:>10.5f}")
    else:
        print("  (none found)")

    print(f"\n  Mains hum bands:")
    for lo, hi, label_band in HUM_BANDS:
        freq, r = band_peak(freqs, rel, lo, hi)
        if freq is None:
            print(f"  {label_band}  [no bins]")
        else:
            print(f"  {label_band}  peak {freq:.2f} Hz  rel mag {r:.5f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the noise floor of a WAV recording."
    )
    parser.add_argument("wav", help="Path to WAV file")
    args = parser.parse_args()

    path = Path(args.wav)
    print(f"File: {path}")

    audio, sr = load_audio(path)

    if audio.ndim == 1:
        analyze_channel(audio, sr, "mono")
    else:
        n_channels = audio.shape[0]
        print(f"Channels: {n_channels}")
        for ch in range(n_channels):
            analyze_channel(audio[ch], sr, f"channel {ch + 1}")

    print(f"\n{'─' * 52}")
    print("Done.")


if __name__ == "__main__":
    main()
