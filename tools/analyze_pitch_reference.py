import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import librosa
from scipy.signal import find_peaks

from core.targets import load_targets
from core.pitch import note_to_hz, hz_to_note, cents_between

SAMPLE_RATE = 48000
SUSTAIN_START_OFFSET = 0.30   # seconds after target time
SUSTAIN_END_OFFSET   = 1.50   # seconds after target time

FFT_N_PEAKS  = 8
FFT_MIN_HZ   = 30.0
FFT_MAX_HZ   = 2000.0

PYIN_FMIN = librosa.note_to_hz("C1")   # ~32.7 Hz — below any bass note
PYIN_FMAX = librosa.note_to_hz("G5")

OCTAVE_CONFUSION_CENTS = 40.0


# ── pyin ────────────────────────────────────────────────────────────────────

def analyze_sustain_pyin(segment):
    f0, _, _ = librosa.pyin(
        segment,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=SAMPLE_RATE,
        frame_length=4096,
        hop_length=512,
    )
    valid = f0[~np.isnan(f0)]
    if len(valid) == 0:
        return None, None, 0

    median_freq = float(np.median(valid))
    if len(valid) > 1:
        cents = 1200.0 * np.log2(valid / median_freq)
        stability = float(np.percentile(cents, 75) - np.percentile(cents, 25))
    else:
        stability = 0.0

    return median_freq, stability, len(valid)


# ── FFT ─────────────────────────────────────────────────────────────────────

def fft_peaks(segment, n_peaks=FFT_N_PEAKS, min_hz=FFT_MIN_HZ, max_hz=FFT_MAX_HZ):
    window   = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs    = np.fft.rfftfreq(len(segment), d=1.0 / SAMPLE_RATE)

    mask            = (freqs >= min_hz) & (freqs <= max_hz)
    masked_freqs    = freqs[mask]
    masked_spectrum = spectrum[mask]

    if masked_spectrum.max() == 0:
        return []

    peaks, _ = find_peaks(masked_spectrum, prominence=masked_spectrum.max() * 0.005)
    if len(peaks) == 0:
        return []

    # keep top n by amplitude, then sort by frequency for display
    top_idx = np.argsort(masked_spectrum[peaks])[::-1][:n_peaks]
    top_peaks = sorted(
        [(float(masked_freqs[peaks[i]]), float(masked_spectrum[peaks[i]])) for i in top_idx],
        key=lambda x: x[0],
    )
    return top_peaks


# ── octave confusion check ───────────────────────────────────────────────────

def octave_warnings(median_freq, expected_hz):
    if median_freq is None:
        return []
    warnings = []
    cents_sub = cents_between(median_freq, expected_hz * 0.5)
    cents_sup = cents_between(median_freq, expected_hz * 2.0)
    if abs(cents_sub) <= OCTAVE_CONFUSION_CENTS:
        warnings.append(
            f"WARNING: estimated {median_freq:.1f} Hz is within {abs(cents_sub):.0f}c of "
            f"0.5× expected ({expected_hz * 0.5:.1f} Hz) — possible octave-below confusion"
        )
    if abs(cents_sup) <= OCTAVE_CONFUSION_CENTS:
        warnings.append(
            f"WARNING: estimated {median_freq:.1f} Hz is within {abs(cents_sup):.0f}c of "
            f"2× expected ({expected_hz * 2.0:.1f} Hz) — possible octave-above confusion"
        )
    return warnings


# ── print ────────────────────────────────────────────────────────────────────

def print_target_analysis(target, audio):
    t            = target["time"]
    start_sample = int((t + SUSTAIN_START_OFFSET) * SAMPLE_RATE)
    end_sample   = int((t + SUSTAIN_END_OFFSET)   * SAMPLE_RATE)

    position = target.get("position", "")
    string   = target.get("string", "")
    location = f"  ({string} string, {position})" if string else ""

    print(f"\n{'─' * 56}")
    print(f"  Target: {target['note']}  @ {t:.3f}s{location}")
    print(f"{'─' * 56}")

    if end_sample > len(audio):
        print("  [SKIP] Recording too short to cover this target.")
        return

    segment      = audio[start_sample:end_sample]
    expected_hz  = note_to_hz(target["note"])
    expected_note = target["note"]

    # ── pyin ──
    median_freq, stability, n_voiced = analyze_sustain_pyin(segment)

    print(f"  Expected:   {expected_note:<5}  {expected_hz:.2f} Hz")

    if median_freq is None:
        print("  Estimated:  no voiced frames detected")
    else:
        est_note, _ = hz_to_note(median_freq)
        cents_err   = cents_between(median_freq, expected_hz)
        print(f"  Estimated:  {est_note:<5}  {median_freq:.2f} Hz")
        print(f"  Cents error:  {cents_err:+.1f}c")
        print(f"  Stability:    {stability:.1f}c IQR  ({n_voiced} voiced frames)")

    # ── FFT ──
    peaks = fft_peaks(segment)
    if peaks:
        max_amp = max(amp for _, amp in peaks)
        print(f"\n  FFT peaks ({FFT_MIN_HZ:.0f}–{FFT_MAX_HZ:.0f} Hz):")
        print(f"  {'#':>3}  {'Freq (Hz)':>10}  {'Ratio':>7}  {'Rel. mag':>9}")
        for i, (freq, amp) in enumerate(peaks, 1):
            ratio   = freq / expected_hz
            rel_mag = amp / max_amp
            print(f"  {i:>3}  {freq:>10.1f}  {ratio:>7.2f}  {rel_mag:>9.3f}")
    else:
        print("\n  FFT peaks: none found")

    # ── warnings ──
    for w in octave_warnings(median_freq, expected_hz):
        print(f"\n  ⚠  {w}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze sustained bass reference recordings against target JSON files."
    )
    parser.add_argument("wav",    help="Path to WAV file")
    parser.add_argument("targets", help="Path to target JSON file")
    return parser.parse_args()


def main():
    args    = parse_args()
    wav_path    = Path(args.wav)
    target_path = Path(args.targets)

    print(f"WAV:     {wav_path}")
    print(f"Targets: {target_path}")
    print(f"Sustain window: +{SUSTAIN_START_OFFSET:.2f}s to +{SUSTAIN_END_OFFSET:.2f}s after each target time")

    audio   = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)[0]
    targets = load_targets(target_path)

    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f}s  |  Targets: {len(targets)}")

    for target in targets:
        print_target_analysis(target, audio)

    print(f"\n{'─' * 56}")
    print("Done.")


if __name__ == "__main__":
    main()
