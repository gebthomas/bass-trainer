#!/usr/bin/env python3
"""Render a WAV backing track from a chord progression JSON."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_RATE = 44100

NOTE_SEMITONES = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

CHORD_INTERVALS = {
    'maj7':  [0, 4, 7, 11],
    'm7b5':  [0, 3, 6, 10],
    'dim7':  [0, 3, 6, 9],
    'sus4':  [0, 5, 7],
    'sus2':  [0, 2, 7],
    'maj':   [0, 4, 7],
    'dim':   [0, 3, 6],
    'aug':   [0, 4, 8],
    'm7':    [0, 3, 7, 10],
    'm6':    [0, 3, 7, 9],
    '7':     [0, 4, 7, 10],
    '6':     [0, 4, 7, 9],
    'm':     [0, 3, 7],
    '':      [0, 4, 7],
}

# Jazz guide-tone voicings (3rd + 7th), chosen for minimal voice movement
# across the ii–V–I:  F3 is common to Dm7/G7; B3 is common to G7/Cmaj7
GUIDE_TONE_VOICINGS = {
    'Dm7':   [65, 72],   # F4 (m3),  C5 (m7)
    'G7':    [65, 71],   # F4 (m7),  B4 (M3)
    'Cmaj7': [64, 71],   # E4 (M3),  B4 (M7)
}


# ── Pitch helpers ─────────────────────────────────────────────────────────────

def _midi_to_freq(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _parse_root_suffix(name):
    for n in (2, 1):
        root = name[:n]
        if root in NOTE_SEMITONES:
            suffix = name[n:]
            if suffix not in CHORD_INTERVALS:
                raise ValueError(f"Unknown chord quality {suffix!r} in {name!r}")
            return root, suffix
    raise ValueError(f"Cannot parse chord root in {name!r}")


def _generic_midis(root, suffix, prev_midis):
    """Root-position chord MIDI notes, voice-led from prev_midis when given."""
    intervals  = CHORD_INTERVALS[suffix]
    root_midi  = 48 + NOTE_SEMITONES[root]   # anchor at C3

    if prev_midis is None:
        return [root_midi + iv for iv in intervals]

    # Greedy nearest-neighbour voice leading: each previous voice pulls its
    # nearest available chord tone (any octave, MIDI 36–84).
    used   = set()
    result = []
    for prev in sorted(prev_midis):
        best, best_dist = None, 999
        for iv in intervals:
            base = root_midi + iv
            for shift in range(-2, 3):
                candidate = base + shift * 12
                if 36 <= candidate <= 84 and candidate not in used:
                    dist = abs(candidate - prev)
                    if dist < best_dist:
                        best_dist = dist
                        best = candidate
        if best is not None:
            used.add(best)
            result.append(best)
    return result


def _chord_midis(name, prev_midis=None):
    """Return MIDI note list for a chord voicing."""
    if name in GUIDE_TONE_VOICINGS:
        return list(GUIDE_TONE_VOICINGS[name])
    root, suffix = _parse_root_suffix(name)
    return _generic_midis(root, suffix, prev_midis)


# ── Timing ────────────────────────────────────────────────────────────────────

def _beat_sample(beat, beat_s, sr):
    """Beat number → sample index (round, not int, for exact grid alignment)."""
    return round(beat * beat_s * sr)


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _make_click(freq, n_samples, amplitude, sr):
    t   = np.arange(n_samples) / sr
    env = np.linspace(1.0, 0.0, n_samples) ** 0.5
    return amplitude * env * np.sin(2.0 * np.pi * freq * t)


def _make_pad(freqs, n_samples, amplitude, sr):
    if n_samples == 0:
        return np.zeros(0)
    t    = np.arange(n_samples) / sr
    wave = sum(np.sin(2.0 * np.pi * f * t) for f in freqs) / len(freqs)

    attack_n  = min(round(0.05 * sr), n_samples // 4)
    release_n = min(round(0.20 * sr), n_samples // 4)
    sustain_n = max(n_samples - attack_n - release_n, 0)

    env = np.concatenate([
        np.linspace(0.0, 1.0, attack_n),
        np.ones(sustain_n),
        np.linspace(1.0, 0.0, release_n),
    ])[:n_samples]

    return amplitude * env * wave


def _mix_in(buf, signal, start_sample):
    end = min(start_sample + len(signal), len(buf))
    if end <= start_sample:
        return
    buf[start_sample:end] += signal[:end - start_sample]


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_track(progression, bpm, count_in_beats, duration_beats=None,
                 time_sig=4, pad_amp=0.3, click_amp=0.5, sr=SAMPLE_RATE):
    """
    Render backing track.

    progression entries use 'start'/'end' as beat numbers (not seconds).
    BPM is the single source of truth for all timing.
    """
    beat_s        = 60.0 / bpm
    prog_end_beat = max((seg["end"] for seg in progression), default=0.0)
    total_prog_b  = duration_beats if duration_beats is not None else prog_end_beat
    total_beats   = count_in_beats + total_prog_b
    total_n       = _beat_sample(total_beats, beat_s, sr) + 1

    buf = np.zeros(total_n)

    # Count-in clicks — same beat clock as everything else
    click_n = round(0.03 * sr)
    for i in range(count_in_beats):
        freq = 1000.0 if (i % time_sig == 0) else 750.0
        _mix_in(buf, _make_click(freq, click_n, click_amp, sr),
                _beat_sample(i, beat_s, sr))

    # Chord pads — beat-aligned, voice-led
    prev_midis = None
    for seg in progression:
        start_beat = count_in_beats + seg["start"]
        end_beat   = count_in_beats + seg["end"]
        start_n    = _beat_sample(start_beat, beat_s, sr)
        end_n      = _beat_sample(end_beat,   beat_s, sr)
        dur_n      = end_n - start_n

        midis      = _chord_midis(seg["chord"], prev_midis)
        freqs      = [_midi_to_freq(m) for m in midis]
        _mix_in(buf, _make_pad(freqs, dur_n, pad_amp, sr), start_n)
        prev_midis = midis

    # Metronome over progression — same beat clock
    beat_idx = 0
    while True:
        abs_beat = count_in_beats + beat_idx
        if abs_beat >= total_beats:
            break
        freq = 1000.0 if (beat_idx % time_sig == 0) else 750.0
        _mix_in(buf, _make_click(freq, click_n, click_amp, sr),
                _beat_sample(abs_beat, beat_s, sr))
        beat_idx += 1

    peak = np.max(np.abs(buf))
    if peak > 0.95:
        buf *= 0.95 / peak

    return buf


# ── CLI ───────────────────────────────────────────────────────────────────────

def _default_output(json_path, bpm):
    stem    = Path(json_path).stem
    out_dir = PROJECT_ROOT / "tests" / "audio" / "practice_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{int(bpm)}bpm.wav"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Render a WAV backing track from a chord progression JSON."
    )
    p.add_argument("progression",  help="Path to progression JSON")
    p.add_argument("--bpm",        type=float, default=120.0,
                   help="Tempo in BPM (default: 120)")
    p.add_argument("--count-in",   type=int,   default=4,    metavar="BEATS",
                   help="Count-in beats (default: 4)")
    p.add_argument("--duration",   type=float, default=None, metavar="BEATS",
                   help="Override total progression length in beats")
    p.add_argument("--time-sig",   type=int,   default=4,    metavar="N",
                   help="Beats per bar for downbeat accent (default: 4)")
    p.add_argument("--pad-amp",    type=float, default=0.3,
                   help="Chord pad amplitude 0–1 (default: 0.3)")
    p.add_argument("--click-amp",  type=float, default=0.5,
                   help="Metronome click amplitude 0–1 (default: 0.5)")
    p.add_argument("--output",     default=None,
                   help="Output WAV path")
    return p.parse_args()


def main():
    args = _parse_args()

    json_path = Path(args.progression)
    with open(json_path, encoding="utf-8") as fh:
        progression = json.load(fh)

    out_path = Path(args.output) if args.output else _default_output(json_path, args.bpm)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    beat_s = 60.0 / args.bpm
    print(f"Progression: {json_path.name}  ({len(progression)} chords)")
    print(f"BPM:         {args.bpm}  ({beat_s:.4f}s/beat)")
    print(f"Count-in:    {args.count_in} beats")
    if args.duration:
        print(f"Duration:    {args.duration} beats (override)")

    # Log voicings for verification
    prev_midis = None
    for seg in progression:
        midis = _chord_midis(seg["chord"], prev_midis)
        note_names = _midis_to_names(midis)
        start_s = (args.count_in + seg["start"]) * beat_s
        print(f"  beat {seg['start']:>4g} ({start_s:.3f}s)  {seg['chord']:<8} {note_names}")
        prev_midis = midis

    audio = render_track(
        progression,
        bpm            = args.bpm,
        count_in_beats = args.count_in,
        duration_beats = args.duration,
        time_sig       = args.time_sig,
        pad_amp        = args.pad_amp,
        click_amp      = args.click_amp,
    )

    sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Output:      {out_path}  ({len(audio) / SAMPLE_RATE:.2f}s)")


def _midis_to_names(midis):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    parts = []
    for m in midis:
        octave = m // 12 - 1
        parts.append(f"{names[m % 12]}{octave}")
    return " ".join(parts)


if __name__ == "__main__":
    main()
