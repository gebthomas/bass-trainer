#!/usr/bin/env python3
"""Minimal live feedback demo — prints target feedback events in real time.

Experimental script only.  No core modules are modified.

Run from the project root:
    python scripts/live_feedback_demo.py
    python scripts/live_feedback_demo.py --device 2
    python scripts/live_feedback_demo.py --device "Scarlett"
    python scripts/live_feedback_demo.py --device 2 --samplerate 44100
    python scripts/live_feedback_demo.py --level-check
    python scripts/live_feedback_demo.py --no-click
    python scripts/live_feedback_demo.py --adaptive-timing
    python scripts/live_feedback_demo.py --adaptive-timing --adaptive-window-shift 0.3
    python scripts/live_feedback_demo.py --adaptive-timing --max-window-shift-beats 0.25

Output format (fixed-grid):
    [t_now s] beat N  SEVERITY  ±XX ms  rms=...  peak=...  message

Output format (adaptive timing):
    [t_now s] beat N  SEVERITY  ±XX ms  rms=...  peak=...  message  [bpm=61.2 win=−5ms]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_pipeline import process_realtime_audio
from core.practice_session import PracticeSession
from core.tempo_tracker import TempoTracker

# ── Config ────────────────────────────────────────────────────────────────────

TARGETS = [
    {"time": 0},
    {"time": 1},
    {"time": 2},
    {"time": 3},
]

BPM          = 60
COUNT_IN     = 2
BLOCK_FRAMES = 512    # ~10 ms per block at 48 kHz
METER_SECS   = 5.0    # level-meter duration
METER_PERIOD = 0.25   # RMS/peak print interval

ACCENT_FREQ  = 880.0  # Hz — beat 1 of count-in
BEAT_FREQ    = 440.0  # Hz — remaining count-in beats
CLICK_S      = 0.05   # click duration in seconds
CLICK_AMP    = 0.7    # peak amplitude of click waveform


# ── Device helpers ────────────────────────────────────────────────────────────

def _list_input_devices() -> None:
    print("Input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            sr = int(dev["default_samplerate"])
            print(f"  [{i:2d}] {dev['name']}  (default SR: {sr} Hz)")
    print()


def _resolve_device(spec: str | None) -> tuple[int | None, int]:
    """Return (device_id, sample_rate) for *spec*.

    *spec* may be None (system default), an integer string ("2"), or a
    partial device-name substring ("Scarlett").
    """
    if spec is None:
        dev = sd.query_devices(kind="input")
        return None, int(dev["default_samplerate"])

    try:
        idx = int(spec)
        dev = sd.query_devices(idx)
        return idx, int(dev["default_samplerate"])
    except ValueError:
        pass

    needle = spec.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
            return i, int(dev["default_samplerate"])

    print(f"Warning: no input device matching {spec!r} found; using system default.",
          file=sys.stderr)
    dev = sd.query_devices(kind="input")
    return None, int(dev["default_samplerate"])


# ── Level meter ───────────────────────────────────────────────────────────────

def _run_level_meter(device, sample_rate: int) -> None:
    """Stream audio for METER_SECS and print RMS/peak every METER_PERIOD seconds."""
    print(f"Input level meter — {METER_SECS:.0f}s  (play something to verify signal)\n")

    period_samples = int(sample_rate * METER_PERIOD)
    chunk_buf      = np.zeros(period_samples, dtype=np.float64)
    chunk_pos      = 0
    total_samples  = 0
    stop_samples   = int(sample_rate * METER_SECS)

    def _print_meter(samples: np.ndarray) -> None:
        rms  = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        bar  = int(min(peak, 1.0) * 40)
        print(f"  rms={rms:.4f}  peak={peak:.4f}  |{'█' * bar}{' ' * (40 - bar)}|")

    with sd.InputStream(
        device=device, samplerate=sample_rate, channels=1,
        dtype="float32", blocksize=BLOCK_FRAMES,
    ) as stream:
        while total_samples < stop_samples:
            raw, _ = stream.read(BLOCK_FRAMES)
            mono   = raw[:, 0].astype(np.float64) if raw.ndim == 2 else raw.astype(np.float64)

            remaining = period_samples - chunk_pos
            if len(mono) >= remaining:
                chunk_buf[chunk_pos:] = mono[:remaining]
                _print_meter(chunk_buf)
                chunk_pos = 0
                leftover  = mono[remaining:]
                if len(leftover):
                    chunk_buf[:len(leftover)] = leftover
                    chunk_pos = len(leftover)
            else:
                chunk_buf[chunk_pos : chunk_pos + len(mono)] = mono
                chunk_pos += len(mono)

            total_samples += len(mono)

    print()


# ── Count-in clicks ───────────────────────────────────────────────────────────

def _make_click(freq: float, sample_rate: int) -> np.ndarray:
    """Short decaying sine-wave click, CLICK_S seconds long."""
    n = int(CLICK_S * sample_rate)
    t = np.arange(n) / sample_rate
    # Decay envelope: reaches ~4% amplitude at end of click
    envelope = np.exp(-t / (CLICK_S * 0.3))
    return (CLICK_AMP * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)


def _make_count_in_audio(bpm: float, count_in_beats: int, sample_rate: int) -> np.ndarray:
    """One click per beat; beat 0 uses ACCENT_FREQ, remaining use BEAT_FREQ."""
    beat_s        = 60.0 / bpm
    total_samples = int(count_in_beats * beat_s * sample_rate)
    audio         = np.zeros(total_samples, dtype=np.float32)

    for beat in range(count_in_beats):
        freq  = ACCENT_FREQ if beat == 0 else BEAT_FREQ
        click = _make_click(freq, sample_rate)
        start = int(beat * beat_s * sample_rate)
        end   = start + len(click)
        audio[start : min(end, total_samples)] = click[: min(len(click), total_samples - start)]

    return audio


def _play_count_in(bpm: float, count_in_beats: int, sample_rate: int) -> None:
    """Start non-blocking count-in click playback; warn on failure."""
    try:
        audio = _make_count_in_audio(bpm, count_in_beats, sample_rate)
        sd.play(audio, samplerate=sample_rate)
    except Exception as exc:
        print(f"Warning: click playback failed ({exc}); continuing silently.",
              file=sys.stderr)


# ── Feedback formatting ───────────────────────────────────────────────────────

def _format_event(event: dict, t_now: float) -> str:
    idx    = event["target_index"]
    ev     = event["evaluation"]
    sev    = event["severity"].upper()
    t_err  = event["timing_error_s"]
    ms_str = f"{t_err * 1000:+.0f} ms" if t_err is not None else "-- ms"
    msg    = "  ".join(event["messages"]) if event["messages"] else ""
    line   = (
        f"[{t_now:5.2f}s] beat {idx}  {sev:<5}  {ms_str:>7}"
        f"  rms={ev['rms']:.4f}  peak={ev['peak']:.4f}"
    )
    if msg:
        line += f"  {msg}"
    if "timing_grid" in event:
        bpm    = event["current_bpm"]
        win_ms = event["window_shift_s"] * 1000
        line  += f"  [bpm={bpm:.1f} win={win_ms:+.0f}ms]"
    return line


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live feedback demo")
    p.add_argument(
        "--device", default=None,
        help="Input device: integer index or partial name string",
    )
    p.add_argument(
        "--samplerate", type=int, default=None,
        help="Sample rate in Hz (default: device default)",
    )
    p.add_argument(
        "--level-check", action="store_true",
        help=f"Run a {METER_SECS:.0f}s input level meter before the demo",
    )
    p.add_argument(
        "--no-click", action="store_true",
        help="Disable audible count-in clicks",
    )
    p.add_argument(
        "--adaptive-timing", action="store_true",
        help="Enable adaptive tempo tracking (TempoTracker-adjusted windows and scoring)",
    )
    p.add_argument(
        "--adaptive-window-shift", type=float, default=0.5,
        metavar="FRAC",
        help="Fraction of (adjusted−nominal) gap to shift the extraction window "
             "(0=none, 1=full; default 0.5).  Only used with --adaptive-timing.",
    )
    p.add_argument(
        "--max-window-shift-beats", type=float, default=0.30,
        metavar="BEATS",
        help="Maximum window shift as a fraction of one beat (default 0.30). "
             "Only used with --adaptive-timing.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    _list_input_devices()

    device, device_sr = _resolve_device(args.device)
    sample_rate       = args.samplerate if args.samplerate is not None else device_sr

    dev_info = sd.query_devices(device if device is not None else sd.default.device[0])
    print(f"Using device [{device}]: {dev_info['name']}")
    print(f"Sample rate : {sample_rate} Hz\n")

    if args.level_check:
        _run_level_meter(device, sample_rate)

    # ── Practice session ──────────────────────────────────────────────────────

    session = PracticeSession(TARGETS, float(BPM), COUNT_IN, sample_rate)

    beat_s      = 60.0 / BPM
    count_in_s  = COUNT_IN * beat_s
    last_beat_s = count_in_s + max(t["time"] for t in TARGETS) * beat_s
    stop_s      = last_beat_s + 2.0

    max_samples = int(sample_rate * stop_s) + BLOCK_FRAMES
    buffer      = np.zeros(max_samples, dtype=np.float64)
    write_pos   = 0

    # ── Adaptive timing setup ─────────────────────────────────────────────────

    tracker            = TempoTracker(float(BPM)) if args.adaptive_timing else None
    max_window_shift_s = args.max_window_shift_beats * beat_s if args.adaptive_timing else None

    # ── Header ────────────────────────────────────────────────────────────────

    click_label = "  (clicks disabled)" if args.no_click else ""
    print(f"BPM {BPM}  |  count-in {COUNT_IN} beats  |  {len(TARGETS)} targets{click_label}")
    print(f"Count-in ends at {count_in_s:.1f}s — last target at {last_beat_s:.1f}s")
    if args.adaptive_timing:
        print(
            f"Adaptive timing ON  "
            f"shift={args.adaptive_window_shift:.2f}  "
            f"max={args.max_window_shift_beats:.2f} beats"
        )
    print("Ctrl-C to stop early.\n")

    try:
        with sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_FRAMES,
        ) as stream:
            if not args.no_click:
                _play_count_in(float(BPM), COUNT_IN, sample_rate)

            while write_pos / sample_rate < stop_s:
                raw, _overflowed = stream.read(BLOCK_FRAMES)
                mono = raw[:, 0].astype(np.float64) if raw.ndim == 2 else raw.astype(np.float64)

                n = len(mono)
                buffer[write_pos : write_pos + n] = mono
                write_pos += n

                t_now = write_pos / sample_rate
                for event in process_realtime_audio(
                    buffer[:write_pos], write_pos, session,
                    tempo_tracker=tracker,
                    adaptive_window_shift=args.adaptive_window_shift,
                    max_window_shift_s=max_window_shift_s,
                ):
                    print(_format_event(event, t_now))

    except KeyboardInterrupt:
        print("\nStopped early.")

    missed = set(range(len(TARGETS))) - session.evaluated_indices
    print(
        f"\nEvaluated {len(session.evaluated_indices)}/{len(TARGETS)} targets."
        + (f"  Not reached: beats {sorted(missed)}" if missed else "")
    )


if __name__ == "__main__":
    main()
