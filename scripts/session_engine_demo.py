#!/usr/bin/env python3
"""Disposable live demo: OnsetAdapter + SessionEngine + TempoTracker.

Sounddevice read-loop only — no callbacks, no threads, no pitch detection.
Plays an audible count-in, then listens for bass attacks and prints timing
feedback to the console using the sample clock as the sole time reference.

Run from the project root:
    python scripts/session_engine_demo.py
    python scripts/session_engine_demo.py --device 2
    python scripts/session_engine_demo.py --device "Scarlett"
    python scripts/session_engine_demo.py --bpm 100
    python scripts/session_engine_demo.py --debug
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.onset_adapter import OnsetAdapter
from core.session_engine import SessionEngine
from core.tempo_tracker import TempoTracker

# ── Configuration ─────────────────────────────────────────────────────────────

BLOCK_FRAMES = 1_024     # ~21 ms per block at 48 kHz
CHANNELS     = 1
DTYPE        = "float32"

BPM_DEFAULT    = 120.0
COUNT_IN_BEATS = 2       # audible only — played before the stream opens

# Four quarter notes. With count_in_beats=0 these land at
# 0.5 s / 1.0 s / 1.5 s / 2.0 s from stream open.
TARGETS = [
    {"time": 1, "note": "?"},
    {"time": 2, "note": "?"},
    {"time": 3, "note": "?"},
    {"time": 4, "note": "?"},
]

MIN_RMS      = 0.018
MIN_PEAK     = 0.15
REFRACTORY_S = 0.150

ACCENT_FREQ = 880.0   # Hz — first count-in click (higher pitch = beat 1)
BEAT_FREQ   = 440.0   # Hz — remaining count-in clicks
CLICK_S     = 0.040   # click duration in seconds
CLICK_AMP   = 0.7

SEVERITY_ICON = {"good": "✓", "warn": "~", "miss": "✗"}


# ── Device helpers ─────────────────────────────────────────────────────────────

def _list_input_devices() -> None:
    print("Available input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i:2d}] {dev['name']}  ({int(dev['default_samplerate'])} Hz)")
    print()


def _resolve_device(spec: str | None) -> tuple[int | None, int]:
    """Return (device_id, sample_rate). spec is None, an int string, or a name fragment."""
    if spec is None:
        dev = sd.query_devices(kind="input")
        return None, int(dev["default_samplerate"])
    try:
        idx = int(spec)
        return idx, int(sd.query_devices(idx)["default_samplerate"])
    except ValueError:
        pass
    needle = spec.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
            return i, int(dev["default_samplerate"])
    print(f"Warning: no device matching {spec!r}; using system default.", file=sys.stderr)
    dev = sd.query_devices(kind="input")
    return None, int(dev["default_samplerate"])


# ── Count-in ──────────────────────────────────────────────────────────────────

def _make_count_in(bpm: float, beats: int, sample_rate: int) -> np.ndarray:
    beat_s   = 60.0 / bpm
    total    = int(beats * beat_s * sample_rate)
    audio    = np.zeros(total, dtype=np.float32)
    click_n  = int(CLICK_S * sample_rate)
    t        = np.arange(click_n) / sample_rate
    env      = np.exp(-t / (CLICK_S * 0.3)).astype(np.float32)
    for b in range(beats):
        freq  = ACCENT_FREQ if b == 0 else BEAT_FREQ
        click = CLICK_AMP * np.sin(2 * np.pi * freq * t).astype(np.float32) * env
        s     = int(b * beat_s * sample_rate)
        end   = min(s + click_n, total)
        audio[s:end] = click[: end - s]
    return audio


# ── Feedback display ───────────────────────────────────────────────────────────

def _print_event(ev: dict, t_now: float) -> None:
    icon    = SEVERITY_ICON[ev["severity"]]
    err     = ev["timing_error_s"]
    err_str = f"{err * 1000:+.0f} ms" if err is not None else "MISS"
    msg     = "  ".join(ev["messages"]) if ev["messages"] else ""
    line    = f"  [{t_now:5.2f}s] beat {ev['target_idx']}  {icon}  {err_str:>8}"
    if msg:
        line += f"  {msg}"
    print(line)


def _print_summary(events: list[dict], tracker: TempoTracker) -> None:
    hits   = [e for e in events if e["detected_note"] is not None]
    misses = [e for e in events if e["detected_note"] is None]
    print("\n── Session summary ──────────────────────────────────")
    print(f"  Hits  : {len(hits)} / {len(events)}")
    print(f"  Misses: {len(misses)}")
    if hits:
        errs = [abs(e["timing_error_s"]) for e in hits if e["timing_error_s"] is not None]
        if errs:
            print(f"  Mean |timing error| : {sum(errs) / len(errs) * 1000:.0f} ms")
    if tracker.has_anchor:
        print(f"  Tracker BPM   : {tracker.current_tempo_bpm():.1f}")
        print(f"  Tempo ratio   : {tracker.tempo_ratio:.3f}  "
              f"({'faster' if tracker.tempo_ratio < 1 else 'slower'} than nominal)")
    else:
        print("  Tracker: no anchor (no matched targets)")
    print("─────────────────────────────────────────────────────")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SessionEngine live demo")
    p.add_argument("--device", default=None,
                   help="Input device: integer index or name substring")
    p.add_argument("--bpm", type=float, default=BPM_DEFAULT,
                   help=f"Tempo in BPM (default {BPM_DEFAULT})")
    p.add_argument("--debug", action="store_true",
                   help="Print block-level RMS/peak and onset timestamps")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    bpm  = args.bpm

    _list_input_devices()
    device, sample_rate = _resolve_device(args.device)

    dev_info = sd.query_devices(device if device is not None else sd.default.device[0])
    print(f"Device : [{device}] {dev_info['name']}")
    print(f"SR     : {sample_rate} Hz")
    print(f"BPM    : {bpm}")
    print(f"Targets: {len(TARGETS)} quarter notes at beats "
          f"{[t['time'] for t in TARGETS]}\n")

    beat_s        = 60.0 / bpm
    # One beat of headroom after the last target before auto-stop.
    session_end_s = max(t["time"] for t in TARGETS) * beat_s + beat_s + 0.5

    tracker = TempoTracker(bpm)
    engine  = SessionEngine(TARGETS, bpm=bpm, count_in_beats=0, tracker=tracker)
    adapter = OnsetAdapter(sample_rate=sample_rate,
                           min_rms=MIN_RMS, min_peak=MIN_PEAK,
                           refractory_s=REFRACTORY_S)

    # Play count-in and block until it finishes; the InputStream opens after,
    # so sample 0 = the moment the player should play beat 1.
    print(f"Count-in ({COUNT_IN_BEATS} beats at {bpm:.0f} BPM)...")
    try:
        sd.play(_make_count_in(bpm, COUNT_IN_BEATS, sample_rate), samplerate=sample_rate)
        sd.wait()
    except Exception as exc:
        print(f"  Warning: count-in failed ({exc}); continuing silently.", file=sys.stderr)

    print("Listening — play along  (Ctrl-C to stop early)\n")

    all_events: list[dict] = []
    write_pos              = 0
    block_n                = 0

    try:
        with sd.InputStream(device=device, samplerate=sample_rate,
                            channels=CHANNELS, dtype=DTYPE,
                            blocksize=BLOCK_FRAMES) as stream:
            while True:
                # Capture block-start before reading so onset times are
                # anchored to the sample at the beginning of this block.
                block_start_sample = write_pos

                block, overflowed = stream.read(BLOCK_FRAMES)
                if overflowed:
                    print(f"\n  [WARN] buffer overrun at block {block_n} "
                          f"(t={write_pos / sample_rate:.2f}s)")

                mono = block[:, 0] if block.ndim == 2 else block

                # ── debug: periodic signal-level readout ───────────────────
                if args.debug and block_n % 50 == 0:
                    rms  = float(np.sqrt(np.mean(mono ** 2)))
                    peak = float(np.max(np.abs(mono)))
                    print(f"  [dbg blk {block_n:4d}]"
                          f"  t={block_start_sample / sample_rate:.2f}s"
                          f"  rms={rms:.4f}  peak={peak:.4f}"
                          f"  bpm={tracker.current_tempo_bpm():.1f}",
                          end="\r")

                # ── onset detection ────────────────────────────────────────
                for onset_t in adapter.process_block(block_start_sample, mono):
                    if args.debug:
                        print(f"\n  [onset] t={onset_t:.3f}s", end="  ")
                    for ev in engine.on_onset(onset_t):
                        _print_event(ev, block_start_sample / sample_rate)
                        all_events.append(ev)

                # ── advance sample clock, then check for missed targets ────
                write_pos += len(mono)
                t_now      = write_pos / sample_rate

                for ev in engine.update_time(t_now):
                    _print_event(ev, t_now)
                    all_events.append(ev)

                block_n += 1

                # ── termination: timeout or all targets evaluated ──────────
                all_done = len(engine.evaluated_indices) == len(TARGETS)
                if t_now >= session_end_s or all_done:
                    break

    except KeyboardInterrupt:
        print("\nStopped early.")

    _print_summary(all_events, tracker)


if __name__ == "__main__":
    main()
