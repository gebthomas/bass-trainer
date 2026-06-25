#!/usr/bin/env python3
"""Song Grid Practice — experimental free-form timing mode.

Play along with a commercial recording (e.g. Spotify on a phone) while the
laptop records only the bass input.  Timing is evaluated against a beat grid
derived from user taps, not from a fixed exercise target list.

Workflow
--------
1. **Tap phase** — listen to the song and tap a key for 16–32 beats.
   BPM and phase are estimated from the taps.
2. **Recording phase** — play bass along with the song.  Each detected onset
   is matched to the nearest grid point and its timing error is logged.
3. **Diagnostics** — session stats are printed and optionally saved.

Usage examples
--------------
    # Default: eighth-note grid, straight feel
    python scripts/song_grid_practice.py

    # Shuffle groove
    python scripts/song_grid_practice.py --feel shuffle

    # Quarter-note pulse only
    python scripts/song_grid_practice.py --subdivision 1

    # Sixteenth-note grid, save log and audio
    python scripts/song_grid_practice.py --subdivision 4 \\
        --save-log sessions/my_session.json \\
        --save-audio sessions/my_session.wav

    # Override latency
    python scripts/song_grid_practice.py --latency-ms 120.0

    # Disable latency compensation
    python scripts/song_grid_practice.py --no-calibration

Time base
---------
All session times are in seconds from the moment the audio stream opens
(``session_wall_start``).

    t_raw       = sample_pos / sample_rate          (WAV position of onset)
    t_comp      = t_raw - latency_ms/1000           (latency-compensated onset)
    phase_s     = tap_anchor_wall - session_wall_start  (negative; taps before recording)
    error_s, _  = nearest_grid_error_s(t_comp, bpm, phase_s, ...)

Events are logged as TARGET_HIT with ``time_sec = t_comp`` and
``value = error_s``.  This is the same convention as the metronome practice
mode, so existing log tools (session_log_summary, practice_replay_viewer) can
read these sessions.
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio_calibration import load_input_latency
from core.onset_adapter import OnsetAdapter
from core.session_log import (
    TARGET_HIT,
    SessionEvent,
    SessionLog,
    append_event,
    save_session_log_file,
)
from pocket_lab.tap_grid import (
    estimate_bpm_from_taps,
    grid_session_stats,
    grid_unit_name,
    nearest_grid_error_s,
)

# ── Audio constants ────────────────────────────────────────────────────────────

_SAMPLE_RATE = 44100
_BLOCK_SIZE  = 512
_CHANNELS    = 1


# ── Latency helpers ────────────────────────────────────────────────────────────

def _resolve_latency(args: argparse.Namespace) -> tuple[float, str]:
    if args.no_calibration:
        return 0.0, "--no-calibration"
    if args.latency_ms is not None:
        return args.latency_ms, "--latency-ms"
    return load_input_latency(), "audio_calibration.json"


def _compensate(onset_times_s: list[float], latency_ms: float) -> list[float]:
    if not onset_times_s:
        return []
    offset_s = latency_ms / 1000.0
    return [t - offset_s for t in onset_times_s]


# ── Audio save ─────────────────────────────────────────────────────────────────

def _save_audio(audio: np.ndarray, path: str) -> None:
    try:
        import soundfile as sf
        sf.write(path, audio, _SAMPLE_RATE)
        print(f"  Audio saved → {path}")
    except ImportError:
        try:
            from scipy.io import wavfile
            wavfile.write(path, _SAMPLE_RATE,
                          (audio * 32767).astype(np.int16))
            print(f"  Audio saved → {path}")
        except ImportError:
            print("  soundfile/scipy not available — audio not saved.",
                  file=sys.stderr)


# ── Tap capture ────────────────────────────────────────────────────────────────

def _tap_capture(min_taps: int = 8, max_taps: int = 32) -> list[float]:
    """Collect tap timestamps in raw terminal mode.

    The user taps any key for each beat.  Pressing ``q`` or Enter stops
    collection once ``min_taps`` have been recorded.

    Returns
    -------
    list[float]
        Wall-clock timestamps (from ``time.perf_counter()``) for each tap.
    """
    print()
    print("  ── Tap phase ───────────────────────────────────────────")
    print(f"  Tap any key for each beat ({min_taps}–{max_taps} taps).")
    print("  Press 'q' or Enter when done (minimum taps required first).")
    print()

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    taps: list[float] = []

    try:
        tty.setcbreak(fd)
        while len(taps) < max_taps:
            ch = sys.stdin.read(1)
            if ch in ('q', '\n', '\r', '\x03'):       # q / Enter / Ctrl-C
                if len(taps) >= min_taps:
                    break
                print(f"\r  Need at least {min_taps} taps (have {len(taps)}).  "
                      "Keep tapping…   ", end='', flush=True)
                continue
            taps.append(time.perf_counter())
            print(f"\r  Taps: {len(taps)}", end='', flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print()
    return taps


# ── Recording loop ─────────────────────────────────────────────────────────────

def _run_session(
    bpm: float,
    phase_session_s: float,
    subdivision: int,
    feel: str,
    latency_ms: float,
    save_audio_path: str | None,
    device: int | None,
    threshold: float,
) -> tuple[SessionLog, np.ndarray | None]:
    """Open the audio stream, collect onsets, and return log + optional audio."""
    import sounddevice as sd

    log = SessionLog(
        schema_version=1,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    adapter    = OnsetAdapter(sample_rate=_SAMPLE_RATE,
                               min_rms=threshold, min_peak=threshold * 6)
    lock       = threading.Lock()
    onset_queue: list[float] = []
    audio_chunks: list[np.ndarray] = []
    sample_pos = [0]
    _capture   = save_audio_path is not None

    def callback(indata: np.ndarray, frames: int,
                 time_info, status) -> None:
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        if _capture:
            audio_chunks.append(mono.copy())
        block_start = sample_pos[0]
        sample_pos[0] += frames
        onsets = adapter.process_block(block_start, mono)
        if onsets:
            with lock:
                onset_queue.extend(onsets)

    print()
    print("  ── Recording ───────────────────────────────────────────")
    print("  Playing bass now. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_BLOCK_SIZE,
            channels=_CHANNELS,
            device=device,
            callback=callback,
            dtype='float32',
        ):
            sample_pos[0] = 0
            try:
                while True:
                    time.sleep(0.010)
                    with lock:
                        raw_onsets = list(onset_queue)
                        onset_queue.clear()

                    for t_comp in _compensate(raw_onsets, latency_ms):
                        error_s, grid_idx = nearest_grid_error_s(
                            t_comp, bpm, phase_session_s, subdivision, feel
                        )
                        append_event(log, SessionEvent(
                            time_sec     = max(0.0, t_comp),
                            event_type   = TARGET_HIT,
                            target_index = max(0, grid_idx),
                            value        = error_s,
                        ))

            except KeyboardInterrupt:
                pass

    except Exception as exc:
        print(f"\n  Audio stream error: {exc}", file=sys.stderr)

    log.ended_at = datetime.now(timezone.utc).isoformat()

    audio: np.ndarray | None = None
    if _capture and audio_chunks:
        audio = np.concatenate(audio_chunks)

    return log, audio


# ── Diagnostics ────────────────────────────────────────────────────────────────

def _print_stats(stats: dict, bpm: float, n_intervals: int,
                 unit: str, latency_ms: float) -> None:
    print()
    print("  ── Session summary ─────────────────────────────────────")
    print(f"  BPM              : {bpm:.1f}")
    print(f"  Grid             : {unit}")
    print(f"  Tap intervals    : {n_intervals}")
    print(f"  Latency comp.    : {latency_ms:.1f} ms")
    print(f"  Bass onsets      : {stats['n_onsets']}")
    if stats['n_onsets'] == 0:
        print("  No onsets detected.")
        return
    print(f"  Mean (signed)    : {stats['mean_signed_ms']:+.1f} ms")
    print(f"  Mean (abs)       : {stats['mean_abs_ms']:.1f} ms")
    print(f"  Std dev          : {stats['std_ms']:.1f} ms")
    print(f"  Within ±30 ms    : {stats['pct_within_30ms']:.0f}%")
    print(f"  Within ±60 ms    : {stats['pct_within_60ms']:.0f}%")
    print(f"  Within ±100 ms   : {stats['pct_within_100ms']:.0f}%")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Song Grid Practice — tap BPM, then record bass timing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--subdivision', type=int, choices=[1, 2, 4], default=2,
        help='Grid subdivision: 1=quarter, 2=eighth, 4=sixteenth',
    )
    parser.add_argument(
        '--feel', choices=['straight', 'shuffle'], default='straight',
        help='Rhythmic feel (only differs from straight at subdivision=2)',
    )
    parser.add_argument(
        '--taps', type=int, default=32, metavar='N',
        help='Maximum number of beats to tap',
    )
    parser.add_argument(
        '--min-taps', type=int, default=8, metavar='N',
        help='Minimum taps before stop is allowed',
    )

    lat = parser.add_mutually_exclusive_group()
    lat.add_argument(
        '--latency-ms', type=float, default=None, dest='latency_ms',
        metavar='MS',
        help='Input latency override (ms)',
    )
    lat.add_argument(
        '--no-calibration', action='store_true',
        help='Disable latency compensation (0 ms)',
    )

    parser.add_argument(
        '--save-log', type=str, default=None, metavar='PATH',
        help='Write session log JSON to PATH',
    )
    parser.add_argument(
        '--save-audio', type=str, default=None, metavar='PATH',
        help='Write recorded audio WAV to PATH',
    )
    parser.add_argument(
        '--device', type=int, default=None,
        help='sounddevice input device index',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.018, metavar='RMS',
        help='Onset detection RMS threshold',
    )

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    latency_ms, latency_src = _resolve_latency(args)
    unit = grid_unit_name(args.subdivision, args.feel)

    print()
    print("  Bass Trainer — Song Grid Practice (experimental)")
    print(f"  Grid      : {unit}  (subdivision={args.subdivision}, feel={args.feel})")
    print(f"  Latency   : {latency_ms:.1f} ms  ({latency_src})")
    print()
    print("  Instructions")
    print("  ─────────────────────────────────────────────────────────")
    print("  1. Start the song on your external device.")
    print("  2. When ready, tap along with the beat (tap phase below).")
    print("  3. After tapping, play your bass line — recording starts")
    print("     automatically after the tap phase.")
    print()

    # ── Phase 1: tap capture ──────────────────────────────────────────────────
    tap_times: list[float] = []
    bpm = 0.0
    tap_anchor_wall = 0.0
    n_intervals = 0

    while True:
        tap_times = _tap_capture(min_taps=args.min_taps, max_taps=args.taps)
        try:
            bpm, tap_anchor_wall, n_intervals = estimate_bpm_from_taps(tap_times)
        except ValueError as exc:
            print(f"\n  BPM estimation failed: {exc}")
            print("  Please tap again.\n")
            continue

        print(f"\n  Estimated BPM    : {bpm:.1f}")
        print(f"  Tap intervals    : {n_intervals}")
        print(f"  Grid             : {unit}")
        print()
        print("  Press Enter to start recording, or 'r' + Enter to re-tap.")
        choice = input("  > ").strip().lower()
        if choice != 'r':
            break
        print()

    # ── Phase 2: recording ────────────────────────────────────────────────────
    #
    # The tap anchor is a wall-clock beat reference. When the audio stream
    # opens, we compute phase_session_s = tap_anchor_wall - session_wall_start.
    # Since tapping precedes recording, phase_session_s is always negative.
    # nearest_grid_error_s handles negative phase values correctly.
    #
    # We can't capture session_wall_start before the stream opens, so we pass
    # tap_anchor_wall to the recording function and let it compute the phase.

    session_wall_start = time.perf_counter()   # approximate; refined inside

    # Open the audio stream and record
    import sounddevice as sd

    log = SessionLog(
        schema_version=1,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    adapter    = OnsetAdapter(sample_rate=_SAMPLE_RATE,
                               min_rms=args.threshold,
                               min_peak=args.threshold * 6)
    lock       = threading.Lock()
    onset_queue: list[float] = []
    audio_chunks: list[np.ndarray] = []
    sample_pos = [0]
    _capture   = args.save_audio is not None

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        if _capture:
            audio_chunks.append(mono.copy())
        block_start = sample_pos[0]
        sample_pos[0] += frames
        onsets = adapter.process_block(block_start, mono)
        if onsets:
            with lock:
                onset_queue.extend(onsets)

    print()
    print("  ── Recording ───────────────────────────────────────────")
    print("  Play your bass line now. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_BLOCK_SIZE,
            channels=_CHANNELS,
            device=args.device,
            callback=callback,
            dtype='float32',
        ):
            session_wall_start = time.perf_counter()
            sample_pos[0] = 0

            # Phase in session time (negative: taps happened before recording)
            phase_session_s = tap_anchor_wall - session_wall_start

            try:
                while True:
                    time.sleep(0.010)
                    with lock:
                        raw_onsets = list(onset_queue)
                        onset_queue.clear()

                    for t_comp in _compensate(raw_onsets, latency_ms):
                        error_s, grid_idx = nearest_grid_error_s(
                            t_comp,
                            bpm,
                            phase_session_s,
                            args.subdivision,
                            args.feel,
                        )
                        append_event(log, SessionEvent(
                            time_sec     = max(0.0, t_comp),
                            event_type   = TARGET_HIT,
                            target_index = max(0, grid_idx),
                            value        = error_s,
                        ))

            except KeyboardInterrupt:
                pass

    except Exception as exc:
        print(f"\n  Audio stream error: {exc}", file=sys.stderr)
        sys.exit(1)

    log.ended_at = datetime.now(timezone.utc).isoformat()

    # ── Phase 3: diagnostics ──────────────────────────────────────────────────

    errors_ms = [
        ev.value * 1000.0
        for ev in log.events
        if ev.event_type == TARGET_HIT and ev.value is not None
    ]
    stats = grid_session_stats(errors_ms)

    _print_stats(stats, bpm, n_intervals, unit, latency_ms)

    # Persist stats in log metrics for downstream tools
    if stats['n_onsets'] > 0:
        log.metrics.update({k: round(v, 4) for k, v in stats.items()
                            if isinstance(v, float)})
        log.metrics['n_onsets'] = stats['n_onsets']

    # Persist session parameters in log metadata
    log.metadata.update({
        'mode':          'song_grid',
        'bpm':           str(round(bpm, 3)),
        'bpm_source':    'tap',
        'n_tap_intervals': str(n_intervals),
        'phase_session_s': str(round(phase_session_s, 6)),
        'subdivision':   str(args.subdivision),
        'feel':          args.feel,
        'grid_unit':     unit,
        'latency_ms':    str(latency_ms),
        'latency_src':   latency_src,
    })

    if args.save_log:
        Path(args.save_log).parent.mkdir(parents=True, exist_ok=True)
        save_session_log_file(log, args.save_log)
        print(f"  Log saved → {args.save_log}")

    if args.save_audio and audio_chunks:
        audio = np.concatenate(audio_chunks)
        _save_audio(audio, args.save_audio)


if __name__ == '__main__':
    main()
