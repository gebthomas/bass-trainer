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
    python scripts/live_feedback_demo.py --click
    python scripts/live_feedback_demo.py --click --bpm 60 --beats 16 --count-in 4
    python scripts/live_feedback_demo.py --adaptive-timing
    python scripts/live_feedback_demo.py --adaptive-timing --adaptive-window-shift 0.3
    python scripts/live_feedback_demo.py --adaptive-timing --max-window-shift-beats 0.25
    python scripts/live_feedback_demo.py --bpm 80 --beats 8 --count-in 4
    python scripts/live_feedback_demo.py --device MiniMe --bpm 60 --beats 16 --count-in 4
    python scripts/live_feedback_demo.py --bpm 60 --beats 16 --count-in 4 --click --save-session-log sessions/

Output format (fixed-grid):
    [t_now s] beat N  SEVERITY  ±XX ms  rms=...  peak=...  message

Output format (adaptive timing):
    [t_now s] beat N  SEVERITY  ±XX ms  rms=...  peak=...  message  [bpm=61.2 win=−5ms]

Output format (--debug-onsets, printed before each evaluation line):
      → onset @4.230s  beat 0 @ 4.000s  err=+230ms  rms=0.0423  peak=0.1234
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_pipeline import process_realtime_audio
from core.log_metrics import compute_log_metrics, log_metrics_to_dict
from core.practice_session import PracticeSession
from core.session_log import (
    SCHEMA_VERSION,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
)
from core.session_store import SessionStoreConfig
from core.session_store import save_session_log as _store_save
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


# ── Click helpers (pure — importable for tests) ───────────────────────────────

def _make_click(
    freq: float,
    sample_rate: int,
    click_s: float = CLICK_S,
    click_amp: float = CLICK_AMP,
) -> np.ndarray:
    """Short decaying sine-wave click of length *click_s* seconds."""
    n        = int(click_s * sample_rate)
    t        = np.arange(n) / sample_rate
    envelope = np.exp(-t / (click_s * 0.3))   # ~4% amplitude at end
    return (click_amp * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)


def click_schedule(
    bpm: float,
    count_in: int,
    n_beats: int,
    accent_freq: float = ACCENT_FREQ,
    beat_freq: float = BEAT_FREQ,
) -> list[tuple[float, float]]:
    """Return (time_s, frequency_hz) for every click in the session.

    Covers *count_in* count-in beats followed by *n_beats* target beats.
    Beat index 0 (the very first click) uses *accent_freq*; all others use
    *beat_freq*.  Times start at 0.0 (the moment the session clock starts).

    Parameters
    ----------
    bpm         : tempo in beats per minute.
    count_in    : number of count-in beats before the first target.
    n_beats     : number of target beats to include in the schedule.
                  Pass 0 to generate a count-in-only schedule.
    accent_freq : frequency (Hz) for the first click.
    beat_freq   : frequency (Hz) for all subsequent clicks.

    Returns
    -------
    list[tuple[float, float]] — one (time_s, freq_hz) pair per click,
    sorted by ascending time.
    """
    beat_s = 60.0 / bpm
    total  = count_in + n_beats
    return [
        (i * beat_s, accent_freq if i == 0 else beat_freq)
        for i in range(total)
    ]


def render_click_track(
    schedule: list[tuple[float, float]],
    sample_rate: int,
    click_s: float = CLICK_S,
    click_amp: float = CLICK_AMP,
) -> np.ndarray:
    """Render a click schedule to a mono float32 audio array.

    Parameters
    ----------
    schedule    : list of (time_s, freq_hz) pairs as returned by
                  :func:`click_schedule`.
    sample_rate : audio sample rate in Hz.
    click_s     : duration of each individual click in seconds.
    click_amp   : peak amplitude of each click (0–1 scale).

    Returns
    -------
    np.ndarray, dtype float32.  Length covers the last click plus *click_s*
    plus a short 50 ms tail of silence.  Returns an empty array when
    *schedule* is empty.
    """
    if not schedule:
        return np.array([], dtype=np.float32)

    last_t = schedule[-1][0]
    n      = int((last_t + click_s + 0.05) * sample_rate)
    audio  = np.zeros(n, dtype=np.float32)

    for time_s, freq in schedule:
        click = _make_click(freq, sample_rate, click_s, click_amp)
        start = int(time_s * sample_rate)
        end   = min(start + len(click), n)
        audio[start:end] += click[: end - start]

    return audio


def _play_clicks(
    bpm: float,
    count_in: int,
    n_beats: int,
    sample_rate: int,
) -> None:
    """Non-blocking playback of a click track via the default output device.

    Plays *count_in* count-in clicks followed by *n_beats* target-beat clicks.
    Pass ``n_beats=0`` for a count-in-only click track (the default session
    behaviour).  Warns to stderr on failure and continues silently.
    """
    try:
        schedule = click_schedule(bpm, count_in, n_beats)
        audio    = render_click_track(schedule, sample_rate)
        sd.play(audio, samplerate=sample_rate)
    except Exception as exc:
        print(f"Warning: click playback failed ({exc}); continuing silently.",
              file=sys.stderr)


# ── Feedback formatting ───────────────────────────────────────────────────────

def _format_onset_debug(
    event: dict,
    targets: list[dict],
    count_in_s: float,
    beat_s: float,
) -> str | None:
    """Return a debug line for a detected onset, or None if no onset in this window.

    Reconstructs the absolute onset time as nominal_target_time + timing_error_s.
    Only called when --debug-onsets is active; output is printed before the
    normal evaluation line so the onset is visible before the verdict.
    """
    ev = event["evaluation"]
    if not ev.get("onset_found"):
        return None

    idx           = event["target_index"]
    target_time_s = count_in_s + targets[idx]["time"] * beat_s
    timing_err_s  = event["timing_error_s"] or 0.0
    onset_time_s  = target_time_s + timing_err_s

    return (
        f"  → onset @{onset_time_s:.3f}s"
        f"  beat {idx} @ {target_time_s:.3f}s"
        f"  err={timing_err_s * 1000:+.0f}ms"
        f"  rms={ev['rms']:.4f}"
        f"  peak={ev['peak']:.4f}"
    )


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


# ── WAV recording ─────────────────────────────────────────────────────────────

def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Clip float64 audio to [-1, 1] and scale to int16."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def _save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a float64 mono array to a 16-bit WAV file using scipy."""
    from pathlib import Path as _Path
    from scipy.io import wavfile
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, _float_to_int16(audio))
    duration_s = len(audio) / sample_rate
    print(f"Saved WAV: {path}  ({duration_s:.2f}s  {sample_rate} Hz  {len(audio)} samples)")


# ── Session log helpers (pure — importable for tests) ─────────────────────────

def event_from_pipeline_result(
    pipeline_event: dict,
    targets: list[dict],
    bpm: float,
    count_in: int,
) -> SessionEvent:
    """Convert one process_realtime_audio event dict to a SessionEvent.

    Classification
    --------------
    ``timing_error_s`` is not None → onset detected → ``TARGET_HIT``
    ``timing_error_s`` is None     → no onset       → ``TARGET_MISS``

    time_sec
    --------
    ``TARGET_HIT``:  detected onset time = target_beat_s + timing_error_s.
                     Clamped to 0.0 to satisfy the non-negative constraint in
                     the rare case the pre-roll window extends before t=0.
    ``TARGET_MISS``: nominal beat time  = target_beat_s  (most meaningful
                     reference when no onset was found).

    value
    -----
    ``TARGET_HIT``:  timing_error_s (seconds, signed; positive = late).
    ``TARGET_MISS``: None.
    """
    beat_s        = 60.0 / bpm
    idx           = pipeline_event["target_index"]
    target_beat_s = count_in * beat_s + targets[idx]["time"] * beat_s
    error_s       = pipeline_event["timing_error_s"]

    if error_s is not None:
        return SessionEvent(
            time_sec     = max(0.0, target_beat_s + error_s),
            event_type   = TARGET_HIT,
            target_index = idx,
            value        = error_s,
        )
    return SessionEvent(
        time_sec     = target_beat_s,
        event_type   = TARGET_MISS,
        target_index = idx,
        value        = None,
    )


def session_events_from_run(
    pipeline_events: list[dict],
    unevaluated_indices: set[int] | frozenset[int],
    targets: list[dict],
    bpm: float,
    count_in: int,
) -> list[SessionEvent]:
    """Build a time-sorted list of SessionEvents from a completed run.

    Parameters
    ----------
    pipeline_events     : all dicts returned by process_realtime_audio.
    unevaluated_indices : target indices that never fired a pipeline event
                          because the session ended before their readiness
                          window opened.  These become TARGET_MISS events.
    targets             : the target list used for the run.
    bpm                 : session tempo in beats per minute.
    count_in            : count-in beats before the first target.

    Returns
    -------
    list[SessionEvent] sorted ascending by time_sec.
    """
    beat_s = 60.0 / bpm
    events = [
        event_from_pipeline_result(e, targets, bpm, count_in)
        for e in pipeline_events
    ]
    for idx in sorted(unevaluated_indices):
        target_beat_s = count_in * beat_s + targets[idx]["time"] * beat_s
        events.append(SessionEvent(
            time_sec     = target_beat_s,
            event_type   = TARGET_MISS,
            target_index = idx,
            value        = None,
        ))
    events.sort(key=lambda e: e.time_sec)
    return events


def _persist_session_log(
    root_dir: str,
    pipeline_events: list[dict],
    unevaluated: set[int],
    targets: list[dict],
    bpm: float,
    count_in: int,
    started_at: str,
    ended_at: str,
    device_label: str,
) -> None:
    """Convert run results to a SessionLog, save it, and print a summary."""
    events = session_events_from_run(
        pipeline_events, unevaluated, targets, bpm, count_in,
    )
    log = SessionLog(
        schema_version = SCHEMA_VERSION,
        started_at     = started_at,
        ended_at       = ended_at,
        events         = events,
        metadata       = {
            "bpm":      str(bpm),
            "count_in": str(count_in),
            "beats":    str(len(targets)),
            "device":   device_label,
            "detector": "live_rise_based",
        },
    )
    m = compute_log_metrics(log)
    log.metrics = {
        k: v for k, v in log_metrics_to_dict(m).items() if v is not None
    }
    config = SessionStoreConfig(root_dir=root_dir)
    path   = _store_save(log, config)

    ms_s = f"{m.mean_signed_error_s * 1000:+.1f}" if m.mean_signed_error_s is not None else "—"
    ms_a = f"{m.mean_abs_error_s * 1000:.1f}"     if m.mean_abs_error_s    is not None else "—"
    print(
        f"\nSession log : {path}"
        f"\n  hits={m.targets_hit}/{m.targets_total}"
        f"  good={m.good_hits}  warn={m.warn_hits}"
        f"  missed={m.targets_missed}"
        f"  mean_err={ms_s} ms  mean|err|={ms_a} ms"
    )


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
        help="Disable all audible clicks",
    )
    p.add_argument(
        "--click", action="store_true",
        help="Play a metronome click on every beat (count-in + all targets). "
             "Default plays clicks only during count-in.",
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
    p.add_argument(
        "--bpm", type=float, default=float(BPM),
        metavar="BPM",
        help=f"Tempo in beats per minute (default {BPM})",
    )
    p.add_argument(
        "--beats", type=int, default=len(TARGETS),
        metavar="N",
        help=f"Number of targets, one per beat starting at beat 0 (default {len(TARGETS)})",
    )
    p.add_argument(
        "--count-in", type=int, default=COUNT_IN, dest="count_in",
        metavar="BEATS",
        help=f"Count-in length in beats before the first target (default {COUNT_IN})",
    )
    p.add_argument(
        "--debug-onsets", action="store_true",
        help="Print each detected onset before its evaluation: time, nearest target, "
             "signed error in ms, rms, peak",
    )
    p.add_argument(
        "--record-wav", default=None, metavar="FILE", dest="record_wav",
        help="Save the captured mono audio to a WAV file after the session ends",
    )
    p.add_argument(
        "--save-session-log", default=None, metavar="DIR", dest="save_session_log",
        help="Save evaluated feedback as a .session.json file in DIR",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = _parse_args()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    _list_input_devices()

    device, device_sr = _resolve_device(args.device)
    sample_rate       = args.samplerate if args.samplerate is not None else device_sr

    dev_info     = sd.query_devices(device if device is not None else sd.default.device[0])
    device_label = dev_info.get("name", str(args.device) if args.device is not None else "default")
    print(f"Using device [{device}]: {dev_info['name']}")
    print(f"Sample rate : {sample_rate} Hz\n")

    if args.level_check:
        _run_level_meter(device, sample_rate)

    # ── Session parameters ────────────────────────────────────────────────────

    bpm      = args.bpm
    count_in = args.count_in
    targets  = [{"time": i} for i in range(args.beats)]

    # ── Practice session ──────────────────────────────────────────────────────

    session = PracticeSession(targets, bpm, count_in, sample_rate)

    beat_s      = 60.0 / bpm
    count_in_s  = count_in * beat_s
    last_beat_s = count_in_s + max(t["time"] for t in targets) * beat_s
    stop_s      = last_beat_s + 2.0

    max_samples        = int(sample_rate * stop_s) + BLOCK_FRAMES
    buffer             = np.zeros(max_samples, dtype=np.float64)
    write_pos          = 0
    all_pipeline_events: list[dict] = []

    # ── Adaptive timing setup ─────────────────────────────────────────────────

    tracker            = TempoTracker(bpm) if args.adaptive_timing else None
    max_window_shift_s = args.max_window_shift_beats * beat_s if args.adaptive_timing else None

    # ── Header ────────────────────────────────────────────────────────────────

    if args.no_click:
        click_label = "  (clicks off)"
    elif args.click:
        click_label = "  (click: all beats)"
    else:
        click_label = "  (click: count-in only)"
    print(f"BPM {bpm:.0f}  |  count-in {count_in} beats  |  {len(targets)} targets{click_label}")
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
                n_click_beats = len(targets) if args.click else 0
                _play_clicks(bpm, count_in, n_click_beats, sample_rate)

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
                    all_pipeline_events.append(event)
                    if args.debug_onsets:
                        onset_line = _format_onset_debug(event, targets, count_in_s, beat_s)
                        if onset_line:
                            print(onset_line)
                    print(_format_event(event, t_now))

    except KeyboardInterrupt:
        print("\nStopped early.")

    ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    missed = set(range(len(targets))) - session.evaluated_indices
    print(
        f"\nEvaluated {len(session.evaluated_indices)}/{len(targets)} targets."
        + (f"  Not reached: beats {sorted(missed)}" if missed else "")
    )

    if args.record_wav and write_pos > 0:
        _save_wav(args.record_wav, buffer[:write_pos], sample_rate)

    if args.save_session_log:
        _persist_session_log(
            root_dir        = args.save_session_log,
            pipeline_events = all_pipeline_events,
            unevaluated     = missed,
            targets         = targets,
            bpm             = bpm,
            count_in        = count_in,
            started_at      = started_at,
            ended_at        = ended_at,
            device_label    = device_label,
        )


if __name__ == "__main__":
    main()
