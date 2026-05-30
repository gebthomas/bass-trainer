#!/usr/bin/env python3
"""Real-time bass practice session demo.

Listens on a microphone input, detects note onsets with a simple energy
threshold, and scores them against a built-in exercise using SessionController.
Optionally plays metronome clicks on the output device.

Usage examples
--------------
    # List available audio input/output devices
    python scripts/practice_session_demo.py --list-devices

    # Run at 80 BPM with default settings
    python scripts/practice_session_demo.py --bpm 80

    # Slower tempo, longer count-in, specific input device
    python scripts/practice_session_demo.py --bpm 60 --count-in 4 --device 2

    # Silence clicks
    python scripts/practice_session_demo.py --no-click

    # Route clicks to a specific output device
    python scripts/practice_session_demo.py --output-device 3

    # Override latency compensation (default reads config/audio_calibration.json)
    python scripts/practice_session_demo.py --latency-ms 120.0

    # Disable latency compensation entirely
    python scripts/practice_session_demo.py --no-calibration

    # Raise threshold if spurious onsets are detected
    python scripts/practice_session_demo.py --threshold 0.03
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.audio_calibration import load_input_latency
from core.session_controller import SessionController, SessionPhase
from core.log_metrics import LogMetrics
from core.session_log import (
    TARGET_HIT,
    TARGET_MISS,
    SessionLog,
    save_session_log_file,
)

# ── Built-in exercise ─────────────────────────────────────────────────────────

_EXERCISE_NAME = "Open E"


def build_targets(n_beats: int) -> list[dict]:
    """Generate *n_beats* quarter-note E2 targets at beat positions 0 … n_beats-1."""
    return [
        {"time": float(i), "note": "E2", "label": str(i + 1)}
        for i in range(n_beats)
    ]

# ── Audio constants ───────────────────────────────────────────────────────────

_SAMPLE_RATE = 44100
_BLOCK_SIZE  = 512       # ~12 ms per block at 44100 Hz
_CHANNELS    = 1

# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_error(timing_error_s: float | None) -> str:
    if timing_error_s is None:
        return "     ----"
    ms = timing_error_s * 1000.0
    tag = "late" if ms > 0 else "early"
    return f"{abs(ms):5.0f}ms {tag}"


def _event_prefix(ev: dict) -> str:
    """Map a feedback event to a display label that matches the SessionLog event type.

    severity="miss" in feedback_events covers two distinct cases:
      - onset detected but outside the timing window → TARGET_HIT in the log (LATE)
      - no onset detected at all                    → TARGET_MISS in the log (MISS)

    timing_error_s is the reliable discriminator: it is None only when no onset
    was matched.
    """
    t_err = ev.get("timing_error_s")
    sev   = ev.get("severity", "miss")
    if t_err is None:
        return "[MISS ]"   # TARGET_MISS: no onset detected
    if sev == "good":
        return "[HIT  ]"   # TARGET_HIT: on time
    if sev == "warn":
        return "[WARN ]"   # TARGET_HIT: slightly off
    return "[LATE ]"       # TARGET_HIT: detected but outside timing window


def _print_event(ev: dict, targets: list[dict]) -> None:
    idx    = ev.get("target_idx", 0)
    label  = targets[idx].get("label", str(idx + 1))
    note   = targets[idx].get("note", "?")
    prefix = _event_prefix(ev)
    msgs   = ", ".join(ev.get("messages", []))
    err    = _fmt_error(ev.get("timing_error_s"))
    print(f"  {prefix}  Beat {label} ({note})  {err}  {msgs}")


def _print_summary(m: LogMetrics, total: int) -> None:
    rate = 100.0 * m.targets_hit / total if total else 0.0
    print()
    print("  " + "-" * 38)
    print("  Session Summary")
    print("  " + "-" * 38)
    print(f"  Hit rate :  {m.targets_hit}/{total}  ({rate:.0f}%)")
    print(f"  Good     :  {m.good_hits}")
    print(f"  Warn     :  {m.warn_hits}")
    print(f"  Missed   :  {m.targets_missed}")
    if m.mean_abs_error_s is not None:
        tends = ""
        if m.mean_signed_error_s is not None:
            tends = "  (tends {})".format(
                "late" if m.mean_signed_error_s > 0 else "early"
            )
        print(f"  Avg err  :  {m.mean_abs_error_s * 1000:.1f}ms{tends}")
    print("  " + "-" * 38)
    print()


# ── Onset detection ───────────────────────────────────────────────────────────

def _detect_onset(block: np.ndarray, prev_rms: float, threshold: float) -> tuple[bool, float]:
    """Simple energy-based onset: RMS rises above threshold and is 1.8x the prev block."""
    rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
    onset = rms > threshold and rms > prev_rms * 1.8
    return onset, rms


# ── Latency compensation ──────────────────────────────────────────────────────

def compensate_onset_times(onset_times_s: list[float], latency_ms: float) -> list[float]:
    """Return onset times shifted earlier by the known audio interface latency.

    A positive ``latency_ms`` means the audio interface delivers audio that
    many milliseconds late relative to real time.  Subtracting it from the
    raw detected onset time recovers the moment the note was actually played
    on the session grid.

    Parameters
    ----------
    onset_times_s : Raw session-relative onset times in seconds.
    latency_ms    : Latency to subtract, in milliseconds (positive = late).

    Returns
    -------
    list[float]
        A new list; the input is never mutated.
    """
    if not onset_times_s:
        return []
    offset_s = latency_ms / 1000.0
    return [t - offset_s for t in onset_times_s]


def _resolve_latency(args: argparse.Namespace) -> tuple[float, str]:
    """Return (latency_ms, source_label) from parsed CLI args.

    Priority:
      1. ``--no-calibration`` → 0 ms
      2. ``--latency-ms VALUE`` → VALUE ms
      3. default → ``load_input_latency()`` from audio_calibration.json
    """
    if args.no_calibration:
        return 0.0, "--no-calibration"
    if args.latency_ms is not None:
        return args.latency_ms, "--latency-ms"
    return load_input_latency(), "audio_calibration.json"


# ── Click generation ──────────────────────────────────────────────────────────

def make_click_waveform(
    freq: float = 1000.0,
    duration: float = 0.04,
    sample_rate: int = 44100,
    volume: float = 0.5,
) -> np.ndarray:
    """Return a sine-burst click with exponential decay as a float32 array.

    Parameters
    ----------
    freq        : Frequency in Hz of the sine tone.
    duration    : Click length in seconds.
    sample_rate : Output sample rate in Hz.
    volume      : Peak amplitude (0..1).

    Returns
    -------
    np.ndarray, dtype=float32, shape=(int(sample_rate * duration),)
    """
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    envelope = np.exp(-t / (duration * 0.25))   # fast exponential decay
    return (volume * np.sin(2.0 * np.pi * freq * t) * envelope).astype(np.float32)


def click_beat_times_s(
    bpm: float,
    count_in_beats: int,
    target_beat_positions: list[float],
) -> list[float]:
    """Return session-relative click fire times in seconds.

    Produces one click per count-in beat at 0, beat_s, 2*beat_s, …
    followed by one click per exercise target at count_in_s + pos*beat_s.

    The result is in ascending order when ``target_beat_positions`` is
    non-decreasing (which the exercise schema guarantees).

    Parameters
    ----------
    bpm                   : Tempo in BPM.
    count_in_beats        : Number of count-in beats before the exercise.
    target_beat_positions : Beat positions of exercise targets (beats after count-in).
    """
    beat_s     = 60.0 / bpm
    count_in_s = count_in_beats * beat_s
    times: list[float] = [i * beat_s for i in range(count_in_beats)]
    for pos in target_beat_positions:
        times.append(count_in_s + pos * beat_s)
    return times


class ClickScheduler:
    """Fire ``sd.play()`` clicks at pre-scheduled session times on a background thread.

    ``session_wall_start`` must be ``time.perf_counter()`` recorded at the
    moment ``sample_pos`` was zeroed, so that ``click_times_s`` offsets map
    to the correct wall-clock moments.

    sounddevice is imported lazily inside ``_run()`` so that this class can
    be constructed and tested without a sounddevice installation.
    """

    def __init__(
        self,
        click_times_s: list[float],
        session_wall_start: float,
        click_data: np.ndarray,
        sample_rate: int,
        output_device: int | None = None,
    ) -> None:
        self._times      = sorted(click_times_s)
        self._wall_start = session_wall_start
        self._click      = click_data
        self._sr         = sample_rate
        self._device     = output_device
        self._stop       = threading.Event()
        self._thread     = threading.Thread(
            target=self._run, daemon=True, name="click-scheduler"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import sounddevice as sd
        for t_s in self._times:
            if self._stop.is_set():
                return
            target_wall = self._wall_start + t_s
            # Coarse sleep to below 5 ms of target, then spin for precision
            coarse = target_wall - time.perf_counter() - 0.005
            if coarse > 0:
                time.sleep(coarse)
            while time.perf_counter() < target_wall:
                if self._stop.is_set():
                    return
            try:
                sd.play(self._click, self._sr, device=self._device, blocking=False)
            except Exception:
                pass  # don't crash the thread on transient audio errors


# ── Post-session output ───────────────────────────────────────────────────────

def _onset_plot_data(
    timing_errors: list[tuple[int, float | None]],
    target_times_s: list[float],
    latency_ms: float,
) -> tuple[list[float], list[float], list[float]]:
    """Derive onset position lists for waveform annotation.

    Parameters
    ----------
    timing_errors
        Output of ``extract_timing_errors(log)``.
    target_times_s
        Expected target times in session-relative seconds.
    latency_ms
        Latency compensation applied (positive = audio arrived late).

    Returns
    -------
    compensated_onsets_s
        Onset times after latency correction, one per detected beat.
    raw_onsets_s
        Onset times before correction (``compensated + latency_s``).
        Empty list when ``latency_ms == 0``.
    miss_target_times_s
        Expected target times where no onset was detected.
    """
    offset_s = latency_ms / 1000.0
    compensated: list[float] = []
    raw: list[float] = []
    misses: list[float] = []
    for idx, err_ms in timing_errors:
        if idx >= len(target_times_s):
            continue
        expected_s = target_times_s[idx]
        if err_ms is None:
            misses.append(expected_s)
        else:
            onset_s = expected_s + err_ms / 1000.0
            compensated.append(onset_s)
            if latency_ms != 0.0:
                raw.append(onset_s + offset_s)
    return compensated, raw, misses


def _split_click_times(
    click_times_s: list[float],
    count_in_s: float,
) -> tuple[list[float], list[float]]:
    """Partition click times into count-in and exercise groups.

    Times strictly before ``count_in_s`` are count-in beats; times at or
    after it are exercise beats.  When ``count_in_s == 0`` all clicks are
    exercise (no count-in was scheduled).

    Parameters
    ----------
    click_times_s
        Absolute click times in session-relative seconds, as returned by
        ``click_beat_times_s()``.
    count_in_s
        Duration of the count-in window in seconds (``count_in_beats * beat_s``).

    Returns
    -------
    count_in_clicks
        Click times that fall within the count-in window.
    exercise_clicks
        Click times that fall at or after ``count_in_s``.
    """
    count_in = [t for t in click_times_s if t < count_in_s]
    exercise = [t for t in click_times_s if t >= count_in_s]
    return count_in, exercise


def extract_timing_errors(log: SessionLog) -> list[tuple[int, float | None]]:
    """Return one ``(target_index, timing_error_ms)`` pair per evaluated target.

    ``TARGET_HIT`` events map to ``(index, value * 1000.0)`` — positive means late.
    ``TARGET_MISS`` events map to ``(index, None)``.
    ``EXTRA_ONSET`` events are excluded (no target to plot against).

    The result is sorted by ``target_index`` regardless of event order in the log.
    """
    entries: list[tuple[int, float | None]] = []
    for ev in log.events:
        if ev.event_type == TARGET_HIT:
            entries.append((ev.target_index, ev.value * 1000.0))
        elif ev.event_type == TARGET_MISS:
            entries.append((ev.target_index, None))
    return sorted(entries, key=lambda x: x[0])


def save_timing_plot(
    log: SessionLog,
    path: str,
    *,
    bpm: float,
    count_in: int,
    beats: int,
    latency_ms: float,
) -> None:
    """Render a timing-error chart and write it to *path* as a PNG.

    Imports matplotlib lazily (inside this function) so sessions without
    ``--plot`` never touch the GUI stack.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    except ImportError:
        print("matplotlib not installed — cannot save plot.", file=sys.stderr)
        return

    errors  = extract_timing_errors(log)
    n       = len(errors)
    width   = max(6.0, n * 0.9 + 2.0)
    fig     = Figure(figsize=(width, 5.0))
    FigureCanvas(fig)
    ax      = fig.add_subplot(111)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(0,     color="black",       linewidth=1.0)
    ax.axhline( 50,   color="tab:green",   linewidth=0.9, linestyle="--", alpha=0.8)
    ax.axhline(-50,   color="tab:green",   linewidth=0.9, linestyle="--", alpha=0.8,
               label="±50 ms (good)")
    ax.axhline( 120,  color="tab:orange",  linewidth=0.9, linestyle="--", alpha=0.8)
    ax.axhline(-120,  color="tab:orange",  linewidth=0.9, linestyle="--", alpha=0.8,
               label="±120 ms (warn)")

    # ── Data points ───────────────────────────────────────────────────────────
    hit_x, hit_y, hit_c = [], [], []
    miss_x = []

    for idx, err_ms in errors:
        if err_ms is None:
            miss_x.append(idx)
        else:
            hit_x.append(idx)
            hit_y.append(err_ms)
            if abs(err_ms) <= 50:
                hit_c.append("tab:green")
            elif abs(err_ms) <= 120:
                hit_c.append("tab:orange")
            else:
                hit_c.append("tab:red")

    if hit_x:
        ax.scatter(hit_x, hit_y, c=hit_c, s=80, zorder=5, label="detected onset")

    if miss_x:
        ax.scatter(
            miss_x, [0] * len(miss_x),
            marker="x", s=120, c="tab:red", linewidths=2.5, zorder=5,
            label="no onset (miss)",
        )
        for x in miss_x:
            ax.annotate(
                "MISS", (x, 0),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=7.5, color="tab:red",
            )

    # ── Axes and title ────────────────────────────────────────────────────────
    if n:
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i + 1) for i in range(n)])

    ax.set_xlabel("Beat")
    ax.set_ylabel("Timing error (ms)")
    ax.set_title(
        f"{bpm:.0f} BPM  ·  {count_in}-beat count-in  ·  {beats} beats"
        f"\nLatency compensation: {latency_ms:.1f} ms",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _save_audio(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Write a float32 audio array to *path* as a WAV file.

    Tries ``soundfile`` first (ships with sounddevice); falls back to
    ``scipy.io.wavfile`` if soundfile is unavailable.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(p), audio, sample_rate, subtype="FLOAT")
        return
    except ImportError:
        pass
    try:
        from scipy.io.wavfile import write as wav_write
        wav_write(str(p), sample_rate, audio)
    except ImportError:
        print(
            "soundfile or scipy required to save audio. "
            "Install with: pip install soundfile",
            file=sys.stderr,
        )


def save_waveform_plot(
    audio: np.ndarray,
    sample_rate: int,
    path: str,
    *,
    target_times_s: list[float],
    timing_errors: list[tuple[int, float | None]],
    count_in_s: float,
    bpm: float,
    latency_ms: float,
    click_times_s: list[float],
) -> None:
    """Render waveform + beat/onset/click overlays; write PNG to *path*.

    Single panel layout:
    - Grey waveform amplitude trace.
    - Dashed grey target beat lines (red for missed beats).
    - Short tick marks at the very top of the plot for each metronome click
      (count-in clicks in blue, exercise clicks in purple), drawn in axes
      fraction coordinates so they sit above the waveform regardless of amplitude.
    - Solid coloured lines for latency-compensated onsets (green/orange/red by
      severity).
    - Faint dotted lines for raw pre-correction onsets, only when
      ``latency_ms != 0``.

    Imports matplotlib lazily so sessions without ``--waveform-plot``
    never touch the display stack.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib.lines as mlines
    except ImportError:
        print("matplotlib not installed — cannot save waveform plot.", file=sys.stderr)
        return

    compensated_s, raw_s_list, _ = _onset_plot_data(
        timing_errors, target_times_s, latency_ms
    )
    count_in_clicks, exercise_clicks = _split_click_times(click_times_s, count_in_s)

    duration_s = len(audio) / sample_rate if len(audio) > 0 and sample_rate > 0 else 0.0
    if duration_s == 0.0 and target_times_s:
        duration_s = max(target_times_s) + 60.0 / bpm

    width = max(10.0, duration_s * 1.5 + 2.0)
    fig = Figure(figsize=(width, 3.5))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # x=data coordinates, y=axes fraction (0=bottom edge, 1=top edge)
    trans = ax.get_xaxis_transform()

    # ── Count-in region ───────────────────────────────────────────────────────
    if count_in_s > 0:
        ax.axvspan(0, count_in_s, alpha=0.06, color="tab:blue")

    # ── Waveform ──────────────────────────────────────────────────────────────
    if len(audio) > 0:
        t_wave = np.arange(len(audio), dtype=np.float32) / sample_rate
        ax.plot(t_wave, audio, color="#999999", linewidth=0.4, alpha=0.7)

    # ── Click tick marks (top 6 % of axes height) ─────────────────────────────
    _C_COUNT_IN = "tab:blue"
    _C_EXERCISE = "#7c4dbd"
    for t in count_in_clicks:
        ax.plot([t, t], [0.94, 1.00], transform=trans,
                color=_C_COUNT_IN, linewidth=2.5, alpha=0.85, solid_capstyle="butt")
    for t in exercise_clicks:
        ax.plot([t, t], [0.94, 1.00], transform=trans,
                color=_C_EXERCISE, linewidth=2.5, alpha=0.85, solid_capstyle="butt")

    # ── Target beat lines ─────────────────────────────────────────────────────
    miss_set = {idx for idx, err_ms in timing_errors if err_ms is None}
    for i, t in enumerate(target_times_s):
        is_miss = i in miss_set
        color = "tab:red" if is_miss else "#cccccc"
        ax.axvline(t, color=color, linewidth=0.9, linestyle="--", alpha=0.8, zorder=2)
        if is_miss:
            ax.text(t, 0.88, f"{i + 1} MISS", transform=trans,
                    ha="center", va="top", fontsize=6.5, color=color)
        else:
            ax.text(t, 0.88, str(i + 1), transform=trans,
                    ha="center", va="top", fontsize=6.5, color="#aaaaaa")

    # ── Raw onset lines — only when latency correction is active ──────────────
    for t in raw_s_list:
        ax.axvline(t, color="tab:blue", linewidth=0.8, linestyle=":", alpha=0.4, zorder=3)

    # ── Compensated onset lines ───────────────────────────────────────────────
    err_by_idx = dict(timing_errors)
    for i, t in enumerate(target_times_s):
        err_ms = err_by_idx.get(i)
        if err_ms is None:
            continue
        onset_s = t + err_ms / 1000.0
        sev_color = (
            "tab:green"  if abs(err_ms) <= 50
            else "tab:orange" if abs(err_ms) <= 120
            else "tab:red"
        )
        ax.axvline(onset_s, color=sev_color, linewidth=1.5, alpha=0.9, zorder=5)
        tag = f"+{abs(err_ms):.0f}ms" if err_ms > 0 else f"-{abs(err_ms):.0f}ms"
        ax.text(onset_s + 0.01, 0.03, tag, transform=trans,
                ha="left", va="bottom", fontsize=6, color=sev_color)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    if count_in_clicks:
        legend_handles.append(
            mlines.Line2D([], [], color=_C_COUNT_IN, linewidth=2.5,
                          label="click (count-in)")
        )
    if exercise_clicks:
        legend_handles.append(
            mlines.Line2D([], [], color=_C_EXERCISE, linewidth=2.5,
                          label="click (exercise)")
        )
    if compensated_s:
        legend_handles.append(
            mlines.Line2D([], [], color="tab:green", linewidth=1.5, label="onset")
        )
    if raw_s_list:
        legend_handles.append(
            mlines.Line2D([], [], color="tab:blue", linewidth=0.8, linestyle=":",
                          label=f"onset raw (+{latency_ms:.0f} ms)")
        )
    if miss_set:
        legend_handles.append(
            mlines.Line2D([], [], color="tab:red", linewidth=0.9, linestyle="--",
                          label="missed beat")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.8)

    # ── Axes and title ────────────────────────────────────────────────────────
    ax.set_xlim(0, duration_s or 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(axis="x", alpha=0.2)
    comp_label = f"{latency_ms:.0f} ms" if latency_ms != 0.0 else "none"
    ax.set_title(
        f"{bpm:.0f} BPM  ·  {len(target_times_s)} beats  ·  Latency: {comp_label}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bass practice session demo — listens on mic, scores against a built-in exercise."
    )
    p.add_argument("--bpm",           type=float,        default=80.0,
                   help="Tempo in BPM (default: 80)")
    p.add_argument("--beats",         type=_positive_int, default=4,
                   help="Number of quarter-note targets (default: 4)")
    p.add_argument("--count-in",      type=int,          default=2,    dest="count_in",
                   help="Count-in beats (default: 2)")
    p.add_argument("--device",        type=int,   default=None,
                   help="Input device index (default: system default)")
    p.add_argument("--threshold",     type=float, default=0.015,
                   help="Onset RMS energy threshold (default: 0.015)")
    lat = p.add_mutually_exclusive_group()
    lat.add_argument("--latency-ms",     type=float, default=None, dest="latency_ms",
                     help="Latency compensation in ms (overrides audio_calibration.json)")
    lat.add_argument("--no-calibration", action="store_true",
                     help="Disable latency compensation (0 ms)")
    p.add_argument("--click",         action=argparse.BooleanOptionalAction, default=True,
                   help="Play audible metronome clicks (default: on)")
    p.add_argument("--output-device", type=int,   default=None, dest="output_device",
                   help="Output device index for click audio (default: system default)")
    p.add_argument("--list-devices",  action="store_true",
                   help="List audio devices and exit")
    p.add_argument("--save-log",       default=None, dest="save_log",       metavar="PATH",
                   help="Write finalized session log to PATH (JSON)")
    p.add_argument("--plot",           default=None,                         metavar="PATH",
                   help="Save timing-error scatter plot to PATH (PNG)")
    p.add_argument("--save-audio",     default=None, dest="save_audio",     metavar="PATH",
                   help="Save captured input audio to PATH (WAV)")
    p.add_argument("--waveform-plot",  default=None, dest="waveform_plot",  metavar="PATH",
                   help="Save waveform + beat/onset overlay plot to PATH (PNG)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except ImportError:
            print("sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
        return

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    bpm      = args.bpm
    count_in = args.count_in
    beat_s   = 60.0 / bpm
    targets  = build_targets(args.beats)

    latency_ms, latency_src = _resolve_latency(args)

    controller = SessionController(
        targets        = targets,
        bpm            = bpm,
        count_in_beats = count_in,
        sample_rate    = _SAMPLE_RATE,
    )

    # Precompute click schedule and waveform
    click_times = click_beat_times_s(bpm, count_in, [t["time"] for t in targets])
    click_data  = make_click_waveform(sample_rate=_SAMPLE_RATE)
    scheduler: ClickScheduler | None = None

    # -- Shared state between audio callback and main thread ------------------
    lock         = threading.Lock()
    onset_queue: list[float] = []
    audio_chunks: list[np.ndarray] = []   # for --save-audio / --waveform-plot
    sample_pos   = [0]     # samples since session start
    prev_rms     = [0.0]
    capturing    = [False]
    _capture_audio = bool(args.save_audio or args.waveform_plot)

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if not capturing[0]:
            sample_pos[0] += frames
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        if _capture_audio:
            audio_chunks.append(mono.copy())
        onset, rms = _detect_onset(mono, prev_rms[0], args.threshold)
        prev_rms[0] = rms
        if onset:
            # Place onset at the centre of the block for better accuracy
            onset_sample = sample_pos[0] + frames // 2
            onset_s = onset_sample / _SAMPLE_RATE
            with lock:
                onset_queue.append(onset_s)
        sample_pos[0] += frames

    # -- Header ---------------------------------------------------------------
    click_dev    = args.output_device if args.output_device is not None else "default"
    click_status = f"on  (output: {click_dev})" if args.click else "off"

    print()
    print(f"  Exercise  : {_EXERCISE_NAME}")
    print(f"  Tempo     : {bpm:.0f} BPM  |  Count-in: {count_in} beats")
    print(f"  Targets   : {len(targets)}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Latency   : {latency_ms:.1f}ms  ({latency_src})")
    print(f"  Clicks    : {click_status}")
    print()
    print("  Press Ctrl+C to stop early.")
    print()

    try:
        with sd.InputStream(
            samplerate = _SAMPLE_RATE,
            blocksize  = _BLOCK_SIZE,
            channels   = _CHANNELS,
            device     = args.device,
            callback   = callback,
            dtype      = "float32",
        ):
            # Record wall time and zero the sample counter atomically so the
            # click scheduler's perf_counter offsets stay aligned to sample_pos.
            session_wall_start = time.perf_counter()
            controller.start()
            capturing[0] = True
            sample_pos[0] = 0

            # Persist session parameters in the log for downstream tools
            if controller.log is not None:
                controller.log.metadata.update({
                    "bpm":        str(bpm),
                    "beats":      str(args.beats),
                    "count_in":   str(count_in),
                    "latency_ms": str(latency_ms),
                    "latency_src": latency_src,
                })

            if args.click:
                scheduler = ClickScheduler(
                    click_times_s      = click_times,
                    session_wall_start = session_wall_start,
                    click_data         = click_data,
                    sample_rate        = _SAMPLE_RATE,
                    output_device      = args.output_device,
                )
                scheduler.start()

            try:
                # -- Count-in loop --------------------------------------------
                last_beat_printed = [0]
                print("  Count-in: ", end="", flush=True)

                while controller.phase == SessionPhase.COUNT_IN:
                    current_s = sample_pos[0] / _SAMPLE_RATE
                    beat_num  = int(current_s / beat_s) + 1

                    if beat_num != last_beat_printed[0] and 1 <= beat_num <= count_in:
                        print(f"{beat_num} ", end="", flush=True)
                        last_beat_printed[0] = beat_num

                    controller.update(sample_pos[0])
                    time.sleep(0.02)

                print("GO!\n")

                # -- Active session loop --------------------------------------
                while not controller.is_complete():
                    current_sample = sample_pos[0]
                    with lock:
                        onsets = list(onset_queue)
                        onset_queue.clear()

                    events = controller.update(
                        current_sample,
                        compensate_onset_times(onsets, latency_ms),
                    )
                    for ev in events:
                        _print_event(ev, targets)

                    if controller.phase == SessionPhase.ABORTED:
                        break

                    time.sleep(0.01)

            finally:
                if scheduler is not None:
                    scheduler.stop()

    except KeyboardInterrupt:
        controller.abort()
        print("\n\n  Session aborted.\n")

    metrics = controller.summary()
    if metrics:
        _print_summary(metrics, len(targets))

    captured_audio = (
        np.concatenate(audio_chunks) if audio_chunks
        else np.zeros(0, dtype=np.float32)
    )

    log = controller.log
    if log is not None:
        if args.save_audio:
            _save_audio(captured_audio, _SAMPLE_RATE, args.save_audio)
            log.metadata["audio_path"] = args.save_audio
            print(f"  Audio saved: {args.save_audio}")
        if args.save_log:
            save_session_log_file(log, args.save_log)
            print(f"  Log saved  : {args.save_log}")
        if args.plot:
            save_timing_plot(
                log, args.plot,
                bpm=bpm, count_in=count_in, beats=args.beats, latency_ms=latency_ms,
            )
            print(f"  Plot saved : {args.plot}")
        if args.waveform_plot:
            count_in_s = count_in * beat_s
            target_times_s = [count_in_s + t["time"] * beat_s for t in targets]
            save_waveform_plot(
                captured_audio, _SAMPLE_RATE, args.waveform_plot,
                target_times_s=target_times_s,
                timing_errors=extract_timing_errors(log),
                count_in_s=count_in_s,
                bpm=bpm,
                latency_ms=latency_ms,
                click_times_s=click_times,
            )
            print(f"  Waveform   : {args.waveform_plot}")


if __name__ == "__main__":
    main()
