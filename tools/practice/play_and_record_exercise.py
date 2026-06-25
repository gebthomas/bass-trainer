#!/usr/bin/env python3
"""
Synchronized practice transport.

Plays a rendered backing track and records bass input simultaneously,
sharing a single audio stream clock.

WAV timeline (both mono and stereo outputs):
  sample 0          = start of count-in
  sample downbeat_n = beat 1 of the progression  (printed on run)

Stereo session mode (--stereo-session):
  LEFT  channel = dry bass input only          (performance signal)
  RIGHT channel = backing/click reference only (transport clock)
  Timing analysis compares left against right.
  Channels are NEVER mixed — right is copied from the pre-rendered
  backing buffer, not captured from the ADC.
"""

import argparse
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.practice.render_practice_track import render_track, _beat_sample


# ── Progression helpers ───────────────────────────────────────────────────────

def _extend_to_duration(progression, duration_beats):
    """Tile progression segments to fill duration_beats."""
    if not progression:
        return []
    prog_span = max(seg["end"] for seg in progression)
    if prog_span <= 0:
        return []
    n_repeats = math.ceil(duration_beats / prog_span)
    extended = []
    for i in range(n_repeats):
        offset = i * prog_span
        for seg in progression:
            start = seg["start"] + offset
            end   = seg["end"]   + offset
            if start >= duration_beats:
                break
            extended.append({
                "start": start,
                "end":   min(end, duration_beats),
                "chord": seg["chord"],
            })
    return extended


# ── Device helpers ────────────────────────────────────────────────────────────

def _resolve_device(device, kind):
    """Return integer device index, resolving None → system default."""
    if device is not None:
        return device
    return sd.query_devices(kind=kind)["index"]


# ── Diagnostics ───────────────────────────────────────────────────────────────

_BASS_LO_HZ        = 40
_BASS_HI_HZ        = 250
_GUIDE_LO_HZ       = 300   # E4 ≈ 330 Hz, F4 ≈ 349 Hz
_GUIDE_HI_HZ       = 600   # B4 ≈ 494 Hz, C5 ≈ 523 Hz
_BASS_RMS_WARN     = 0.002  # very low — likely no signal
_BLEED_RATIO_WARN  = 2.5    # guide-tone / bass energy ratio above this → spectral warning
_CORR_WARN         = 0.50   # |channel correlation| above this → bleed warning


def _diagnostics_stereo(bass_ch, monitor_ch, sr, downbeat_n):
    """Report and warn from stereo session data."""
    bass_peak = float(np.max(np.abs(bass_ch)))
    bass_rms  = float(np.sqrt(np.mean(bass_ch ** 2)))
    mon_peak  = float(np.max(np.abs(monitor_ch)))
    mon_rms   = float(np.sqrt(np.mean(monitor_ch ** 2)))

    print(f"Bass (L) peak:    {bass_peak:.4f}")
    print(f"Bass (L) RMS:     {bass_rms:.6f}")
    print(f"Monitor (R) peak: {mon_peak:.4f}")
    print(f"Monitor (R) RMS:  {mon_rms:.6f}")

    if bass_rms < _BASS_RMS_WARN:
        print("WARNING: bass (L) RMS very low — check input gain, cable, and device.")

    # Channel correlation: high correlation means backing may have bled into input.
    seg_start = min(downbeat_n, max(0, len(bass_ch) - sr))
    seg_end   = min(seg_start + 4 * sr, len(bass_ch))
    if seg_end > seg_start + sr:
        b = bass_ch[seg_start:seg_end]
        m = monitor_ch[seg_start:seg_end]
        if np.std(b) > 1e-6 and np.std(m) > 1e-6:
            corr = float(np.corrcoef(b, m)[0, 1])
            print(f"Channel corr:     {corr:+.3f}")
            if abs(corr) > _CORR_WARN:
                print(
                    f"WARNING: |correlation| = {abs(corr):.2f} > {_CORR_WARN} — "
                    "possible backing bleed into bass input. "
                    "Disable hardware direct monitoring and try --silent-backing to confirm."
                )


def _diagnostics_mono(bass_wav, sr, downbeat_n, has_monitor, backing):
    """Report and warn from mono session data."""
    n = len(bass_wav)
    bass_peak = float(np.max(np.abs(bass_wav))) if n else 0.0
    bass_rms  = float(np.sqrt(np.mean(bass_wav ** 2))) if n else 0.0

    print(f"Bass WAV peak:    {bass_peak:.4f}")
    print(f"Bass WAV RMS:     {bass_rms:.6f}")

    if has_monitor:
        mon = backing[:n]
        print(f"Monitor peak:     {float(np.max(np.abs(mon))):.4f}")
        print(f"Monitor RMS:      {float(np.sqrt(np.mean(mon ** 2))):.6f}")

    if bass_rms < _BASS_RMS_WARN:
        print("WARNING: bass RMS very low — check input gain, cable, and device.")

    # Spectral bleed check: guide-tone energy vs bass energy.
    seg_start = downbeat_n
    seg_end   = min(seg_start + 4 * sr, n)
    if seg_end > seg_start + sr:
        seg   = bass_wav[seg_start:seg_end]
        mag   = np.abs(rfft(seg))
        freqs = rfftfreq(len(seg), d=1.0 / sr)
        bass_e  = float(np.sum(mag[(freqs >= _BASS_LO_HZ)  & (freqs <= _BASS_HI_HZ)]  ** 2))
        guide_e = float(np.sum(mag[(freqs >= _GUIDE_LO_HZ) & (freqs <= _GUIDE_HI_HZ)] ** 2))
        if bass_e > 0 and guide_e / bass_e > _BLEED_RATIO_WARN:
            dom_freq = float(freqs[np.argmax(mag)])
            print(
                f"WARNING: dominant FFT energy in guide-tone range "
                f"({_GUIDE_LO_HZ}–{_GUIDE_HI_HZ} Hz, peak {dom_freq:.0f} Hz); "
                f"bass-range energy low (ratio {guide_e / bass_e:.1f}×). "
                "Possible backing-track bleed — try --silent-backing to isolate."
            )


# ── Transport ─────────────────────────────────────────────────────────────────

def run_transport(
    progression, bpm, count_in_beats, duration_s,
    time_sig, pad_amp, click_amp,
    sr, input_device, output_device, input_channel,
    output_path,
    stereo_session=False,
    save_bass_only=None,
    silent_backing=False,
    monitor_path=None,
):
    beat_s = 60.0 / bpm

    # Build (possibly looped) progression to fill the requested duration
    if duration_s is not None:
        duration_beats = duration_s / beat_s
        prog = _extend_to_duration(progression, duration_beats)
    else:
        prog = progression
        duration_beats = max((seg["end"] for seg in prog), default=0.0)

    # Render backing track at the session sample rate
    backing = render_track(
        prog,
        bpm            = bpm,
        count_in_beats = count_in_beats,
        duration_beats = duration_beats,
        time_sig       = time_sig,
        pad_amp        = pad_amp,
        click_amp      = click_amp,
        sr             = sr,
    ).astype(np.float32)

    total_n    = len(backing)
    downbeat_n = _beat_sample(count_in_beats, beat_s, sr)

    # Resolve devices
    in_dev  = _resolve_device(input_device,  "input")
    out_dev = _resolve_device(output_device, "output")

    in_info  = sd.query_devices(in_dev)
    out_info = sd.query_devices(out_dev)

    in_channels  = min(2, in_info["max_input_channels"])
    out_channels = min(2, out_info["max_output_channels"])

    if in_channels == 0:
        sys.exit(f"Device [{in_dev}] '{in_info['name']}' has no input channels.")
    if out_channels == 0:
        sys.exit(f"Device [{out_dev}] '{out_info['name']}' has no output channels.")

    rec_channel = min(input_channel, in_channels - 1)
    if rec_channel != input_channel:
        print(f"Warning: input channel {input_channel} not available; using channel {rec_channel}.")

    # Pre-allocate recording buffer.
    # Stereo: shape (total_n, 2) — col 0 = bass input, col 1 = backing reference.
    # Mono:   shape (total_n,)   — bass input only.
    if stereo_session:
        rec_buf = np.zeros((total_n, 2), dtype=np.float32)
    else:
        rec_buf = np.zeros(total_n, dtype=np.float32)

    pos_arr  = np.array([0], dtype=np.int64)
    finished = threading.Event()

    def _callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}", flush=True)

        p = int(pos_arr[0])
        remaining = total_n - p

        if remaining <= 0:
            outdata[:] = 0.0
            finished.set()
            raise sd.CallbackStop()

        n = min(frames, remaining)

        # ── Output ────────────────────────────────────────────────────────────
        # Backing track broadcast to all output channels, or silence for
        # diagnostic mode.  This path has no connection to rec_buf.
        if silent_backing:
            outdata[:] = 0.0
        else:
            outdata[:n, :] = backing[p : p + n, np.newaxis]
            if n < frames:
                outdata[n:] = 0.0

        # ── Record ────────────────────────────────────────────────────────────
        # Source is ALWAYS indata (ADC), never outdata or backing.
        if stereo_session:
            # LEFT  = dry bass input only  (performance signal)
            rec_buf[p : p + n, 0] = indata[:n, rec_channel]
            # RIGHT = backing reference    (transport/timing signal, not from ADC)
            rec_buf[p : p + n, 1] = backing[p : p + n]
        else:
            rec_buf[p : p + n] = indata[:n, rec_channel]

        pos_arr[0] += n

        if pos_arr[0] >= total_n:
            finished.set()
            raise sd.CallbackStop()

    # ── Session info ──────────────────────────────────────────────────────────
    print(f"Input device:     [{in_dev}] {in_info['name']}  (channel {rec_channel})")
    print(f"Output device:    [{out_dev}] {out_info['name']}")
    print(f"Mode:             {'stereo session (L=bass, R=backing)' if stereo_session else 'mono bass-only'}")
    if silent_backing:
        print("Backing:          SILENT (diagnostic mode — output muted)")
    print(f"Sample rate:      {sr} Hz")
    print(f"BPM:              {bpm}")
    print(f"Count-in:         {count_in_beats} beats  ({count_in_beats * beat_s:.3f}s)")
    print(f"Progression:      {duration_beats:.1f} beats  ({duration_beats * beat_s:.2f}s)")
    print(f"Total duration:   {total_n / sr:.3f}s  ({total_n} samples)")
    print(f"Downbeat in WAV:  sample {downbeat_n}  ({downbeat_n / sr:.4f}s from start)")
    print(f"Output:           {output_path}")
    if stereo_session and save_bass_only:
        print(f"Bass-only:        {save_bass_only}")
    print()

    input("Press Enter to start...")
    print("Recording — press Ctrl+C to stop early.\n")

    try:
        with sd.Stream(
            samplerate = sr,
            blocksize  = 512,
            dtype      = "float32",
            device     = (in_dev, out_dev),
            channels   = (in_channels, out_channels),
            callback   = _callback,
        ):
            finished.wait()
    except KeyboardInterrupt:
        print("\nStopped early.")

    # ── Save ─────────────────────────────────────────────────────────────────
    recorded_n = int(pos_arr[0])
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if stereo_session:
        recorded = rec_buf[:recorded_n, :]          # shape (N, 2)
        sf.write(str(out_path), recorded, sr, subtype="PCM_16")

        if save_bass_only:
            bass_path = Path(save_bass_only)
            bass_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(bass_path), recorded[:, 0], sr, subtype="PCM_16")
            print(f"Bass-only saved:  {bass_path}  (mono, analysis-ready)")

        print(f"Stereo saved:     {out_path}  (L=bass, R=backing)")
    else:
        recorded = rec_buf[:recorded_n]             # shape (N,)
        sf.write(str(out_path), recorded, sr, subtype="PCM_16")

        if monitor_path:
            mon_path = Path(monitor_path)
            mon_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(mon_path), backing[:recorded_n], sr, subtype="PCM_16")
            print(f"Monitor saved:    {mon_path}")

        print(f"Bass saved:       {out_path}  (mono)")

    print(f"Expected samples: {total_n}")
    print(f"Recorded samples: {recorded_n}")
    print(f"Downbeat in WAV:  sample {downbeat_n}  ({downbeat_n / sr:.4f}s)")
    print()

    if stereo_session:
        _diagnostics_stereo(recorded[:, 0], recorded[:, 1], sr, downbeat_n)
    else:
        _diagnostics_mono(recorded, sr, downbeat_n, monitor_path is not None, backing[:recorded_n])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Play backing track and record bass simultaneously."
    )
    p.add_argument("--progression",       required=True,
                   help="Path to progression JSON")
    p.add_argument("--bpm",               type=float, default=120.0,
                   help="Tempo in BPM (default: 120)")
    p.add_argument("--duration",          type=float, default=None, metavar="SECONDS",
                   help="Progression duration in seconds; progression loops to fill "
                        "(default: one pass through the JSON)")
    p.add_argument("--output",            required=True,
                   help="Output WAV path (stereo session or mono bass-only)")
    p.add_argument("--count-in-beats",    type=int, default=4,
                   help="Count-in beats before downbeat (default: 4)")
    p.add_argument("--beats-per-measure", type=int, default=4,
                   help="Beats per bar for downbeat accent (default: 4)")
    p.add_argument("--sample-rate",       type=int, default=48000,
                   help="Sample rate in Hz (default: 48000)")
    p.add_argument("--input-device",      type=int, default=None,
                   help="sounddevice input device index (default: system default)")
    p.add_argument("--output-device",     type=int, default=None,
                   help="sounddevice output device index (default: system default)")
    p.add_argument("--input-channel",     type=int, default=0,
                   help="Input channel to record, 0-indexed (default: 0)")
    p.add_argument("--pad-amp",           type=float, default=0.25,
                   help="Chord pad amplitude 0–1 (default: 0.25)")
    p.add_argument("--click-amp",         type=float, default=0.45,
                   help="Metronome click amplitude 0–1 (default: 0.45)")
    p.add_argument("--stereo-session",    action="store_true",
                   help="Save stereo WAV: L=dry bass input, R=backing/click reference. "
                        "Channels are never mixed. Use for timing analysis and review.")
    p.add_argument("--save-bass-only",    default=None, metavar="FILE",
                   help="(with --stereo-session) Also save L channel as a separate mono WAV "
                        "for analysis tools.")
    p.add_argument("--silent-backing",    action="store_true",
                   help="Mute output entirely; record input as normal. "
                        "Use to confirm bass-only capture / diagnose hardware loopback.")
    p.add_argument("--save-monitor",      default=None, metavar="FILE",
                   help="(mono mode) Save pre-rendered backing track as a separate WAV.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.save_bass_only and not args.stereo_session:
        sys.exit("--save-bass-only requires --stereo-session.")

    with open(args.progression, encoding="utf-8") as fh:
        progression = json.load(fh)

    run_transport(
        progression    = progression,
        bpm            = args.bpm,
        count_in_beats = args.count_in_beats,
        duration_s     = args.duration,
        time_sig       = args.beats_per_measure,
        pad_amp        = args.pad_amp,
        click_amp      = args.click_amp,
        sr             = args.sample_rate,
        input_device   = args.input_device,
        output_device  = args.output_device,
        input_channel  = args.input_channel,
        output_path    = args.output,
        stereo_session = args.stereo_session,
        save_bass_only = args.save_bass_only,
        silent_backing = args.silent_backing,
        monitor_path   = args.save_monitor,
    )


if __name__ == "__main__":
    main()
