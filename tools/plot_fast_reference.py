# Usage:
#   python tools/plot_fast_reference.py <wav> <targets.json> [options]
#
#   --apply-calibration   load input_latency_ms from config/audio_calibration.json
#   --target-index N      zoom panels 1–2 around target N (0-based)
#   --save FILE           save to PNG instead of showing interactively
#
# Developer diagnostic — not polished UI.  Squint and iterate.

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
import librosa

from analyze_fast_reference import (
    SAMPLE_RATE,
    PYIN_FMIN, PYIN_FMAX, PYIN_FRAME_LENGTH, PYIN_HOP_LENGTH,
    analyze_target,
)
from plot_onsets import compute_rms_blocks, detect_onsets
from core.targets import load_targets
from core.pitch import note_to_hz, hz_to_note
from core.audio_calibration import load_input_latency

# ── Confidence-tier colour map ────────────────────────────────────────────────

_TIER_COLOR = {
    "high":      "#2ecc71",   # green
    "medium":    "#f1c40f",   # yellow
    "low":       "#e67e22",   # orange
    "uncertain": "#95a5a6",   # gray
}


def _tier_color(tier):
    return _TIER_COLOR.get(tier, "#cccccc")


def _semi_to_note(semitone):
    hz = 440.0 * 2.0 ** ((semitone - 69) / 12.0)
    name, _ = hz_to_note(hz)
    return name


# ── Status colour map ─────────────────────────────────────────────────────────

_STATUS_COLOR = {
    "ok":                   "#2ecc71",
    "pitch_marginal":       "#f39c12",
    "pitch_error":          "#e74c3c",
    "pitch_uncertain":      "#e67e22",
    "pitch_low_confidence": "#f1c40f",
    "missed":               "#95a5a6",
    "no_pitch":             "#bdc3c7",
}


def _status_color(status):
    return _STATUS_COLOR.get(status, "#cccccc")


# ── Full-audio pyin trace ─────────────────────────────────────────────────────

def _pyin_full(audio):
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=SAMPLE_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=PYIN_HOP_LENGTH,
    )
    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=SAMPLE_RATE, hop_length=PYIN_HOP_LENGTH
    )
    return times, f0, voiced_flag


# ── X range ───────────────────────────────────────────────────────────────────

def _x_range(target_index, targets, results, audio_dur, pad=0.40):
    if target_index is None:
        return 0.0, audio_dur
    lo = (targets[target_index - 1]["time"] - pad) if target_index > 0 else 0.0
    if target_index + 1 < len(targets):
        hi = targets[target_index + 1]["time"] + pad
    else:
        hi = results[target_index]["win_end"] + pad
    return max(0.0, lo), min(audio_dur, hi)


# ── Panel 1: waveform ─────────────────────────────────────────────────────────

def _draw_waveform(ax, t_audio, audio, results, x_lo, x_hi, input_latency_ms, onsets):
    ax.plot(t_audio, audio, color="#aaaaaa", linewidth=0.35, zorder=1)
    ax.set_ylabel("Amplitude")
    ax.set_xlim(x_lo, x_hi)

    # x: data coords, y: axes coords (0–1) — lets us pin labels to top edge
    xform = ax.get_xaxis_transform()

    for i, r in enumerate(results):
        # attack-guard zone
        ax.axvspan(r["effective_time"], r["win_start"],
                   alpha=0.10, color="#888888", zorder=2)
        # analysis window coloured by confidence tier
        ax.axvspan(r["win_start"], r["win_end"],
                   alpha=0.22, color=_tier_color(r["confidence_tier"]), zorder=2)
        # musical target time
        ax.axvline(r["target_time"],
                   color="#bb3333", linewidth=0.9, linestyle="--", zorder=3)
        # latency-shifted effective time (only when meaningfully different)
        if abs(input_latency_ms) > 0.5:
            ax.axvline(r["effective_time"],
                       color="#3366bb", linewidth=0.9, linestyle=":", zorder=3)
        # note label pinned to top
        ax.text(r["target_time"], 0.97, f"{i + 1}:{r['target_note']}",
                transform=xform, fontsize=6, color="#880000",
                ha="center", va="top", clip_on=True)

    # detected onset markers
    for t in onsets:
        ax.axvline(t, color="#cc7700", linewidth=0.9, linestyle="--", alpha=0.75, zorder=4)

    legend = [
        mpatches.Patch(color="#bb3333", label="target time (musical)"),
        mpatches.Patch(color="#888888", alpha=0.35, label="attack guard"),
        mpatches.Patch(color="#2ecc71", alpha=0.55, label="window: high confidence"),
        mpatches.Patch(color="#f1c40f", alpha=0.55, label="window: medium"),
        mpatches.Patch(color="#e67e22", alpha=0.55, label="window: low"),
        mpatches.Patch(color="#95a5a6", alpha=0.55, label="window: uncertain"),
        mpatches.Patch(color="#cc7700", label="detected onset"),
    ]
    if abs(input_latency_ms) > 0.5:
        legend.insert(1, mpatches.Patch(color="#3366bb", label="effective time (latency-shifted)"))
    ax.legend(handles=legend, fontsize=6, loc="upper right", ncol=2)


# ── Panel 2: pitch trace ──────────────────────────────────────────────────────

def _draw_pitch(ax, ft, f0, voiced, results, x_lo, x_hi):
    voiced_mask = voiced & ~np.isnan(f0) & (f0 > 0)
    ax.scatter(ft[voiced_mask], f0[voiced_mask],
               s=3, color="#2255aa", zorder=3, label="pyin f0 (voiced)")
    ax.set_ylabel("Hz (log scale)")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)

    # x: axes coords (0–1), y: data coords (Hz) — pins note labels to left edge
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)

    seen = {}
    for r in results:
        ax.axvline(r["target_time"],
                   color="#bb3333", linewidth=0.6, linestyle="--", alpha=0.40, zorder=2)
        note = r["target_note"]
        hz   = note_to_hz(note)
        if note not in seen:
            ax.axhline(hz, color="#bb3333", linewidth=0.7,
                       linestyle="--", alpha=0.55, zorder=2)
            ax.text(0.01, hz, note, transform=y_trans,
                    fontsize=6, color="#bb3333", va="center", ha="left", clip_on=True)
            seen[note] = True

    # Y limits from voiced frames in the visible window
    in_view = voiced_mask & (ft >= x_lo) & (ft <= x_hi)
    if np.any(in_view):
        ax.set_ylim(
            max(PYIN_FMIN, float(np.min(f0[in_view])) * 0.85),
            min(PYIN_FMAX, float(np.max(f0[in_view])) * 1.20),
        )
    else:
        hz_all = [note_to_hz(r["target_note"]) for r in results]
        if hz_all:
            ax.set_ylim(min(hz_all) * 0.80, max(hz_all) * 1.30)

    ax.legend(fontsize=6, loc="upper right")


# ── Panel 3: per-target status blocks ────────────────────────────────────────

def _draw_status(ax, results, selected_index):
    n = len(results)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")

    for i, r in enumerate(results):
        row = n - 1 - i  # top-to-bottom display order
        color = _status_color(r["pitch_status"])

        rect = mpatches.FancyBboxPatch(
            (0.005, row + 0.08), 0.990, 0.84,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="#666666", linewidth=0.5, alpha=0.85,
        )
        ax.add_patch(rect)

        cand  = r["candidate_note"] or "—"
        cents = f"{r['cents_error']:+.0f}c" if r["cents_error"] is not None else "—"
        trunc = " [trunc]" if r["truncated"] else ""
        line = (
            f"#{i + 1:02d}  {r['target_note']:<5} @{r['target_time']:.3f}s"
            f"  →{cand:<5} {cents:>6}"
            f"  vf={r['voiced_fraction'] * 100:.0f}%"
            f"  cons={r['pitch_consensus'] * 100:.0f}%"
            f"  {r['confidence_tier']:<9}  {r['pitch_status']}{trunc}"
        )
        ax.text(0.012, row + 0.65, line,
                fontsize=7, va="center", ha="left",
                fontfamily="monospace", clip_on=True)

        votes = r.get("semitone_votes")
        if votes:
            votes_str = ", ".join(
                f"{_semi_to_note(s)} {frac * 100:.0f}%" for s, frac in votes
            )
            ax.text(0.012, row + 0.27, votes_str,
                    fontsize=6, va="center", ha="left",
                    fontfamily="monospace", color="#444444", clip_on=True)

        chord  = r.get("current_chord")
        hclass = r.get("harmonic_class")
        if chord is not None:
            _HARM_COLOR = {"chord": "#1a7a3a", "scale": "#7a6a00",
                           "out": "#8a0000", "pitch_uncertain": "#666666"}
            badge_color = _HARM_COLOR.get(hclass, "#444444")
            badge = f"{chord}→{hclass}" if hclass else f"{chord}"
            ax.text(0.988, row + 0.27, badge,
                    fontsize=6, va="center", ha="right",
                    fontfamily="monospace", color=badge_color,
                    fontweight="bold", clip_on=True)

    # Highlight the zoomed target
    if selected_index is not None and 0 <= selected_index < n:
        row = n - 1 - selected_index
        border = mpatches.FancyBboxPatch(
            (0.002, row + 0.03), 0.996, 0.94,
            boxstyle="round,pad=0.01",
            facecolor="none", edgecolor="#000000", linewidth=2.0,
        )
        ax.add_patch(border)


# ── Figure assembly ───────────────────────────────────────────────────────────

def _build_figure(wav_path, targets, audio, results, ft, f0, voiced,
                  input_latency_ms, onsets, args):
    n = len(results)
    audio_dur = len(audio) / SAMPLE_RATE
    t_audio = np.linspace(0, audio_dur, len(audio))
    x_lo, x_hi = _x_range(args.target_index, targets, results, audio_dur)

    wave_h   = 4.0
    pitch_h  = 3.0
    status_h = max(2.0, n * 0.30)
    fig_h    = wave_h + pitch_h + status_h + 1.2  # 1.2 for suptitle + margins

    fig = plt.figure(figsize=(14, fig_h))
    title = (
        f"{wav_path.name}  ·  {Path(args.targets).name}"
        + (f"  ·  latency {input_latency_ms:+.1f} ms" if abs(input_latency_ms) > 0.5 else "")
        + (f"  ·  zoom → target {args.target_index + 1}" if args.target_index is not None else "")
    )
    fig.suptitle(title, fontsize=9)

    gs = GridSpec(3, 1,
                  height_ratios=[wave_h, pitch_h, status_h],
                  hspace=0.38, top=0.94, bottom=0.03)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    _draw_waveform(ax1, t_audio, audio, results, x_lo, x_hi, input_latency_ms, onsets)
    _draw_pitch(ax2, ft, f0, voiced, results, x_lo, x_hi)
    ax2.set_xlabel("Time (s)")
    _draw_status(ax3, results, args.target_index)

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Developer diagnostic: plot fast-reference windows, pitch trace, and target status."
    )
    p.add_argument("wav",     help="WAV file")
    p.add_argument("targets", help="Targets JSON file")
    p.add_argument("--apply-calibration", action="store_true",
                   help="Apply input_latency_ms from config/audio_calibration.json")
    p.add_argument("--target-index", type=int, default=None, metavar="N",
                   help="Zoom panels 1–2 around target N (0-based)")
    p.add_argument("--save", metavar="FILE",
                   help="Save PNG instead of showing interactively")
    return p.parse_args()


def main():
    args = _parse_args()
    wav_path    = Path(args.wav)
    target_path = Path(args.targets)

    input_latency_ms = 0.0
    if args.apply_calibration:
        input_latency_ms = load_input_latency()
        print(f"Calibration: input_latency_ms = {input_latency_ms:+.1f} ms")

    print(f"Loading: {wav_path.name}")
    audio   = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)[0]
    targets = load_targets(target_path)
    audio_dur = len(audio) / SAMPLE_RATE
    print(f"Audio: {audio_dur:.2f}s  |  Targets: {len(targets)}")

    if args.target_index is not None:
        if not (0 <= args.target_index < len(targets)):
            print(
                f"Error: --target-index {args.target_index} out of range "
                f"(0–{len(targets) - 1})",
                file=sys.stderr,
            )
            sys.exit(1)

    print("Analyzing targets...")
    results = [analyze_target(i, targets, audio, input_latency_ms)
               for i in range(len(targets))]

    print("Running pyin on full audio...")
    ft, f0, voiced = _pyin_full(audio)

    print("Detecting onsets...")
    t_rms, rms = compute_rms_blocks(audio)
    onsets, _, _ = detect_onsets(t_rms, rms)
    print(f"Detected {len(onsets)} onsets:")
    for t in onsets:
        print(f"  {t:.3f}s")

    fig = _build_figure(wav_path, targets, audio, results, ft, f0, voiced,
                        input_latency_ms, onsets, args)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
