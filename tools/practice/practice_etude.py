#!/usr/bin/env python3
"""One-command etude practice workflow.

Renders the exercise card, waits for Enter, records a stereo session,
runs jazz exercise analysis, and prints a pass/fail verdict.

Usage:
    python tools/practice_etude.py \\
        tests/targets/jazz/ii_v_i_low_position_quarters.json \\
        --progression tests/progressions/ii_v_i_C_slow.json \\
        --bpm 100 \\
        --duration 12 \\
        --output-prefix tests/real_audio/jazz/simple_etude
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_RENDERER = PROJECT_ROOT / "tools" / "render_exercise_card.py"
_RECORDER = PROJECT_ROOT / "tools" / "play_and_record_exercise.py"
_ANALYZER = PROJECT_ROOT / "tools" / "run_jazz_exercise.py"
_PY       = sys.executable


# ── Sub-step functions ────────────────────────────────────────────────────────

def _render_card(targets: str, progression: str, card_mode: str = "full") -> None:
    subprocess.run(
        [_PY, str(_RENDERER), targets, "--progression", progression,
         "--card-mode", card_mode],
        check=False,
    )


def _record(args: argparse.Namespace, stereo_path: str, bass_path: str) -> bool:
    cmd = [
        _PY, str(_RECORDER),
        "--progression",    args.progression,
        "--bpm",            str(args.bpm),
        "--duration",       str(args.duration),
        "--output",         stereo_path,
        "--stereo-session",
        "--save-bass-only", bass_path,
        "--count-in-beats", str(args.count_in_beats),
        "--sample-rate",    str(args.sample_rate),
    ]
    if args.input_device is not None:
        cmd += ["--input-device", str(args.input_device)]
    if args.output_device is not None:
        cmd += ["--output-device", str(args.output_device)]

    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def _analyze(args: argparse.Namespace, bass_path: str, time_offset: float) -> str:
    """Run the analyzer and return its stdout as a string."""
    cmd = [
        _PY, str(_ANALYZER),
        bass_path,
        args.targets,
        "--progression",      args.progression,
        "--target-time-unit", "beats",
        "--bpm",              str(args.bpm),
        "--time-offset",      f"{time_offset:.6f}",
    ]
    if args.apply_calibration:
        cmd.append("--apply-calibration")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


# ── Pass/fail logic ───────────────────────────────────────────────────────────

# Matches table rows: "  N  ..." where N is the note index
_NOTE_ROW_RE = re.compile(r"^\s{2}\s*\d+\s")

# Matches the pitch-status token and optional cents inside a table row
_STATUS_RE = re.compile(
    r"\b(ok|missed|pitch_error|pitch_marginal|low_amplitude|pitch_uncertain)\b"
    r"(?:\s+\(([+-]?\d+)c\))?"
)

# Notes whose |cents| >= this threshold are treated as detector artefacts,
# not intonation problems, and are excluded from the pass/fail denominator.
_ARTEFACT_CENTS = 900


def _parse_note_results(text: str) -> list[dict]:
    """Return one {status, cents} dict per note row in the analyzer table."""
    notes = []
    for line in text.splitlines():
        if not _NOTE_ROW_RE.match(line):
            continue
        m = _STATUS_RE.search(line)
        if not m:
            continue
        cents = int(m.group(2)) if m.group(2) is not None else None
        notes.append({"status": m.group(1), "cents": cents})
    return notes


def _compute_verdict(notes: list[dict]) -> tuple[bool, list[str], dict]:
    """Apply pass/fail criteria, excluding detector-uncertain notes from scoring.

    A note is detector-uncertain when its pitch_error or pitch_marginal status
    comes with |cents| >= _ARTEFACT_CENTS.  Such notes are excluded from both
    the numerator and denominator of the pitch-OK percentage so they cannot
    cause a FAIL on their own.

    Pass requires all of:
      - pitch OK among evaluable notes >= 90%
      - true pitch errors (100c < |cents| < 900c) = 0
      - missed notes = 0
    """
    n_total = len(notes)

    n_uncertain = sum(
        1 for n in notes
        if n["status"] in ("pitch_error", "pitch_marginal")
        and n["cents"] is not None
        and abs(n["cents"]) >= _ARTEFACT_CENTS
    )
    n_evaluable = n_total - n_uncertain

    n_ok = sum(1 for n in notes if n["status"] == "ok")
    n_missed = sum(1 for n in notes if n["status"] == "missed")
    n_true_errors = sum(
        1 for n in notes
        if n["status"] in ("pitch_error", "pitch_marginal")
        and (n["cents"] is None or abs(n["cents"]) < _ARTEFACT_CENTS)
    )

    pitch_ok_pct = (n_ok / n_evaluable * 100) if n_evaluable > 0 else 0.0

    reasons = []
    if pitch_ok_pct < 90:
        reasons.append(f"pitch OK {pitch_ok_pct:.0f}% of evaluable < 90%")
    if n_true_errors > 0:
        reasons.append(f"pitch errors {n_true_errors}")
    if n_missed > 0:
        reasons.append(f"missed {n_missed}")

    metrics = {
        "n_total":       n_total,
        "n_evaluable":   n_evaluable,
        "n_uncertain":   n_uncertain,
        "n_ok":          n_ok,
        "n_missed":      n_missed,
        "n_true_errors": n_true_errors,
        "pitch_ok_pct":  pitch_ok_pct,
    }

    passed = not reasons and n_evaluable > 0
    return passed, reasons, metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-command etude: render → record → analyze → verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("targets",
                   help="Target JSON file")
    p.add_argument("--progression",       required=True, metavar="FILE",
                   help="Chord progression JSON")
    p.add_argument("--bpm",               required=True, type=float, metavar="BPM",
                   help="Tempo in beats per minute")
    p.add_argument("--duration",          required=True, type=float, metavar="SECONDS",
                   help="Recording duration in seconds")
    p.add_argument("--output-prefix",     required=True, metavar="PREFIX",
                   help="Path prefix for output WAVs  (<prefix>_stereo.wav / _bass.wav)")
    # pass-through: recorder hardware
    p.add_argument("--input-device",      type=int, default=None, metavar="N",
                   help="sounddevice input device index")
    p.add_argument("--output-device",     type=int, default=None, metavar="N",
                   help="sounddevice output device index")
    p.add_argument("--sample-rate",       type=int, default=48000, metavar="HZ",
                   help="Sample rate in Hz (default: 48000)")
    p.add_argument("--count-in-beats",    type=int, default=4, metavar="N",
                   help="Count-in beats before downbeat (default: 4)")
    # pass-through: analyzer calibration
    p.add_argument("--apply-calibration", action="store_true",
                   help="Apply input_latency_ms from audio_calibration.json (analyzer only)")
    p.add_argument("--card-mode", choices=["full", "practice", "memory"],
                   default="full", metavar="{full,practice,memory}",
                   help="Exercise card detail level (default: full)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    stereo_path = f"{args.output_prefix}_stereo.wav"
    bass_path   = f"{args.output_prefix}_bass.wav"
    time_offset = args.count_in_beats * 60.0 / args.bpm

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # 1. Render exercise card
    _render_card(args.targets, args.progression, card_mode=args.card_mode)

    # 2. Prompt
    try:
        input("  Press Enter to start recording…\n")
    except KeyboardInterrupt:
        print("\n  Aborted.")
        sys.exit(0)

    # 3. Record
    print()
    ok = _record(args, stereo_path, bass_path)
    if not ok:
        print("  [recorder reported an error — check output above]")

    if not Path(bass_path).exists():
        sys.exit(f"  Bass WAV not found: {bass_path}\n  Cannot analyze.")

    # 4. Analyze
    print()
    analysis = _analyze(args, bass_path, time_offset)
    print(analysis, end="")

    # 5. Verdict
    notes = _parse_note_results(analysis)
    passed, reasons, metrics = _compute_verdict(notes)

    print("─" * 50)
    if metrics["n_uncertain"]:
        print(f"  ({metrics['n_uncertain']} note(s) excluded from scoring"
              f" — detector-uncertain |cents| ≥ {_ARTEFACT_CENTS}c)")
    if not notes:
        print("  INCONCLUSIVE  —  no note results found in analyzer output")
    elif metrics["n_evaluable"] == 0:
        print("  INCONCLUSIVE  —  all notes detector-uncertain")
    elif passed:
        print("  PASS")
    else:
        print("  FAIL  —  " + ";  ".join(reasons))
    print()


if __name__ == "__main__":
    main()
