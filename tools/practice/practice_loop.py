#!/usr/bin/env python3
"""Adaptive BPM practice loop.

Calls practice_etude.py repeatedly.  On PASS, advances to the next BPM.
On FAIL or INCONCLUSIVE, retries the same BPM.  After each attempt the user
can override with r/s/q.

Usage:
    python tools/practice/practice_loop.py \\
        tests/targets/jazz/ii_v_i_low_position_quarters.json \\
        --progression tests/progressions/ii_v_i_C_slow.json \\
        --start-bpm 90 \\
        --max-bpm 120 \\
        --bpm-step 5 \\
        --duration 12 \\
        --output-dir tests/real_audio/jazz/practice_loop
"""

import argparse
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_ETUDE    = PROJECT_ROOT / "tools" / "practice_etude.py"
_RENDERER = PROJECT_ROOT / "tools" / "render_exercise_card.py"
_PY       = sys.executable

_ARTEFACT_CENTS = 900

# Matches note table rows emitted by run_jazz_exercise.py (--target-time-unit beats format):
#   "  N  musbeat  wavtime  target_note  detected  status..."
_NOTE_FULL_RE = re.compile(
    r"^\s{2,}\d+\s+"
    r"\S+\s+\S+\s+"          # musbeat + wavtime columns
    r"(\S+)\s+"              # target_note  (group 1)
    r"\S+\s+"                # detected
    r"\b(ok|missed|pitch_error|pitch_marginal|low_amplitude|pitch_uncertain)\b"
    r"(?:\s+\(([+-]?\d+)c\))?"
)


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run_etude(cmd: list[str]) -> str:
    """Run practice_etude.py, streaming output to the terminal while capturing it.

    The subprocess stdin is inherited so interactive prompts (Press Enter…)
    still reach the user.  -u disables Python's output buffering so lines
    appear immediately even though stdout is piped.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)

    captured: list[str] = []

    def _tee() -> None:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)

    thr = threading.Thread(target=_tee, daemon=True)
    thr.start()
    proc.wait()
    thr.join()
    return "".join(captured)


def _run_etude_quiet(cmd: list[str], targets: str, progression: str, card_mode: str = "full") -> str:
    """Show card + our own prompt, run etude silently, return captured stdout."""
    subprocess.run(
        [_PY, str(_RENDERER), targets, "--progression", progression, "--card-mode", card_mode],
        check=False,
    )
    try:
        input("  Press Enter to start recording…\n")
    except (KeyboardInterrupt, EOFError):
        print("\n  Aborted.")
        sys.exit(0)
    print("  Recording…")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        result = subprocess.run(
            cmd,
            input="\n",       # auto-advance past practice_etude's own Enter prompt
            capture_output=True,
            text=True,
            env=env,
        )
    except KeyboardInterrupt:
        print("\n  Aborted.")
        sys.exit(0)
    return result.stdout


def _parse_metrics(text: str) -> dict:
    """Re-parse the note table from captured etude output to produce compact metrics."""
    n_total = n_ok = n_uncertain = n_missed = n_true_errors = 0
    focus_notes: list[str] = []

    for line in text.splitlines():
        m = _NOTE_FULL_RE.match(line)
        if not m:
            continue
        target_note, status, cents_str = m.group(1), m.group(2), m.group(3)
        cents = int(cents_str) if cents_str is not None else None
        n_total += 1
        if status == "ok":
            n_ok += 1
        elif status == "missed":
            n_missed += 1
            focus_notes.append(target_note)
        elif status in ("pitch_error", "pitch_marginal"):
            if cents is not None and abs(cents) >= _ARTEFACT_CENTS:
                n_uncertain += 1
            else:
                n_true_errors += 1
                focus_notes.append(target_note)

    n_evaluable = n_total - n_uncertain
    pitch_ok_pct = (n_ok / n_evaluable * 100) if n_evaluable > 0 else 0.0
    return {
        "n_total":       n_total,
        "n_evaluable":   n_evaluable,
        "n_ok":          n_ok,
        "n_uncertain":   n_uncertain,
        "n_missed":      n_missed,
        "n_true_errors": n_true_errors,
        "pitch_ok_pct":  pitch_ok_pct,
        "focus_notes":   focus_notes,
    }


def _print_compact_summary(verdict: str, metrics: dict) -> None:
    print(f"  Result: {verdict}")
    print(f"  Evaluable: {metrics['n_evaluable']}/{metrics['n_total']}")
    print(f"  OK: {metrics['n_ok']}/{metrics['n_evaluable']}")
    print(f"  Detector uncertain: {metrics['n_uncertain']}")
    print(f"  Real errors: {metrics['n_true_errors']}")
    if verdict == "FAIL" and metrics["focus_notes"]:
        print(f"  Focus: {', '.join(metrics['focus_notes'])}")


def _print_status_line(bpm: float, attempt: int, last: dict | None, highest_passed: float | None) -> None:
    if last is None:
        last_str = "—"
    else:
        last_str = f"{last['verdict']}, {last['pitch_ok_pct']:.0f}% evaluable OK"
    highest_str = f"{highest_passed:.0f}" if highest_passed is not None else "—"
    print(f"  BPM {bpm:.0f} | attempt {attempt} | last: {last_str} | highest passed: {highest_str}")


def _build_etude_cmd(
    args: argparse.Namespace, bpm: float, prefix: str
) -> list[str]:
    cmd = [
        _PY, "-u", str(_ETUDE),
        args.targets,
        "--progression",    args.progression,
        "--bpm",            str(bpm),
        "--duration",       str(args.duration),
        "--output-prefix",  prefix,
        "--count-in-beats", str(args.count_in_beats),
        "--sample-rate",    str(args.sample_rate),
    ]
    if args.input_device is not None:
        cmd += ["--input-device", str(args.input_device)]
    if args.output_device is not None:
        cmd += ["--output-device", str(args.output_device)]
    if args.apply_calibration:
        cmd.append("--apply-calibration")
    cmd += ["--card-mode", args.card_mode]
    return cmd


# ── Verdict parsing ───────────────────────────────────────────────────────────

def _parse_verdict(text: str) -> str:
    """Return 'PASS', 'FAIL', 'INCONCLUSIVE', or 'UNKNOWN' from etude output."""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s == "PASS":
            return "PASS"
        if s.startswith("FAIL"):
            return "FAIL"
        if s.startswith("INCONCLUSIVE"):
            return "INCONCLUSIVE"
    return "UNKNOWN"


# ── User prompt ───────────────────────────────────────────────────────────────

def _prompt_choice(verdict: str) -> str:
    """Prompt for loop control.  Returns 'continue' | 'retry' | 'skip' | 'quit'."""
    default_desc = "advance BPM" if verdict == "PASS" else "retry"
    print(f"  [Enter] {default_desc}   [r] retry same BPM   [s] skip to next BPM   [q] quit")
    try:
        raw = input("  > ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "quit"

    return {"r": "retry", "s": "skip", "q": "quit"}.get(raw, "continue")


# ── BPM sequence ──────────────────────────────────────────────────────────────

def _bpm_sequence(start: float, max_bpm: float, step: float):
    bpm = start
    while bpm <= max_bpm + 1e-9:
        yield bpm
        bpm = round(bpm + step, 6)


# ── File naming ───────────────────────────────────────────────────────────────

def _output_prefix(output_dir: Path, stem: str, bpm: float, attempt: int) -> str:
    return str(output_dir / f"{stem}_{f'{bpm:.0f}'.zfill(3)}_attempt{attempt:02d}")


# ── Session summary ───────────────────────────────────────────────────────────

def _print_summary(log: list[dict]) -> None:
    print()
    print("═" * 50)
    print("  SESSION SUMMARY")
    print("═" * 50)

    highest = None
    for entry in log:
        mark = "PASS" if entry["passed"] else "—   "
        print(f"  {entry['bpm']:>6.0f} BPM  {entry['attempts']:>2} attempt(s)  {mark}")
        if entry["passed"]:
            highest = entry["bpm"]

    print()
    if highest is not None:
        print(f"  Highest BPM passed: {highest:.0f}")
    else:
        print("  No BPM levels passed this session.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive BPM loop: repeat until PASS, then advance tempo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("targets",
                   help="Target JSON file")
    p.add_argument("--progression",       required=True, metavar="FILE",
                   help="Chord progression JSON")
    p.add_argument("--start-bpm",         required=True, type=float, metavar="BPM",
                   help="Starting tempo")
    p.add_argument("--max-bpm",           required=True, type=float, metavar="BPM",
                   help="Stop after passing this tempo")
    p.add_argument("--bpm-step",          type=float, default=5.0, metavar="STEP",
                   help="BPM increment on each pass (default: 5)")
    p.add_argument("--duration",          required=True, type=float, metavar="SECONDS",
                   help="Recording duration per attempt in seconds")
    p.add_argument("--output-dir",        required=True, metavar="DIR",
                   help="Directory for WAV output files")
    # pass-through hardware options
    p.add_argument("--input-device",      type=int, default=None, metavar="N",
                   help="sounddevice input device index")
    p.add_argument("--output-device",     type=int, default=None, metavar="N",
                   help="sounddevice output device index")
    p.add_argument("--sample-rate",       type=int, default=48000, metavar="HZ",
                   help="Sample rate in Hz (default: 48000)")
    p.add_argument("--count-in-beats",    type=int, default=4, metavar="N",
                   help="Count-in beats before downbeat (default: 4)")
    p.add_argument("--apply-calibration", action="store_true",
                   help="Pass --apply-calibration to the analyzer")
    p.add_argument("--quiet-practice", action="store_true",
                   help="Suppress etude output; show compact musician-facing summary")
    p.add_argument("--card-mode", choices=["full", "practice", "memory"],
                   default="full", metavar="{full,practice,memory}",
                   help="Exercise card detail level (default: full)")
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.start_bpm > args.max_bpm:
        sys.exit("error: --start-bpm must be <= --max-bpm")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_stem = Path(args.targets).stem

    session_log: list[dict] = []
    highest_passed: float | None = None

    for bpm in _bpm_sequence(args.start_bpm, args.max_bpm, args.bpm_step):
        print()
        print("═" * 50)
        print(f"  BPM: {bpm:.0f}")
        print("═" * 50)

        attempt = 1
        last_attempt: dict | None = None
        session_log.append({"bpm": bpm, "attempts": 0, "passed": False})

        while True:
            session_log[-1]["attempts"] = attempt
            prefix = _output_prefix(output_dir, target_stem, bpm, attempt)
            cmd    = _build_etude_cmd(args, bpm, prefix)

            if args.quiet_practice:
                print()
                _print_status_line(bpm, attempt, last_attempt, highest_passed)
                output  = _run_etude_quiet(cmd, args.targets, args.progression, args.card_mode)
                verdict = _parse_verdict(output)
                metrics = _parse_metrics(output)
                print()
                _print_compact_summary(verdict, metrics)
                last_attempt = {"verdict": verdict, "pitch_ok_pct": metrics["pitch_ok_pct"]}
            else:
                output  = _run_etude(cmd)
                verdict = _parse_verdict(output)

            if verdict == "PASS":
                session_log[-1]["passed"] = True
                if highest_passed is None or bpm > highest_passed:
                    highest_passed = bpm

            print()
            choice = _prompt_choice(verdict)
            print()

            if choice == "quit":
                _print_summary(session_log)
                sys.exit(0)

            advance = (choice == "continue" and verdict == "PASS") or choice == "skip"
            if advance:
                break

            # retry: keep same BPM, increment attempt counter
            attempt += 1

    _print_summary(session_log)


if __name__ == "__main__":
    main()
