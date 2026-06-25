#!/usr/bin/env python3
"""Diagnose pyin pitch detection across all reference audio files.

Runs the analysis engine from tools/audio/analyze_fast_reference on every known
(wav, target) pair and prints a summary table.  Highlights undertone events
(detected pitch ≥ 900c below expected) and overtone events (≥ 900c above).
Does not modify any algorithm.

Usage:
    python tools/diagnostics/diagnose_pitch_accuracy.py
    python tools/diagnostics/diagnose_pitch_accuracy.py --filter A_string
    python tools/diagnostics/diagnose_pitch_accuracy.py --filter fast
    python tools/diagnostics/diagnose_pitch_accuracy.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np

from tools.audio.analyze_fast_reference import analyze_target, SAMPLE_RATE


# ── Manifest ──────────────────────────────────────────────────────────────────

def _manifest(root: Path) -> list[tuple[Path, Path, str]]:
    """Return (wav, target_json, label) for every known reference pair."""
    ref   = root / "tests/real_audio/reference"
    tgts  = root / "tests/targets/reference"
    fref  = root / "tests/real_audio/reference/fast_reference"
    ftgts = root / "tests/targets/fast_reference"

    cases: list[tuple[Path, Path, str]] = []

    # Sustained single-string reference recordings
    for inst in ("fretted", "fretless", "upright_bow", "upright_pizz"):
        for string in ("A", "D", "E", "G"):
            slug = f"{inst}_{string}_string_reference"
            wav  = ref  / inst / f"{slug}.wav"
            tgt  = tgts / f"{slug}.json"
            if wav.exists() and tgt.exists():
                cases.append((wav, tgt, f"{inst}/{string}"))

    # Fast-passage reference recordings
    fast_slugs = [
        "upright_bow_A_string_medium_low",
        "upright_pizz_D_string_fast_mid",
        "fretless_G_string_fast_high",
        "fretted_G_string_fast_high",
    ]
    for slug in fast_slugs:
        wav = fref  / f"{slug}.wav"
        tgt = ftgts / f"{slug}.json"
        if wav.exists() and tgt.exists():
            cases.append((wav, tgt, f"fast/{slug}"))

    # Repeated A1 fretless (regression for A-string undertone checks)
    cases.append((
        root / "tests/real_audio/fretless_finger/repeated_60_A1_fretless.wav",
        root / "tests/targets/repeat_A1.json",
        "fretless_finger/repeat_A1",
    ))

    return cases


# ── Per-note result ───────────────────────────────────────────────────────────

def _flag(cents_error: float | None) -> str:
    if cents_error is None:
        return ""
    if cents_error <= -900:
        return "UNDERTONE"
    if cents_error >= 900:
        return "OVERTONE"
    if abs(cents_error) > 100:
        return "error"
    if abs(cents_error) > 50:
        return "marginal"
    return ""


def _run_file(wav: Path, tgt: Path, label: str, verbose: bool) -> list[dict]:
    """Run per-target analysis and return a list of result dicts."""
    import json
    with open(tgt) as fh:
        targets = json.load(fh)
    audio = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)[0]

    rows = []
    for i, t in enumerate(targets):
        r = analyze_target(i, targets, audio, 0.0)
        cents = r["cents_error"]
        rows.append({
            "label":        label,
            "note_idx":     i,
            "target_note":  t["note"],
            "detected":     r["candidate_note"] or "—",
            "cents_error":  cents,
            "confidence":   r["confidence_tier"],
            "pitch_status": r["pitch_status"],
            "voiced_frac":  r["voiced_fraction"],
            "consensus":    r["pitch_consensus"],
            "flag":         _flag(cents),
            "votes":        r.get("semitone_votes", []),
        })

    if verbose:
        _print_votes(label, targets, rows)

    return rows


def _print_votes(label: str, targets: list[dict], rows: list[dict]) -> None:
    print(f"\n  Votes for {label}:")
    for r in rows:
        t_note = r["target_note"]
        votes = r["votes"]
        vote_str = "  ".join(f"{n}:{v*100:.0f}%" for n, v in votes) if votes else "—"
        print(f"    {t_note:>5} → {r['detected']:>5}  {vote_str}")


# ── Table printing ────────────────────────────────────────────────────────────

_COL_LABEL = 32
_COL_IDX   = 3
_COL_NOTE  = 5
_COL_CENTS = 6
_COL_TIER  = 9
_COL_STAT  = 22


def _header() -> str:
    return (
        f"{'File':<{_COL_LABEL}}  {'#':>{_COL_IDX}}"
        f"  {'Tgt':>{_COL_NOTE}}  {'Det':>{_COL_NOTE}}  {'¢':>{_COL_CENTS}}"
        f"  {'Confidence':<{_COL_TIER}}  {'Status'}"
    )


def _row(r: dict, first_in_file: bool) -> str:
    label = r["label"] if first_in_file else ""
    cents_s = f"{r['cents_error']:+.0f}" if r["cents_error"] is not None else "—"
    flag    = f"  [{r['flag']}]" if r["flag"] else ""
    return (
        f"{label:<{_COL_LABEL}}  {r['note_idx']+1:>{_COL_IDX}}"
        f"  {r['target_note']:>{_COL_NOTE}}  {r['detected']:>{_COL_NOTE}}  {cents_s:>{_COL_CENTS}}"
        f"  {r['confidence']:<{_COL_TIER}}  {r['pitch_status']}{flag}"
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(all_rows: list[dict]) -> None:
    total       = len(all_rows)
    ok          = sum(1 for r in all_rows if r["pitch_status"] == "ok")
    marginal    = sum(1 for r in all_rows if r["pitch_status"] == "pitch_marginal")
    errors      = sum(1 for r in all_rows if r["pitch_status"] == "pitch_error")
    missed      = sum(1 for r in all_rows if r["pitch_status"] in ("missed", "pitch_uncertain", "pitch_low_confidence"))
    undertones  = sum(1 for r in all_rows if r["flag"] == "UNDERTONE")
    overtones   = sum(1 for r in all_rows if r["flag"] == "OVERTONE")

    print()
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print(f"  Total targets:   {total}")
    print(f"  OK:              {ok}  ({ok/total*100:.0f}%)")
    if marginal:
        print(f"  Marginal >50c:   {marginal}")
    if errors:
        print(f"  Pitch errors:    {errors}  (>100c or status=pitch_error)")
    if missed:
        print(f"  Missed/uncertain:{missed}")
    print()
    if undertones:
        print(f"  ⬇ UNDERTONE events (detected ≥ 900c below expected):  {undertones}")
    if overtones:
        print(f"  ⬆ OVERTONE events (detected ≥ 900c above expected):   {overtones}")
    if not undertones and not overtones:
        print("  No undertone or overtone events detected.")

    issues = [r for r in all_rows if r["flag"] in ("UNDERTONE", "OVERTONE", "error")]
    if issues:
        print()
        print("  Notable events:")
        for r in issues:
            cents_s = f"{r['cents_error']:+.0f}c" if r["cents_error"] is not None else "—"
            tag     = f"[{r['flag']}]" if r["flag"] else ""
            print(f"    {r['label']:<32}  #{r['note_idx']+1}  "
                  f"{r['target_note']:>5} → {r['detected']:>5}  {cents_s:>7}  "
                  f"{r['confidence']:<9}  {r['pitch_status']}  {tag}")

    # A-string open A1 summary
    a1_rows = [r for r in all_rows if r["target_note"] == "A1" and r["note_idx"] == 0]
    if a1_rows:
        print()
        print("  Open A1 (55 Hz) detection across all A-string recordings:")
        for r in a1_rows:
            cents_s = f"{r['cents_error']:+.0f}c" if r["cents_error"] is not None else "—"
            print(f"    {r['label']:<32}  {r['detected']:>5}  {cents_s:>7}"
                  f"  {r['confidence']:<9}  {r['pitch_status']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose pyin pitch detection on reference recordings")
    p.add_argument(
        "--filter", default="",
        metavar="SUBSTR",
        help="Only process files whose label contains SUBSTR (e.g. 'A_string', 'fast', 'pizz')",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print semitone vote distribution for every note",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cases = _manifest(PROJECT_ROOT)
    if args.filter:
        cases = [(w, t, lbl) for w, t, lbl in cases if args.filter.lower() in lbl.lower()]
        if not cases:
            print(f"No cases match filter {args.filter!r}")
            return

    print(_header())
    print("─" * 85)

    all_rows: list[dict] = []
    for wav, tgt, label in cases:
        rows = _run_file(wav, tgt, label, args.verbose)
        first = True
        for r in rows:
            print(_row(r, first))
            first = False
        all_rows.extend(rows)

    _print_summary(all_rows)
    print()


if __name__ == "__main__":
    main()
