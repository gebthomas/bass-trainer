"""
Generate target JSON files for bass practice scales.

Usage:
    python tools/generate_scale_targets.py --root C
    python tools/generate_scale_targets.py --root F# --bpm 80 --subdivision quarter
    python tools/generate_scale_targets.py --cycle-fifths
    python tools/generate_scale_targets.py --cycle-fifths --bpm 72 --output my_scales.json
"""

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TARGETS_DIR = ROOT_DIR / "tests" / "targets" / "scales"

CYCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "Ab", "Eb", "Bb", "F"]

# Seven scale-degree note names for each major key.
# Enharmonic simplifications: E# → F, B# → C (keeps names bass-readable).
MAJOR_SCALE_NAMES: dict[str, list[str]] = {
    "C":  ["C",  "D",  "E",  "F",  "G",  "A",  "B"],
    "G":  ["G",  "A",  "B",  "C",  "D",  "E",  "F#"],
    "D":  ["D",  "E",  "F#", "G",  "A",  "B",  "C#"],
    "A":  ["A",  "B",  "C#", "D",  "E",  "F#", "G#"],
    "E":  ["E",  "F#", "G#", "A",  "B",  "C#", "D#"],
    "B":  ["B",  "C#", "D#", "E",  "F#", "G#", "A#"],
    "F#": ["F#", "G#", "A#", "B",  "C#", "D#", "F"],   # E# → F
    "C#": ["C#", "D#", "F",  "F#", "G#", "A#", "C"],   # E# → F, B# → C
    "Ab": ["Ab", "Bb", "C",  "Db", "Eb", "F",  "G"],
    "Eb": ["Eb", "F",  "G",  "Ab", "Bb", "C",  "D"],
    "Bb": ["Bb", "C",  "D",  "Eb", "F",  "G",  "A"],
    "F":  ["F",  "G",  "A",  "Bb", "C",  "D",  "E"],
}

# Bass-friendly root MIDI numbers (C4 = 60, so C1 = 24, C2 = 36).
ROOT_MIDI: dict[str, int] = {
    "C": 36, "G": 31, "D": 38, "A": 33, "E": 28, "B": 35,
    "F#": 30, "C#": 37, "Ab": 32, "Eb": 39, "Bb": 34, "F": 29,
}

# Major scale semitone offsets for degrees 1-8 (inclusive of octave).
MAJOR_SEMITONES = [0, 2, 4, 5, 7, 9, 11, 12]

SUBDIVISIONS: dict[str, int] = {"eighth": 2, "quarter": 1, "sixteenth": 4}
SUBDIVISION_LABEL: dict[str, str] = {
    "eighth": "eighths", "quarter": "quarters", "sixteenth": "sixteenths",
}


def note_with_octave(midi: int, name: str) -> str:
    """Return note name + octave derived from MIDI number (C4 = 60 convention)."""
    octave = midi // 12 - 1
    return f"{name}{octave}"


def major_scale_notes(root: str) -> list[str]:
    """15 note names for one octave ascending then descending (root ... octave ... root)."""
    names = MAJOR_SCALE_NAMES[root]
    root_midi = ROOT_MIDI[root]

    ascending = [
        note_with_octave(root_midi + MAJOR_SEMITONES[i], names[i] if i < 7 else names[0])
        for i in range(8)
    ]
    descending = [
        note_with_octave(root_midi + MAJOR_SEMITONES[i], names[i])
        for i in range(6, -1, -1)
    ]
    return ascending + descending


def beat_spacing(bpm: int, subdivision: str) -> float:
    return (60.0 / bpm) / SUBDIVISIONS[subdivision]


def build_targets(notes: list[str], bpm: int, subdivision: str,
                  start_time: float = 1.0) -> tuple[list[dict], float]:
    """Return (target list, time after last note)."""
    spacing = beat_spacing(bpm, subdivision)
    targets = []
    t = start_time
    for note in notes:
        targets.append({"time": round(t, 6), "note": note})
        t = round(t + spacing, 6)
    return targets, t


def target_filename(tag: str, bpm: int, subdivision: str) -> str:
    return f"major_{tag}_{bpm}bpm_{SUBDIVISION_LABEL[subdivision]}.json"


def save_targets(targets: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(targets, fh, indent=2)
    print(f"Saved {len(targets):3d} targets → {path}")


def cmd_root(args: argparse.Namespace) -> None:
    root = args.root
    if root not in MAJOR_SCALE_NAMES:
        valid = ", ".join(CYCLE_OF_FIFTHS)
        raise SystemExit(f"Unknown root: {root!r}. Valid roots: {valid}")

    notes = major_scale_notes(root)
    targets, _ = build_targets(notes, args.bpm, args.subdivision)

    out = (Path(args.output) if args.output
           else TARGETS_DIR / target_filename(root, args.bpm, args.subdivision))
    save_targets(targets, out)


def cmd_cycle(args: argparse.Namespace) -> None:
    beat_dur = 60.0 / args.bpm
    gap_sec = 2 * beat_dur   # two-beat rest between keys in the combined file

    combined: list[dict] = []
    current_time = 1.0

    for root in CYCLE_OF_FIFTHS:
        notes = major_scale_notes(root)

        # Individual file always starts at 1.0s
        ind_targets, _ = build_targets(notes, args.bpm, args.subdivision, start_time=1.0)
        ind_path = TARGETS_DIR / target_filename(root, args.bpm, args.subdivision)
        save_targets(ind_targets, ind_path)

        # Combined file continues from current position
        part, next_t = build_targets(notes, args.bpm, args.subdivision, start_time=current_time)
        combined.extend(part)
        current_time = round(next_t + gap_sec, 6)

    combined_path = (Path(args.output) if args.output
                     else TARGETS_DIR / target_filename("cycle_5ths", args.bpm, args.subdivision))
    save_targets(combined, combined_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bass scale target JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", metavar="NOTE",
                       help="Generate one major scale (e.g. C, G, F#, Ab)")
    group.add_argument("--cycle-fifths", action="store_true",
                       help="Generate all 12 major scales in cycle-of-fifths order")
    parser.add_argument("--bpm", type=int, default=60,
                        help="Tempo in BPM (default: 60)")
    parser.add_argument("--subdivision", choices=list(SUBDIVISIONS), default="eighth",
                        help="Note spacing: eighth, quarter, or sixteenth (default: eighth)")
    parser.add_argument("--output", metavar="PATH",
                        help="Override the output file path (combined file for --cycle-fifths)")

    args = parser.parse_args()
    if args.cycle_fifths:
        cmd_cycle(args)
    else:
        cmd_root(args)


if __name__ == "__main__":
    main()
