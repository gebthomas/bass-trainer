#!/usr/bin/env python3
"""
Compact jazz bass exercise card renderer.

Displays chord names and scale degrees colour-coded by intended string.
No note names, fret numbers, or tablature in the compact view.

Usage:
    python tools/render_exercise_card.py <targets.json> \\
        --progression <progression.json>
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.targets import load_targets
from core.constraints import chord_at_time, PITCH_CLASS


# ── ANSI colours ──────────────────────────────────────────────────────────────

_ANSI_STRING: dict[str, str] = {
    "E": "\033[91m",   # bright red
    "A": "\033[94m",   # bright blue
    "D": "\033[92m",   # bright green
    "G": "\033[95m",   # bright magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"

_STRING_LABEL: dict[str, str] = {
    "E": "red",
    "A": "blue",
    "D": "green",
    "G": "magenta",
}


# ── Scale degree tables ───────────────────────────────────────────────────────

# Semitone interval from chord root → degree name, per chord quality.
# Chord tones use plain numbers; extensions use jazz convention.
# Notes outside the mode are labelled by their chromatic function.
_DEGREE_LABELS: dict[str, dict[int, str]] = {
    "m7": {    # Dorian
        0:  "R",   1: "♭9",  2: "9",   3: "♭3", 4: "♮3",
        5:  "11",  6: "♭5",  7: "5",   8: "♭6", 9: "13",
        10: "♭7",  11: "♯7",
    },
    "7": {     # Mixolydian
        0:  "R",   1: "♭9",  2: "9",   3: "♯9", 4: "3",
        5:  "11",  6: "♯11", 7: "5",   8: "♭13",9: "13",
        10: "♭7",  11: "♮7",
    },
    "maj7": {  # Ionian
        0:  "R",   1: "♭9",  2: "9",   3: "♭3", 4: "3",
        5:  "11",  6: "♯11", 7: "5",   8: "♭6", 9: "13",
        10: "♭7",  11: "7",
    },
}

# Fallback for unknown chord qualities
_CHROMATIC_LABELS: dict[int, str] = {
    0: "R",  1: "♭2", 2: "2",  3: "♭3", 4: "3",
    5: "4",  6: "♭5", 7: "5",  8: "♭6", 9: "6",
    10: "♭7", 11: "7",
}

_NOTE_RE  = re.compile(r"^([A-G][#b]?)")
_CHORD_RE = re.compile(r"^([A-G][#b]?)(maj7|m7|7)$")


# ── Helper functions ──────────────────────────────────────────────────────────

def degree_label(note: str, chord: str) -> str:
    """Return the scale degree label for note played over chord.

    Examples:
        degree_label("F2",  "Dm7")   → "♭3"
        degree_label("B2",  "Dm7")   → "13"
        degree_label("C#2", "G7")    → "♯11"
        degree_label("B3",  "Cmaj7") → "7"
    """
    note_m  = _NOTE_RE.match(note)
    chord_m = _CHORD_RE.match(chord)
    if not note_m or not chord_m:
        return "?"
    note_pc = PITCH_CLASS.get(note_m.group(1))
    root_pc = PITCH_CLASS.get(chord_m.group(1))
    quality = chord_m.group(2)
    if note_pc is None or root_pc is None:
        return "?"
    interval = (note_pc - root_pc) % 12
    return _DEGREE_LABELS.get(quality, _CHROMATIC_LABELS).get(interval, "?")


def string_color(string_name: str, use_color: bool = True) -> tuple[str, str]:
    """Return (ANSI prefix, ANSI suffix) for a bass string name (E/A/D/G)."""
    if not use_color:
        return "", ""
    code = _ANSI_STRING.get(string_name.upper(), "")
    return (code, _RESET) if code else ("", "")


def group_targets_by_measure(
    targets: list[dict],
    progression: list[dict],
    beats_per_measure: int = 4,
) -> list[tuple[str, list[dict]]]:
    """Group targets into (chord, [target_list]) pairs.

    A new group starts when the chord changes or beats_per_measure notes
    have accumulated — whichever comes first.  Uses musical target time
    for chord lookup; never applies any WAV time offset.
    """
    groups: list[tuple[str, list]] = []
    current_chord: str | None = None
    current_group: list[dict] = []

    for t in targets:
        chord = chord_at_time(progression, t["time"], loop=False) or "?"
        if chord != current_chord or len(current_group) >= beats_per_measure:
            if current_group:
                groups.append((current_chord, current_group))
            current_chord = chord
            current_group = [t]
        else:
            current_group.append(t)

    if current_group:
        groups.append((current_chord, current_group))

    return groups


# ── Cell and row rendering ────────────────────────────────────────────────────

_CHORD_COL_W = 8   # visible characters reserved for the chord name column
_DEG_COL_W   = 4   # visible characters per degree cell


def _degree_cell(deg: str, string_name: str, use_color: bool) -> str:
    """Return a colour-coded, fixed-width degree cell string."""
    fg, rst = string_color(string_name, use_color)
    # Root is bold so it reads clearly as the anchor note.
    bold = _BOLD if (deg == "R" and use_color and fg) else ""
    end  = _RESET if bold else rst
    label = f"{fg}{bold}{deg}{end}"
    pad   = " " * max(0, _DEG_COL_W - len(deg))
    return label + pad


def render_card(
    groups: list[tuple[str, list[dict]]],
    use_color: bool = True,
) -> list[str]:
    """Return display lines, one per chord/measure group.

    Format:
        Dm7     | R   ♭3  5   ♭7  |
        G7      | 3   R   ♭7  5   |
        Cmaj7   | 3   5   7   5   |
    """
    lines = []
    for chord, targets in groups:
        cells = "".join(
            _degree_cell(
                degree_label(t.get("note", "?"), chord),
                t.get("string", ""),
                use_color,
            )
            for t in targets
        )
        chord_col = chord.ljust(_CHORD_COL_W)
        lines.append(f"{chord_col}| {cells}|")
    return lines


def render_legend(use_color: bool = True) -> str:
    """Return a one-line string legend."""
    parts = []
    for s, desc in _STRING_LABEL.items():
        fg, rst = string_color(s, use_color)
        parts.append(f"{fg}{s}{rst} = {desc}")
    return "Strings:  " + "   ".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a compact harmonic exercise card in the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("targets",
                   help="Target JSON file")
    p.add_argument("--progression", required=True, metavar="FILE",
                   help="Chord progression JSON")
    p.add_argument("--beats-per-measure", type=int, default=4, metavar="N",
                   help="Notes per display row (default: 4)")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour output")
    return p.parse_args()


def main() -> None:
    args      = _parse_args()
    use_color = not args.no_color

    targets = load_targets(args.targets)
    with open(args.progression, encoding="utf-8") as fh:
        progression = json.load(fh)

    groups = group_targets_by_measure(targets, progression, args.beats_per_measure)
    lines  = render_card(groups, use_color)

    print()
    for line in lines:
        print(" ", line)
    print()
    print(" ", render_legend(use_color))
    print()


if __name__ == "__main__":
    main()
