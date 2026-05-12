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


# ── ANSI colour tables ────────────────────────────────────────────────────────

# Foreground-only style
_FG_STRING: dict[str, str] = {
    "E": "\033[91m",   # bright red
    "A": "\033[94m",   # bright blue
    "D": "\033[92m",   # bright green
    "G": "\033[95m",   # bright magenta
}

# Background style: (bg_code, fg_code)
_BG_STRING: dict[str, tuple[str, str]] = {
    "E": ("\033[41m",  "\033[97m"),   # red bg, bright white fg
    "A": ("\033[44m",  "\033[97m"),   # blue bg, bright white fg
    "D": ("\033[42m",  "\033[30m"),   # green bg, black fg
    "G": ("\033[45m",  "\033[97m"),   # magenta bg, bright white fg
}

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_NORM  = "\033[22m"   # normal intensity — turns off bold without resetting colors

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

_SUPER_FINGER: dict[int, str] = {0: "⁰", 1: "¹", 2: "²", 4: "⁴"}
_POS_COL_W = 4   # chars reserved for Simandl position indicator


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


def _degree_cell(
    deg: str,
    string_name: str,
    use_color: bool,
    style: str = "background",
    finger: int | None = None,
    show_finger: bool = False,
    pos_label: str = "",
    show_pos: bool = False,
) -> str:
    """Return a colour-coded degree cell, optionally prefixed by a position column.

    pos_label is the Roman numeral to display (e.g. "III"), or "" when the
    position has not changed.  When show_pos is True the position column is
    always emitted (_POS_COL_W wide) so cells stay aligned.
    """
    # Position prefix — always _POS_COL_W chars when enabled, uncolored.
    pos_col = pos_label.ljust(_POS_COL_W) if show_pos else ""

    # Degree label with optional superscript finger.
    finger_str = _SUPER_FINGER.get(finger, "") if show_finger and finger is not None else ""
    label = deg + finger_str
    pad   = " " * max(0, _DEG_COL_W - len(label))

    if not use_color:
        return pos_col + label + pad

    s = string_name.upper()
    is_root = deg == "R"

    if style == "background":
        bg, fg = _BG_STRING.get(s, ("", ""))
        if not bg:
            return pos_col + label + pad
        bold_on  = _BOLD if is_root else ""
        bold_off = _NORM if is_root else ""
        return pos_col + f"{bg}{fg}{bold_on}{label}{bold_off}{pad}{_RESET}"
    else:  # foreground
        fg = _FG_STRING.get(s, "")
        if not fg:
            return pos_col + label + pad
        bold = _BOLD if is_root else ""
        return pos_col + f"{fg}{bold}{label}{_RESET}{pad}"


def render_card(
    groups: list[tuple[str, list[dict]]],
    use_color: bool = True,
    style: str = "background",
    show_fingers: bool = True,
    show_positions: bool = True,
) -> list[str]:
    """Return display lines, one per chord/measure group.

    Format (with fingers + positions):
        Dm7     |I   R   ♭3⁴ I   5   ♭7⁴|
        G7      |I   R²  III ♭7⁴ ♭7¹ I   R  |
        Cmaj7   |I   3²  III 5⁴  I   7²  5⁴ |
    """
    lines = []
    prev_position: str | None = None

    for chord, targets in groups:
        cells = []
        for t in targets:
            position = t.get("position")

            if show_positions and position and position != prev_position:
                pos_label = position
            else:
                pos_label = ""

            if position:
                prev_position = position

            cells.append(_degree_cell(
                degree_label(t.get("note", "?"), chord),
                t.get("string", ""),
                use_color,
                style,
                finger=t.get("finger"),
                show_finger=show_fingers,
                pos_label=pos_label,
                show_pos=show_positions,
            ))

        chord_col = chord.ljust(_CHORD_COL_W)
        lines.append(f"{chord_col}| {''.join(cells)}|")

    return lines


def render_legend(use_color: bool = True, style: str = "background") -> str:
    """Return a one-line string legend."""
    parts = []
    for s, desc in _STRING_LABEL.items():
        if not use_color:
            parts.append(f"{s} = {desc}")
        elif style == "background":
            bg, fg = _BG_STRING.get(s, ("", ""))
            block = f"{bg}{fg} {s} {_RESET}" if bg else s
            parts.append(f"{block} = {desc}")
        else:
            fg = _FG_STRING.get(s, "")
            parts.append(f"{fg}{s}{_RESET} = {desc}")
    finger_note = "  Fingers: ⁰open  ¹index  ²middle  ⁴pinky"
    return "Strings:  " + "   ".join(parts) + "\n" + finger_note


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
    p.add_argument("--color-style", choices=["foreground", "background"],
                   default="background", metavar="{foreground,background}",
                   help="Color style: background (default) or foreground")
    p.add_argument("--show-fingers", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Show Simandl finger numbers as superscripts (default: on)")
    p.add_argument("--show-positions", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Show position markers when position changes (default: on)")
    return p.parse_args()


def main() -> None:
    args      = _parse_args()
    use_color = not args.no_color
    style     = args.color_style

    targets = load_targets(args.targets)
    with open(args.progression, encoding="utf-8") as fh:
        progression = json.load(fh)

    groups = group_targets_by_measure(targets, progression, args.beats_per_measure)
    lines  = render_card(groups, use_color, style,
                         show_fingers=args.show_fingers,
                         show_positions=args.show_positions)

    print()
    for line in lines:
        print(" ", line)
    print()
    print(" ", render_legend(use_color, style))
    print()


if __name__ == "__main__":
    main()
