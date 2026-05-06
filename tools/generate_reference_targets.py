import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "targets" / "reference"

# MIDI note number for each open string (C4 = 60)
OPEN_MIDI = {"E": 28, "A": 33, "D": 38, "G": 43}

POSITIONS_FRETTED  = [0, 3, 5, 7, 9, 12, 15, 17, 19, 24]
POSITIONS_FRETLESS = [0, 3, 5, 7, 9, 12, 15, 17, 19]

# Upright: major-scale note sequences (~1.5 octaves) per string
UPRIGHT_NOTES = {
    "E": ["E1",  "F#1", "G#1", "A1",  "B1",  "C#2", "D#2", "E2",  "F#2", "G#2", "A2"],
    "A": ["A1",  "B1",  "C#2", "D2",  "E2",  "F#2", "G#2", "A2",  "B2",  "C#3", "D3"],
    "D": ["D2",  "E2",  "F#2", "G2",  "A2",  "B2",  "C#3", "D3",  "E3",  "F#3", "G3"],
    "G": ["G2",  "A2",  "B2",  "C3",  "D3",  "E3",  "F#3", "G3",  "A3",  "B3",  "C4"],
}

FIRST_NOTE_SEC  = 1.0
NOTE_SPACING_SEC = 3.0

# Conventional bass note names (flats preferred for b3, b6, b7)
_PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def midi_to_note(midi):
    pc = midi % 12
    octave = (midi // 12) - 1
    return f"{_PITCH_NAMES[pc]}{octave}"


def build_fret_targets(string_name, positions):
    open_midi = OPEN_MIDI[string_name]
    targets = []
    for i, fret in enumerate(positions):
        targets.append({
            "time":     round(FIRST_NOTE_SEC + i * NOTE_SPACING_SEC, 3),
            "note":     midi_to_note(open_midi + fret),
            "string":   string_name,
            "position": "open" if fret == 0 else fret,
        })
    return targets


def build_upright_targets(string_name):
    return [
        {
            "time":     round(FIRST_NOTE_SEC + i * NOTE_SPACING_SEC, 3),
            "note":     note,
            "string":   string_name,
            "position": f"scale_{i + 1}",
        }
        for i, note in enumerate(UPRIGHT_NOTES[string_name])
    ]


def build_targets(instrument, string_name):
    if instrument == "fretted":
        return build_fret_targets(string_name, POSITIONS_FRETTED)
    elif instrument == "fretless":
        return build_fret_targets(string_name, POSITIONS_FRETLESS)
    else:
        return build_upright_targets(string_name)


# ── README ───────────────────────────────────────────────────────────────────

def _fret_table(positions):
    header = " | ".join("open" if p == 0 else f"fret {p}" for p in positions)
    sep    = "-|-".join("-" * (len("open" if p == 0 else f"fret {p}") + 2) for p in positions)
    rows   = []
    for s in ["E", "A", "D", "G"]:
        notes = [midi_to_note(OPEN_MIDI[s] + p) for p in positions]
        rows.append("| " + s + " | " + " | ".join(f"{n:<6}" for n in notes) + " |")
    return header, sep, "\n".join(rows)


def _upright_table():
    rows = []
    for s in ["E", "A", "D", "G"]:
        notes = ", ".join(UPRIGHT_NOTES[s])
        rows.append(f"  {s}: {notes}")
    return "\n".join(rows)


def build_readme():
    fretted_header, fretted_sep, fretted_rows   = _fret_table(POSITIONS_FRETTED)
    fretless_header, fretless_sep, fretless_rows = _fret_table(POSITIONS_FRETLESS)

    return f"""\
# Reference target files

## Overview

- **Fretted electric** targets are fret/position-based (chromatic landmarks).
- **Fretless electric** targets are also fret-position-based but stop at fret 19
  (the upper register is less commonly used on fretless without physical landmarks).
- **Upright** targets (pizz and bow) are scale-based because there are no frets.
  Note sequences follow a major scale up approximately 1.5 octaves per string.
  Record pizzicato and arco separately.

## Recording protocol

- Leave one second of silence before the first note
- Play each target note at the time listed in the JSON file
- Sustain each note for about 2 seconds
- Leave about 1 second of silence between notes
- Use clean fingerstyle or plucked tone for electric and upright_pizz
- Use a clean bow stroke for upright_bow
- No effects or compression
- Use consistent input gain across all takes for the same instrument

## File naming

`{{instrument}}_{{string}}_string_reference.json`

Instruments: fretted, fretless, upright_pizz, upright_bow
Strings: E, A, D, G

## Fretted electric — positions: {", ".join(str(p) for p in POSITIONS_FRETTED)}

| String | {fretted_header} |
|--------|{fretted_sep}|
{fretted_rows}

## Fretless electric — positions: {", ".join(str(p) for p in POSITIONS_FRETLESS)}

| String | {fretless_header} |
|--------|{fretless_sep}|
{fretless_rows}

## Upright (pizz and bow) — scale-based sequences

Position labels are scale_1, scale_2, … (no fret numbers).

{_upright_table()}
"""


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument in ["fretted", "fretless", "upright_pizz", "upright_bow"]:
        for string_name in OPEN_MIDI:
            filename = f"{instrument}_{string_name}_string_reference.json"
            path = OUTPUT_DIR / filename
            targets = build_targets(instrument, string_name)
            with path.open("w", encoding="utf-8") as f:
                json.dump(targets, f, indent=2)
            print(f"Wrote {path.relative_to(PROJECT_ROOT)}")

    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(build_readme(), encoding="utf-8")
    print(f"Wrote {readme_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
