import re

PITCH_CLASS: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

CHORD_INTERVALS: dict[str, list[int]] = {
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "7":    [0, 4, 7, 10],
}

_NOTE_RE  = re.compile(r"([A-G][#b]?)\d*$")
_CHORD_RE = re.compile(r"([A-G][#b]?)(maj7|m7|7)$")


def note_to_pitch_class(note: str) -> int:
    m = _NOTE_RE.match(note)
    if not m:
        raise ValueError(f"Cannot parse note: {note!r}")
    return PITCH_CLASS[m.group(1)]


def chord_to_pitch_classes(chord: str) -> set[int]:
    m = _CHORD_RE.match(chord)
    if not m:
        raise ValueError(f"Cannot parse chord: {chord!r}")
    root_pc = PITCH_CLASS[m.group(1)]
    return {(root_pc + i) % 12 for i in CHORD_INTERVALS[m.group(2)]}


def is_note_allowed(note: str, chord: str) -> bool:
    return note_to_pitch_class(note) in chord_to_pitch_classes(chord)
