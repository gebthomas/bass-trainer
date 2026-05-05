import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constraints import (
    note_to_pitch_class,
    chord_to_pitch_classes,
    chord_to_scale_pitch_classes,
    classify_note_against_chord,
    is_note_chord_tone,
    is_note_allowed,
)


def test_note_to_pitch_class():
    assert note_to_pitch_class("C2")  == 0
    assert note_to_pitch_class("C3")  == 0
    assert note_to_pitch_class("C#2") == 1
    assert note_to_pitch_class("Db2") == 1
    assert note_to_pitch_class("F#1") == 6
    assert note_to_pitch_class("Gb3") == 6
    assert note_to_pitch_class("Bb2") == 10


def test_chord_to_pitch_classes():
    assert chord_to_pitch_classes("Dm7")    == {2, 5, 9, 0}
    assert chord_to_pitch_classes("G7")     == {7, 11, 2, 5}
    assert chord_to_pitch_classes("Cmaj7")  == {0, 4, 7, 11}
    assert chord_to_pitch_classes("Bbmaj7") == {10, 2, 5, 9}
    assert chord_to_pitch_classes("F#m7")   == {6, 9, 1, 4}


def test_chord_to_scale_pitch_classes():
    # D dorian: D E F G A B C
    assert chord_to_scale_pitch_classes("Dm7")   == {2, 4, 5, 7, 9, 11, 0}
    # G mixolydian: G A B C D E F
    assert chord_to_scale_pitch_classes("G7")    == {7, 9, 11, 0, 2, 4, 5}
    # C ionian: C D E F G A B
    assert chord_to_scale_pitch_classes("Cmaj7") == {0, 2, 4, 5, 7, 9, 11}


def test_classify_note_against_chord():
    # Dm7
    assert classify_note_against_chord("E2",  "Dm7") == "scale"   # dorian 2nd
    assert classify_note_against_chord("F2",  "Dm7") == "chord"   # b3
    assert classify_note_against_chord("F#2", "Dm7") == "out"

    # G7
    assert classify_note_against_chord("A2",  "G7")  == "scale"   # mixolydian 2nd
    assert classify_note_against_chord("B2",  "G7")  == "chord"   # 3rd
    assert classify_note_against_chord("Ab2", "G7")  == "out"

    # Cmaj7
    assert classify_note_against_chord("D2",  "Cmaj7") == "scale"  # ionian 2nd
    assert classify_note_against_chord("E2",  "Cmaj7") == "chord"  # 3rd
    assert classify_note_against_chord("Bb2", "Cmaj7") == "out"


def test_is_note_chord_tone():
    assert     is_note_chord_tone("F2",  "Dm7")   # b3
    assert not is_note_chord_tone("E2",  "Dm7")   # scale tone, not chord tone
    assert not is_note_chord_tone("F#2", "Dm7")   # outside


def test_is_note_allowed():
    # chord tones → allowed
    assert is_note_allowed("F2",  "Dm7")
    assert is_note_allowed("B2",  "G7")
    assert is_note_allowed("B2",  "Cmaj7")
    # scale tones → also allowed
    assert is_note_allowed("E2",  "Dm7")    # dorian 2nd
    assert is_note_allowed("C2",  "G7")     # mixolydian b7 (was "outside" under old strict def)
    assert is_note_allowed("D2",  "Cmaj7")  # ionian 2nd
    # outside notes → not allowed
    assert not is_note_allowed("F#2", "Dm7")
    assert not is_note_allowed("Ab2", "G7")
    assert not is_note_allowed("Bb2", "Cmaj7")


def test_invalid_inputs():
    for fn, arg in [
        (note_to_pitch_class,    "H2"),
        (chord_to_pitch_classes, "C13"),
        (chord_to_pitch_classes, "Cmin7"),
    ]:
        try:
            fn(arg)
            raise AssertionError(f"Expected ValueError for {fn.__name__}({arg!r})")
        except ValueError:
            pass


def run_all_tests():
    test_note_to_pitch_class()
    test_chord_to_pitch_classes()
    test_chord_to_scale_pitch_classes()
    test_classify_note_against_chord()
    test_is_note_chord_tone()
    test_is_note_allowed()
    test_invalid_inputs()


if __name__ == "__main__":
    run_all_tests()
    print("All constraint tests passed.")
