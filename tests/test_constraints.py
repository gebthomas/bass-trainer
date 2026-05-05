import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constraints import note_to_pitch_class, chord_to_pitch_classes, is_note_allowed


def test_note_to_pitch_class():
    assert note_to_pitch_class("C2")  == 0
    assert note_to_pitch_class("C3")  == 0
    assert note_to_pitch_class("C#2") == 1
    assert note_to_pitch_class("Db2") == 1
    assert note_to_pitch_class("F#1") == 6
    assert note_to_pitch_class("Gb3") == 6
    assert note_to_pitch_class("Bb2") == 10


def test_chord_to_pitch_classes():
    assert chord_to_pitch_classes("Dm7")   == {2, 5, 9, 0}
    assert chord_to_pitch_classes("G7")    == {7, 11, 2, 5}
    assert chord_to_pitch_classes("Cmaj7") == {0, 4, 7, 11}
    assert chord_to_pitch_classes("Bbmaj7") == {10, 2, 5, 9}
    assert chord_to_pitch_classes("F#m7")  == {6, 9, 1, 4}


def test_is_note_allowed():
    assert     is_note_allowed("F2",  "Dm7")    # chord tone (b3)
    assert not is_note_allowed("F#2", "Dm7")    # outside
    assert     is_note_allowed("B2",  "G7")     # chord tone (3)
    assert not is_note_allowed("C2",  "G7")     # outside
    assert     is_note_allowed("B2",  "Cmaj7")  # chord tone (7)
    assert not is_note_allowed("Bb2", "Cmaj7")  # outside (b7, not in maj7)


def test_invalid_inputs():
    for fn, arg in [
        (note_to_pitch_class,   "H2"),
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
    test_is_note_allowed()
    test_invalid_inputs()


if __name__ == "__main__":
    run_all_tests()
    print("All constraint tests passed.")
