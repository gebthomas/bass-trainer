import json
import re

# "M-N/D"  →  measure M (1-indexed), beat offset N/D of the measure
_TIME_RE = re.compile(r"^(\d+)-(\d+)/(\d+)$")

# "N/D"    →  N/D of one measure
_DUR_RE  = re.compile(r"^(\d+)/(\d+)$")


def _parse_time(val, beats_per_measure: float) -> tuple[float, str | None]:
    """Return (absolute_beats, source_string_or_None).

    Numeric values pass through unchanged.
    String "M-N/D": absolute_beats = (M-1)*bpm + (N/D)*bpm
    """
    if isinstance(val, (int, float)):
        return float(val), None
    m = _TIME_RE.match(str(val))
    if not m:
        raise ValueError(f"Unrecognised time format: {val!r}  (expected int/float or 'M-N/D')")
    measure, num, denom = int(m.group(1)), int(m.group(2)), int(m.group(3))
    beats = (measure - 1) * beats_per_measure + (num / denom) * beats_per_measure
    return beats, str(val)


def _parse_duration(val, beats_per_measure: float) -> tuple[float, str | None]:
    """Return (duration_beats, source_string_or_None).

    Numeric values pass through unchanged.
    String "N/D": duration_beats = (N/D) * beats_per_measure
      "1/1"  = whole measure
      "1/2"  = half measure
      "1/4"  = quarter note  (1 beat in 4/4)
      "1/8"  = eighth note   (0.5 beat in 4/4)
      "1/16" = sixteenth note (0.25 beat in 4/4)
    """
    if isinstance(val, (int, float)):
        return float(val), None
    m = _DUR_RE.match(str(val))
    if not m:
        raise ValueError(f"Unrecognised duration format: {val!r}  (expected int/float or 'N/D')")
    num, denom = int(m.group(1)), int(m.group(2))
    return (num / denom) * beats_per_measure, str(val)


def load_targets(filename="targets.json", beats_per_measure: int = 4) -> list[dict]:
    """Load and normalise a target JSON file.

    Numeric "time" values are preserved as-is (cast to float).
    String "time" values ("M-N/D") are converted to absolute beat positions.

    Optional "duration" field (numeric or "N/D" string) is converted to
    "duration_beats" (float).  The original string is kept in "source_time"
    or "source_duration" when the input was symbolic.

    No file is written; normalisation happens in memory only.
    """
    with open(filename, "r") as f:
        raw = json.load(f)

    result = []
    for entry in raw:
        t = dict(entry)

        raw_time = t["time"]
        beats, src_time = _parse_time(raw_time, beats_per_measure)
        t["time"] = beats
        if src_time is not None:
            t["source_time"] = src_time

        if "duration" in t:
            raw_dur = t.pop("duration")
            dur_beats, src_dur = _parse_duration(raw_dur, beats_per_measure)
            t["duration_beats"] = dur_beats
            if src_dur is not None:
                t["source_duration"] = src_dur

        result.append(t)

    return result


def nearest_target(onset_time, targets):
    return min(targets, key=lambda t: abs(t["time"] - onset_time))

def timing_error_ms(onset_time, target_time):
    return round((onset_time - target_time) * 1000)

def compare_note(detected_note, target_note):
    return detected_note == target_note
