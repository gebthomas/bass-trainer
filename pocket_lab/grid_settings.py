"""Grid source metadata, settings I/O, and annotation labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GridSource:
    """Describes how the beat grid was generated.

    Future grid sources may include:
      - fixed_bpm              (current: user-supplied tempo and meter)
      - manually_adjusted      (user-corrected beat_zero_s)
      - song_audio_beat_track  (estimated from song channel)
      - annotated_beat_map     (hand-annotated beat positions)
    """
    method: str
    description: str


GRID_SOURCE_FIXED_BPM = GridSource(
    method="fixed_bpm",
    description=(
        "Beat locations are generated from user-supplied BPM, meter, "
        "shuffle fraction, and beat-zero time. They are not yet "
        "estimated from the song audio."
    ),
)


def load_grid_settings(path: str | Path) -> dict:
    """Load grid settings from a JSON file."""
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_grid_settings(
    path: str | Path,
    *,
    bpm: float,
    beats_per_measure: int,
    shuffle_fraction: float,
    beat_zero_s: float,
    source_file: str,
    notes: str = "",
) -> None:
    """Write grid settings to a JSON file."""
    import json
    data = {
        "bpm": bpm,
        "beats_per_measure": beats_per_measure,
        "shuffle_fraction": shuffle_fraction,
        "beat_zero_s": beat_zero_s,
        "source_file": source_file,
        "notes": notes,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


ANNOTATION_LABELS = [
    "true_attack", "string_noise", "passing_note", "downbeat",
    "beat3", "fill", "ignore", "uncertain",
]
