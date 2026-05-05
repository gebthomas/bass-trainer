import csv
from datetime import datetime
from pathlib import Path
import numpy as np


FIELDNAMES = [
    "date", "time", "mode", "exercise_name", "bpm",
    "n_hits", "n_misses", "n_extras",
    "pitch_accuracy_pct", "mean_timing_error_ms", "mean_abs_timing_error_ms",
    "chord_pct", "scale_pct", "out_pct", "total_score", "score_per_note",
]


def append_practice_log(results_logger, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    results = results_logger.results
    hits   = [r for r in results if r.get("event_type") == "hit"]
    misses = [r for r in results if r.get("event_type") == "missed"]
    extras = [r for r in results if r.get("event_type") == "extra"]

    n_hits   = len(hits)
    n_misses = len(misses)
    n_extras = len(extras)

    corrected = [r["corrected_timing_error_ms"] for r in hits
                 if r.get("corrected_timing_error_ms") not in ("", None)]
    pitch_ok  = sum(1 for r in hits if r.get("pitch_ok"))

    classified = [r for r in hits if r.get("constraint_classification") in ("chord", "scale", "out")]
    n_classified = len(classified)
    if n_classified:
        n_chord = sum(1 for r in classified if r["constraint_classification"] == "chord")
        n_scale = sum(1 for r in classified if r["constraint_classification"] == "scale")
        n_out   = sum(1 for r in classified if r["constraint_classification"] == "out")
        scores  = [r["constraint_score"] for r in classified if r.get("constraint_score") not in ("", None)]
        total_score   = sum(scores) if scores else ""
        score_per_note = round(total_score / n_classified, 2) if scores else ""
        chord_pct = round(n_chord / n_classified * 100, 1)
        scale_pct = round(n_scale / n_classified * 100, 1)
        out_pct   = round(n_out   / n_classified * 100, 1)
    else:
        chord_pct = scale_pct = out_pct = total_score = score_per_note = ""

    row = {
        "date":                    datetime.now().strftime("%Y-%m-%d"),
        "time":                    datetime.now().strftime("%H:%M:%S"),
        "mode":                    metadata.get("mode", ""),
        "exercise_name":           metadata.get("exercise_name", ""),
        "bpm":                     metadata.get("bpm", ""),
        "n_hits":                  n_hits,
        "n_misses":                n_misses,
        "n_extras":                n_extras,
        "pitch_accuracy_pct":      round(pitch_ok / n_hits * 100, 1) if n_hits else "",
        "mean_timing_error_ms":    round(float(np.mean(corrected)), 1) if corrected else "",
        "mean_abs_timing_error_ms": round(float(np.mean(np.abs(corrected))), 1) if corrected else "",
        "chord_pct":               chord_pct,
        "scale_pct":               scale_pct,
        "out_pct":                 out_pct,
        "total_score":             total_score,
        "score_per_note":          score_per_note,
    }

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Practice log updated: {path}")
