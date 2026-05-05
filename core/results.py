import csv
from datetime import datetime
from pathlib import Path
import numpy as np


class ResultsLogger:
    FIELDNAMES = [
        "event_type",
        "target_note",
        "target_time",
        "played_note",
        "played_time",
        "raw_timing_error_ms",
        "corrected_timing_error_ms",
        "pitch_ok",
        "timing_label",
        "detected_freq_hz",
        "target_freq_hz",
        "cents_error",
        "pitch_stability_cents",
        "pitch_match_ratio",
        "voiced_ratio",
        "target_window_status",
        "adaptive_offset_ms",
        "current_chord",
        "constraint_classification",
        "constraint_score",
    ]

    def __init__(self, results_dir):
        self.results = []
        self.results_dir = Path(results_dir)
        self._onset_constraint = {}   # onset_time -> {chord, classification, score}
        self._target_chord = {}       # target_time -> chord (for misses)

    def register_constraint(self, onset_time, chord, classification, score_value):
        self._onset_constraint[onset_time] = {
            "current_chord": chord,
            "constraint_classification": classification,
            "constraint_score": score_value,
        }

    def append_hit(self, target, onset_time, raw_error, corrected_error, pitch_ok, timing_label, note,
                   detected_freq_hz="", target_freq_hz="", cents_error="", pitch_stability_cents="",
                   pitch_match_ratio="", voiced_ratio="", target_window_status="", adaptive_offset_ms="",
                   current_chord="", constraint_classification="", constraint_score=""):
        constraint = self._onset_constraint.pop(onset_time, {})
        self.results.append({
            "event_type": "hit",
            "target_note": target["note"],
            "target_time": target["time"],
            "played_note": note,
            "played_time": onset_time,
            "raw_timing_error_ms": raw_error,
            "corrected_timing_error_ms": corrected_error,
            "pitch_ok": pitch_ok,
            "timing_label": timing_label,
            "detected_freq_hz": detected_freq_hz,
            "target_freq_hz": target_freq_hz,
            "cents_error": cents_error,
            "pitch_stability_cents": pitch_stability_cents,
            "pitch_match_ratio": pitch_match_ratio,
            "voiced_ratio": voiced_ratio,
            "target_window_status": target_window_status,
            "adaptive_offset_ms": adaptive_offset_ms,
            "current_chord": current_chord or constraint.get("current_chord", ""),
            "constraint_classification": constraint_classification or constraint.get("constraint_classification", ""),
            "constraint_score": constraint_score if constraint_score != "" else constraint.get("constraint_score", ""),
        })

    def append_miss(self, target, target_window_status="", adaptive_offset_ms="",
                    current_chord="", constraint_classification="", constraint_score=""):
        self.results.append({
            "event_type": "missed",
            "target_note": target["note"],
            "target_time": target["time"],
            "played_note": "",
            "played_time": "",
            "raw_timing_error_ms": "",
            "corrected_timing_error_ms": "",
            "pitch_ok": False,
            "timing_label": "missed",
            "detected_freq_hz": "",
            "target_freq_hz": "",
            "cents_error": "",
            "pitch_stability_cents": "",
            "pitch_match_ratio": "",
            "voiced_ratio": "",
            "target_window_status": target_window_status,
            "adaptive_offset_ms": adaptive_offset_ms,
            "current_chord": current_chord or self._target_chord.get(target["time"], ""),
            "constraint_classification": constraint_classification,
            "constraint_score": constraint_score,
        })

    def append_extra(self, note, onset_time, detected_freq_hz="", pitch_stability_cents="",
                     current_chord="", constraint_classification="", constraint_score=""):
        self.results.append({
            "event_type": "extra",
            "target_note": "",
            "target_time": "",
            "played_note": note,
            "played_time": onset_time,
            "raw_timing_error_ms": "",
            "corrected_timing_error_ms": "",
            "pitch_ok": False,
            "timing_label": "extra",
            "detected_freq_hz": detected_freq_hz,
            "target_freq_hz": "",
            "cents_error": "",
            "pitch_stability_cents": pitch_stability_cents,
            "pitch_match_ratio": "",
            "voiced_ratio": "",
            "target_window_status": "",
            "adaptive_offset_ms": "",
            "current_chord": current_chord,
            "constraint_classification": constraint_classification,
            "constraint_score": constraint_score,
        })

    def print_summary(self):
        if not self.results:
            print("No notes evaluated.")
            return

        hit_rows = [r for r in self.results if r.get("event_type") == "hit"]
        missed_rows = [r for r in self.results if r.get("event_type") == "missed"]
        extra_rows = [r for r in self.results if r.get("event_type") == "extra"]

        raw_errors = [r["raw_timing_error_ms"] for r in hit_rows]
        corrected_errors = [r["corrected_timing_error_ms"] for r in hit_rows]
        pitch_accuracy = (sum(r["pitch_ok"] for r in hit_rows) / len(hit_rows) * 100) if hit_rows else 0.0

        print("\nSession summary")
        print(f"Hits/evaluated notes: {len(hit_rows)}")
        print(f"Missed notes: {len(missed_rows)}")
        print(f"Extra notes: {len(extra_rows)}")

        if hit_rows:
            wrong_pitch = sum(1 for r in hit_rows if not r["pitch_ok"])
            cents_values = [float(r["cents_error"]) for r in hit_rows
                            if r.get("cents_error") not in ("", None)]
            stability_values = [float(r["pitch_stability_cents"]) for r in hit_rows
                                if r.get("pitch_stability_cents") not in ("", None)]

            print(f"Average raw timing error: {np.mean(raw_errors):+.1f} ms")
            print(f"Average corrected timing error: {np.mean(corrected_errors):+.1f} ms")
            print(f"Average absolute corrected timing error: {np.mean(np.abs(corrected_errors)):.1f} ms")
            print(f"Pitch accuracy: {pitch_accuracy:.1f}%  (wrong pitch: {wrong_pitch})")
            if cents_values:
                print(f"Median cents error: {np.median(cents_values):+.1f} c")
            if stability_values:
                print(f"Median pitch stability: {np.median(stability_values):.1f} c IQR")
        else:
            print("Average raw timing error: N/A")
            print("Average corrected timing error: N/A")
            print("Average absolute corrected timing error: N/A")
            print("Pitch accuracy: N/A")

    def save_csv(self):
        if not self.results:
            return None

        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.results_dir / f"session_{timestamp}.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for row in self.results:
                writer.writerow({f: row.get(f, "") for f in self.FIELDNAMES})

        print(f"Saved session results to {path}")
        return path
