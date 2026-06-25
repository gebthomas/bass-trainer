"""Practice exercise runner: analyze a recording against targets and a chord progression."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import librosa

from core.targets import load_targets
from core.audio_calibration import load_input_latency

# Reuse analysis engine from analyze_fast_reference
from tools.audio.analyze_fast_reference import analyze_target, annotate_harmony, SAMPLE_RATE


# ── Per-note table ────────────────────────────────────────────────────────────

def _print_per_note(results, time_offset=0.0, beat_unit=False):
    show_wav = time_offset != 0.0 or beat_unit
    print()
    if beat_unit:
        print(f"  {'#':>2}  {'MusBeat':>8}  {'WavTime':>8}  {'Target':<6}  {'Detected':<8}  {'Pitch':<22}  {'Chord':<7}  Harm")
        print(f"  {'─' * 82}")
    elif show_wav:
        print(f"  {'#':>2}  {'MusTime':>8}  {'WavTime':>8}  {'Target':<6}  {'Detected':<8}  {'Pitch':<22}  {'Chord':<7}  Harm")
        print(f"  {'─' * 80}")
    else:
        print(f"  {'#':>2}  {'Time':>7}  {'Target':<6}  {'Detected':<8}  {'Pitch':<22}  {'Chord':<7}  Harm")
        print(f"  {'─' * 70}")
    for i, r in enumerate(results):
        detected = r["candidate_note"] if r["candidate_note"] else "—"
        cents    = f" ({r['cents_error']:+.0f}c)" if r["cents_error"] is not None else ""
        pitch    = r["pitch_status"] + cents
        chord    = r.get("current_chord") or "—"
        hclass   = r.get("harmonic_class") or "—"
        if beat_unit:
            wav_t   = r.get("audio_time", 0.0)
            mus_val = r["target_time"]   # stored as beat number
            print(f"  {i + 1:>2}  {mus_val:>7.1f}b  {wav_t:>7.3f}s  {r['target_note']:<6}  {detected:<8}  {pitch:<22}  {chord:<7}  {hclass}")
        elif show_wav:
            wav_t = r.get("audio_time", r["target_time"] + time_offset)
            print(f"  {i + 1:>2}  {r['target_time']:>7.3f}s  {wav_t:>7.3f}s  {r['target_note']:<6}  {detected:<8}  {pitch:<22}  {chord:<7}  {hclass}")
        else:
            print(f"  {i + 1:>2}  {r['target_time']:>6.3f}s  {r['target_note']:<6}  {detected:<8}  {pitch:<22}  {chord:<7}  {hclass}")


# ── Practice suggestion ───────────────────────────────────────────────────────

def _practice_suggestion(n_total, n_ok, n_pitch_err, n_missed, n_out, n_classified,
                          pitch_errors, outside_notes):
    suggestions = []

    pitch_pct = n_ok / n_total * 100 if n_total else 0

    if n_missed > 0:
        suggestions.append(
            f"You missed {n_missed} note(s) entirely — check timing and note placement."
        )

    if n_pitch_err > 0:
        # Partition by cents magnitude — large deviations are likely detection artefacts,
        # not intonation problems the player should act on.
        intonation  = [e for e in pitch_errors if abs(e["cents"]) < 300]
        wrong_or_cx = [e for e in pitch_errors if 300 <= abs(e["cents"]) < 900]
        artefacts   = [e for e in pitch_errors if abs(e["cents"]) >= 900]

        if intonation:
            notes_str = ", ".join(f"{e['target']} ({e['cents']:+.0f}c)" for e in intonation)
            suggestions.append(
                f"Intonation issues on: {notes_str}. Slow down and listen for the centre of the pitch."
            )

        if wrong_or_cx:
            notes_str = ", ".join(f"{e['target']} ({e['cents']:+.0f}c)" for e in wrong_or_cx)
            suggestions.append(
                f"Wrong note or contaminated window: {notes_str}. Check fingering and note choice."
            )

        if artefacts:
            notes_str = ", ".join(f"{e['target']} ({e['cents']:+.0f}c)" for e in artefacts)
            suggestions.append(
                f"Possible octave/subharmonic detection error (not an intonation issue): {notes_str}."
            )

    if n_classified > 0 and n_out > 0:
        out_pct = n_out / n_classified * 100
        if out_pct > 30:
            suggestions.append(
                "Many outside notes — prioritise landing on chord tones (roots, 3rds, 5ths, 7ths)."
            )
        else:
            out_str = ", ".join(f"{o['note']} over {o['chord']}" for o in outside_notes)
            suggestions.append(
                f"Outside note(s): {out_str}. Check chromatic approaches — resolve them by ear."
            )

    if pitch_pct == 100 and n_missed == 0 and n_out == 0:
        suggestions.append("Everything solid — try increasing tempo or adding more outside colour.")
    elif pitch_pct >= 80 and n_out == 0 and not suggestions:
        suggestions.append("Good overall — tighten intonation on the marginal notes.")

    return suggestions if suggestions else ["Keep working — record and listen back critically."]


# ── Musical summary ───────────────────────────────────────────────────────────

def _print_summary(results):
    n = len(results)

    n_ok      = sum(1 for r in results if r["pitch_status"] == "ok")
    n_marg    = sum(1 for r in results if r["pitch_status"] == "pitch_marginal")
    n_err     = sum(1 for r in results if r["pitch_status"] == "pitch_error")
    n_missed  = sum(1 for r in results if r["pitch_status"] == "missed")
    n_other   = n - n_ok - n_marg - n_err - n_missed

    classified = [r for r in results if r.get("harmonic_class") not in (None, "pitch_uncertain")]
    nc         = len(classified)
    n_chord    = sum(1 for r in classified if r["harmonic_class"] == "chord")
    n_scale    = sum(1 for r in classified if r["harmonic_class"] == "scale")
    n_out_r    = sum(1 for r in classified if r["harmonic_class"] == "out")

    pitch_errors  = [
        {"target": r["target_note"], "cents": r["cents_error"]}
        for r in results if r["pitch_status"] in ("pitch_error", "pitch_marginal")
        and r["cents_error"] is not None
    ]
    outside_notes = [
        {"note": r["candidate_note"] or r["target_note"], "chord": r.get("current_chord", "?")}
        for r in classified if r["harmonic_class"] == "out"
    ]

    print()
    print("═" * 50)
    print("  MUSICAL SUMMARY")
    print("═" * 50)
    print(f"  Notes:          {n}")
    print(f"  Pitch OK:       {n_ok}/{n}  ({n_ok / n * 100:.0f}%)")
    if n_marg:
        print(f"  Marginal:       {n_marg}  (>50c but ≤100c)")
    if n_err:
        print(f"  Pitch errors:   {n_err}  (>100c)")
    if n_missed:
        print(f"  Missed:         {n_missed}")
    if n_other:
        print(f"  Low/uncertain:  {n_other}")

    if nc:
        print()
        print(f"  Harmonic  ({nc} classified):")
        print(f"    Chord tones   {n_chord:>3}  ({n_chord / nc * 100:.0f}%)")
        print(f"    Scale tones   {n_scale:>3}  ({n_scale / nc * 100:.0f}%)")
        print(f"    Outside       {n_out_r:>3}  ({n_out_r / nc * 100:.0f}%)")

    if outside_notes:
        print()
        print("  Outside notes:")
        for o in outside_notes:
            print(f"    {o['note']}  over  {o['chord']}")

    if pitch_errors:
        print()
        print("  Pitch errors / marginal:")
        for e in pitch_errors:
            print(f"    {e['target']}  {e['cents']:+.0f}c")

    suggestions = _practice_suggestion(
        n, n_ok, n_err, n_missed, n_out_r, nc, pitch_errors, outside_notes
    )
    print()
    print("  Practice suggestion:")
    for s in suggestions:
        print(f"    → {s}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a jazz bass practice exercise: pitch + harmonic feedback."
    )
    parser.add_argument("wav",     help="Path to WAV recording")
    parser.add_argument("targets", help="Path to target JSON file")
    parser.add_argument(
        "--progression",
        metavar="FILE",
        help="Chord progression JSON (e.g. tests/progressions/ii_v_i_C.json)",
    )
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Apply input_latency_ms from config/audio_calibration.json",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Shift every target time by SECONDS before analysis "
             "(use count-in duration when WAV was recorded with play_and_record_exercise.py)",
    )
    parser.add_argument(
        "--target-time-unit",
        choices=["seconds", "beats"],
        default="seconds",
        metavar="{seconds,beats}",
        help="Unit of the 'time' field in the target JSON (default: seconds)",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        metavar="BPM",
        help="Tempo in beats per minute — required when --target-time-unit=beats",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    beat_unit = args.target_time_unit == "beats"
    if beat_unit and args.bpm is None:
        sys.exit("error: --bpm is required when --target-time-unit=beats")

    wav_path    = Path(args.wav)
    target_path = Path(args.targets)

    print(f"Exercise:  {wav_path.name}")
    print(f"Targets:   {target_path.name}")

    input_latency_ms = 0.0
    if args.apply_calibration:
        input_latency_ms = load_input_latency()
        print(f"Latency:   {input_latency_ms:+.1f}ms")

    if args.time_offset:
        print(f"Offset:    {args.time_offset:+.3f}s")

    if beat_unit:
        beat_s = 60.0 / args.bpm
        print(f"Tempo:     {args.bpm} BPM  ({beat_s:.4f}s/beat)")

    progression = []
    if args.progression:
        with open(args.progression, encoding="utf-8") as fh:
            progression = json.load(fh)
        print(f"Chords:    {Path(args.progression).name}  ({len(progression)} segments)")

    audio   = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)[0]
    targets = load_targets(target_path)
    print(f"Audio:     {len(audio) / SAMPLE_RATE:.2f}s  |  {len(targets)} targets")

    time_offset = args.time_offset

    if beat_unit:
        # Convert beat times to seconds for audio window placement, keep originals for chord lookup.
        beat_s = 60.0 / args.bpm
        audio_targets = [
            {**t, "time": time_offset + t["time"] * beat_s}
            for t in targets
        ]
    elif time_offset:
        audio_targets = [{**t, "time": t["time"] + time_offset} for t in targets]
    else:
        audio_targets = targets

    results = [analyze_target(i, audio_targets, audio, input_latency_ms) for i in range(len(targets))]

    # Restore musical time for chord lookup and reporting; stash WAV position in audio_time.
    if beat_unit or time_offset:
        for i, r in enumerate(results):
            r["audio_time"]  = r["target_time"]    # seconds — where in the WAV we looked
            r["target_time"] = targets[i]["time"]  # original value (beats or seconds)

    for r in results:
        annotate_harmony(r, progression)

    _print_per_note(results, time_offset, beat_unit)
    _print_summary(results)


if __name__ == "__main__":
    main()
