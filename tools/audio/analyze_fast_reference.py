import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import librosa

from core.targets import load_targets
from core.pitch import note_to_hz, hz_to_note, cents_between
from core.audio_calibration import load_input_latency, effective_target_time
from core.constraints import chord_at_time, classify_note_against_chord

# ── Audio / pyin ──────────────────────────────────────────────────────────────

SAMPLE_RATE       = 48000
PYIN_FMIN         = librosa.note_to_hz("C1")
PYIN_FMAX         = librosa.note_to_hz("G5")
PYIN_FRAME_LENGTH = 4096
PYIN_HOP_LENGTH   = 512

# ── Window sizing ─────────────────────────────────────────────────────────────

ATTACK_GUARD_MIN_SEC          = 0.020   # floor on transient skip
ATTACK_GUARD_FRACTION         = 0.08    # fraction of inter-note gap for guard
WINDOW_FRACTION_OF_INTER_NOTE = 0.60    # fraction of gap used for analysis
WINDOW_MAX_SEC                = 0.35    # hard cap (prevents bleeding for slow notes)
WINDOW_MIN_SEC                = 0.050   # floor (safety, rarely active for bass tempos)

# ── Confidence tier thresholds ────────────────────────────────────────────────

CONF_HIGH_VOICED_FRAC = 0.60
CONF_HIGH_CONSENSUS   = 0.75
CONF_HIGH_FRAMES      = 6

CONF_MED_VOICED_FRAC  = 0.40
CONF_MED_CONSENSUS    = 0.60
CONF_MED_FRAMES       = 4

CONF_LOW_VOICED_FRAC  = 0.25
CONF_LOW_FRAMES       = 3

# ── Pitch classification thresholds ──────────────────────────────────────────

PITCH_ERROR_CENTS    = 100.0   # confident + clearly wrong
PITCH_MARGINAL_CENTS = 50.0    # log but don't penalise


# ── Window sizing ─────────────────────────────────────────────────────────────

def _compute_window(target_index, targets, anchor_time):
    # Inter-note gap comes from musical target times (tempo, not hardware).
    # Window placement uses anchor_time, which may be shifted by input latency.
    t = targets[target_index]["time"]
    if target_index + 1 < len(targets):
        inter_note   = targets[target_index + 1]["time"] - t
        attack_guard = max(ATTACK_GUARD_MIN_SEC, inter_note * ATTACK_GUARD_FRACTION)
        raw_window   = inter_note * WINDOW_FRACTION_OF_INTER_NOTE
        window_dur   = max(WINDOW_MIN_SEC, min(WINDOW_MAX_SEC, raw_window))
    else:
        attack_guard = ATTACK_GUARD_MIN_SEC
        window_dur   = WINDOW_MAX_SEC

    win_start = anchor_time + attack_guard
    win_end   = win_start + window_dur
    return win_start, win_end, window_dur, attack_guard


# ── Frame evidence collection ─────────────────────────────────────────────────

def _collect_frames(audio, win_start, win_end):
    start_s = int(win_start * SAMPLE_RATE)
    end_s   = min(int(win_end * SAMPLE_RATE), len(audio))

    if start_s >= len(audio) or end_s <= start_s:
        empty = np.array([])
        return empty, np.array([], dtype=bool), empty

    segment = audio[start_s:end_s]
    f0, voiced_flag, voiced_prob = librosa.pyin(
        segment,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=SAMPLE_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=PYIN_HOP_LENGTH,
    )
    return f0, voiced_flag, voiced_prob


# ── Evidence aggregation ──────────────────────────────────────────────────────

def _aggregate(f0, voiced_flag, voiced_prob):
    n_total = len(f0)
    if n_total == 0:
        return None, None, None, 0.0, 0, 0, 0.0, {}, 0.0

    usable      = voiced_flag & ~np.isnan(f0) & (f0 > 0)
    n_voiced    = int(np.sum(usable))
    voiced_frac = n_voiced / n_total

    if n_voiced == 0:
        return None, None, None, 0.0, 0, n_total, 0.0, {}, 0.0

    usable_f0   = f0[usable]
    usable_prob = voiced_prob[usable]

    # Weighted semitone votes — voiced_prob is the weight per frame.
    semitones = np.round(69.0 + 12.0 * np.log2(usable_f0 / 440.0)).astype(int)
    votes = {}
    for s, p in zip(semitones.tolist(), usable_prob.tolist()):
        votes[s] = votes.get(s, 0.0) + p

    total_weight   = sum(votes.values())
    winner_semi    = max(votes, key=lambda k: votes[k])
    winner_mask    = semitones == winner_semi
    candidate_hz   = float(np.median(usable_f0[winner_mask]))
    consensus      = votes[winner_semi] / total_weight if total_weight > 0 else 0.0
    candidate_note, _ = hz_to_note(candidate_hz)

    return candidate_note, candidate_hz, winner_semi, consensus, n_voiced, n_total, voiced_frac, votes, total_weight


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence_tier(voiced_frac, consensus, n_voiced):
    if (voiced_frac >= CONF_HIGH_VOICED_FRAC
            and consensus >= CONF_HIGH_CONSENSUS
            and n_voiced  >= CONF_HIGH_FRAMES):
        return "high"
    if (voiced_frac >= CONF_MED_VOICED_FRAC
            and consensus >= CONF_MED_CONSENSUS
            and n_voiced  >= CONF_MED_FRAMES):
        return "medium"
    if voiced_frac >= CONF_LOW_VOICED_FRAC or n_voiced >= CONF_LOW_FRAMES:
        return "low"
    return "uncertain"


# ── Classification ────────────────────────────────────────────────────────────

def _timing_status(n_voiced):
    return "present" if n_voiced > 0 else "absent"


def _pitch_status(timing_status, tier, cents_error):
    if timing_status == "absent":
        return "missed"
    if tier == "uncertain":
        return "pitch_uncertain"
    if tier == "low":
        return "pitch_low_confidence"
    if cents_error is None:
        return "no_pitch"
    if abs(cents_error) > PITCH_ERROR_CENTS:
        return "pitch_error"
    if abs(cents_error) > PITCH_MARGINAL_CENTS:
        return "pitch_marginal"
    return "ok"


# ── Per-target analysis ───────────────────────────────────────────────────────

def analyze_target(target_index, targets, audio, input_latency_ms=0.0):
    target      = targets[target_index]
    target_time = target["time"]
    eff_time    = effective_target_time(target_time, input_latency_ms)

    win_start, win_end, win_dur, attack_guard = _compute_window(target_index, targets, eff_time)

    audio_dur  = len(audio) / SAMPLE_RATE
    truncated  = win_end > audio_dur

    f0, voiced_flag, voiced_prob = _collect_frames(audio, win_start, win_end)
    cand_note, cand_hz, _, consensus, n_voiced, n_total, voiced_frac, votes, total_weight = _aggregate(
        f0, voiced_flag, voiced_prob
    )

    if votes and total_weight > 0:
        sorted_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:3]
        semitone_votes = [(s, v / total_weight) for s, v in sorted_votes]
    else:
        semitone_votes = []

    target_hz   = note_to_hz(target["note"])
    cents_error = cents_between(cand_hz, target_hz) if cand_hz is not None else None

    tier    = _confidence_tier(voiced_frac, consensus, n_voiced)
    timing  = _timing_status(n_voiced)
    pitch   = _pitch_status(timing, tier, cents_error)

    return {
        "target_note":      target["note"],
        "target_time":      target_time,
        "effective_time":   eff_time,
        "input_latency_ms": input_latency_ms,
        "string":           target.get("string", ""),
        "position":         target.get("position", ""),
        "win_start":        win_start,
        "win_end":          win_end,
        "win_dur":          win_dur,
        "attack_guard":     attack_guard,
        "truncated":        truncated,
        "candidate_note":   cand_note,
        "candidate_hz":     cand_hz,
        "cents_error":      cents_error,
        "voiced_frames":    n_voiced,
        "total_frames":     n_total,
        "voiced_fraction":  voiced_frac,
        "pitch_consensus":  consensus,
        "confidence_tier":  tier,
        "timing_status":    timing,
        "pitch_status":     pitch,
        "semitone_votes":   semitone_votes,
    }


# ── Harmonic annotation ───────────────────────────────────────────────────────

def annotate_harmony(result, progression):
    """Attach current_chord and harmonic_class to a result dict in-place.

    harmonic_class values: "chord" | "scale" | "out" | "pitch_uncertain" | None
      None            — no progression supplied, or no chord at this time
      "pitch_uncertain" — confidence_tier is not "medium" or "high"
      "chord"/"scale"/"out" — classification of candidate_note against the chord
    """
    if not progression:
        result["current_chord"]  = None
        result["harmonic_class"] = None
        return result

    chord = chord_at_time(progression, result["target_time"], loop=True)
    result["current_chord"] = chord

    if chord is None or result["candidate_note"] is None:
        result["harmonic_class"] = None
    elif result["confidence_tier"] not in ("medium", "high"):
        result["harmonic_class"] = "pitch_uncertain"
    else:
        result["harmonic_class"] = classify_note_against_chord(result["candidate_note"], chord)

    return result


# ── Print: per-target block ───────────────────────────────────────────────────

_DIV = "─" * 60


def _print_target(idx, r):
    loc = ""
    if r["string"] and r["position"]:
        loc = f"  ({r['string']} string, {r['position']})"
    elif r["position"]:
        loc = f"  ({r['position']})"

    print(f"\n{_DIV}")
    print(f"  Target {idx + 1}:  {r['target_note']:<5} @ {r['target_time']:.3f}s{loc}")
    print(_DIV)

    if r["input_latency_ms"] != 0.0:
        print(f"  Target time:     {r['target_time']:.3f}s  (musical)")
        print(f"  Effective time:  {r['effective_time']:.3f}s"
              f"  (latency {r['input_latency_ms']:+.1f}ms)")

    trunc_note = "  [window truncated — recording ends early]" if r["truncated"] else ""
    win_ms   = r["win_dur"] * 1000
    guard_ms = r["attack_guard"] * 1000
    print(f"  Window:          {r['win_start']:.3f}s – {r['win_end']:.3f}s"
          f"  ({win_ms:.0f}ms, guard {guard_ms:.0f}ms){trunc_note}")

    if r["candidate_note"] is not None:
        print(f"  Candidate:       {r['candidate_note']:<5}  ({r['candidate_hz']:.1f} Hz)")
        print(f"  Cents error:     {r['cents_error']:+.1f}c")
    else:
        print("  Candidate:       —")
        print("  Cents error:     —")

    vf_pct = r["voiced_fraction"] * 100
    print(f"  Voiced frames:   {r['voiced_frames']} / {r['total_frames']}  ({vf_pct:.1f}%)")
    print(f"  Pitch consensus: {r['pitch_consensus'] * 100:.1f}%")
    print(f"  Confidence:      {r['confidence_tier']}")
    print(f"  Timing status:   {r['timing_status']}")
    print(f"  Pitch status:    {r['pitch_status']}")

    if r.get("current_chord") is not None:
        hclass = r["harmonic_class"] or "—"
        print(f"  Chord:           {r['current_chord']:<6}  →  {hclass}")


# ── Print: summary table ──────────────────────────────────────────────────────

def _print_summary(results):
    has_harmony = any(r.get("current_chord") is not None for r in results)

    width = 103 if has_harmony else 92
    print(f"\n{'═' * width}")
    print("  SUMMARY")
    print(f"{'═' * width}")

    hdr = (f"  {'#':>2}  {'Note':<5}  {'Time':>7}  {'WinDur':>6}  "
           f"{'Candidate':<6}  {'Cents':>7}  {'V/Tot':>9}  "
           f"{'Consensus':>9}  {'Tier':<9}  {'Timing':<8}  Pitch")
    if has_harmony:
        hdr += f"  {'Chord':<6}  Harm"
    print(hdr)
    print(f"  {'─' * (width - 2)}")

    for i, r in enumerate(results):
        cand  = r["candidate_note"] if r["candidate_note"] else "—"
        cents = f"{r['cents_error']:+.1f}c" if r["cents_error"] is not None else "—"
        vtot  = f"{r['voiced_frames']}/{r['total_frames']} {r['voiced_fraction'] * 100:.0f}%"
        cons  = f"{r['pitch_consensus'] * 100:.1f}%"
        trunc = "*" if r["truncated"] else " "
        row = (
            f"  {i + 1:>2}  {r['target_note']:<5}  {r['target_time']:>6.3f}s"
            f"  {r['win_dur'] * 1000:>4.0f}ms  "
            f"{cand:<6}  {cents:>7}  {vtot:>9}  "
            f"{cons:>9}  {r['confidence_tier']:<9}  "
            f"{r['timing_status']:<8}  {r['pitch_status']}{trunc}"
        )
        if has_harmony:
            chord  = r.get("current_chord") or "—"
            hclass = r.get("harmonic_class") or "—"
            row += f"  {chord:<6}  {hclass}"
        print(row)

    n = len(results)
    counts = {
        "ok":                   sum(1 for r in results if r["pitch_status"] == "ok"),
        "pitch_marginal":       sum(1 for r in results if r["pitch_status"] == "pitch_marginal"),
        "pitch_error":          sum(1 for r in results if r["pitch_status"] == "pitch_error"),
        "pitch_uncertain":      sum(1 for r in results if r["pitch_status"] == "pitch_uncertain"),
        "pitch_low_confidence": sum(1 for r in results if r["pitch_status"] == "pitch_low_confidence"),
        "missed":               sum(1 for r in results if r["pitch_status"] == "missed"),
    }

    print()
    print(f"  Targets:              {n}")
    print(f"  OK:                   {counts['ok']}  ({counts['ok'] / n * 100:.0f}%)")
    if counts["pitch_marginal"]:
        print(f"  Marginal (>50c):      {counts['pitch_marginal']}")
    if counts["pitch_error"]:
        print(f"  Pitch errors (>100c): {counts['pitch_error']}")
    if counts["pitch_uncertain"]:
        print(f"  Pitch uncertain:      {counts['pitch_uncertain']}")
    if counts["pitch_low_confidence"]:
        print(f"  Low confidence:       {counts['pitch_low_confidence']}")
    if counts["missed"]:
        print(f"  Missed (no energy):   {counts['missed']}")

    # Cents stats over OK + marginal at medium/high confidence
    evaluable = [r for r in results
                 if r["cents_error"] is not None
                 and r["confidence_tier"] in ("high", "medium")]
    if evaluable:
        errors = [r["cents_error"] for r in evaluable]
        print(f"\n  Cents error over {len(evaluable)} medium/high-confidence targets:")
        print(f"    median  {float(np.median(errors)):+.1f}c")
        print(f"    mean    {float(np.mean(errors)):+.1f}c")
        print(f"    max abs {float(np.max(np.abs(errors))):.1f}c")

    if any(r["truncated"] for r in results):
        print("\n  * window truncated — recording ends before window close")

    # Harmonic breakdown — only when a progression was supplied
    classified = [r for r in results
                  if r.get("harmonic_class") not in (None, "pitch_uncertain")]
    uncertain  = [r for r in results if r.get("harmonic_class") == "pitch_uncertain"]
    if classified or uncertain:
        nc = len(classified)
        n_chord = sum(1 for r in classified if r["harmonic_class"] == "chord")
        n_scale = sum(1 for r in classified if r["harmonic_class"] == "scale")
        n_out   = sum(1 for r in classified if r["harmonic_class"] == "out")
        print(f"\n  Harmonic classification  ({nc} medium/high-confidence targets):")
        if nc:
            print(f"    chord tones   {n_chord:>3}  ({n_chord / nc * 100:.0f}%)")
            print(f"    scale tones   {n_scale:>3}  ({n_scale / nc * 100:.0f}%)")
            print(f"    outside       {n_out:>3}  ({n_out / nc * 100:.0f}%)")
        if uncertain:
            print(f"    pitch_uncertain (excluded from above): {len(uncertain)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze fast-passage bass reference recordings with onset-centered "
                    "windows, weighted evidence aggregation, and confidence scoring."
    )
    parser.add_argument("wav",     help="Path to WAV file")
    parser.add_argument("targets", help="Path to target JSON file")
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Shift analysis windows by input_latency_ms from config/audio_calibration.json",
    )
    parser.add_argument(
        "--progression",
        metavar="FILE",
        help="Chord progression JSON for harmonic classification (e.g. tests/progressions/ii_v_i_C.json)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    wav_path    = Path(args.wav)
    target_path = Path(args.targets)

    print(f"WAV:     {wav_path.name}")
    print(f"Targets: {target_path.name}")
    print(f"Window:  {WINDOW_FRACTION_OF_INTER_NOTE:.0%} of inter-note gap"
          f"  |  guard ≥ {ATTACK_GUARD_MIN_SEC * 1000:.0f}ms"
          f"  |  cap {WINDOW_MAX_SEC * 1000:.0f}ms")

    input_latency_ms = 0.0
    if args.apply_calibration:
        input_latency_ms = load_input_latency()
        print(f"Calibration:  input_latency_ms = {input_latency_ms:+.1f}ms")
    else:
        print("Calibration:  off (--apply-calibration not set)")

    progression = []
    if args.progression:
        with open(args.progression, encoding="utf-8") as fh:
            progression = json.load(fh)
        print(f"Progression:  {Path(args.progression).name}  ({len(progression)} segments)")
    else:
        print("Progression:  none (--progression not set)")

    audio   = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)[0]
    targets = load_targets(target_path)

    print(f"Audio:   {len(audio) / SAMPLE_RATE:.2f}s  |  Targets: {len(targets)}")

    results = [analyze_target(i, targets, audio, input_latency_ms) for i in range(len(targets))]
    for r in results:
        annotate_harmony(r, progression)

    for i, r in enumerate(results):
        _print_target(i, r)

    _print_summary(results)
    print()


if __name__ == "__main__":
    main()
