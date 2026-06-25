"""Snapshot regression tests for pyin pitch detection on reference audio files.

These tests freeze CURRENT BEHAVIOUR as of 2026-05-15.  They are not
necessarily "correct" — several entries document known detection issues
(marked with # KNOWN ISSUE).  The goal is to catch accidental regressions
if pyin parameters, window sizing, or the analysis engine changes.

Run:
    python -m pytest tests/test_pitch_reference_regression.py -v

Skip policy: every test is skipped automatically when its audio file is
absent, so the suite stays green on machines that only have a subset of
reference recordings.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

librosa = pytest.importorskip("librosa")
from tools.audio.analyze_fast_reference import analyze_target, SAMPLE_RATE


# ── Snapshot dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Snap:
    """Expected outcome for one note in one reference recording."""
    label: str           # human-readable ID used in test parametrize IDs
    wav:   str           # path relative to PROJECT_ROOT
    tgt:   str           # path relative to PROJECT_ROOT
    idx:   int           # 0-based target index in the JSON
    expected_note: str   # note name in the target JSON
    pitch_status:  str   # expected pitch_status string
    cents_lo: int        # inclusive lower bound on cents_error (None accepted if status is missed/uncertain)
    cents_hi: int        # inclusive upper bound on cents_error


def _s(label, wav, tgt, idx, note, status, lo, hi):
    return Snap(label, wav, tgt, idx, note, status, lo, hi)


REF  = "tests/real_audio/reference"
TREF = "tests/targets/reference"
FREF = "tests/real_audio/reference/fast_reference"
FTGT = "tests/targets/fast_reference"


# ── Snapshot table ────────────────────────────────────────────────────────────
#
# Each row encodes:  label, wav, target_json, note_index, expected_note,
#                   expected_pitch_status, cents_lo, cents_hi
#
# Bounds are ±30c wider than the observed value to allow for minor float
# rounding differences across librosa versions, while still catching large
# regressions.  Undertone/overtone cases use ±200c.
#
# KNOWN ISSUE rows document current wrong behaviour.  They will FAIL if the
# algorithm is fixed — update them when that happens.

SNAPSHOTS: list[Snap] = [

    # ── Fretted A string ──────────────────────────────────────────────────────
    _s("fretted_A/A1",  f"{REF}/fretted/fretted_A_string_reference.wav",
                        f"{TREF}/fretted_A_string_reference.json",
       0, "A1",  "ok", -30, 30),
    _s("fretted_A/C2",  f"{REF}/fretted/fretted_A_string_reference.wav",
                        f"{TREF}/fretted_A_string_reference.json",
       1, "C2",  "ok", -30, 30),
    _s("fretted_A/D2",  f"{REF}/fretted/fretted_A_string_reference.wav",
                        f"{TREF}/fretted_A_string_reference.json",
       2, "D2",  "ok", -30, 30),
    # KNOWN ISSUE: last note (A3) is detected as F#3 at -300c.
    _s("fretted_A/A3_KNOWN_ERROR",
                        f"{REF}/fretted/fretted_A_string_reference.wav",
                        f"{TREF}/fretted_A_string_reference.json",
       9, "A3",  "pitch_error", -400, -200),

    # ── Fretted D string ──────────────────────────────────────────────────────
    _s("fretted_D/D2",  f"{REF}/fretted/fretted_D_string_reference.wav",
                        f"{TREF}/fretted_D_string_reference.json",
       0, "D2",  "ok", -30, 30),
    # KNOWN ISSUE: last note (D4) detected as B3 at -290c.
    _s("fretted_D/D4_KNOWN_ERROR",
                        f"{REF}/fretted/fretted_D_string_reference.wav",
                        f"{TREF}/fretted_D_string_reference.json",
       9, "D4",  "pitch_error", -400, -200),

    # ── Fretted E string ──────────────────────────────────────────────────────
    _s("fretted_E/E1",  f"{REF}/fretted/fretted_E_string_reference.wav",
                        f"{TREF}/fretted_E_string_reference.json",
       0, "E1",  "ok", -30, 10),
    _s("fretted_E/A1",  f"{REF}/fretted/fretted_E_string_reference.wav",
                        f"{TREF}/fretted_E_string_reference.json",
       2, "A1",  "ok", -30, 10),
    # KNOWN ISSUE: last note (E3) detected as C#3 at -310c.
    _s("fretted_E/E3_KNOWN_ERROR",
                        f"{REF}/fretted/fretted_E_string_reference.wav",
                        f"{TREF}/fretted_E_string_reference.json",
       9, "E3",  "pitch_error", -400, -200),

    # ── Fretted G string ──────────────────────────────────────────────────────
    _s("fretted_G/G2",  f"{REF}/fretted/fretted_G_string_reference.wav",
                        f"{TREF}/fretted_G_string_reference.json",
       0, "G2",  "ok", -30, 30),
    # KNOWN ISSUE: last note (G4) detected as E4 at -300c.
    _s("fretted_G/G4_KNOWN_ERROR",
                        f"{REF}/fretted/fretted_G_string_reference.wav",
                        f"{TREF}/fretted_G_string_reference.json",
       9, "G4",  "pitch_error", -400, -200),

    # ── Fretless A string ─────────────────────────────────────────────────────
    _s("fretless_A/A1", f"{REF}/fretless/fretless_A_string_reference.wav",
                        f"{TREF}/fretless_A_string_reference.json",
       0, "A1",  "ok", -30, 30),
    _s("fretless_A/D2", f"{REF}/fretless/fretless_A_string_reference.wav",
                        f"{TREF}/fretless_A_string_reference.json",
       2, "D2",  "ok", -30, 30),

    # ── Fretless D string ─────────────────────────────────────────────────────
    _s("fretless_D/D2", f"{REF}/fretless/fretless_D_string_reference.wav",
                        f"{TREF}/fretless_D_string_reference.json",
       0, "D2",  "ok", -30, 30),
    # KNOWN ISSUE: last note (A3) detected as G3 at -190c.
    _s("fretless_D/A3_KNOWN_ERROR",
                        f"{REF}/fretless/fretless_D_string_reference.wav",
                        f"{TREF}/fretless_D_string_reference.json",
       8, "A3",  "pitch_error", -280, -100),

    # ── Fretless E string ─────────────────────────────────────────────────────
    _s("fretless_E/E1", f"{REF}/fretless/fretless_E_string_reference.wav",
                        f"{TREF}/fretless_E_string_reference.json",
       0, "E1",  "ok", -30, 30),
    _s("fretless_E/A1", f"{REF}/fretless/fretless_E_string_reference.wav",
                        f"{TREF}/fretless_E_string_reference.json",
       2, "A1",  "ok", -30, 40),

    # ── Fretless G string ─────────────────────────────────────────────────────
    _s("fretless_G/G2", f"{REF}/fretless/fretless_G_string_reference.wav",
                        f"{TREF}/fretless_G_string_reference.json",
       0, "G2",  "ok", -30, 40),

    # ── Upright bow A string ──────────────────────────────────────────────────
    _s("upright_bow_A/A1",
                        f"{REF}/upright_bow/upright_bow_A_string_reference.wav",
                        f"{TREF}/upright_bow_A_string_reference.json",
       0, "A1",  "ok", -30, 30),
    _s("upright_bow_A/B1",
                        f"{REF}/upright_bow/upright_bow_A_string_reference.wav",
                        f"{TREF}/upright_bow_A_string_reference.json",
       1, "B1",  "ok", -30, 30),
    _s("upright_bow_A/E2",
                        f"{REF}/upright_bow/upright_bow_A_string_reference.wav",
                        f"{TREF}/upright_bow_A_string_reference.json",
       4, "E2",  "ok", -30, 30),
    # Near miss: D3 is pitch_marginal at -50c.
    _s("upright_bow_A/D3_marginal",
                        f"{REF}/upright_bow/upright_bow_A_string_reference.wav",
                        f"{TREF}/upright_bow_A_string_reference.json",
       10, "D3", "pitch_marginal", -80, -20),

    # ── Upright bow E string ──────────────────────────────────────────────────
    _s("upright_bow_E/E1",
                        f"{REF}/upright_bow/upright_bow_E_string_reference.wav",
                        f"{TREF}/upright_bow_E_string_reference.json",
       0, "E1",  "ok", -30, 30),

    # ── Upright bow G string ──────────────────────────────────────────────────
    _s("upright_bow_G/G2",
                        f"{REF}/upright_bow/upright_bow_G_string_reference.wav",
                        f"{TREF}/upright_bow_G_string_reference.json",
       0, "G2",  "ok", -30, 30),

    # ── Upright pizz A string ─────────────────────────────────────────────────
    _s("upright_pizz_A/A1",
                        f"{REF}/upright_pizz/upright_pizz_A_string_reference.wav",
                        f"{TREF}/upright_pizz_A_string_reference.json",
       0, "A1",  "ok", -30, 30),
    _s("upright_pizz_A/B1",
                        f"{REF}/upright_pizz/upright_pizz_A_string_reference.wav",
                        f"{TREF}/upright_pizz_A_string_reference.json",
       1, "B1",  "ok", -30, 30),

    # ── Upright pizz D string — overtone event ────────────────────────────────
    # KNOWN ISSUE: note #5 (A2) is detected as A3 (+1230c) — high confidence
    # overtone confusion.  This is a real pyin failure, not a recording problem.
    _s("upright_pizz_D/A2_OVERTONE",
                        f"{REF}/upright_pizz/upright_pizz_D_string_reference.wav",
                        f"{TREF}/upright_pizz_D_string_reference.json",
       4, "A2",  "pitch_error", 1000, 1500),

    # ── Upright pizz E string — undertone event ───────────────────────────────
    # KNOWN ISSUE: note #10 (G#2) is detected as G#1 (-1210c) — high confidence
    # UNDERTONE confusion.  Open-A-string undertone problem class.
    _s("upright_pizz_E/Gs2_UNDERTONE",
                        f"{REF}/upright_pizz/upright_pizz_E_string_reference.wav",
                        f"{TREF}/upright_pizz_E_string_reference.json",
       9, "G#2", "pitch_error", -1500, -900),

    # ── Upright pizz G string ─────────────────────────────────────────────────
    _s("upright_pizz_G/G2",
                        f"{REF}/upright_pizz/upright_pizz_G_string_reference.wav",
                        f"{TREF}/upright_pizz_G_string_reference.json",
       0, "G2",  "ok", -30, 30),

    # ── Fast reference: upright bow A string ──────────────────────────────────
    _s("fast/upright_bow_A/A1",
                        f"{FREF}/upright_bow_A_string_medium_low.wav",
                        f"{FTGT}/upright_bow_A_string_medium_low.json",
       0, "A1",  "ok", -30, 30),
    _s("fast/upright_bow_A/B1",
                        f"{FREF}/upright_bow_A_string_medium_low.wav",
                        f"{FTGT}/upright_bow_A_string_medium_low.json",
       1, "B1",  "ok", 0, 60),
    # KNOWN ISSUE: D2 detected as C#2 (-90c) — musician played flat, or detection drift.
    _s("fast/upright_bow_A/D2_marginal",
                        f"{FREF}/upright_bow_A_string_medium_low.wav",
                        f"{FTGT}/upright_bow_A_string_medium_low.json",
       3, "D2",  "pitch_marginal", -130, -50),
    # KNOWN ISSUE: E2 detected as D2 (-190c) — pitch_error.
    _s("fast/upright_bow_A/E2_KNOWN_ERROR",
                        f"{FREF}/upright_bow_A_string_medium_low.wav",
                        f"{FTGT}/upright_bow_A_string_medium_low.json",
       4, "E2",  "pitch_error", -250, -130),

    # ── Fast reference: fretted G string — massive undertone ─────────────────
    # KNOWN ISSUE: first note (G3) detected as E1 at -2700c (2.5 octaves below).
    _s("fast/fretted_G/G3_MASSIVE_UNDERTONE",
                        f"{FREF}/fretted_G_string_fast_high.wav",
                        f"{FTGT}/fretted_G_string_fast_high.json",
       0, "G3",  "pitch_error", -3000, -2000),

    # ── Fast reference: fretless G string — massive overtone ─────────────────
    # KNOWN ISSUE: first note (G3) detected as G5 at +2360c (2 octaves above).
    _s("fast/fretless_G/G3_MASSIVE_OVERTONE",
                        f"{FREF}/fretless_G_string_fast_high.wav",
                        f"{FTGT}/fretless_G_string_fast_high.json",
       0, "G3",  "pitch_low_confidence", 2000, 3000),

    # ── Repeated A1 fretless ──────────────────────────────────────────────────
    # All 8 repetitions must be detected correctly as A1.
    *[
        _s(f"repeat_A1/rep{i+1}",
           "tests/real_audio/fretless_finger/repeated_60_A1_fretless.wav",
           "tests/targets/repeat_A1.json",
           i, "A1", "ok", -30, 30)
        for i in range(8)
    ],
]


# ── Fixture and test ──────────────────────────────────────────────────────────

def _load(wav_rel: str, tgt_rel: str):
    """Cache-less load; called once per unique (wav, tgt) pair per session."""
    wav = PROJECT_ROOT / wav_rel
    tgt = PROJECT_ROOT / tgt_rel
    if not wav.exists():
        pytest.skip(f"audio not found: {wav_rel}")
    with open(tgt) as fh:
        targets = json.load(fh)
    audio = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)[0]
    return targets, audio


@pytest.mark.parametrize("snap", SNAPSHOTS, ids=[s.label for s in SNAPSHOTS])
def test_pitch_snapshot(snap: Snap) -> None:
    targets, audio = _load(snap.wav, snap.tgt)

    assert snap.idx < len(targets), (
        f"target index {snap.idx} out of range (file has {len(targets)} targets)"
    )
    assert targets[snap.idx]["note"] == snap.expected_note, (
        f"note at index {snap.idx} is {targets[snap.idx]['note']!r},"
        f" expected {snap.expected_note!r}"
    )

    result = analyze_target(snap.idx, targets, audio, 0.0)

    assert result["pitch_status"] == snap.pitch_status, (
        f"{snap.label}: pitch_status={result['pitch_status']!r},"
        f" expected {snap.pitch_status!r}"
        f"  (detected={result['candidate_note']}, cents={result['cents_error']})"
    )

    if result["cents_error"] is not None:
        assert snap.cents_lo <= result["cents_error"] <= snap.cents_hi, (
            f"{snap.label}: cents_error={result['cents_error']:.1f}"
            f" outside [{snap.cents_lo}, {snap.cents_hi}]"
        )


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "pytest", __file__, "-v"],
            cwd=PROJECT_ROOT,
        ).returncode
    )
