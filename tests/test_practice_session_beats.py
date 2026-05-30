"""Tests for --beats CLI option and build_targets() in practice_session_demo.py.

No audio hardware required.

Test matrix
-----------
build_targets
    1.  Returns the requested number of targets.
    2.  Beat positions are 0.0, 1.0, … n-1 (quarter notes).
    3.  Every target has note == "E2".
    4.  Labels are "1", "2", … n (1-indexed strings).
    5.  Single-beat exercise is valid.

--beats argument
    6.  Default is 4.
    7.  --beats 8 produces args.beats == 8.
    8.  --beats 0 is rejected (SystemExit).
    9.  --beats -1 is rejected (SystemExit).
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import pytest

_ROOT    = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from practice_session_demo import build_targets, _parse_args


def _parse(*argv: str):
    with unittest.mock.patch("sys.argv", ["practice_session_demo.py", *argv]):
        return _parse_args()


# ── build_targets ─────────────────────────────────────────────────────────────

def test_build_targets_count():
    assert len(build_targets(8)) == 8


def test_build_targets_beat_positions():
    positions = [t["time"] for t in build_targets(4)]
    assert positions == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_build_targets_note_is_e2():
    assert all(t["note"] == "E2" for t in build_targets(6))


def test_build_targets_labels_are_1_indexed_strings():
    labels = [t["label"] for t in build_targets(4)]
    assert labels == ["1", "2", "3", "4"]


def test_build_targets_single_beat():
    targets = build_targets(1)
    assert len(targets) == 1
    assert targets[0]["time"] == pytest.approx(0.0)
    assert targets[0]["note"] == "E2"
    assert targets[0]["label"] == "1"


# ── --beats argument ──────────────────────────────────────────────────────────

def test_default_beats_is_4():
    assert _parse().beats == 4


def test_beats_8():
    assert _parse("--beats", "8").beats == 8


def test_beats_zero_rejected():
    with pytest.raises(SystemExit):
        _parse("--beats", "0")


def test_beats_negative_rejected():
    with pytest.raises(SystemExit):
        _parse("--beats", "-1")
