"""Tests for latency compensation helpers in scripts/practice_session_demo.py.

No audio hardware required.

Test matrix
-----------
compensate_onset_times
    1.  Positive latency shifts onsets earlier (subtracts offset).
    2.  Input list is not mutated.
    3.  Empty list returns empty list.
    4.  Zero latency returns values unchanged.
    5.  Negative latency shifts onsets later (unusual but valid).

_resolve_latency
    6.  Default (no flags) calls load_input_latency() and returns
        "audio_calibration.json" as the source label.
    7.  --latency-ms overrides load_input_latency() and returns "--latency-ms".
    8.  --no-calibration forces 0 ms regardless of load_input_latency().
    9.  --latency-ms and --no-calibration are mutually exclusive (argparse error).
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

import practice_session_demo as demo
from practice_session_demo import compensate_onset_times, _resolve_latency, _parse_args


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse(*argv: str):
    with unittest.mock.patch("sys.argv", ["practice_session_demo.py", *argv]):
        return _parse_args()


# ── compensate_onset_times ────────────────────────────────────────────────────

def test_compensation_shifts_onsets_earlier():
    result = compensate_onset_times([1.0, 1.5, 2.0], latency_ms=100.0)
    assert result == pytest.approx([0.9, 1.4, 1.9])


def test_compensation_does_not_mutate_input():
    original = [1.0, 2.0, 3.0]
    snapshot = list(original)
    compensate_onset_times(original, latency_ms=50.0)
    assert original == snapshot


def test_empty_list_stays_empty():
    assert compensate_onset_times([], latency_ms=100.0) == []


def test_zero_latency_is_identity():
    times = [0.5, 1.0, 1.5]
    assert compensate_onset_times(times, latency_ms=0.0) == pytest.approx(times)


def test_negative_latency_shifts_onsets_later():
    # Unusual but should work: negative latency means audio arrives early
    result = compensate_onset_times([1.0], latency_ms=-50.0)
    assert result == pytest.approx([1.05])


# ── _resolve_latency ──────────────────────────────────────────────────────────

def test_default_calls_load_input_latency(monkeypatch):
    monkeypatch.setattr(demo, "load_input_latency", lambda: 75.0)
    args = _parse()
    latency_ms, source = _resolve_latency(args)
    assert latency_ms == pytest.approx(75.0)
    assert source == "audio_calibration.json"


def test_latency_ms_flag_overrides_config(monkeypatch):
    monkeypatch.setattr(demo, "load_input_latency", lambda: 999.0)
    args = _parse("--latency-ms", "150.0")
    latency_ms, source = _resolve_latency(args)
    assert latency_ms == pytest.approx(150.0)
    assert source == "--latency-ms"


def test_no_calibration_forces_zero(monkeypatch):
    monkeypatch.setattr(demo, "load_input_latency", lambda: 999.0)
    args = _parse("--no-calibration")
    latency_ms, source = _resolve_latency(args)
    assert latency_ms == pytest.approx(0.0)
    assert source == "--no-calibration"


def test_latency_ms_and_no_calibration_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parse("--latency-ms", "100.0", "--no-calibration")
