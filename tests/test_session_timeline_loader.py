"""Tests for session-file extraction in scripts/plot_session_timeline.py.

All tests use in-memory SessionLog objects — no file I/O, no audio hardware.

Test matrix
-----------
scenario_from_session_log — extraction
  1.  target_hit: expected_time = time_sec - value
  2.  target_miss: expected_time = time_sec (nominal target time)
  3.  extra_onset contributes an onset time (no target)
  4.  Empty log returns empty arrays
  5.  Only misses: no onset times, all targets present
  6.  Only hits: all targets and all onsets present
  7.  Mixed hits, misses, extra onsets
  8.  Targets are sorted by target_index, not by event order
  9.  Onset times are sorted ascending
  10. Target with target_hit value=None is skipped (malformed hit)
  11. Return type is TimingScenario (NamedTuple with four fields)
  12. title contains the session date from started_at
  13. description contains BPM from metadata when present
  14. description contains device from metadata when present

scenario_from_session_log — value round-trip
  15. signed error: late hit (positive value) reconstructs correct expected_time
  16. signed error: early hit (negative value) reconstructs correct expected_time
  17. Round-trip: expected_time recovered from (time_sec, value) within float tolerance

Demo scenarios
  18. All five built-in scenarios return TimingScenario with non-empty arrays
  19. Demo scenarios unpack as 4-tuples (NamedTuple is a tuple)

CLI argument parsing
  20. No positional arg → args.file is None (demo mode)
  21. Positional arg → args.file is the given path (file mode)
  22. --scenario flag is accepted in demo mode
  23. --output flag is accepted
  24. --tolerance-ms and --on-time-ms are parsed as floats
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# scripts/ on path so we can import plot_session_timeline as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.session_log import (
    EXTRA_ONSET,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
)
from plot_session_timeline import (
    TimingScenario,
    _ALL_SCENARIOS,
    _parse_args,
    scenario_from_session_log,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_log(events: list[SessionEvent] | None = None,
                 metadata: dict | None = None) -> SessionLog:
    """Construct a minimal valid SessionLog with optional events and metadata."""
    return SessionLog(
        schema_version = 1,
        started_at     = "2026-05-27T10:00:00",
        events         = events or [],
        metadata       = metadata or {},
    )


def _hit(target_index: int, onset_time: float, error_s: float) -> SessionEvent:
    """Build a target_hit event. error_s = onset_time - expected_time."""
    return SessionEvent(
        time_sec     = onset_time,
        event_type   = TARGET_HIT,
        target_index = target_index,
        value        = error_s,
    )


def _miss(target_index: int, expected_time: float) -> SessionEvent:
    """Build a target_miss event. time_sec = nominal target time."""
    return SessionEvent(
        time_sec     = expected_time,
        event_type   = TARGET_MISS,
        target_index = target_index,
    )


def _extra(onset_time: float) -> SessionEvent:
    """Build an extra_onset event."""
    return SessionEvent(
        time_sec   = onset_time,
        event_type = EXTRA_ONSET,
    )


# ── Extraction: target times from hit events ──────────────────────────────────

class TestHitExtraction:
    def test_expected_time_derived_from_hit(self):
        # expected_time = onset_time - error = 2.0 - 0.05 = 1.95
        log = _minimal_log([_hit(0, onset_time=2.0, error_s=0.05)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == pytest.approx([1.95])

    def test_onset_time_comes_from_hit_time_sec(self):
        log = _minimal_log([_hit(0, onset_time=2.0, error_s=0.05)])
        scenario = scenario_from_session_log(log)
        assert scenario.onset_times_s == pytest.approx([2.0])

    def test_late_hit_positive_error(self):
        # onset 50 ms late: error = +0.05, onset = 1.05, expected = 1.0
        log = _minimal_log([_hit(0, onset_time=1.05, error_s=0.05)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s[0] == pytest.approx(1.0)

    def test_early_hit_negative_error(self):
        # onset 30 ms early: error = -0.03, onset = 0.97, expected = 1.0
        log = _minimal_log([_hit(0, onset_time=0.97, error_s=-0.03)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s[0] == pytest.approx(1.0)

    def test_exact_hit_zero_error(self):
        log = _minimal_log([_hit(0, onset_time=1.0, error_s=0.0)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s[0] == pytest.approx(1.0)
        assert scenario.onset_times_s[0]  == pytest.approx(1.0)

    def test_malformed_hit_with_none_value_is_skipped(self):
        # Hit where value=None should not produce a target or onset entry.
        bad_hit = SessionEvent(
            time_sec=2.0, event_type=TARGET_HIT, target_index=0, value=None
        )
        log = _minimal_log([bad_hit])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == []
        assert scenario.onset_times_s  == []


# ── Extraction: target times from miss events ─────────────────────────────────

class TestMissExtraction:
    def test_miss_expected_time_is_time_sec(self):
        log = _minimal_log([_miss(0, expected_time=1.5)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == pytest.approx([1.5])

    def test_miss_produces_no_onset(self):
        log = _minimal_log([_miss(0, expected_time=1.5)])
        scenario = scenario_from_session_log(log)
        assert scenario.onset_times_s == []

    def test_multiple_misses(self):
        log = _minimal_log([_miss(0, 1.0), _miss(1, 2.0), _miss(2, 3.0)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == pytest.approx([1.0, 2.0, 3.0])
        assert scenario.onset_times_s  == []


# ── Extraction: extra onsets ──────────────────────────────────────────────────

class TestExtraOnsetExtraction:
    def test_extra_onset_contributes_onset_time(self):
        log = _minimal_log([_extra(onset_time=0.5)])
        scenario = scenario_from_session_log(log)
        assert scenario.onset_times_s == pytest.approx([0.5])

    def test_extra_onset_does_not_produce_target(self):
        log = _minimal_log([_extra(onset_time=0.5)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == []

    def test_multiple_extra_onsets(self):
        log = _minimal_log([_extra(0.5), _extra(1.5)])
        scenario = scenario_from_session_log(log)
        assert scenario.onset_times_s == pytest.approx([0.5, 1.5])


# ── Mixed events ──────────────────────────────────────────────────────────────

class TestMixedEvents:
    def test_empty_log(self):
        scenario = scenario_from_session_log(_minimal_log())
        assert scenario.target_times_s == []
        assert scenario.onset_times_s  == []

    def test_hit_and_miss(self):
        # target 0 hit at 1.0 (error 0), target 1 missed at 2.0
        log = _minimal_log([_hit(0, 1.0, 0.0), _miss(1, 2.0)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == pytest.approx([1.0, 2.0])
        assert scenario.onset_times_s  == pytest.approx([1.0])

    def test_hit_miss_and_extra(self):
        log = _minimal_log([
            _hit(0, 1.0, 0.0),
            _miss(1, 2.0),
            _extra(2.4),
        ])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == pytest.approx([1.0, 2.0])
        assert scenario.onset_times_s  == pytest.approx([1.0, 2.4])

    def test_only_extra_onsets(self):
        log = _minimal_log([_extra(0.5), _extra(1.0)])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == []
        assert scenario.onset_times_s  == pytest.approx([0.5, 1.0])


# ── Ordering guarantees ───────────────────────────────────────────────────────

class TestOrdering:
    def test_targets_sorted_by_index_not_event_order(self):
        # Events arrive in reverse target-index order.
        log = _minimal_log([
            _hit(2, onset_time=3.0, error_s=0.0),
            _hit(0, onset_time=1.0, error_s=0.0),
            _hit(1, onset_time=2.0, error_s=0.0),
        ])
        scenario = scenario_from_session_log(log)
        # Should be [1.0, 2.0, 3.0] (index 0 → 1 → 2), not [3.0, 1.0, 2.0]
        assert scenario.target_times_s == pytest.approx([1.0, 2.0, 3.0])

    def test_onset_times_sorted_ascending(self):
        # Hits arrive out of time order.
        log = _minimal_log([
            _hit(1, onset_time=2.0, error_s=0.0),
            _hit(0, onset_time=1.0, error_s=0.0),
        ])
        scenario = scenario_from_session_log(log)
        assert scenario.onset_times_s == pytest.approx([1.0, 2.0])

    def test_mixed_event_types_targets_still_sorted_by_index(self):
        log = _minimal_log([
            _miss(3, 4.0),
            _hit(1, 2.0, 0.0),
            _miss(2, 3.0),
            _hit(0, 1.0, 0.0),
        ])
        scenario = scenario_from_session_log(log)
        assert scenario.target_times_s == pytest.approx([1.0, 2.0, 3.0, 4.0])


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_timing_scenario(self):
        scenario = scenario_from_session_log(_minimal_log())
        assert isinstance(scenario, TimingScenario)

    def test_has_four_fields(self):
        scenario = scenario_from_session_log(_minimal_log())
        targets, onsets, title, description = scenario   # tuple unpacking
        assert isinstance(targets,     list)
        assert isinstance(onsets,      list)
        assert isinstance(title,       str)
        assert isinstance(description, str)

    def test_field_access_by_name(self):
        scenario = scenario_from_session_log(_minimal_log())
        assert scenario.target_times_s is not None
        assert scenario.onset_times_s  is not None
        assert scenario.title          is not None
        assert scenario.description    is not None


# ── Title and description ─────────────────────────────────────────────────────

class TestLabels:
    def test_title_contains_session_date(self):
        log = _minimal_log()   # started_at = "2026-05-27T10:00:00"
        scenario = scenario_from_session_log(log)
        assert "2026-05-27" in scenario.title

    def test_description_includes_bpm_from_metadata(self):
        log = _minimal_log(metadata={"bpm": "120.0"})
        scenario = scenario_from_session_log(log)
        assert "120.0" in scenario.description

    def test_description_includes_device_from_metadata(self):
        log = _minimal_log(metadata={"device": "Apogee MiniMe"})
        scenario = scenario_from_session_log(log)
        assert "Apogee MiniMe" in scenario.description

    def test_description_omits_missing_device(self):
        log = _minimal_log(metadata={"bpm": "90.0"})
        scenario = scenario_from_session_log(log)
        # No crash; device field just absent from description
        assert isinstance(scenario.description, str)


# ── Value round-trip precision ────────────────────────────────────────────────

class TestRoundTrip:
    def test_expected_time_recovered_within_float_tolerance(self):
        # Simulate: expected = 2.667, onset = 2.721, error = +0.054
        expected = 2.666_666_666_666_667
        onset    = 2.72125
        error    = onset - expected

        log = _minimal_log([_hit(0, onset_time=onset, error_s=error)])
        scenario = scenario_from_session_log(log)

        recovered = scenario.target_times_s[0]
        assert abs(recovered - expected) < 1e-9

    def test_multiple_hits_round_trip(self):
        pairs = [
            (0, 1.0, 0.054),
            (1, 2.0, -0.028),
            (2, 3.0, 0.063),
        ]
        events = [_hit(idx, onset, err) for idx, onset, err in pairs]
        log    = _minimal_log(events)
        scenario = scenario_from_session_log(log)

        for (idx, onset, err), recovered_target in zip(pairs, scenario.target_times_s):
            expected = onset - err
            assert abs(recovered_target - expected) < 1e-9


# ── Demo scenarios ────────────────────────────────────────────────────────────

class TestDemoScenarios:
    @pytest.mark.parametrize("name", list(_ALL_SCENARIOS))
    def test_demo_scenario_returns_timing_scenario(self, name):
        scenario = _ALL_SCENARIOS[name]()
        assert isinstance(scenario, TimingScenario)

    @pytest.mark.parametrize("name", list(_ALL_SCENARIOS))
    def test_demo_scenario_non_empty_targets(self, name):
        scenario = _ALL_SCENARIOS[name]()
        assert len(scenario.target_times_s) > 0

    @pytest.mark.parametrize("name", list(_ALL_SCENARIOS))
    def test_demo_scenario_tuple_unpack(self, name):
        targets, onsets, title, desc = _ALL_SCENARIOS[name]()
        assert isinstance(targets, list)
        assert isinstance(title,   str)


# ── CLI argument parsing ──────────────────────────────────────────────────────

class TestArgParsing:
    def test_no_args_file_is_none(self):
        args = _parse_args([])
        assert args.file is None

    def test_positional_file_arg_sets_file(self):
        args = _parse_args(["some/session.json"])
        assert args.file == "some/session.json"

    def test_scenario_flag_accepted(self):
        args = _parse_args(["--scenario", "perfect", "mixed"])
        assert args.scenario == ["perfect", "mixed"]

    def test_output_flag(self):
        args = _parse_args(["-o", "out/diag.png"])
        assert args.output == "out/diag.png"

    def test_tolerance_ms_parsed_as_float(self):
        args = _parse_args(["--tolerance-ms", "100"])
        assert args.tolerance_ms == pytest.approx(100.0)

    def test_on_time_ms_parsed_as_float(self):
        args = _parse_args(["--on-time-ms", "20"])
        assert args.on_time_ms == pytest.approx(20.0)

    def test_file_and_output_together(self):
        args = _parse_args(["session.json", "-o", "diag.png"])
        assert args.file   == "session.json"
        assert args.output == "diag.png"

    def test_file_mode_detected_when_positional_present(self):
        args = _parse_args(["any.session.json"])
        # file mode: args.file is not None
        assert args.file is not None

    def test_demo_mode_detected_when_no_positional(self):
        args = _parse_args([])
        # demo mode: args.file is None
        assert args.file is None
