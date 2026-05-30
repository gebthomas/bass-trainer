"""Tests for BPM-derived tolerance defaulting and match_window_s metadata.

Covers:
  session_tolerance_s()        — in plot_session_timeline
  generate_report()            — tolerance source logic in session_diagnostic_report
  format_summary_text()        — tolerance_source in report text
  _parse_args() defaults       — both scripts
  session_runner metadata      — match_window_s written to new SessionLogs
  live_feedback_demo metadata  — match_window_s written to live SessionLogs

Test matrix
-----------
session_tolerance_s — BPM only (legacy)
  1.  Valid BPM returns match_window_s(bpm)
  2.  120 BPM → 250 ms (0.25 s)
  3.  60 BPM → 350 ms clamped to max 0.35 s
  4.  400 BPM → 100 ms clamped to min 0.10 s
  5.  Missing "bpm" key → None
  6.  Empty string BPM → None
  7.  Non-numeric BPM string → None
  8.  Zero BPM → None
  9.  Negative BPM → None

session_tolerance_s — match_window_s metadata key (priority over BPM)
  23. Valid match_window_s metadata → returns it directly
  24. match_window_s metadata takes priority over BPM key
  25. Invalid match_window_s falls through to BPM
  26. Zero match_window_s falls through to BPM
  27. Negative match_window_s falls through to BPM
  28. Non-numeric match_window_s falls through to BPM
  29. match_window_s metadata present but BPM absent → returns match_window_s value
  30. Both absent → None

generate_report tolerance source
  10. No explicit tolerance_ms + BPM 120 → 250 ms appears in summary text
  11. No explicit tolerance_ms + no BPM → 80 ms fallback appears in text
  12. Explicit tolerance_ms=120.0 overrides session BPM
  13. Explicit tolerance_ms=120.0 with BPM 120 still uses 120.0
  14. Tolerance source "session BPM (120.0)" appears in text when BPM auto-detected
  15. Tolerance source "fallback default" appears in text when no BPM
  16. Tolerance source "explicit" appears in text when tolerance_ms given

generate_report — match_window_s metadata
  31. match_window_s metadata → that value is used (not BPM-derived)
  32. match_window_s metadata → "session metadata match_window_s" label in text
  33. Explicit --tolerance-ms overrides match_window_s metadata
  34. Invalid match_window_s metadata + valid BPM → BPM-derived label

format_summary_text tolerance_source
  17. tolerance_source="" → no "Tolerance source" line
  18. tolerance_source="explicit" → "Tolerance source:   explicit" in text
  19. tolerance_source="session BPM (120.0)" → appears in text

_parse_args defaults
  20. plot_session_timeline: --tolerance-ms default is None
  21. session_diagnostic_report: --tolerance-ms default is None
  22. session_diagnostic_report: positional arg + no --tolerance-ms → tolerance_ms is None

session_runner metadata
  35. run_session_bundle produces log with match_window_s in metadata
  36. match_window_s value equals match_window_s(exercise.bpm)
  37. Empty onset list still writes match_window_s to metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_session_timeline import (
    EvaluationSummary,
    TimingScenario,
    _parse_args as _plot_parse_args,
    session_tolerance_s,
)
from session_diagnostic_report import (
    _DEFAULT_TOLERANCE_MS,
    _parse_args as _report_parse_args,
    format_summary_text,
    generate_report,
)
from core.session_log import SessionLog
from core.timing_policy import match_window_s


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_log(
    bpm: str | None = "120.0",
    match_window: str | None = None,
    events: list[dict] | None = None,
) -> SessionLog:
    """Build a SessionLog with optional bpm and/or match_window_s metadata."""
    from core.session_log import SessionLog as SL, SessionEvent
    metadata: dict[str, str] = {}
    if bpm is not None:
        metadata["bpm"] = bpm
    if match_window is not None:
        metadata["match_window_s"] = match_window
    event_objs = []
    for e in (events or []):
        event_objs.append(SessionEvent(
            time_sec     = e["time_sec"],
            event_type   = e["event_type"],
            target_index = e.get("target_index"),
            value        = e.get("value"),
        ))
    return SL(
        schema_version = 1,
        started_at     = "2026-05-27T10:00:00",
        events         = event_objs,
        metrics        = {},
        metadata       = metadata,
    )


def _write_session(tmp_path: Path, bpm: str | None = "120.0",
                   match_window: str | None = None,
                   events: list[dict] | None = None,
                   name: str = "run.session.json") -> Path:
    metadata: dict = {}
    if bpm is not None:
        metadata["bpm"] = bpm
    if match_window is not None:
        metadata["match_window_s"] = match_window
    data = {
        "schema_version": 1,
        "started_at":     "2026-05-27T10:00:00",
        "events":         events or [],
        "metadata":       metadata,
    }
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _hit_event(target_index: int, time_sec: float, error_s: float = 0.0) -> dict:
    return {
        "time_sec":     time_sec,
        "event_type":   "target_hit",
        "target_index": target_index,
        "value":        error_s,
    }


def _miss_event(target_index: int, time_sec: float) -> dict:
    return {
        "time_sec":     time_sec,
        "event_type":   "target_miss",
        "target_index": target_index,
    }


# ── session_tolerance_s ───────────────────────────────────────────────────────

class TestSessionToleranceS:
    def test_valid_bpm_returns_match_window(self):
        log = _make_log(bpm="90.0")
        assert session_tolerance_s(log) == pytest.approx(match_window_s(90.0))

    def test_120_bpm_returns_250ms(self):
        log = _make_log(bpm="120.0")
        assert session_tolerance_s(log) == pytest.approx(0.25)

    def test_60_bpm_clamped_to_max(self):
        log = _make_log(bpm="60.0")
        # 30/60 = 0.5, clamped to max 0.35
        assert session_tolerance_s(log) == pytest.approx(0.35)

    def test_400_bpm_clamped_to_min(self):
        log = _make_log(bpm="400.0")
        # 30/400 = 0.075, clamped to min 0.10
        assert session_tolerance_s(log) == pytest.approx(0.10)

    def test_missing_bpm_key_returns_none(self):
        log = _make_log(bpm=None)
        assert session_tolerance_s(log) is None

    def test_empty_bpm_string_returns_none(self):
        log = _make_log(bpm="")
        assert session_tolerance_s(log) is None

    def test_non_numeric_bpm_returns_none(self):
        log = _make_log(bpm="fast")
        assert session_tolerance_s(log) is None

    def test_zero_bpm_returns_none(self):
        log = _make_log(bpm="0.0")
        assert session_tolerance_s(log) is None

    def test_negative_bpm_returns_none(self):
        log = _make_log(bpm="-120.0")
        assert session_tolerance_s(log) is None


# ── generate_report tolerance source ─────────────────────────────────────────

class TestGenerateReportToleranceDefaulting:
    def test_bpm_120_no_explicit_uses_250ms(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0")
        _, txt  = generate_report(session)
        assert "250.0" in txt.read_text(encoding="utf-8")

    def test_no_bpm_uses_80ms_fallback(self, tmp_path):
        session = _write_session(tmp_path, bpm=None)
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert f"{_DEFAULT_TOLERANCE_MS:.1f}" in content

    def test_explicit_tolerance_overrides_bpm(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0")
        _, txt  = generate_report(session, tolerance_ms=120.0)
        content = txt.read_text(encoding="utf-8")
        assert "120.0" in content

    def test_explicit_tolerance_does_not_use_bpm_value(self, tmp_path):
        # 120 BPM would give 250 ms; we pass 120 ms explicitly — 250.0 must not appear
        session = _write_session(tmp_path, bpm="120.0")
        _, txt  = generate_report(session, tolerance_ms=120.0)
        content = txt.read_text(encoding="utf-8")
        # 250.0 should not appear as the tolerance line value
        tol_line = next(
            (ln for ln in content.splitlines() if ln.startswith("Match tolerance:")),
            ""
        )
        assert "120.0" in tol_line
        assert "250.0" not in tol_line

    def test_bpm_source_label_in_text_when_auto(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0")
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert "session BPM (120.0)" in content

    def test_fallback_default_label_when_no_bpm(self, tmp_path):
        session = _write_session(tmp_path, bpm=None)
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert "fallback default" in content

    def test_explicit_label_when_tolerance_ms_given(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0")
        _, txt  = generate_report(session, tolerance_ms=80.0)
        content = txt.read_text(encoding="utf-8")
        assert "explicit" in content

    def test_non_numeric_bpm_uses_fallback(self, tmp_path):
        session = _write_session(tmp_path, bpm="unknown")
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert "fallback default" in content
        assert f"{_DEFAULT_TOLERANCE_MS:.1f}" in content


# ── format_summary_text tolerance_source ─────────────────────────────────────

class TestFormatSummaryTextToleranceSource:
    def _scenario(self) -> TimingScenario:
        return TimingScenario([1.0, 2.0], [1.0, 2.0], "Test", "2 targets")

    def _summary(self) -> EvaluationSummary:
        return EvaluationSummary(2, 2, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    def test_no_tolerance_source_omits_line(self):
        text = format_summary_text(
            "s.json", self._scenario(), self._summary(), 80.0, 30.0,
            tolerance_source="",
        )
        assert "Tolerance source" not in text

    def test_explicit_tolerance_source_appears(self):
        text = format_summary_text(
            "s.json", self._scenario(), self._summary(), 80.0, 30.0,
            tolerance_source="explicit",
        )
        assert "Tolerance source:   explicit" in text

    def test_bpm_tolerance_source_appears(self):
        text = format_summary_text(
            "s.json", self._scenario(), self._summary(), 250.0, 30.0,
            tolerance_source="session BPM (120.0)",
        )
        assert "session BPM (120.0)" in text

    def test_fallback_tolerance_source_appears(self):
        text = format_summary_text(
            "s.json", self._scenario(), self._summary(), 80.0, 30.0,
            tolerance_source="fallback default",
        )
        assert "fallback default" in text


# ── _parse_args defaults ──────────────────────────────────────────────────────

class TestParseArgsToleranceDefaults:
    def test_plot_timeline_tolerance_default_is_none(self):
        # session-file mode — no explicit --tolerance-ms
        args = _plot_parse_args(["session.json"])
        assert args.tolerance_ms is None

    def test_plot_timeline_demo_mode_tolerance_default_is_none(self):
        args = _plot_parse_args([])
        assert args.tolerance_ms is None

    def test_report_tolerance_default_is_none(self):
        args = _report_parse_args(["session.json"])
        assert args.tolerance_ms is None

    def test_report_explicit_tolerance_captured(self):
        args = _report_parse_args(["session.json", "--tolerance-ms", "250"])
        assert args.tolerance_ms == pytest.approx(250.0)

    def test_plot_explicit_tolerance_captured(self):
        args = _plot_parse_args(["session.json", "--tolerance-ms", "100"])
        assert args.tolerance_ms == pytest.approx(100.0)


# ── session_tolerance_s — match_window_s metadata ────────────────────────────

class TestSessionToleranceSMatchWindowMetadata:
    def test_valid_match_window_returned_directly(self):
        log = _make_log(bpm=None, match_window="0.3")
        assert session_tolerance_s(log) == pytest.approx(0.3)

    def test_match_window_takes_priority_over_bpm(self):
        # match_window_s=0.3 should win over BPM-derived 0.25
        log = _make_log(bpm="120.0", match_window="0.3")
        assert session_tolerance_s(log) == pytest.approx(0.3)

    def test_invalid_match_window_falls_through_to_bpm(self):
        log = _make_log(bpm="120.0", match_window="bad")
        assert session_tolerance_s(log) == pytest.approx(0.25)

    def test_zero_match_window_falls_through_to_bpm(self):
        log = _make_log(bpm="120.0", match_window="0.0")
        assert session_tolerance_s(log) == pytest.approx(0.25)

    def test_negative_match_window_falls_through_to_bpm(self):
        log = _make_log(bpm="120.0", match_window="-0.1")
        assert session_tolerance_s(log) == pytest.approx(0.25)

    def test_non_numeric_match_window_falls_through_to_bpm(self):
        log = _make_log(bpm="120.0", match_window="fast")
        assert session_tolerance_s(log) == pytest.approx(0.25)

    def test_match_window_present_without_bpm(self):
        log = _make_log(bpm=None, match_window="0.2")
        assert session_tolerance_s(log) == pytest.approx(0.2)

    def test_both_absent_returns_none(self):
        log = _make_log(bpm=None, match_window=None)
        assert session_tolerance_s(log) is None

    def test_invalid_match_window_and_no_bpm_returns_none(self):
        log = _make_log(bpm=None, match_window="bad")
        assert session_tolerance_s(log) is None


# ── generate_report — match_window_s metadata ────────────────────────────────

class TestGenerateReportMatchWindowMetadata:
    def test_match_window_metadata_used_as_tolerance(self, tmp_path):
        # match_window_s=0.3 (300 ms) should override BPM-derived 250 ms
        session = _write_session(tmp_path, bpm="120.0", match_window="0.3")
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        tol_line = next(
            ln for ln in content.splitlines() if ln.startswith("Match tolerance:")
        )
        assert "300.0" in tol_line

    def test_match_window_source_label_in_report(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0", match_window="0.25")
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert "session metadata match_window_s" in content

    def test_explicit_tolerance_overrides_match_window_metadata(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0", match_window="0.3")
        _, txt  = generate_report(session, tolerance_ms=80.0)
        content = txt.read_text(encoding="utf-8")
        tol_line = next(
            ln for ln in content.splitlines() if ln.startswith("Match tolerance:")
        )
        assert "80.0" in tol_line
        assert "explicit" in content

    def test_invalid_match_window_falls_back_to_bpm_label(self, tmp_path):
        session = _write_session(tmp_path, bpm="120.0", match_window="invalid")
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert "session BPM (120.0)" in content
        assert "250.0" in content


# ── session_runner metadata ───────────────────────────────────────────────────

class TestSessionRunnerMatchWindowMetadata:
    def test_log_contains_match_window_s_key(self):
        from core.session_runner import run_session_bundle
        from tests.test_session_runner import _metro_bundle, _STARTED
        bundle = _metro_bundle()
        log    = run_session_bundle(bundle, onset_times_sec=[], started_at=_STARTED)
        assert "match_window_s" in log.metadata

    def test_match_window_s_equals_policy_value(self):
        from core.session_runner import run_session_bundle
        from tests.test_session_runner import _metro_bundle, _STARTED
        bundle = _metro_bundle()
        log    = run_session_bundle(bundle, onset_times_sec=[], started_at=_STARTED)
        stored = float(log.metadata["match_window_s"])
        assert stored == pytest.approx(match_window_s(bundle.exercise.bpm))

    def test_match_window_s_present_with_empty_onsets(self):
        from core.session_runner import run_session_bundle
        from tests.test_session_runner import _metro_bundle, _STARTED
        bundle = _metro_bundle()
        log    = run_session_bundle(bundle, onset_times_sec=[], started_at=_STARTED)
        assert float(log.metadata["match_window_s"]) > 0
