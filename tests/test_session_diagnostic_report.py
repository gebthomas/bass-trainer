"""Tests for scripts/session_diagnostic_report.py.

All tests use in-memory data or pytest's tmp_path fixture — no persistent
file I/O, no audio hardware.

Test matrix
-----------
derive_output_paths
  1.  Default dir == session file's parent
  2.  .session.json suffix is stripped to form the stem
  3.  Non-.session.json extension: stem from Path.stem (last suffix only)
  4.  --out-dir overrides the parent directory
  5.  Nested --out-dir path is used as-is (no extra subdirectories added)
  6.  PNG path has .timeline.png extension
  7.  TXT path has .summary.txt extension
  8.  Both paths share the same parent when default dir is used
  9.  Both paths share the same parent when --out-dir is given

format_summary_text
  10. Contains the session file path (filename at minimum)
  11. Contains the scenario title
  12. Contains match tolerance in ms
  13. Contains on-time window in ms
  14. Contains total target count
  15. Contains on-time count
  16. Contains early count
  17. Contains late count
  18. Contains miss count
  19. Contains extra onset count
  20. Contains mean signed error
  21. Contains mean abs error
  22. Contains max abs error
  23. All-misses → error fields show "—"
  24. Zero targets → hit rate shows 0.0%
  25. Output is deterministic (same inputs → identical strings)

generate_report
  26. PNG file is created
  27. TXT file is created
  28. Both returned paths are resolved absolute paths
  29. PNG file is non-empty
  30. TXT content contains the session filename
  31. Custom --out-dir: files land in the given directory
  32. Custom --out-dir: missing directories are created
  33. Custom tolerance reflected in summary text
  34. Custom on-time-ms reflected in summary text
  35. All-misses session: TXT shows "—" for error fields

CLI argument parsing
  36. Positional session arg is captured
  37. --out-dir is captured
  38. --tolerance-ms is parsed as float
  39. --on-time-ms is parsed as float
  40. Defaults match the constants from plot_session_timeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from session_diagnostic_report import (
    _DEFAULT_ON_TIME_MS,
    _DEFAULT_TOLERANCE_MS,
    _parse_args,
    derive_output_paths,
    format_summary_text,
    generate_report,
)
from plot_session_timeline import EvaluationSummary, TimingScenario


# ── Fixtures and helpers ──────────────────────────────────────────────────────

def _minimal_session_json(events: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "started_at":     "2026-05-27T10:00:00",
        "events":         events or [],
        "metadata":       {"bpm": "120.0"},
    }


def _write_session(tmp_path: Path, name: str = "example.session.json",
                   events: list[dict] | None = None) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(_minimal_session_json(events)), encoding="utf-8")
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


def _make_summary(
    n_targets: int = 4,
    n_on_time: int = 2,
    n_early: int = 1,
    n_late: int = 0,
    n_miss: int = 1,
    n_unmatched: int = 0,
    mean_signed: float | None = 5.0,
    mean_abs: float | None = 15.0,
    max_abs: float | None = 25.0,
) -> EvaluationSummary:
    return EvaluationSummary(
        n_targets            = n_targets,
        n_on_time            = n_on_time,
        n_early              = n_early,
        n_late               = n_late,
        n_miss               = n_miss,
        n_unmatched_onsets   = n_unmatched,
        mean_signed_error_ms = mean_signed,
        mean_abs_error_ms    = mean_abs,
        max_abs_error_ms     = max_abs,
    )


def _make_scenario(title: str = "Test session", desc: str = "4 targets · 120.0 BPM") -> TimingScenario:
    return TimingScenario(
        target_times_s = [1.0, 2.0, 3.0, 4.0],
        onset_times_s  = [1.0, 2.0, 3.0],
        title          = title,
        description    = desc,
    )


# ── derive_output_paths ───────────────────────────────────────────────────────

class TestDeriveOutputPaths:
    def test_default_dir_is_session_parent(self):
        png, txt = derive_output_paths("diagnostics/example.session.json")
        assert png.parent == Path("diagnostics")
        assert txt.parent == Path("diagnostics")

    def test_stem_strips_session_json_suffix(self):
        png, txt = derive_output_paths("diagnostics/example.session.json")
        assert png.name == "example.timeline.png"
        assert txt.name == "example.summary.txt"

    def test_non_standard_extension_uses_path_stem(self):
        png, txt = derive_output_paths("some/file.json")
        assert png.name == "file.timeline.png"
        assert txt.name == "file.summary.txt"

    def test_out_dir_overrides_parent(self):
        png, txt = derive_output_paths("diagnostics/example.session.json",
                                       out_dir="reports/")
        assert png.parent == Path("reports")
        assert txt.parent == Path("reports")

    def test_out_dir_nested_used_as_is(self):
        png, _ = derive_output_paths("example.session.json",
                                     out_dir="a/b/c")
        assert png.parent == Path("a/b/c")

    def test_png_extension(self):
        png, _ = derive_output_paths("session.session.json")
        assert png.suffix == ".png"
        assert png.name.endswith(".timeline.png")

    def test_txt_extension(self):
        _, txt = derive_output_paths("session.session.json")
        assert txt.suffix == ".txt"
        assert txt.name.endswith(".summary.txt")

    def test_both_share_parent_default(self):
        png, txt = derive_output_paths("data/foo.session.json")
        assert png.parent == txt.parent

    def test_both_share_parent_with_out_dir(self):
        png, txt = derive_output_paths("data/foo.session.json", out_dir="out/")
        assert png.parent == txt.parent


# ── format_summary_text ───────────────────────────────────────────────────────

class TestFormatSummaryText:
    def _text(self, *, path="session.json", title="My session",
              desc="4 targets · 120 BPM", summary=None,
              tolerance_ms=80.0, on_time_ms=30.0) -> str:
        s = summary or _make_summary()
        sc = TimingScenario([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0], title, desc)
        return format_summary_text(path, sc, s, tolerance_ms, on_time_ms)

    def test_contains_session_filename(self):
        text = self._text(path="diagnostics/my_session.session.json")
        assert "my_session.session.json" in text

    def test_contains_scenario_title(self):
        text = self._text(title="Session 2026-05-27")
        assert "Session 2026-05-27" in text

    def test_contains_tolerance_ms(self):
        text = self._text(tolerance_ms=100.0)
        assert "100.0" in text

    def test_contains_on_time_ms(self):
        text = self._text(on_time_ms=20.0)
        assert "20.0" in text

    def test_contains_total_targets(self):
        s    = _make_summary(n_targets=6)
        text = self._text(summary=s)
        assert "6" in text

    def test_contains_on_time_count(self):
        s    = _make_summary(n_on_time=3)
        text = self._text(summary=s)
        assert "3" in text

    def test_contains_early_count(self):
        s    = _make_summary(n_early=2)
        text = self._text(summary=s)
        assert "2" in text

    def test_contains_late_count(self):
        s    = _make_summary(n_late=1)
        text = self._text(summary=s)
        assert "1" in text

    def test_contains_miss_count(self):
        s    = _make_summary(n_miss=2)
        text = self._text(summary=s)
        assert "2" in text

    def test_contains_extra_onset_count(self):
        s    = _make_summary(n_unmatched=3)
        text = self._text(summary=s)
        assert "3" in text

    def test_contains_mean_signed_error(self):
        s    = _make_summary(mean_signed=12.5)
        text = self._text(summary=s)
        assert "12.5" in text

    def test_contains_mean_abs_error(self):
        s    = _make_summary(mean_abs=18.3)
        text = self._text(summary=s)
        assert "18.3" in text

    def test_contains_max_abs_error(self):
        s    = _make_summary(max_abs=42.7)
        text = self._text(summary=s)
        assert "42.7" in text

    def test_all_misses_shows_dash_for_error_fields(self):
        s = _make_summary(
            n_targets=2, n_on_time=0, n_early=0, n_late=0, n_miss=2,
            mean_signed=None, mean_abs=None, max_abs=None,
        )
        text = self._text(summary=s)
        assert text.count("—") >= 3

    def test_zero_targets_shows_zero_hit_rate(self):
        s = EvaluationSummary(0, 0, 0, 0, 0, 0, None, None, None)
        sc = TimingScenario([], [], "empty", "")
        text = format_summary_text("s.json", sc, s, 80.0, 30.0)
        assert "0.0%" in text

    def test_is_deterministic(self):
        s  = _make_summary()
        sc = _make_scenario()
        t1 = format_summary_text("s.json", sc, s, 80.0, 30.0)
        t2 = format_summary_text("s.json", sc, s, 80.0, 30.0)
        assert t1 == t2

    def test_signed_error_has_plus_prefix_for_positive(self):
        s    = _make_summary(mean_signed=8.0)
        text = self._text(summary=s)
        assert "+8.0 ms" in text

    def test_signed_error_has_minus_prefix_for_negative(self):
        s    = _make_summary(mean_signed=-12.5)
        text = self._text(summary=s)
        assert "-12.5 ms" in text


# ── generate_report ───────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_creates_png_file(self, tmp_path):
        session = _write_session(tmp_path)
        png, _  = generate_report(session)
        assert png.exists()

    def test_creates_txt_file(self, tmp_path):
        session = _write_session(tmp_path)
        _, txt  = generate_report(session)
        assert txt.exists()

    def test_returns_resolved_absolute_paths(self, tmp_path):
        session  = _write_session(tmp_path)
        png, txt = generate_report(session)
        assert png.is_absolute()
        assert txt.is_absolute()

    def test_png_is_non_empty(self, tmp_path):
        session = _write_session(tmp_path)
        png, _  = generate_report(session)
        assert png.stat().st_size > 0

    def test_txt_contains_session_filename(self, tmp_path):
        session = _write_session(tmp_path, name="myrun.session.json")
        _, txt  = generate_report(session)
        assert "myrun.session.json" in txt.read_text(encoding="utf-8")

    def test_custom_out_dir_files_land_in_that_dir(self, tmp_path):
        session  = _write_session(tmp_path)
        out_dir  = tmp_path / "reports"
        png, txt = generate_report(session, out_dir=out_dir)
        assert png.parent == out_dir.resolve()
        assert txt.parent == out_dir.resolve()

    def test_custom_out_dir_creates_missing_directories(self, tmp_path):
        session = _write_session(tmp_path)
        out_dir = tmp_path / "nested" / "output"
        assert not out_dir.exists()
        generate_report(session, out_dir=out_dir)
        assert out_dir.exists()

    def test_custom_tolerance_reflected_in_summary_text(self, tmp_path):
        session = _write_session(tmp_path)
        _, txt  = generate_report(session, tolerance_ms=120.0)
        assert "120.0" in txt.read_text(encoding="utf-8")

    def test_custom_on_time_reflected_in_summary_text(self, tmp_path):
        session = _write_session(tmp_path)
        _, txt  = generate_report(session, on_time_ms=15.0)
        assert "15.0" in txt.read_text(encoding="utf-8")

    def test_all_misses_session_error_fields_are_dash(self, tmp_path):
        events = [
            _miss_event(0, 1.0),
            _miss_event(1, 2.0),
        ]
        session = _write_session(tmp_path, events=events)
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        assert "—" in content

    def test_hits_session_error_fields_are_numeric(self, tmp_path):
        events = [
            _hit_event(0, time_sec=1.01, error_s=0.01),
            _hit_event(1, time_sec=2.02, error_s=0.02),
        ]
        session = _write_session(tmp_path, events=events)
        _, txt  = generate_report(session)
        content = txt.read_text(encoding="utf-8")
        # No dash for errors when there are matched hits
        lines = {ln.split(":")[0].strip(): ln for ln in content.splitlines() if ":" in ln}
        assert "—" not in lines.get("Mean signed error", "——")

    def test_stem_derived_correctly_for_default_dir(self, tmp_path):
        session  = _write_session(tmp_path, name="run42.session.json")
        png, txt = generate_report(session)
        assert png.name == "run42.timeline.png"
        assert txt.name == "run42.summary.txt"


# ── CLI argument parsing ──────────────────────────────────────────────────────

class TestCLIArgParsing:
    def test_positional_session_arg(self):
        args = _parse_args(["session.json"])
        assert args.session == "session.json"

    def test_out_dir_flag(self):
        args = _parse_args(["session.json", "--out-dir", "reports/"])
        assert args.out_dir == "reports/"

    def test_out_dir_default_is_none(self):
        args = _parse_args(["session.json"])
        assert args.out_dir is None

    def test_tolerance_ms_flag(self):
        args = _parse_args(["session.json", "--tolerance-ms", "100"])
        assert args.tolerance_ms == pytest.approx(100.0)

    def test_on_time_ms_flag(self):
        args = _parse_args(["session.json", "--on-time-ms", "20"])
        assert args.on_time_ms == pytest.approx(20.0)

    def test_defaults_match_module_constants(self):
        args = _parse_args(["session.json"])
        assert args.tolerance_ms == pytest.approx(_DEFAULT_TOLERANCE_MS)
        assert args.on_time_ms   == pytest.approx(_DEFAULT_ON_TIME_MS)

    def test_all_flags_together(self):
        args = _parse_args([
            "my.session.json",
            "--out-dir", "out/",
            "--tolerance-ms", "60",
            "--on-time-ms", "25",
        ])
        assert args.session      == "my.session.json"
        assert args.out_dir      == "out/"
        assert args.tolerance_ms == pytest.approx(60.0)
        assert args.on_time_ms   == pytest.approx(25.0)
