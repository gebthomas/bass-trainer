"""Tests for figure file output in scripts/plot_session_timeline.py.

Covers save_figure(), the --output CLI flag, and end-to-end save paths for
both demo mode and session-file mode.  All tests use pytest's tmp_path
fixture — no permanent files are written.

Test matrix
-----------
save_figure() — file creation
  1.  Output file is created at the given path
  2.  Created file is non-empty (contains image data)
  3.  Returns a Path object
  4.  Returned path is the resolved absolute path

save_figure() — parent directory creation
  5.  Creates a single missing parent directory
  6.  Creates multiple levels of missing parent directories
  7.  Does not fail when the parent directory already exists

CLI parsing — --output flag
  8.  Long form --output sets args.output
  9.  Short form -o sets args.output
  10. Default value is "diagnostic_timeline.png" when flag omitted

End-to-end: demo mode
  11. build_figure() + save_figure() produces an existing PNG for a single demo
  12. build_figure() + save_figure() produces an existing PNG for all demos

End-to-end: session-file mode
  13. scenario_from_session_file() → build_figure() → save_figure() with an
      in-memory fixture written to tmp_path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root on path so core.* imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# scripts/ on path so plot_session_timeline is importable as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_session_timeline import (
    _ALL_SCENARIOS,
    _parse_args,
    _scenario_perfect,
    build_figure,
    save_figure,
    scenario_from_session_file,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_fig() -> plt.Figure:
    """Minimal matplotlib figure for testing save behaviour."""
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    return fig


def _minimal_session_json() -> dict:
    """Minimal valid session-log dict (no events)."""
    return {
        "schema_version": 1,
        "started_at":     "2026-05-27T10:00:00",
        "events":         [],
        "metadata":       {"bpm": "120.0"},
    }


# ── save_figure(): file creation ──────────────────────────────────────────────

class TestSaveFigureCreatesFile:
    def test_output_file_is_created(self, tmp_path):
        fig = _tiny_fig()
        out = tmp_path / "result.png"
        save_figure(fig, out)
        plt.close(fig)
        assert out.exists()

    def test_output_file_is_non_empty(self, tmp_path):
        fig = _tiny_fig()
        out = tmp_path / "result.png"
        save_figure(fig, out)
        plt.close(fig)
        assert out.stat().st_size > 0

    def test_returns_a_path_object(self, tmp_path):
        fig = _tiny_fig()
        result = save_figure(fig, tmp_path / "result.png")
        plt.close(fig)
        assert isinstance(result, Path)

    def test_returned_path_is_resolved_absolute(self, tmp_path):
        fig   = _tiny_fig()
        out   = tmp_path / "result.png"
        result = save_figure(fig, out)
        plt.close(fig)
        assert result.is_absolute()
        assert result == out.resolve()


# ── save_figure(): parent directory creation ──────────────────────────────────

class TestSaveFigureCreatesParents:
    def test_creates_single_missing_parent(self, tmp_path):
        fig = _tiny_fig()
        out = tmp_path / "subdir" / "result.png"
        assert not out.parent.exists()
        save_figure(fig, out)
        plt.close(fig)
        assert out.exists()

    def test_creates_nested_missing_parents(self, tmp_path):
        fig = _tiny_fig()
        out = tmp_path / "a" / "b" / "c" / "result.png"
        save_figure(fig, out)
        plt.close(fig)
        assert out.exists()

    def test_does_not_fail_when_parent_exists(self, tmp_path):
        fig = _tiny_fig()
        out = tmp_path / "result.png"   # tmp_path already exists
        save_figure(fig, out)           # should not raise
        plt.close(fig)
        assert out.exists()


# ── CLI parsing: --output flag ────────────────────────────────────────────────

class TestOutputArgParsing:
    def test_long_form_output_flag(self):
        args = _parse_args(["--output", "out/diag.png"])
        assert args.output == "out/diag.png"

    def test_short_form_output_flag(self):
        args = _parse_args(["-o", "out/diag.png"])
        assert args.output == "out/diag.png"

    def test_default_output_when_flag_omitted(self):
        args = _parse_args([])
        assert args.output == "diagnostic_timeline.png"

    def test_output_with_session_file(self):
        args = _parse_args(["session.json", "-o", "diag.png"])
        assert args.file   == "session.json"
        assert args.output == "diag.png"

    def test_output_accepts_subdirectory_path(self):
        args = _parse_args(["-o", "diagnostics/2026-05-27/timeline.png"])
        assert args.output == "diagnostics/2026-05-27/timeline.png"


# ── End-to-end: demo mode ─────────────────────────────────────────────────────

class TestDemoOutput:
    def test_single_demo_scenario_saves_png(self, tmp_path):
        scenario = _scenario_perfect()
        fig = build_figure([scenario])
        out = save_figure(fig, tmp_path / "perfect.png")
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 0

    @pytest.mark.parametrize("name", list(_ALL_SCENARIOS))
    def test_each_demo_scenario_saves_png(self, name, tmp_path):
        scenario = _ALL_SCENARIOS[name]()
        fig = build_figure([scenario])
        out = save_figure(fig, tmp_path / f"{name}.png")
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_demos_single_figure_saves_png(self, tmp_path):
        rows = [fn() for fn in _ALL_SCENARIOS.values()]
        fig  = build_figure(rows)
        out  = save_figure(fig, tmp_path / "all_demos.png")
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 0


# ── End-to-end: session-file mode ────────────────────────────────────────────

class TestSessionFileOutput:
    def _write_fixture(self, tmp_path: Path) -> Path:
        """Write a minimal .session.json fixture and return its path."""
        p = tmp_path / "fixture.session.json"
        p.write_text(json.dumps(_minimal_session_json()))
        return p

    def test_session_file_scenario_saves_png(self, tmp_path):
        fixture = self._write_fixture(tmp_path)
        scenario = scenario_from_session_file(fixture)
        fig = build_figure([scenario])
        out = save_figure(fig, tmp_path / "session.png")
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_session_file_output_to_subdirectory(self, tmp_path):
        fixture = self._write_fixture(tmp_path)
        scenario = scenario_from_session_file(fixture)
        fig = build_figure([scenario])
        out = save_figure(fig, tmp_path / "subdir" / "session.png")
        plt.close(fig)
        assert out.exists()

    def test_session_file_returned_path_is_absolute(self, tmp_path):
        fixture = self._write_fixture(tmp_path)
        scenario = scenario_from_session_file(fixture)
        fig = build_figure([scenario])
        result = save_figure(fig, tmp_path / "out.png")
        plt.close(fig)
        assert result.is_absolute()
