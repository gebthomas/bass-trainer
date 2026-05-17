"""Regression tests for core/session_replay.py.

Each JSON fixture under tests/fixtures/sessions/ is loaded and replayed.
The test asserts counts from the "expected" key when present, and does a
full field-by-field comparison against "golden_events" when present.

To add a new scenario: drop a JSON file into tests/fixtures/sessions/ with
at least "bpm", "count_in_beats", "targets", "onsets", and "expected".
No code changes needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_replay import replay_session_data

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sessions"


def _fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda p: p.stem)
def test_replay_fixture(fixture_path: Path) -> None:
    data   = json.loads(fixture_path.read_text())
    events = replay_session_data(data)

    expected = data.get("expected", {})

    if "total" in expected:
        assert len(events) == expected["total"], (
            f"total events: got {len(events)}, expected {expected['total']}"
        )

    if "hits" in expected:
        hits = [e for e in events if e["detected_note"] is not None]
        assert len(hits) == expected["hits"], (
            f"hits: got {len(hits)}, expected {expected['hits']}"
        )

    if "misses" in expected:
        misses = [e for e in events if e["detected_note"] is None]
        assert len(misses) == expected["misses"], (
            f"misses: got {len(misses)}, expected {expected['misses']}"
        )

    if "good" in expected:
        good = sum(1 for e in events if e["severity"] == "good")
        assert good == expected["good"], (
            f"good count: got {good}, expected {expected['good']}"
        )

    if "warn" in expected:
        warn = sum(1 for e in events if e["severity"] == "warn")
        assert warn == expected["warn"], (
            f"warn count: got {warn}, expected {expected['warn']}"
        )

    golden = data.get("golden_events")
    if golden is not None:
        assert len(events) == len(golden), (
            f"golden_events length mismatch: got {len(events)}, expected {len(golden)}"
        )
        for i, (actual, gold) in enumerate(zip(events, golden)):
            for key, gold_val in gold.items():
                actual_val = actual.get(key)
                if isinstance(gold_val, float):
                    assert actual_val == pytest.approx(gold_val, abs=1e-9), (
                        f"event[{i}][{key!r}]: got {actual_val}, expected {gold_val}"
                    )
                else:
                    assert actual_val == gold_val, (
                        f"event[{i}][{key!r}]: got {actual_val!r}, expected {gold_val!r}"
                    )
