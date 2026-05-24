"""Cross-module consistency guard for timing severity thresholds.

feedback_events.py and severity_policy.py both define timing severity
thresholds independently, with no shared constant or import between them.
While both coexist, this test is the only thing preventing them from
silently diverging.

See docs/runtime_migration_plan.md §6.2 for the documented risk and the
intended resolution (remove local thresholds from feedback_events.py once
the live display is migrated to severity_policy in Phase 5).
"""

# Importing private constants from feedback_events intentionally.
# This is a transition-protection test — these names are not part of the
# public API of that module.  When feedback_events.py is retired or its
# thresholds are removed (Phase 5 of the migration plan), this test should
# be updated or deleted alongside that change.
from core.feedback_events import _TIMING_GOOD as _FE_GOOD, _TIMING_WARN as _FE_WARN

from core.severity_policy import (
    DEFAULT_GOOD_THRESHOLD_S,
    DEFAULT_WARN_THRESHOLD_S,
    SEVERITY_GOOD,
    SEVERITY_WARN,
    SEVERITY_MISS,
    timing_severity,
)
from core.feedback_events import feedback_event


_EXPECTED_GOOD_S = 0.05
_EXPECTED_WARN_S = 0.12


class TestThresholdValues:
    """Both modules must agree on the canonical threshold values."""

    def test_severity_policy_good_threshold(self):
        assert DEFAULT_GOOD_THRESHOLD_S == _EXPECTED_GOOD_S

    def test_severity_policy_warn_threshold(self):
        assert DEFAULT_WARN_THRESHOLD_S == _EXPECTED_WARN_S

    def test_feedback_events_good_threshold(self):
        assert _FE_GOOD == _EXPECTED_GOOD_S

    def test_feedback_events_warn_threshold(self):
        assert _FE_WARN == _EXPECTED_WARN_S

    def test_thresholds_match_each_other(self):
        assert DEFAULT_GOOD_THRESHOLD_S == _FE_GOOD
        assert DEFAULT_WARN_THRESHOLD_S == _FE_WARN


class TestBehavioralConsistency:
    """Both modules must produce the same severity label at key boundary values.

    This catches threshold disagreement that survives a constant rename or
    copy-paste error — e.g. both constants read 0.05 but one is used as
    a strict inequality and the other as non-strict.
    """

    _TARGET = {"note": "E1", "time": 1.0}

    def _fe_severity(self, timing_error_s: float) -> str:
        """Extract timing-only severity from feedback_event (no pitch, no confidence)."""
        result = {
            "detected_note":     "E1",
            "timing_error_s":    timing_error_s,
            "pitch_error_cents": None,
            "confidence":        None,
        }
        return feedback_event(0, self._TARGET, result)["severity"]

    def test_exact_good_boundary_agrees(self):
        error = _EXPECTED_GOOD_S
        assert timing_severity(error) == SEVERITY_GOOD
        assert self._fe_severity(error) == SEVERITY_GOOD

    def test_just_inside_good_agrees(self):
        error = _EXPECTED_GOOD_S - 0.001
        assert timing_severity(error) == SEVERITY_GOOD
        assert self._fe_severity(error) == SEVERITY_GOOD

    def test_just_outside_good_agrees(self):
        error = _EXPECTED_GOOD_S + 0.001
        assert timing_severity(error) == SEVERITY_WARN
        assert self._fe_severity(error) == SEVERITY_WARN

    def test_exact_warn_boundary_agrees(self):
        error = _EXPECTED_WARN_S
        assert timing_severity(error) == SEVERITY_WARN
        assert self._fe_severity(error) == SEVERITY_WARN

    def test_just_inside_warn_agrees(self):
        error = _EXPECTED_WARN_S - 0.001
        assert timing_severity(error) == SEVERITY_WARN
        assert self._fe_severity(error) == SEVERITY_WARN

    def test_just_outside_warn_agrees(self):
        error = _EXPECTED_WARN_S + 0.001
        assert timing_severity(error) == SEVERITY_MISS
        assert self._fe_severity(error) == SEVERITY_MISS

    def test_early_errors_symmetric(self):
        for error in (_EXPECTED_GOOD_S, _EXPECTED_WARN_S, 0.20):
            assert timing_severity(-error) == timing_severity(error)
            assert self._fe_severity(-error) == self._fe_severity(error)
