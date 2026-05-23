"""Tests for core/severity_policy.py — derived timing severity classification."""

import pytest

from core.session_log import EXTRA_ONSET, TARGET_HIT, TARGET_MISS, SessionEvent
from core.severity_policy import (
    DEFAULT_GOOD_THRESHOLD_S,
    DEFAULT_WARN_THRESHOLD_S,
    SEVERITY_GOOD,
    SEVERITY_MISS,
    SEVERITY_WARN,
    event_timing_severity,
    timing_severity,
)


class TestTimingSeverity:
    def test_exact_good_boundary(self):
        assert timing_severity(DEFAULT_GOOD_THRESHOLD_S) == SEVERITY_GOOD

    def test_exact_warn_boundary(self):
        assert timing_severity(DEFAULT_WARN_THRESHOLD_S) == SEVERITY_WARN

    def test_early_and_late_symmetric(self):
        assert timing_severity(-0.04) == timing_severity(0.04) == SEVERITY_GOOD

    def test_miss_beyond_warn(self):
        assert timing_severity(0.20) == SEVERITY_MISS

    def test_zero_is_good(self):
        assert timing_severity(0.0) == SEVERITY_GOOD

    def test_inside_good_window(self):
        assert timing_severity(0.049) == SEVERITY_GOOD

    def test_inside_warn_window(self):
        assert timing_severity(0.051) == SEVERITY_WARN

    def test_negative_inside_warn_window(self):
        assert timing_severity(-0.10) == SEVERITY_WARN

    def test_invalid_good_s_zero(self):
        with pytest.raises(ValueError, match="good_s must be positive"):
            timing_severity(0.05, good_s=0.0, warn_s=0.12)

    def test_invalid_good_s_negative(self):
        with pytest.raises(ValueError, match="good_s must be positive"):
            timing_severity(0.05, good_s=-0.01, warn_s=0.12)

    def test_invalid_warn_s_zero(self):
        with pytest.raises(ValueError, match="warn_s must be positive"):
            timing_severity(0.05, good_s=0.05, warn_s=0.0)

    def test_invalid_good_greater_than_warn(self):
        with pytest.raises(ValueError, match="good_s must be <= warn_s"):
            timing_severity(0.05, good_s=0.15, warn_s=0.10)

    def test_equal_thresholds_allowed(self):
        # When good_s == warn_s the warn band collapses: at threshold → good, above → miss.
        assert timing_severity(0.05, good_s=0.05, warn_s=0.05) == SEVERITY_GOOD
        assert timing_severity(0.06, good_s=0.05, warn_s=0.05) == SEVERITY_MISS

    def test_custom_thresholds(self):
        assert timing_severity(0.08, good_s=0.03, warn_s=0.06) == SEVERITY_MISS
        assert timing_severity(0.05, good_s=0.03, warn_s=0.06) == SEVERITY_WARN
        assert timing_severity(0.02, good_s=0.03, warn_s=0.06) == SEVERITY_GOOD


class TestEventTimingSeverity:
    def test_target_hit_value_good(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=0.02)
        assert event_timing_severity(event) == SEVERITY_GOOD

    def test_target_hit_value_warn(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=0.08)
        assert event_timing_severity(event) == SEVERITY_WARN

    def test_target_hit_value_miss(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=0.20)
        assert event_timing_severity(event) == SEVERITY_MISS

    def test_target_hit_exact_good_boundary(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=DEFAULT_GOOD_THRESHOLD_S)
        assert event_timing_severity(event) == SEVERITY_GOOD

    def test_target_hit_exact_warn_boundary(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=DEFAULT_WARN_THRESHOLD_S)
        assert event_timing_severity(event) == SEVERITY_WARN

    def test_target_hit_missing_value_raises(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=None)
        with pytest.raises(ValueError, match="value=None"):
            event_timing_severity(event)

    def test_target_miss_returns_miss(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_MISS)
        assert event_timing_severity(event) == SEVERITY_MISS

    def test_extra_onset_returns_none(self):
        event = SessionEvent(time_sec=1.0, event_type=EXTRA_ONSET)
        assert event_timing_severity(event) is None

    def test_negative_timing_error_symmetric(self):
        early = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=-0.04)
        late  = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=0.04)
        assert event_timing_severity(early) == event_timing_severity(late) == SEVERITY_GOOD

    def test_custom_thresholds_forwarded(self):
        event = SessionEvent(time_sec=1.0, event_type=TARGET_HIT, value=0.08)
        # With tighter thresholds 0.08 exceeds warn_s=0.06 → miss
        assert event_timing_severity(event, good_s=0.03, warn_s=0.06) == SEVERITY_MISS
