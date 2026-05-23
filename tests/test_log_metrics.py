"""Tests for core/log_metrics.py — structured metrics from canonical SessionLog."""

import pytest

from core.session_log import EXTRA_ONSET, TARGET_HIT, TARGET_MISS, SessionEvent, SessionLog
from core.log_metrics import LogMetrics, compute_log_metrics, log_metrics_to_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_log(*events: SessionEvent) -> SessionLog:
    return SessionLog(
        schema_version=1,
        started_at="2026-05-23T10:00:00",
        events=list(events),
    )


def _hit(value: float, target_index: int = 0) -> SessionEvent:
    return SessionEvent(time_sec=1.0, event_type=TARGET_HIT, target_index=target_index, value=value)


def _miss(target_index: int = 0) -> SessionEvent:
    return SessionEvent(time_sec=1.0, event_type=TARGET_MISS, target_index=target_index)


def _extra() -> SessionEvent:
    return SessionEvent(time_sec=1.0, event_type=EXTRA_ONSET)


# ── compute_log_metrics ───────────────────────────────────────────────────────

class TestComputeLogMetrics:
    def test_empty_log(self):
        m = compute_log_metrics(_make_log())
        assert m.targets_total == 0
        assert m.targets_hit == 0
        assert m.targets_missed == 0
        assert m.extra_onsets == 0
        assert m.good_hits == 0
        assert m.warn_hits == 0
        assert m.miss_severity_events == 0
        assert m.mean_signed_error_s is None
        assert m.mean_abs_error_s is None

    def test_all_hits_good(self):
        log = _make_log(_hit(0.02, 0), _hit(-0.03, 1), _hit(0.01, 2))
        m = compute_log_metrics(log)
        assert m.targets_hit == 3
        assert m.targets_missed == 0
        assert m.targets_total == 3
        assert m.good_hits == 3
        assert m.warn_hits == 0
        assert m.miss_severity_events == 0

    def test_mixed_good_warn_miss_hits(self):
        log = _make_log(
            _hit(0.02, 0),   # good  (abs <= 0.05)
            _hit(0.08, 1),   # warn  (0.05 < abs <= 0.12)
            _hit(0.20, 2),   # miss severity (abs > 0.12)
        )
        m = compute_log_metrics(log)
        assert m.targets_hit == 3
        assert m.good_hits == 1
        assert m.warn_hits == 1
        assert m.miss_severity_events == 1

    def test_missed_targets(self):
        log = _make_log(_miss(0), _miss(1))
        m = compute_log_metrics(log)
        assert m.targets_total == 2
        assert m.targets_hit == 0
        assert m.targets_missed == 2
        assert m.miss_severity_events == 2
        assert m.mean_signed_error_s is None
        assert m.mean_abs_error_s is None

    def test_extra_onsets(self):
        log = _make_log(_extra(), _extra())
        m = compute_log_metrics(log)
        assert m.extra_onsets == 2
        assert m.targets_total == 0
        assert m.targets_hit == 0

    def test_targets_total_is_hits_plus_misses(self):
        log = _make_log(_hit(0.02, 0), _miss(1), _miss(2), _extra())
        m = compute_log_metrics(log)
        assert m.targets_total == m.targets_hit + m.targets_missed
        assert m.targets_total == 3

    def test_signed_mean_cancels_to_zero(self):
        log = _make_log(_hit(-0.04, 0), _hit(0.04, 1))
        m = compute_log_metrics(log)
        assert m.mean_signed_error_s == pytest.approx(0.0)
        assert m.mean_abs_error_s == pytest.approx(0.04)

    def test_signed_mean_non_zero(self):
        # -0.10 and +0.02  →  mean signed = -0.04,  mean abs = 0.06
        log = _make_log(_hit(-0.10, 0), _hit(0.02, 1))
        m = compute_log_metrics(log)
        assert m.mean_signed_error_s == pytest.approx(-0.04)
        assert m.mean_abs_error_s == pytest.approx(0.06)

    def test_single_hit_means(self):
        log = _make_log(_hit(0.03, 0))
        m = compute_log_metrics(log)
        assert m.mean_signed_error_s == pytest.approx(0.03)
        assert m.mean_abs_error_s == pytest.approx(0.03)

    def test_no_hits_means_are_none(self):
        log = _make_log(_miss(0), _extra())
        m = compute_log_metrics(log)
        assert m.mean_signed_error_s is None
        assert m.mean_abs_error_s is None

    def test_miss_severity_includes_bad_timing_hit(self):
        log = _make_log(
            _hit(0.20, 0),   # onset was detected, but timing is miss severity
            _miss(1),        # no onset at all
        )
        m = compute_log_metrics(log)
        assert m.targets_hit == 1       # onset was present
        assert m.targets_missed == 1
        assert m.miss_severity_events == 2  # both count

    def test_negative_hit_error_symmetry(self):
        log_late  = _make_log(_hit(0.08, 0))
        log_early = _make_log(_hit(-0.08, 0))
        assert compute_log_metrics(log_late).warn_hits == 1
        assert compute_log_metrics(log_early).warn_hits == 1

    def test_invalid_log_raises(self):
        bad_log = SessionLog(schema_version=0, started_at="2026-05-23T10:00:00")
        with pytest.raises(ValueError):
            compute_log_metrics(bad_log)

    def test_mixed_all_event_types_combined(self):
        log = _make_log(
            _hit(0.01, 0),   # good
            _hit(0.09, 1),   # warn
            _miss(2),        # missed
            _extra(),        # extra
        )
        m = compute_log_metrics(log)
        assert m.targets_total == 3
        assert m.targets_hit == 2
        assert m.targets_missed == 1
        assert m.extra_onsets == 1
        assert m.good_hits == 1
        assert m.warn_hits == 1
        assert m.miss_severity_events == 1   # only the TARGET_MISS


# ── log_metrics_to_dict ───────────────────────────────────────────────────────

class TestLogMetricsToDict:
    _EXPECTED_KEYS = {
        "targets_total",
        "targets_hit",
        "targets_missed",
        "extra_onsets",
        "good_hits",
        "warn_hits",
        "miss_severity_events",
        "mean_signed_error_s",
        "mean_abs_error_s",
    }

    def test_round_trip_shape(self):
        m = compute_log_metrics(_make_log(_hit(0.02, 0), _miss(1)))
        assert set(log_metrics_to_dict(m).keys()) == self._EXPECTED_KEYS

    def test_dict_values_match_fields(self):
        m = compute_log_metrics(_make_log(_hit(0.02, 0), _miss(1)))
        d = log_metrics_to_dict(m)
        assert d["targets_hit"] == m.targets_hit
        assert d["targets_missed"] == m.targets_missed
        assert d["good_hits"] == m.good_hits
        assert d["mean_signed_error_s"] == pytest.approx(m.mean_signed_error_s)

    def test_none_means_serialise_as_none(self):
        m = compute_log_metrics(_make_log(_miss(0)))
        d = log_metrics_to_dict(m)
        assert d["mean_signed_error_s"] is None
        assert d["mean_abs_error_s"] is None

    def test_numeric_means_serialise_correctly(self):
        m = compute_log_metrics(_make_log(_hit(-0.04, 0), _hit(0.04, 1)))
        d = log_metrics_to_dict(m)
        assert d["mean_signed_error_s"] == pytest.approx(0.0)
        assert d["mean_abs_error_s"] == pytest.approx(0.04)
