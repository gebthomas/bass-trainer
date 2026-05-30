"""Tests for extract_replay_data() in scripts/practice_replay_viewer.py.

All tests are pure — no audio hardware, no browser, no file I/O.

Test matrix
-----------
extract_replay_data
    1.  Default metadata (no metadata set) → defaults: bpm=120, beats=4,
        count_in=2, latency_ms=0.
    2.  Custom metadata is parsed correctly (bpm, beats, count_in, latency_ms).
    3.  beat_s == 60 / bpm and count_in_s == count_in * beat_s.
    4.  target_times_s: count_in_s + i*beat_s for i in range(beats).
    5.  onset_data length always equals beats (from metadata), regardless of
        how many events are in the log.
    6.  Single TARGET_HIT → onset_s, err_ms populated; is_miss=False.
    7.  Single TARGET_MISS → is_miss=True; onset_s=None; err_ms=None.
    8.  Beat with no matching event is treated as a miss.
    9.  latency_ms == 0 → raw_onset_times_s is empty.
    10. latency_ms != 0 → raw_onset_times_s has one entry per hit.
    11. raw onset = compensated onset + latency_s (for every hit).
    12. EXTRA_ONSET events collected in extra_onset_times_s.
    13. EXTRA_ONSET events do NOT appear in onset_data.
    14. click_times_s length == count_in + beats.
    15. Count-in clicks are strictly before count_in_s;
        exercise clicks are at or after count_in_s.
    16. total_duration == count_in_s + beats*beat_s + beat_s.
    17. Mixed hits and misses produce correct onset_data entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT    = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from practice_replay_viewer import extract_replay_data
from core.session_log import (
    EXTRA_ONSET,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_log(metadata: dict | None = None, *events: SessionEvent) -> SessionLog:
    log = SessionLog(schema_version="1", started_at="2026-01-01T00:00:00+00:00")
    if metadata:
        log.metadata.update(metadata)
    log.events.extend(events)
    return log


def _hit(target_index: int, value_s: float, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=TARGET_HIT,
        target_index=target_index,
        value=value_s,
    )


def _miss(target_index: int, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=TARGET_MISS,
        target_index=target_index,
        value=None,
    )


def _extra(time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=EXTRA_ONSET,
        target_index=None,
        value=None,
    )


_META_4 = {"bpm": "120", "beats": "4", "count_in": "2", "latency_ms": "0"}


# ── Default metadata ──────────────────────────────────────────────────────────

def test_default_bpm():
    result = extract_replay_data(_make_log())
    assert result["bpm"] == pytest.approx(120.0)


def test_default_beats():
    result = extract_replay_data(_make_log())
    assert result["beats"] == 4


def test_default_count_in():
    result = extract_replay_data(_make_log())
    assert result["count_in"] == 2


def test_default_latency_ms():
    result = extract_replay_data(_make_log())
    assert result["latency_ms"] == pytest.approx(0.0)


# ── Custom metadata parsing ───────────────────────────────────────────────────

def test_metadata_bpm_parsed():
    log = _make_log({"bpm": "80", "beats": "4", "count_in": "2", "latency_ms": "0"})
    assert extract_replay_data(log)["bpm"] == pytest.approx(80.0)


def test_metadata_beats_parsed():
    log = _make_log({"bpm": "120", "beats": "8", "count_in": "2", "latency_ms": "0"})
    assert extract_replay_data(log)["beats"] == 8


def test_metadata_count_in_parsed():
    log = _make_log({"bpm": "120", "beats": "4", "count_in": "4", "latency_ms": "0"})
    assert extract_replay_data(log)["count_in"] == 4


def test_metadata_latency_ms_parsed():
    log = _make_log({"bpm": "120", "beats": "4", "count_in": "2", "latency_ms": "75.5"})
    assert extract_replay_data(log)["latency_ms"] == pytest.approx(75.5)


# ── Derived timing values ─────────────────────────────────────────────────────

def test_beat_s_derived_from_bpm():
    log = _make_log({"bpm": "60", "beats": "4", "count_in": "2", "latency_ms": "0"})
    result = extract_replay_data(log)
    assert result["beat_s"] == pytest.approx(1.0)


def test_count_in_s_derived():
    # 120 BPM → beat_s=0.5; count_in=2 → count_in_s=1.0
    log = _make_log(_META_4)
    result = extract_replay_data(log)
    assert result["count_in_s"] == pytest.approx(1.0)


def test_target_times_s():
    # 120 BPM, count_in=2 (count_in_s=1.0), beats=4
    # targets: 1.0, 1.5, 2.0, 2.5
    log = _make_log(_META_4)
    result = extract_replay_data(log)
    assert result["target_times_s"] == pytest.approx([1.0, 1.5, 2.0, 2.5])


def test_total_duration():
    # count_in_s + beats*beat_s + beat_s = 1.0 + 4*0.5 + 0.5 = 3.5
    log = _make_log(_META_4)
    result = extract_replay_data(log)
    assert result["total_duration"] == pytest.approx(3.5)


# ── onset_data length ─────────────────────────────────────────────────────────

def test_onset_data_length_equals_beats():
    log = _make_log(_META_4, _hit(0, 0.020))  # only one event, 4 beats declared
    assert len(extract_replay_data(log)["onset_data"]) == 4


def test_onset_data_length_with_no_events():
    log = _make_log(_META_4)
    assert len(extract_replay_data(log)["onset_data"]) == 4


# ── Single TARGET_HIT ─────────────────────────────────────────────────────────

def test_single_hit_is_not_miss():
    log = _make_log(_META_4, _hit(0, 0.030))
    d = extract_replay_data(log)["onset_data"][0]
    assert d["is_miss"] is False


def test_single_hit_err_ms():
    # value_s = 0.030 → err_ms = 30.0
    log = _make_log(_META_4, _hit(0, 0.030))
    d = extract_replay_data(log)["onset_data"][0]
    assert d["err_ms"] == pytest.approx(30.0)


def test_single_hit_onset_s():
    # target_times_s[0]=1.0 (120 BPM, 2 count-in), err_s=0.030 → onset=1.030
    log = _make_log(_META_4, _hit(0, 0.030))
    d = extract_replay_data(log)["onset_data"][0]
    assert d["onset_s"] == pytest.approx(1.030)


def test_single_hit_target_s():
    log = _make_log(_META_4, _hit(0, 0.030))
    d = extract_replay_data(log)["onset_data"][0]
    assert d["target_s"] == pytest.approx(1.0)


# ── Single TARGET_MISS ────────────────────────────────────────────────────────

def test_single_miss_is_miss():
    log = _make_log(_META_4, _miss(1))
    d = extract_replay_data(log)["onset_data"][1]
    assert d["is_miss"] is True


def test_single_miss_onset_s_is_none():
    log = _make_log(_META_4, _miss(1))
    d = extract_replay_data(log)["onset_data"][1]
    assert d["onset_s"] is None


def test_single_miss_err_ms_is_none():
    log = _make_log(_META_4, _miss(1))
    d = extract_replay_data(log)["onset_data"][1]
    assert d["err_ms"] is None


# ── Unrecorded beat → miss ────────────────────────────────────────────────────

def test_unrecorded_beat_treated_as_miss():
    # No events at all; all 4 beats should be misses
    log = _make_log(_META_4)
    for d in extract_replay_data(log)["onset_data"]:
        assert d["is_miss"] is True


# ── Latency: zero ─────────────────────────────────────────────────────────────

def test_zero_latency_raw_onset_times_empty():
    log = _make_log(_META_4, _hit(0, 0.030))
    assert extract_replay_data(log)["raw_onset_times_s"] == []


# ── Latency: non-zero ─────────────────────────────────────────────────────────

def test_nonzero_latency_raw_onset_populated():
    meta = {**_META_4, "latency_ms": "100"}
    log  = _make_log(meta, _hit(0, 0.030))
    raw  = extract_replay_data(log)["raw_onset_times_s"]
    assert len(raw) == 1


def test_raw_onset_equals_onset_plus_latency():
    latency_ms = 120.0
    meta = {**_META_4, "latency_ms": str(latency_ms)}
    log  = _make_log(meta, _hit(0, 0.030), _hit(1, -0.020))
    result     = extract_replay_data(log)
    offset_s   = latency_ms / 1000.0
    expected_raw = [
        d["onset_s"] + offset_s
        for d in result["onset_data"]
        if d["onset_s"] is not None
    ]
    assert result["raw_onset_times_s"] == pytest.approx(expected_raw)


def test_nonzero_latency_miss_not_in_raw():
    # Miss has no onset_s, so should not contribute a raw onset entry
    meta = {**_META_4, "latency_ms": "100"}
    log  = _make_log(meta, _miss(0), _hit(1, 0.040))
    raw  = extract_replay_data(log)["raw_onset_times_s"]
    assert len(raw) == 1  # only the hit, not the miss


# ── EXTRA_ONSET events ────────────────────────────────────────────────────────

def test_extra_onsets_collected():
    log = _make_log(_META_4, _extra(time_sec=1.3), _extra(time_sec=1.8))
    result = extract_replay_data(log)
    assert result["extra_onset_times_s"] == pytest.approx([1.3, 1.8])


def test_extra_onsets_not_in_onset_data():
    log = _make_log(_META_4, _extra(time_sec=1.3))
    onset_data = extract_replay_data(log)["onset_data"]
    # All onset_data entries should be misses (no hits recorded)
    assert all(d["is_miss"] for d in onset_data)


# ── Click times ───────────────────────────────────────────────────────────────

def test_click_times_count():
    # count_in=2, beats=4 → 6 total clicks
    log = _make_log(_META_4)
    clicks = extract_replay_data(log)["click_times_s"]
    assert len(clicks) == 6


def test_count_in_clicks_before_count_in_s():
    log    = _make_log(_META_4)
    result = extract_replay_data(log)
    ci_s   = result["count_in_s"]
    for t in result["click_times_s"]:
        if t < ci_s:
            pass  # count-in click, fine
    count_in_clicks = [t for t in result["click_times_s"] if t < ci_s]
    assert len(count_in_clicks) == result["count_in"]


def test_exercise_clicks_at_or_after_count_in_s():
    log    = _make_log(_META_4)
    result = extract_replay_data(log)
    ci_s   = result["count_in_s"]
    exercise_clicks = [t for t in result["click_times_s"] if t >= ci_s]
    assert len(exercise_clicks) == result["beats"]


# ── Mixed hits and misses ─────────────────────────────────────────────────────

def test_mixed_hits_and_misses():
    # 4 beats; 0 and 2 hit; 1 miss; 3 unrecorded (miss)
    log = _make_log(
        _META_4,
        _hit(0, 0.020),
        _miss(1),
        _hit(2, -0.015),
    )
    onset_data = extract_replay_data(log)["onset_data"]

    assert onset_data[0]["is_miss"] is False
    assert onset_data[0]["err_ms"]  == pytest.approx(20.0)

    assert onset_data[1]["is_miss"] is True
    assert onset_data[1]["onset_s"] is None

    assert onset_data[2]["is_miss"] is False
    assert onset_data[2]["err_ms"]  == pytest.approx(-15.0)

    assert onset_data[3]["is_miss"] is True  # unrecorded
