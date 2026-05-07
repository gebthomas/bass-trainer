import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio_calibration import load_input_latency, effective_target_time


# ── load_input_latency ────────────────────────────────────────────────────────

def test_load_valid_config():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "audio_calibration.json"
        cfg.write_text(json.dumps({"input_latency_ms": 184.375}))
        assert load_input_latency(cfg) == 184.375


def test_load_zero_is_valid():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "audio_calibration.json"
        cfg.write_text(json.dumps({"input_latency_ms": 0}))
        assert load_input_latency(cfg) == 0.0


def test_load_negative_is_valid():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "audio_calibration.json"
        cfg.write_text(json.dumps({"input_latency_ms": -50.0}))
        assert load_input_latency(cfg) == -50.0


def test_load_missing_file_returns_zero():
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "does_not_exist.json"
        result = load_input_latency(missing)
        assert result == 0.0


def test_load_missing_key_returns_zero():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "audio_calibration.json"
        cfg.write_text(json.dumps({"timing_offset_ms": -100}))  # wrong key
        result = load_input_latency(cfg)
        assert result == 0.0


def test_load_malformed_json_returns_zero():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "audio_calibration.json"
        cfg.write_text("{ not valid json !!!")
        result = load_input_latency(cfg)
        assert result == 0.0


def test_load_integer_value_returned_as_float():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "audio_calibration.json"
        cfg.write_text(json.dumps({"input_latency_ms": 100}))
        result = load_input_latency(cfg)
        assert isinstance(result, float)
        assert result == 100.0


def test_load_uses_default_path_when_no_arg():
    # Smoke test only: confirms the call completes and returns a float.
    # The default config/audio_calibration.json may or may not exist.
    result = load_input_latency()
    assert isinstance(result, float)


# ── effective_target_time ─────────────────────────────────────────────────────

def test_effective_time_positive_latency():
    # 100 ms latency shifts analysis window 0.1 s forward
    assert abs(effective_target_time(2.0, 100.0) - 2.1) < 1e-9


def test_effective_time_zero_latency():
    assert effective_target_time(2.0, 0.0) == 2.0


def test_effective_time_negative_latency():
    # Negative latency shifts window back
    assert abs(effective_target_time(2.0, -50.0) - 1.95) < 1e-9


def test_effective_time_fractional_ms():
    assert abs(effective_target_time(1.0, 184.375) - 1.184375) < 1e-9


def test_effective_time_zero_target():
    assert abs(effective_target_time(0.0, 200.0) - 0.2) < 1e-9


# ── runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    test_load_valid_config()
    test_load_zero_is_valid()
    test_load_negative_is_valid()
    test_load_missing_file_returns_zero()
    test_load_missing_key_returns_zero()
    test_load_malformed_json_returns_zero()
    test_load_integer_value_returned_as_float()
    test_load_uses_default_path_when_no_arg()
    test_effective_time_positive_latency()
    test_effective_time_zero_latency()
    test_effective_time_negative_latency()
    test_effective_time_fractional_ms()
    test_effective_time_zero_target()


if __name__ == "__main__":
    run_all_tests()
    print("All audio_calibration tests passed.")
