import json
from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "audio_calibration.json"


def load_input_latency(config_path=None):
    """Return input_latency_ms from config/audio_calibration.json.

    Falls back to 0 with a printed warning when the file is absent, the key is
    missing, or the file cannot be parsed.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG

    if not path.exists():
        print(f"Warning: audio calibration not found at {path} — using input_latency_ms = 0")
        return 0.0

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as ex:
        print(f"Warning: could not parse {path}: {ex} — using input_latency_ms = 0")
        return 0.0

    if "input_latency_ms" not in data:
        print(f"Warning: 'input_latency_ms' not found in {path} — using input_latency_ms = 0")
        return 0.0

    return float(data["input_latency_ms"])


def effective_target_time(target_time, input_latency_ms):
    """Return the audio-buffer time that corresponds to a target played at target_time.

    A positive input_latency_ms means the device delivers audio later than real
    time, so the note appears further into the buffer than its nominal time.
    """
    return target_time + input_latency_ms / 1000.0
