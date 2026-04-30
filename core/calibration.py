import json
import numpy as np


def load_calibration(config_path, default_offset_ms):
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            return config.get("timing_offset_ms", default_offset_ms)
        except Exception as ex:
            print(f"Warning: failed to load calibration: {ex}")
    return default_offset_ms


def save_calibration(offset_ms, config_path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({"timing_offset_ms": offset_ms}, handle, indent=2)


def run_calibration_summary(calibration_errors, save_calibration_fn, calibration_config_path):
    if not calibration_errors:
        print("No calibration notes were recorded.")
        return

    average_raw_error_ms = float(np.mean(calibration_errors))
    timing_offset_ms = -average_raw_error_ms
    save_calibration_fn(timing_offset_ms, calibration_config_path)
    print("\nCalibration complete")
    print(f"Average raw timing error: {average_raw_error_ms:+.1f} ms")
    print(f"Saved TIMING_OFFSET_MS = {timing_offset_ms:+.1f} ms to {calibration_config_path}")
