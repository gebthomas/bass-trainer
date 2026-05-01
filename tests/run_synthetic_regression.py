import csv
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "realtime" / "onset_pitch_realtime.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


@dataclass
class RegressionCase:
    name: str
    audio: Path
    targets: Path
    expected_hits: int
    expected_missed: int
    expected_extra: int
    expected_pitch_accuracy: float
    expected_mean_raw_timing_error_range: tuple[float, float]


CASES = [
    RegressionCase(
        name="slow_perfect",
        audio=ROOT / "tests" / "audio" / "slow_perfect.wav",
        targets=ROOT / "tests" / "targets" / "slow_quarter.json",
        expected_hits=3,
        expected_missed=0,
        expected_extra=0,
        expected_pitch_accuracy=100.0,
        expected_mean_raw_timing_error_range=(-20.0, 20.0),
    ),
    RegressionCase(
        name="slow_late_150ms",
        audio=ROOT / "tests" / "audio" / "slow_late_150ms.wav",
        targets=ROOT / "tests" / "targets" / "slow_quarter.json",
        expected_hits=3,
        expected_missed=0,
        expected_extra=0,
        expected_pitch_accuracy=100.0,
        expected_mean_raw_timing_error_range=(120.0, 180.0),
    ),
    RegressionCase(
        name="slow_early_100ms",
        audio=ROOT / "tests" / "audio" / "slow_early_100ms.wav",
        targets=ROOT / "tests" / "targets" / "slow_quarter.json",
        expected_hits=3,
        expected_missed=0,
        expected_extra=0,
        expected_pitch_accuracy=100.0,
        expected_mean_raw_timing_error_range=(-140.0, -60.0),
    ),
    RegressionCase(
        name="slow_missed_first",
        audio=ROOT / "tests" / "audio" / "slow_missed_first.wav",
        targets=ROOT / "tests" / "targets" / "slow_quarter.json",
        expected_hits=2,
        expected_missed=1,
        expected_extra=0,
        expected_pitch_accuracy=100.0,
        expected_mean_raw_timing_error_range=(-20.0, 20.0),
    ),
    RegressionCase(
        name="slow_extra_between",
        audio=ROOT / "tests" / "audio" / "slow_extra_between.wav",
        targets=ROOT / "tests" / "targets" / "slow_quarter.json",
        expected_hits=3,
        expected_missed=0,
        expected_extra=1,
        expected_pitch_accuracy=100.0,
        expected_mean_raw_timing_error_range=(-30.0, 100.0),
    ),
    RegressionCase(
				name="pentatonic_perfect",
				audio=ROOT / "tests" / "audio" / "pentatonic_perfect.wav",
				targets=ROOT / "tests" / "targets" / "pentatonic_60.json",
				expected_hits=9,
				expected_missed=0,
				expected_extra=0,
				expected_pitch_accuracy=100.0,
				expected_mean_raw_timing_error_range=(-20.0, 20.0),
		),
		RegressionCase(
				name="pentatonic_late_100ms",
				audio=ROOT / "tests" / "audio" / "pentatonic_late_100ms.wav",
				targets=ROOT / "tests" / "targets" / "pentatonic_60.json",
				expected_hits=9,
				expected_missed=0,
				expected_extra=0,
				expected_pitch_accuracy=100.0,
				expected_mean_raw_timing_error_range=(70.0, 130.0),
		),
]


def substitute_constant(text: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name} = {value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text


def run_case(case: RegressionCase) -> tuple[bool, str]:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Python executable not found: {VENV_PYTHON}")

    original_text = SCRIPT.read_text(encoding="utf-8")
    patched_text = original_text
    patched_text = substitute_constant(patched_text, "PROJECT_ROOT", f"Path(r'{ROOT}')")
    patched_text = substitute_constant(patched_text, "OFFLINE_MODE", "True")
    patched_text = substitute_constant(patched_text, "OFFLINE_AUDIO_FILE", f"Path(r'{case.audio}')")
    patched_text = substitute_constant(patched_text, "OFFLINE_TARGET_FILE", f"Path(r'{case.targets}')")
    patched_text = substitute_constant(patched_text, "APPLY_CALIBRATION_IN_OFFLINE_MODE", "False")

    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        temp_script = temp_root / "onset_pitch_realtime.py"
        temp_results = temp_root / "results"

        patched_text = substitute_constant(
            patched_text,
            "RESULTS_DIR",
            f"Path(r'{temp_results}')",
        )

        temp_script.write_text(patched_text, encoding="utf-8")

        proc = subprocess.run(
            [str(VENV_PYTHON), str(temp_script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            return False, f"Script failed with return code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        csv_files = sorted(temp_results.glob("session_*.csv"))
        if not csv_files:
            return False, f"No result CSV produced. STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        results = parse_csv(csv_files[-1])
        passed, message = evaluate_results(case, results)
        status = "PASS" if passed else "FAIL"
        return passed, f"{status}: {case.name} - {message}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def parse_csv(csv_path: Path) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    hit_rows = [r for r in rows if r["event_type"] == "hit"]
    missed_rows = [r for r in rows if r["event_type"] == "missed"]
    extra_rows = [r for r in rows if r["event_type"] == "extra"]

    pitch_ok_values = [r["pitch_ok"].strip().lower() == "true" for r in hit_rows]
    raw_errors = [float(r["raw_timing_error_ms"]) for r in hit_rows if r["raw_timing_error_ms"].strip()]

    mean_raw_error = sum(raw_errors) / len(raw_errors) if raw_errors else None
    pitch_accuracy = (sum(pitch_ok_values) / len(pitch_ok_values) * 100.0) if pitch_ok_values else 0.0

    return {
        "hits": len(hit_rows),
        "missed": len(missed_rows),
        "extra": len(extra_rows),
        "pitch_accuracy": pitch_accuracy,
        "mean_raw_error": mean_raw_error,
    }


def evaluate_results(case: RegressionCase, results: dict) -> tuple[bool, str]:
    mismatches = []

    if results["hits"] != case.expected_hits:
        mismatches.append(f"hits={results['hits']} (expected {case.expected_hits})")
    if results["missed"] != case.expected_missed:
        mismatches.append(f"missed={results['missed']} (expected {case.expected_missed})")
    if results["extra"] != case.expected_extra:
        mismatches.append(f"extra={results['extra']} (expected {case.expected_extra})")

    if abs(results["pitch_accuracy"] - case.expected_pitch_accuracy) > 1e-6:
        mismatches.append(
            f"pitch_accuracy={results['pitch_accuracy']:.1f}% (expected {case.expected_pitch_accuracy:.1f}%)"
        )

    mean_raw = results["mean_raw_error"]
    if mean_raw is None:
        mismatches.append("mean_raw_error=N/A")
    elif not (case.expected_mean_raw_timing_error_range[0] <= mean_raw <= case.expected_mean_raw_timing_error_range[1]):
        mismatches.append(
            f"mean_raw_error={mean_raw:.1f} ms (expected range {case.expected_mean_raw_timing_error_range})"
        )

    if mismatches:
        return False, "; ".join(mismatches)
    return True, (
        f"hits={results['hits']} missed={results['missed']} extra={results['extra']} "
        f"pitch_accuracy={results['pitch_accuracy']:.1f}% mean_raw_error={results['mean_raw_error']:.1f} ms"
    )


def main() -> int:
    passed_count = 0
    for case in CASES:
        passed, message = run_case(case)
        print(message)
        if passed:
            passed_count += 1

    summary = f"{passed_count}/{len(CASES)} regression cases passed."
    print(summary)
    return 0 if passed_count == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
