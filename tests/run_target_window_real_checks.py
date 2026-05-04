import csv
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "realtime" / "onset_pitch_realtime.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


@dataclass
class TWCase:
    name: str
    audio: Path
    targets: Path


CASES = [
    TWCase(
        name="fretless_pentatonic_clean",
        audio=ROOT / "tests" / "real_audio" / "fretless_finger" / "pentatonic_60_fretless_clean.wav",
        targets=ROOT / "tests" / "targets" / "pentatonic_60_twopass.json",
    ),
    TWCase(
        name="fretless_pentatonic_expressive",
        audio=ROOT / "tests" / "real_audio" / "fretless_finger" / "pentatonic_60_fretless_expressive.wav",
        targets=ROOT / "tests" / "targets" / "pentatonic_60_twopass.json",
    ),
    TWCase(
        name="fretless_pentatonic_legato",
        audio=ROOT / "tests" / "real_audio" / "fretless_finger" / "pentatonic_60_fretless_legato.wav",
        targets=ROOT / "tests" / "targets" / "pentatonic_60_twopass.json",
    ),
    TWCase(
        name="fretless_repeat_A1",
        audio=ROOT / "tests" / "real_audio" / "fretless_finger" / "repeated_60_A1_fretless.wav",
        targets=ROOT / "tests" / "targets" / "repeat_A1.json",
    ),
]


def substitute_constant(text: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name} = {value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text


def run_case(case: TWCase) -> dict:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Python executable not found: {VENV_PYTHON}")
    if not case.audio.exists():
        return {"error": f"Audio file not found: {case.audio}"}

    original_text = SCRIPT.read_text(encoding="utf-8")
    patched = original_text
    patched = substitute_constant(patched, "PROJECT_ROOT", f"Path(r'{ROOT}')")
    patched = substitute_constant(patched, "TARGET_WINDOW_MODE", "True")
    patched = substitute_constant(patched, "OFFLINE_MODE", "True")
    patched = substitute_constant(patched, "OFFLINE_AUDIO_FILE", f"Path(r'{case.audio}')")
    patched = substitute_constant(patched, "OFFLINE_TARGET_FILE", f"Path(r'{case.targets}')")
    patched = substitute_constant(patched, "APPLY_CALIBRATION_IN_OFFLINE_MODE", "False")

    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        temp_script = temp_root / "onset_pitch_realtime.py"
        temp_results = temp_root / "results"

        patched = substitute_constant(patched, "RESULTS_DIR", f"Path(r'{temp_results}')")
        temp_script.write_text(patched, encoding="utf-8")

        proc = subprocess.run(
            [str(VENV_PYTHON), str(temp_script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if proc.returncode != 0:
            return {
                "error": f"Script exited {proc.returncode}",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        csv_files = sorted(temp_results.glob("session_*.csv"))
        if not csv_files:
            return {
                "error": "No result CSV produced",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        return parse_csv(csv_files[-1])


def _float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_csv(csv_path: Path) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    hit_rows    = [r for r in rows if r["event_type"] == "hit"]
    missed_rows = [r for r in rows if r["event_type"] == "missed"]

    strong = sum(1 for r in hit_rows if r.get("target_window_status") == "strong")
    weak   = sum(1 for r in hit_rows if r.get("target_window_status") == "weak")

    corrected_errors = [v for r in hit_rows if (v := _float(r["corrected_timing_error_ms"])) is not None]
    cents_errors     = [v for r in hit_rows if (v := _float(r["cents_error"])) is not None]
    stabilities      = [v for r in hit_rows if (v := _float(r["pitch_stability_cents"])) is not None]
    offsets          = [v for r in rows     if (v := _float(r.get("adaptive_offset_ms", ""))) is not None]

    return {
        "hits":             len(hit_rows),
        "missed":           len(missed_rows),
        "strong":           strong,
        "weak":             weak,
        "median_offset_ms": statistics.median(offsets) if offsets else None,
        "mean_abs_corrected_ms": (
            sum(abs(e) for e in corrected_errors) / len(corrected_errors)
            if corrected_errors else None
        ),
        "median_cents_error":   statistics.median(cents_errors)  if cents_errors  else None,
        "median_stability_cents": statistics.median(stabilities) if stabilities   else None,
    }


def verdict(missed: int) -> str:
    if missed == 0:
        return "PASS"
    if missed <= 2:
        return "REVIEW"
    return "FAIL"


def fmt(val, spec) -> str:
    return format(val, spec) if val is not None else "N/A"


def print_case_result(case: TWCase, result: dict) -> str:
    if "error" in result:
        label = "ERROR"
        print(f"{label}: {case.name}")
        print(f"  {result['error']}")
        if "stdout" in result:
            print(f"  STDOUT: {result['stdout'][-400:]}")
        if "stderr" in result:
            print(f"  STDERR: {result['stderr'][-400:]}")
        return label

    v = verdict(result["missed"])
    total = result["hits"] + result["missed"]
    print(
        f"{v}: {case.name}"
        f"  hits={result['hits']}/{total}"
        f"  strong={result['strong']}  weak={result['weak']}  missed={result['missed']}"
    )
    print(
        f"     offset={fmt(result['median_offset_ms'], '+.0f')} ms"
        f"  |adj timing|={fmt(result['mean_abs_corrected_ms'], '.1f')} ms"
        f"  cents={fmt(result['median_cents_error'], '+.1f')}"
        f"  stability={fmt(result['median_stability_cents'], '.1f')} c"
    )
    return v


def main() -> int:
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "ERROR": 0}
    for case in CASES:
        result = run_case(case)
        v = print_case_result(case, result)
        counts[v] = counts.get(v, 0) + 1

    print(
        f"\n{counts['PASS']} PASS  {counts['REVIEW']} REVIEW"
        f"  {counts['FAIL']} FAIL  {counts['ERROR']} ERROR"
        f"  (of {len(CASES)} cases)"
    )
    return 0 if counts["FAIL"] == 0 and counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
