import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "realtime" / "onset_pitch_realtime.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
REAL_AUDIO = ROOT / "tests" / "real_audio" / "fretted_finger"
TARGETS = ROOT / "tests" / "targets"


@dataclass
class RealAudioCase:
    name: str
    audio: Path
    targets: Path
    min_hits: int
    max_hits: int
    min_missed: int
    max_missed: int
    min_extra: int
    max_extra: int


CASES = [
    RealAudioCase(
        name="slow_quarter_clean",
        audio=REAL_AUDIO / "slow_quarter_clean.wav",
        targets=TARGETS / "slow_quarter.json",
        min_hits=3, max_hits=3,
        min_missed=0, max_missed=0,
        min_extra=0, max_extra=0,
    ),
    RealAudioCase(
        name="slow_quarter_missed",
        audio=REAL_AUDIO / "slow_quarter_missed.wav",
        targets=TARGETS / "slow_quarter.json",
        min_hits=2, max_hits=2,
        min_missed=1, max_missed=1,
        min_extra=0, max_extra=0,
    ),
    RealAudioCase(
        name="slow_quarter_extra",
        audio=REAL_AUDIO / "slow_quarter_extra.wav",
        targets=TARGETS / "slow_quarter.json",
        min_hits=3, max_hits=3,
        min_missed=0, max_missed=0,
        min_extra=1, max_extra=99,
    ),
    RealAudioCase(
        name="slow_quarter_legato",
        audio=REAL_AUDIO / "slow_quarter_legato.wav",
        targets=TARGETS / "slow_quarter.json",
        min_hits=3, max_hits=3,
        min_missed=0, max_missed=0,
        min_extra=0, max_extra=1,
    ),
    RealAudioCase(
        name="slow_quarter_soft",
        audio=REAL_AUDIO / "slow_quarter_soft.wav",
        targets=TARGETS / "slow_quarter.json",
        min_hits=2, max_hits=3,
        min_missed=0, max_missed=1,
        min_extra=0, max_extra=1,
    ),
    RealAudioCase(
        name="pentatonic_60_clean",
        audio=REAL_AUDIO / "pentatonic_60_clean.wav",
        targets=TARGETS / "pentatonic_60.json",
        min_hits=7, max_hits=9,
        min_missed=0, max_missed=2,
        min_extra=0, max_extra=2,
    ),
]


def substitute_constant(text: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name} = {value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text


def run_case(case: RealAudioCase) -> tuple[str, str]:
    if not case.audio.exists():
        return "SKIP", f"audio file not found: {case.audio.name}"

    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Python executable not found: {VENV_PYTHON}")

    original_text = SCRIPT.read_text(encoding="utf-8")
    patched_text = original_text
    patched_text = substitute_constant(patched_text, "PROJECT_ROOT", f"Path(r'{ROOT}')")
    patched_text = substitute_constant(patched_text, "OFFLINE_MODE", "True")
    patched_text = substitute_constant(patched_text, "OFFLINE_AUDIO_FILE", f"Path(r'{case.audio}')")
    patched_text = substitute_constant(patched_text, "OFFLINE_TARGET_FILE", f"Path(r'{case.targets}')")
    patched_text = substitute_constant(patched_text, "APPLY_CALIBRATION_IN_OFFLINE_MODE", "True")
    patched_text = substitute_constant(patched_text, "ONSET_DETECTOR_MODE", '"decay_break"')

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
            return "FAIL", f"script error (rc={proc.returncode}): {proc.stderr.strip()[-200:]}"

        csv_files = sorted(temp_results.glob("session_*.csv"))
        if not csv_files:
            return "FAIL", "no result CSV produced"

        results = parse_csv(csv_files[-1])

    verdict, detail = evaluate(case, results)
    return verdict, detail


def parse_csv(csv_path: Path) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    hit_rows    = [r for r in rows if r["event_type"] == "hit"]
    missed_rows = [r for r in rows if r["event_type"] == "missed"]
    extra_rows  = [r for r in rows if r["event_type"] == "extra"]

    pitch_ok = [r["pitch_ok"].strip().lower() == "true" for r in hit_rows]
    pitch_accuracy = (sum(pitch_ok) / len(pitch_ok) * 100.0) if pitch_ok else 0.0

    return {
        "hits":           len(hit_rows),
        "missed":         len(missed_rows),
        "extra":          len(extra_rows),
        "pitch_accuracy": pitch_accuracy,
    }


def _deviation(actual: int, lo: int, hi: int) -> int:
    if actual < lo:
        return lo - actual
    if actual > hi:
        return actual - hi
    return 0


def evaluate(case: RealAudioCase, results: dict) -> tuple[str, str]:
    hits   = results["hits"]
    missed = results["missed"]
    extra  = results["extra"]
    pitch  = results["pitch_accuracy"]

    dev_hits   = _deviation(hits,   case.min_hits,   case.max_hits)
    dev_missed = _deviation(missed, case.min_missed, case.max_missed)
    dev_extra  = _deviation(extra,  case.min_extra,  case.max_extra)
    total_dev  = dev_hits + dev_missed + dev_extra

    detail = (
        f"hits={hits} missed={missed} extra={extra} pitch={pitch:.0f}%"
        f"  [expect hits {case.min_hits}–{case.max_hits},"
        f" missed {case.min_missed}–{case.max_missed},"
        f" extra {case.min_extra}–{case.max_extra}]"
    )

    if total_dev == 0:
        verdict = "PASS"
    elif total_dev == 1:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    return verdict, detail


def main() -> int:
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "SKIP": 0}

    for case in CASES:
        verdict, detail = run_case(case)
        counts[verdict] += 1
        label = f"[{verdict}]".ljust(8)
        print(f"{label} {case.name:<26}  {detail}")

    print()
    print(
        f"{counts['PASS']} PASS  "
        f"{counts['REVIEW']} REVIEW  "
        f"{counts['FAIL']} FAIL  "
        f"{counts['SKIP']} SKIP"
        f"  (of {len(CASES)} cases)"
    )

    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
