from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path("/Users/gthomas/Documents/Audio Software/bass-trainer")
LIBRARY = PROJECT / "music-library"
WORK = PROJECT / "_demucs_work"

DRY_RUN = False
SKIP_EXISTING = True

def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd))
    if not DRY_RUN:
        subprocess.run(cmd, check=True)

def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        print(f"WARNING: missing {src}")
        return
    print(f"COPY: {src.name} -> {dst}")
    if not DRY_RUN:
        shutil.copy2(src, dst)

song_dirs = sorted(p for p in LIBRARY.iterdir() if p.is_dir())

for song_dir in song_dirs:
    master = song_dir / "master.wav"
    if not master.exists():
        continue

    wanted = [
        song_dir / "bass.wav",
        song_dir / "drums.wav",
        song_dir / "vocals.wav",
        song_dir / "other.wav",
        song_dir / "no_bass.wav",
    ]

    if SKIP_EXISTING and all(p.exists() for p in wanted):
        print(f"SKIP complete: {song_dir.name}")
        continue

    print(f"\n=== {song_dir.name} ===")

    song_work = WORK / song_dir.name
    full_out = song_work / "full"
    nobass_out = song_work / "nobass"

    # Full 4-stem separation: bass, drums, vocals, other
    run([
        sys.executable, "-m", "demucs",
        "--out", str(full_out),
        str(master),
    ])

    full_stems = full_out / "htdemucs" / "master"

    copy_if_exists(full_stems / "bass.wav", song_dir / "bass.wav")
    copy_if_exists(full_stems / "drums.wav", song_dir / "drums.wav")
    copy_if_exists(full_stems / "vocals.wav", song_dir / "vocals.wav")
    copy_if_exists(full_stems / "other.wav", song_dir / "other.wav")

    # Separate again in two-stem mode to get no_bass.wav directly.
    run([
        sys.executable, "-m", "demucs",
        "--two-stems", "bass",
        "--out", str(nobass_out),
        str(master),
    ])

    nobass_stems = nobass_out / "htdemucs" / "master"

    copy_if_exists(nobass_stems / "no_bass.wav", song_dir / "no_bass.wav")

print("\nDone.")