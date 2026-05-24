"""Session persistence with protected-log retention.

Standard library only — no audio dependencies.

Saves SessionLog files into a managed directory and enforces configurable
limits (max_logs, max_total_bytes) by deleting oldest unprotected logs first.
Protected logs (metadata flags keep/milestone/best/starred = "true") are never
deleted by retention regardless of limits.

Safety invariants honoured unconditionally:
- Never deletes outside root_dir.
- Never follows or deletes symlinks.
- Only ever deletes files ending in .session.json.
- Treats any unreadable log as protected.

Typical usage
-------------
    from core.session_store import SessionStoreConfig, save_session_log

    config = SessionStoreConfig(root_dir="sessions/")
    path = save_session_log(log, config)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.session_log import (
    SessionLog,
    load_session_log_file,
    session_log_to_json,
    validate_session_log,
)

_SUFFIX = ".session.json"
_PROTECTION_KEYS = frozenset({"keep", "milestone", "best", "starred"})


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SessionStoreConfig:
    """Configuration for the session store.

    Parameters
    ----------
    root_dir
        Root directory under which session files are written.  Must be
        non-empty.  The directory is created on first save if it does not
        exist.
    max_logs
        Maximum number of ``.session.json`` files allowed under root_dir.
        Must be >= 1.
    max_total_bytes
        Maximum combined size of all ``.session.json`` files in bytes.
        Must be >= 1.
    date_subdirs
        When True, each session is written into a ``YYYY-MM-DD`` subdirectory
        derived from ``log.started_at``.  When False, all files go directly
        into root_dir.
    """

    root_dir:        str
    max_logs:        int  = 200
    max_total_bytes: int  = 50_000_000
    date_subdirs:    bool = True


def validate_session_store_config(config: SessionStoreConfig) -> None:
    """Raise ValueError if config is invalid."""
    if not config.root_dir or not config.root_dir.strip():
        raise ValueError("root_dir must be non-empty")
    if config.max_logs < 1:
        raise ValueError(f"max_logs must be >= 1, got {config.max_logs}")
    if config.max_total_bytes < 1:
        raise ValueError(f"max_total_bytes must be >= 1, got {config.max_total_bytes}")


# ── Filename helpers ──────────────────────────────────────────────────────────

def session_log_filename(log: SessionLog) -> str:
    """Return a filesystem-safe filename for *log*, always ending in .session.json.

    Uses ``log.started_at`` as the base.  Any character that is not
    alphanumeric, a hyphen, or an underscore is replaced with ``_``;
    consecutive underscores are collapsed to one.
    """
    base = log.started_at or "session"
    base = re.sub(r"[^\w-]", "_", base)
    base = re.sub(r"_+", "_", base).strip("_") or "session"
    return base + _SUFFIX


def _unique_dest(save_dir: Path, base_name: str) -> Path:
    """Return a path in save_dir that does not yet exist, adding _N if needed."""
    dest = save_dir / base_name
    if not dest.exists() and not dest.is_symlink():
        return dest
    stem = base_name[: -len(_SUFFIX)]
    counter = 1
    while True:
        candidate = save_dir / f"{stem}_{counter}{_SUFFIX}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        counter += 1


def _extract_date(started_at: str) -> str:
    """Return YYYY-MM-DD from an ISO 8601 timestamp, or 'unknown'."""
    if started_at and len(started_at) >= 10 and re.match(r"\d{4}-\d{2}-\d{2}", started_at):
        return started_at[:10]
    return "unknown"


# ── Protection ────────────────────────────────────────────────────────────────

def is_protected_log(log: SessionLog) -> bool:
    """Return True if *log* carries any protection flag set to ``"true"`` (case-insensitive).

    Protection flags are metadata keys: ``keep``, ``milestone``, ``best``,
    ``starred``.
    """
    for key in _PROTECTION_KEYS:
        if log.metadata.get(key, "").lower() == "true":
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def save_session_log(log: SessionLog, config: SessionStoreConfig) -> Path:
    """Validate, persist, and enforce retention for *log*.

    Parameters
    ----------
    log
        A valid SessionLog to save.
    config
        Store configuration.

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    ValueError
        If config or log validation fails.
    """
    validate_session_store_config(config)
    validate_session_log(log)

    root = Path(config.root_dir)
    save_dir = root / _extract_date(log.started_at) if config.date_subdirs else root
    save_dir.mkdir(parents=True, exist_ok=True)

    dest = _unique_dest(save_dir, session_log_filename(log))
    dest.write_text(session_log_to_json(log), encoding="utf-8")

    enforce_retention(config)
    return dest


def list_session_logs(config: SessionStoreConfig) -> list[Path]:
    """Return all ``.session.json`` files under root_dir, oldest mtime first.

    Symlinks are excluded.  Returns an empty list if root_dir does not exist.
    """
    root = Path(config.root_dir)
    if not root.exists() or root.is_symlink() or not root.is_dir():
        return []

    paths = [p for p in root.rglob(f"*{_SUFFIX}") if not p.is_symlink()]
    paths.sort(key=lambda p: p.stat().st_mtime)
    return paths


def enforce_retention(config: SessionStoreConfig) -> list[Path]:
    """Delete oldest unprotected logs until within configured limits.

    Iterates candidates oldest-first.  For each candidate, loads the log to
    check protection flags; skips it if protected or unreadable.  Stops as
    soon as both ``max_logs`` and ``max_total_bytes`` are satisfied.

    Safety guarantees (unconditional):
    - Never deletes outside root_dir.
    - Never deletes symlinks.
    - Never deletes files not ending in .session.json.
    - Never deletes a log whose metadata carries a protection flag.
    - Treats any log that cannot be loaded as protected.

    Returns
    -------
    list[Path]
        Paths that were deleted, in deletion order.
    """
    validate_session_store_config(config)
    root = Path(config.root_dir).resolve()

    candidates = list_session_logs(config)

    sizes: dict[Path, int] = {}
    for p in candidates:
        try:
            sizes[p] = p.stat().st_size
        except OSError:
            sizes[p] = 0

    total_count = len(candidates)
    total_bytes = sum(sizes.values())
    deleted: list[Path] = []

    for path in candidates:  # oldest first
        if total_count <= config.max_logs and total_bytes <= config.max_total_bytes:
            break

        # Belt-and-suspenders: verify containment (list_session_logs already scopes to root)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not _is_under_root(resolved, root):
            continue

        if path.is_symlink():
            continue

        if not path.name.endswith(_SUFFIX):
            continue

        try:
            log = load_session_log_file(path)
            if is_protected_log(log):
                continue
        except Exception:
            continue  # unreadable → treat as protected

        try:
            path.unlink()
        except OSError:
            continue

        deleted.append(path)
        total_count -= 1
        total_bytes -= sizes.get(path, 0)

    return deleted


# ── Private helpers ───────────────────────────────────────────────────────────

def _is_under_root(resolved_path: Path, resolved_root: Path) -> bool:
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False
