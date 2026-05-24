"""Tests for core/session_store.py — session persistence with retention."""

import os
import time
from pathlib import Path

import pytest

from core.session_log import SessionLog, load_session_log_file, session_log_to_json
from core.session_store import (
    SessionStoreConfig,
    enforce_retention,
    is_protected_log,
    list_session_logs,
    save_session_log,
    session_log_filename,
    validate_session_store_config,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_log(started_at: str = "2026-05-24T10:00:00", **metadata) -> SessionLog:
    return SessionLog(
        schema_version=1,
        started_at=started_at,
        metadata={k: str(v) for k, v in metadata.items()},
    )


def _write_at(path: Path, log: SessionLog, mtime: float) -> None:
    """Write a session log file bypassing save_session_log, with an explicit mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session_log_to_json(log), encoding="utf-8")
    os.utime(path, (mtime, mtime))


# ── validate_session_store_config ─────────────────────────────────────────────

class TestValidateSessionStoreConfig:
    def test_empty_root_dir_raises(self):
        with pytest.raises(ValueError, match="root_dir"):
            validate_session_store_config(SessionStoreConfig(root_dir=""))

    def test_whitespace_root_dir_raises(self):
        with pytest.raises(ValueError, match="root_dir"):
            validate_session_store_config(SessionStoreConfig(root_dir="   "))

    def test_max_logs_zero_raises(self):
        with pytest.raises(ValueError, match="max_logs"):
            validate_session_store_config(SessionStoreConfig(root_dir="/tmp", max_logs=0))

    def test_max_total_bytes_zero_raises(self):
        with pytest.raises(ValueError, match="max_total_bytes"):
            validate_session_store_config(SessionStoreConfig(root_dir="/tmp", max_total_bytes=0))

    def test_valid_config_passes(self):
        validate_session_store_config(SessionStoreConfig(root_dir="/tmp"))


# ── session_log_filename ──────────────────────────────────────────────────────

class TestSessionLogFilename:
    def test_always_ends_with_suffix(self):
        assert session_log_filename(_make_log()).endswith(".session.json")

    def test_colons_sanitized(self):
        name = session_log_filename(_make_log("2026-05-24T10:30:00"))
        assert ":" not in name

    def test_spaces_sanitized(self):
        name = session_log_filename(_make_log("2026 05 24"))
        assert " " not in name

    def test_slashes_sanitized(self):
        name = session_log_filename(_make_log("2026/05/24"))
        assert "/" not in name

    def test_timestamp_info_preserved(self):
        name = session_log_filename(_make_log("2026-05-24T10:00:00"))
        assert "2026" in name
        assert "05" in name
        assert "24" in name

    def test_result_is_nonempty(self):
        assert session_log_filename(_make_log()) != ".session.json"


# ── is_protected_log ──────────────────────────────────────────────────────────

class TestIsProtectedLog:
    def test_keep_true(self):
        assert is_protected_log(_make_log(keep="true"))

    def test_milestone_true(self):
        assert is_protected_log(_make_log(milestone="true"))

    def test_best_true(self):
        assert is_protected_log(_make_log(best="true"))

    def test_starred_true(self):
        assert is_protected_log(_make_log(starred="true"))

    def test_case_insensitive(self):
        assert is_protected_log(_make_log(keep="True"))
        assert is_protected_log(_make_log(keep="TRUE"))

    def test_false_value_not_protected(self):
        assert not is_protected_log(_make_log(keep="false"))

    def test_no_flags_not_protected(self):
        assert not is_protected_log(_make_log())

    def test_unrelated_metadata_not_protected(self):
        assert not is_protected_log(_make_log(exercise="walking_bass", bpm="120"))


# ── save_session_log ──────────────────────────────────────────────────────────

class TestSaveSessionLog:
    def test_creates_session_json_file(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        path = save_session_log(_make_log(), config)
        assert path.exists()
        assert path.name.endswith(".session.json")

    def test_saved_file_reloads_as_original(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        log = _make_log("2026-05-24T10:00:00")
        path = save_session_log(log, config)
        assert load_session_log_file(path) == log

    def test_date_subdirs_creates_dated_directory(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=True)
        path = save_session_log(_make_log("2026-05-24T10:00:00"), config)
        assert path.parent.name == "2026-05-24"

    def test_no_date_subdirs_saves_in_root(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        path = save_session_log(_make_log("2026-05-24T10:00:00"), config)
        assert path.parent == tmp_path

    def test_collision_avoidance_produces_distinct_paths(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False, max_logs=100)
        log = _make_log("2026-05-24T10:00:00")
        path1 = save_session_log(log, config)
        path2 = save_session_log(log, config)
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()

    def test_invalid_config_raises(self):
        config = SessionStoreConfig(root_dir="", date_subdirs=False)
        with pytest.raises(ValueError):
            save_session_log(_make_log(), config)


# ── list_session_logs ─────────────────────────────────────────────────────────

class TestListSessionLogs:
    def test_returns_session_json_files(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        save_session_log(_make_log(), config)
        paths = list_session_logs(config)
        assert len(paths) == 1
        assert all(p.name.endswith(".session.json") for p in paths)

    def test_ignores_non_session_json_files(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        (tmp_path / "exercise.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("notes")
        save_session_log(_make_log(), config)
        paths = list_session_logs(config)
        assert len(paths) == 1

    def test_ignores_symlinks(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        real_path = save_session_log(_make_log(), config)
        link_path = tmp_path / "linked.session.json"
        link_path.symlink_to(real_path)
        paths = list_session_logs(config)
        assert real_path in paths
        assert link_path not in paths

    def test_missing_root_returns_empty(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path / "nonexistent"))
        assert list_session_logs(config) == []

    def test_sorted_oldest_first(self, tmp_path):
        now = time.time()
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=False)
        older = tmp_path / "a.session.json"
        newer = tmp_path / "b.session.json"
        _write_at(older, _make_log(), now - 100)
        _write_at(newer, _make_log(), now)
        paths = list_session_logs(config)
        assert paths[0] == older
        assert paths[1] == newer

    def test_finds_files_in_subdirectories(self, tmp_path):
        config = SessionStoreConfig(root_dir=str(tmp_path), date_subdirs=True)
        path = save_session_log(_make_log("2026-05-24T10:00:00"), config)
        paths = list_session_logs(config)
        assert path in paths


# ── enforce_retention ─────────────────────────────────────────────────────────

class TestEnforceRetention:
    def test_max_logs_deletes_oldest_unprotected(self, tmp_path):
        now = time.time()
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=2, max_total_bytes=50_000_000, date_subdirs=False
        )
        oldest = tmp_path / "a.session.json"
        middle = tmp_path / "b.session.json"
        newest = tmp_path / "c.session.json"
        _write_at(oldest, _make_log(), now - 200)
        _write_at(middle, _make_log(), now - 100)
        _write_at(newest, _make_log(), now)

        deleted = enforce_retention(config)

        assert oldest in deleted
        assert not oldest.exists()
        assert middle.exists()
        assert newest.exists()

    def test_max_total_bytes_deletes_oldest_unprotected(self, tmp_path):
        now  = time.time()
        log  = _make_log()
        # Measure real file size so we can set a limit between one and two files.
        probe = tmp_path / "probe.session.json"
        probe.write_text(session_log_to_json(log))
        one_file_bytes = probe.stat().st_size
        probe.unlink()

        config = SessionStoreConfig(
            root_dir=str(tmp_path),
            max_logs=1000,
            max_total_bytes=one_file_bytes + 1,  # one file fits; two do not
            date_subdirs=False,
        )
        older = tmp_path / "older.session.json"
        newer = tmp_path / "newer.session.json"
        _write_at(older, log, now - 100)
        _write_at(newer, log, now)

        deleted = enforce_retention(config)

        assert older in deleted
        assert not older.exists()
        assert newer.exists()

    def test_protected_keep_not_deleted(self, tmp_path):
        now = time.time()
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=50_000_000, date_subdirs=False
        )
        protected   = tmp_path / "protected.session.json"
        unprotected = tmp_path / "unprotected.session.json"
        _write_at(protected,   _make_log(keep="true"), now - 100)
        _write_at(unprotected, _make_log(),            now)

        deleted = enforce_retention(config)

        assert protected.exists()
        assert unprotected in deleted

    def test_protected_milestone_not_deleted(self, tmp_path):
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        path = tmp_path / "milestone.session.json"
        _write_at(path, _make_log(milestone="true"), time.time())
        assert path not in enforce_retention(config)
        assert path.exists()

    def test_protected_best_not_deleted(self, tmp_path):
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        path = tmp_path / "best.session.json"
        _write_at(path, _make_log(best="true"), time.time())
        assert path not in enforce_retention(config)
        assert path.exists()

    def test_protected_starred_not_deleted(self, tmp_path):
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        path = tmp_path / "starred.session.json"
        _write_at(path, _make_log(starred="true"), time.time())
        assert path not in enforce_retention(config)
        assert path.exists()

    def test_all_protected_nothing_deleted(self, tmp_path):
        now = time.time()
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        p1 = tmp_path / "p1.session.json"
        p2 = tmp_path / "p2.session.json"
        _write_at(p1, _make_log(keep="true"), now - 100)
        _write_at(p2, _make_log(keep="true"), now)

        deleted = enforce_retention(config)

        assert deleted == []
        assert p1.exists()
        assert p2.exists()

    def test_non_session_files_not_deleted(self, tmp_path):
        now = time.time()
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        session_file = tmp_path / "a.session.json"
        other_json   = tmp_path / "exercise.json"
        other_txt    = tmp_path / "notes.txt"
        _write_at(session_file, _make_log(), now - 100)
        other_json.write_text('{"exercise": true}')
        other_txt.write_text("notes")

        enforce_retention(config)

        assert other_json.exists()
        assert other_txt.exists()

    def test_symlinks_not_deleted(self, tmp_path):
        now = time.time()
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        real_file = tmp_path / "real.session.json"
        link_file = tmp_path / "link.session.json"
        _write_at(real_file, _make_log(), now - 100)
        link_file.symlink_to(real_file)

        deleted = enforce_retention(config)

        # enforce_retention may delete real_file (it's a candidate), but must never
        # delete the symlink entry itself.  is_symlink() returns True even for a
        # broken symlink (target gone), so it proves the entry was not unlinked.
        assert link_file not in deleted
        assert link_file.is_symlink()

    def test_retention_never_deletes_outside_root(self, tmp_path):
        store_root = tmp_path / "store"
        outside    = tmp_path / "outside"
        outside.mkdir()
        config = SessionStoreConfig(
            root_dir=str(store_root), max_logs=1, max_total_bytes=1, date_subdirs=False
        )
        outside_file = outside / "outside.session.json"
        _write_at(outside_file, _make_log(), time.time())

        enforce_retention(config)

        assert outside_file.exists()

    def test_within_limits_deletes_nothing(self, tmp_path):
        config = SessionStoreConfig(
            root_dir=str(tmp_path), max_logs=100, max_total_bytes=50_000_000, date_subdirs=False
        )
        save_session_log(_make_log(), config)
        assert enforce_retention(config) == []
