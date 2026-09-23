import time

import pytest

from app.store import db


def _job(worker_id: str, *, heartbeat_at: float, jtype: str = "expert") -> str:
    with db.connect() as conn:
        row = db.create_job(conn, jtype=jtype, payload={})
        conn.execute(
            "UPDATE jobs SET status='running', worker_id=?, heartbeat_at=? WHERE id=?",
            (worker_id, heartbeat_at, row["id"]))
        return row["id"]


def test_exact_worker_exit_interrupts_only_its_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path))
    db.init_db()
    dead = _job("dead-worker", heartbeat_at=time.time())
    live = _job("live-worker", heartbeat_at=time.time())

    assert db.interrupt_worker_jobs("dead-worker", reason="worker crashed") == 1
    with db.connect() as conn:
        dead_row = conn.execute("SELECT status,error,finished_at FROM jobs WHERE id=?", (dead,)).fetchone()
        live_row = conn.execute("SELECT status FROM jobs WHERE id=?", (live,)).fetchone()
    assert dead_row["status"] == "interrupted"
    assert dead_row["error"] == "worker crashed"
    assert dead_row["finished_at"] is not None
    assert live_row["status"] == "running"


def test_periodic_recovery_repairs_only_stale_heartbeats(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path))
    db.init_db()
    stale = _job("old-worker", heartbeat_at=time.time() - 120)
    fresh = _job("other-worker", heartbeat_at=time.time())

    assert db.recover_stale_jobs("replacement", stale_seconds=60) == 1
    with db.connect() as conn:
        assert conn.execute("SELECT status FROM jobs WHERE id=?", (stale,)).fetchone()[0] == "interrupted"
        assert conn.execute("SELECT status FROM jobs WHERE id=?", (fresh,)).fetchone()[0] == "running"


def test_delete_journal_mode_avoids_wal_shared_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TENNIS_SQLITE_JOURNAL_MODE", "delete")
    db.init_db()

    with db.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not (tmp_path / "app.sqlite-shm").exists()


def test_invalid_journal_mode_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TENNIS_SQLITE_JOURNAL_MODE", "unsafe-value")
    with pytest.raises(RuntimeError, match="仅支持 WAL 或 DELETE"):
        db.init_db()
