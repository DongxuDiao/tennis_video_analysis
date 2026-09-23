"""SQLite 存储：WAL、schema、原子任务领取、崩溃恢复。"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS project_trash (project_id TEXT PRIMARY KEY, deleted_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  original_filename TEXT NOT NULL,
  original_path TEXT NOT NULL,
  video_hash TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  probe_json TEXT,
  proxy_status TEXT NOT NULL DEFAULT 'pending',   -- pending|processing|ready|failed
  proxy_error TEXT,
  width INTEGER, height INTEGER, duration_ms INTEGER,
  nominal_fps REAL, rotation INTEGER DEFAULT 0, has_audio INTEGER DEFAULT 0,
  frame_count INTEGER
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,              -- import|analysis|export|model_download
  status TEXT NOT NULL,            -- queued|running|succeeded|failed|cancelled|interrupted
  stage TEXT,
  progress_json TEXT,              -- {frames_done, frames_total, note}
  project_id TEXT, run_id TEXT, export_id TEXT,
  payload_json TEXT,
  result_json TEXT, error TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  client_request_id TEXT,
  worker_id TEXT, heartbeat_at REAL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL,
  started_at REAL, finished_at REAL,
  UNIQUE(client_request_id)
);
CREATE TRIGGER IF NOT EXISTS block_trashed_job BEFORE INSERT ON jobs
WHEN EXISTS (SELECT 1 FROM project_trash WHERE project_id=NEW.project_id
OR project_id=(SELECT project_id FROM runs WHERE id=NEW.run_id)
OR project_id=(SELECT r.project_id FROM exports e JOIN runs r ON e.run_id=r.id WHERE e.id=NEW.export_id))
BEGIN SELECT RAISE(ABORT, '项目已在回收站，请先恢复'); END;
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS calibrations (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL, revision INTEGER NOT NULL,
  data_json TEXT NOT NULL, created_at REAL NOT NULL,
  UNIQUE(project_id, revision)
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
  parent_run_id TEXT, mode TEXT NOT NULL DEFAULT 'automatic',  -- automatic|assisted
  status TEXT NOT NULL,           -- queued|running|succeeded|failed|cancelled|interrupted
  config_json TEXT NOT NULL, effective_config_json TEXT,
  model_hash TEXT, code_version TEXT, calibration_revision INTEGER,
  segment_start_ms INTEGER NOT NULL, segment_end_ms INTEGER NOT NULL,
  cache_key TEXT, run_dir TEXT, reuses_frames_from TEXT,
  performance_json TEXT, summary_json TEXT,
  created_at REAL NOT NULL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE TABLE IF NOT EXISTS corrections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL, revision INTEGER NOT NULL,
  op_json TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corr_run ON corrections(run_id, revision);
CREATE TABLE IF NOT EXISTS ground_truth_sets (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
  name TEXT NOT NULL, version INTEGER NOT NULL, locked INTEGER NOT NULL DEFAULT 0,
  data_json TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, gt_set_id TEXT NOT NULL,
  params_json TEXT NOT NULL, result_json TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS exports (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
  view TEXT NOT NULL, correction_revision INTEGER NOT NULL DEFAULT 0,
  config_json TEXT, status TEXT NOT NULL DEFAULT 'queued',
  dir TEXT, artifacts_json TEXT, error TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_connections (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,              -- glm
  account_kind TEXT NOT NULL,          -- api | coding_plan
  protocol TEXT NOT NULL,              -- openai_compatible | anthropic | mcp
  display_name TEXT,
  model_id TEXT,                       -- 探测成功后冻结
  secret_ref TEXT NOT NULL,            -- Keychain 服务引用；DB 不存 Key
  capabilities_json TEXT NOT NULL,
  eligibility_status TEXT NOT NULL,    -- confirmed | pending | rejected
  eligibility_evidence_ref TEXT,
  enabled INTEGER NOT NULL DEFAULT 0,
  consent_version TEXT,
  consented_at REAL,
  config_revision INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS enhancement_requests (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL, connection_id TEXT NOT NULL,
  config_revision INTEGER NOT NULL,
  evidence_hash TEXT NOT NULL, skill_hash TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  round INTEGER NOT NULL,
  status TEXT NOT NULL,                -- pending|sent|succeeded|failed|timeout_unknown|cancelled|deduped
  attempt_count INTEGER NOT NULL DEFAULT 0,
  provider_request_id TEXT,
  started_at REAL, finished_at REAL,
  usage_json TEXT, error_code TEXT, response_artifact TEXT,
  created_at REAL NOT NULL,
  UNIQUE(job_id, evidence_hash, connection_id, config_revision, prompt_version, round)
);
CREATE INDEX IF NOT EXISTS idx_enh_job ON enhancement_requests(job_id);
"""


def new_id() -> str:
    return uuid.uuid4().hex


def journal_mode() -> str:
    """Return the configured SQLite journal mode.

    WAL is fast on a native local filesystem, but its ``-shm`` mmap file is
    unsafe on Docker Desktop's macOS bind mounts and can terminate a process
    with SIGBUS.  The container therefore selects DELETE through compose.
    """
    mode = os.environ.get("TENNIS_SQLITE_JOURNAL_MODE", "WAL").strip().upper()
    if mode not in {"WAL", "DELETE"}:
        raise RuntimeError("TENNIS_SQLITE_JOURNAL_MODE 仅支持 WAL 或 DELETE")
    return mode


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # The value is allow-listed above; PRAGMA journal_mode cannot use a bound
    # parameter.  DELETE avoids SQLite shared-memory mmap on macOS bind mounts.
    conn.execute(f"PRAGMA journal_mode={journal_mode()}")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.db_path().parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)


def uj(s: str | None) -> Any:
    return json.loads(s) if s else None


def create_job(conn: sqlite3.Connection, *, jtype: str, payload: dict[str, Any],
               project_id: str | None = None, run_id: str | None = None,
               export_id: str | None = None, client_request_id: str | None = None) -> sqlite3.Row | None:
    """创建任务；client_request_id 重复时返回既有任务。"""
    now = time.time()
    jid = new_id()
    try:
        conn.execute(
            "INSERT INTO jobs (id,type,status,project_id,run_id,export_id,payload_json,client_request_id,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (jid, jtype, "queued", project_id, run_id, export_id, j(payload),
             client_request_id, now, now))
    except sqlite3.IntegrityError:
        if client_request_id:
            row = conn.execute("SELECT * FROM jobs WHERE client_request_id=?", (client_request_id,)).fetchone()
            if row:
                return row
        raise
    return conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()


def claim_next_job(worker_id: str) -> sqlite3.Row | None:
    """原子领取一个 queued 任务（单槽：先到先得）。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            return None
        cur = conn.execute(
            "UPDATE jobs SET status='running', worker_id=?, heartbeat_at=?, started_at=COALESCE(started_at,?), updated_at=? "
            "WHERE id=? AND status='queued'",
            (worker_id, time.time(), time.time(), time.time(), row["id"]))
        if cur.rowcount != 1:
            return None
        return conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()


def _interrupt_related(conn: sqlite3.Connection, job: sqlite3.Row, error: str) -> None:
    """任务中断时联动修正其派生资源的状态（runs / exports / projects）。

    import 任务会把 projects.proxy_status 置为 processing；不在这里回滚的话，
    项目会永久卡在 processing，create_run 一直 409。顺带清理转码残留的
    临时文件（transcode_proxy 只在取消路径主动 unlink）。
    """
    if job["type"] in ("analysis", "rederive") and job["run_id"]:
        conn.execute("UPDATE runs SET status='interrupted' WHERE id=? AND status='running'",
                     (job["run_id"],))
    if job["type"] == "export" and job["export_id"]:
        conn.execute("UPDATE exports SET status='interrupted', error=? WHERE id=? AND status='running'",
                     (error + "，导出中断", job["export_id"]))
    if job["type"] == "import" and job["project_id"]:
        conn.execute("UPDATE projects SET proxy_status='failed', proxy_error=? "
                     "WHERE id=? AND proxy_status='processing'",
                     (error + "，导入中断", job["project_id"]))
        row = conn.execute("SELECT original_path FROM projects WHERE id=?",
                           (job["project_id"],)).fetchone()
        if row and row["original_path"]:
            (Path(row["original_path"]).parent / "proxy.tmp.mp4").unlink(missing_ok=True)


def recover_stale_jobs(worker_id: str, stale_seconds: float = 60.0) -> int:
    """将失去心跳的 running 任务标记 interrupted（含其他 worker 的遗留）。

    ``worker_id`` is retained for the existing caller/API.  Recovery is based on
    the persisted heartbeat so a replacement worker can repair jobs owned by a
    process that no longer exists.
    """
    now = time.time()
    with connect() as conn:
        stale = conn.execute(
            "SELECT type,project_id,run_id,export_id FROM jobs WHERE status='running' "
            "AND (heartbeat_at IS NULL OR heartbeat_at < ?)",
            (now - stale_seconds,)).fetchall()
        cur = conn.execute(
            "UPDATE jobs SET status='interrupted', error='worker 心跳丢失（崩溃或被杀），任务中断', updated_at=?, finished_at=? "
            "WHERE status='running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)",
            (now, now, now - stale_seconds))
        for job in stale:
            _interrupt_related(conn, job, "worker 心跳丢失（崩溃或被杀）")
        return cur.rowcount


def interrupt_worker_jobs(worker_id: str, *, reason: str = "worker 异常退出，任务中断") -> int:
    """Immediately interrupt jobs owned by a worker known to have exited.

    The API supervisor calls this before starting the replacement process.  An
    exact owner match keeps independently managed workers isolated and removes
    the 60-second stale-heartbeat race.
    """
    if not worker_id:
        return 0
    now = time.time()
    with connect() as conn:
        owned = conn.execute(
            "SELECT type,project_id,run_id,export_id FROM jobs WHERE status='running' AND worker_id=?",
            (worker_id,)).fetchall()
        cur = conn.execute(
            "UPDATE jobs SET status='interrupted', error=?, updated_at=?, finished_at=? "
            "WHERE status='running' AND worker_id=?",
            (reason, now, now, worker_id))
        for job in owned:
            _interrupt_related(conn, job, reason)
        return cur.rowcount


def heartbeat(job_id: str, worker_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE jobs SET heartbeat_at=?, updated_at=? WHERE id=? AND worker_id=? AND status='running'",
                     (time.time(), time.time(), job_id, worker_id))


def is_cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
    return bool(row and row["cancel_requested"])


class CancelledError(RuntimeError):
    pass


def check_cancel(conn: sqlite3.Connection, job_id: str) -> None:
    if is_cancel_requested(conn, job_id):
        raise CancelledError("用户请求取消")


def update_job_progress(conn: sqlite3.Connection, job_id: str, *, stage: str | None = None,
                        progress: dict[str, Any] | None = None) -> None:
    # 进度更新同时刷新心跳：崩溃恢复用 heartbeat_at 判定任务是否仍存活
    sets, vals = ["updated_at=?", "heartbeat_at=?"], [time.time(), time.time()]
    if stage is not None:
        sets.append("stage=?"); vals.append(stage)
    if progress is not None:
        sets.append("progress_json=?"); vals.append(j(progress))
    vals.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
