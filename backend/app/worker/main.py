"""独立 worker 进程：单个重任务执行槽。

- 原子领取 queued 任务（SQLite 单条 UPDATE）
- 心跳；崩溃/被杀后由重启的 worker 将失心跳任务标记 interrupted
- 逐帧/逐批检查取消标记
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

from app import config
from app.store import db
from app.store.db import CancelledError, claim_next_job, check_cancel, heartbeat, recover_stale_jobs, update_job_progress

WORKER_ID = os.environ.get("TENNIS_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"
POLL_INTERVAL_S = 0.5
RECOVERY_INTERVAL_S = 5.0


def log(msg: str) -> None:
    print(f"[worker {WORKER_ID}] {msg}", flush=True)


def _finish(conn, job_id: str, status: str, *, result=None, error=None):
    conn.execute(
        "UPDATE jobs SET status=?, result_json=?, error=?, updated_at=?, finished_at=? WHERE id=?",
        (status, db.j(result) if result is not None else None, error, time.time(), time.time(), job_id))


# ---------------- 任务处理器 ----------------

def handle_model_download(payload: dict, job: dict, conn) -> dict:
    import hashlib
    import urllib.request
    manifest = json.loads(config.manifest_path().read_text(encoding="utf-8"))
    results = []
    for item in manifest["models"]:
        dst = config.models_dir() / item["filename"]
        if dst.exists():
            h = hashlib.sha256(dst.read_bytes()).hexdigest()
            if h == item["sha256"]:
                results.append({"key": item["key"], "status": "already_present"})
                continue
        url = item["url"]
        tmp = dst.with_suffix(".tmp")
        log(f"downloading {item['key']} from {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        h = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if h != item["sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{item['key']} 校验失败：sha256 {h} != {item['sha256']}")
        os.replace(tmp, dst)
        results.append({"key": item["key"], "status": "downloaded", "sha256": h})
        update_job_progress(conn, job["id"], progress={"downloaded": results})
    return {"models": results}


def handle_import(payload: dict, job: dict, conn) -> dict:
    from app.video import proxy as proxy_mod
    from app.video.probe import analysis_dims, file_hash, probe_video
    project_id = payload["project_id"]
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if proj is None:
        raise RuntimeError("项目不存在")
    update_job_progress(conn, job["id"], stage="probing")
    probe = probe_video(Path(proj["original_path"]))
    vhash = file_hash(Path(proj["original_path"]))
    conn.execute(
        "UPDATE projects SET probe_json=?, video_hash=?, width=?, height=?, duration_ms=?, "
        "nominal_fps=?, rotation=?, has_audio=?, proxy_status='processing' WHERE id=?",
        (db.j(probe), vhash, probe["width"], probe["height"], probe["duration_ms"],
         probe.get("nominal_fps"), probe.get("rotation", 0), 1 if probe.get("has_audio") else 0,
         project_id))
    conn.commit()
    update_job_progress(conn, job["id"], stage="preparing")

    def should_continue() -> bool:
        from app.store.db import is_cancel_requested
        return not is_cancel_requested(conn, job["id"])

    proxy_path = Path(proj["original_path"]).parent / "proxy.mp4"
    meta = proxy_mod.transcode_proxy(
        proj["original_path"], proxy_path,
        progress=lambda d, t: update_job_progress(
            conn, job["id"], stage="preparing", progress={"frames_done": d, "frames_total_est": t}),
        should_continue=should_continue)
    # 缩略图（代理首帧）
    thumb_path = Path(proj["original_path"]).parent / "thumb.jpg"
    try:
        import av as avlib
        import cv2
        import numpy as np
        with avlib.open(str(proxy_path)) as c:
            for frame in c.decode(video=0):
                img = frame.to_ndarray(format="bgr24")
                cv2.imwrite(str(thumb_path), cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))))
                break
    except Exception:
        thumb_path = None  # 缩略图失败不影响导入
    conn.execute("UPDATE projects SET proxy_status='ready', proxy_error=NULL WHERE id=?", (project_id,))
    check_cancel(conn, job["id"])
    analysis_id = None
    if payload.get("auto_analyze"):
        if (probe.get("duration_ms") or 0) > 60000:
            raise RuntimeError("自动分析支持60秒以内的视频，请剪短后重新上传")
        analysis_id = db.create_job(conn, jtype="expert", project_id=project_id,
            payload={"action": "automatic", "source": None, "spec": {"hand": "auto", **payload.get("analysis_spec", {"analysis_mode": "local"})}},
            client_request_id=f"auto-import:{job['id']}")["id"]
    return {"proxy": meta, "thumb": str(thumb_path) if thumb_path else None, "probe": probe,
            "analysis_job_id": analysis_id}


def handle_analysis(payload: dict, job: dict, conn) -> dict:
    from app.vision.pipeline import execute_analysis
    run_id = payload["run_id"]
    run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        raise RuntimeError("run 不存在")
    conn.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
    conn.commit()
    update_job_progress(conn, job["id"], stage="preparing")
    job_dict = {"id": job["id"], "worker_id": WORKER_ID}
    try:
        out = execute_analysis(job=job_dict, run=dict(run))
    except Exception:
        conn.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))
        conn.commit()
        raise
    # 复用帧的派生 run：写入事件与摘要（execute_analysis 内部已写入 run_dir）
    from pathlib import Path as _P
    run_dir = _P(run["run_dir"])
    (_run_dims := run_dir / "analysis_dims.json")
    if not _run_dims.exists():
        _run_dims.write_text(json.dumps(out["analysis_dims"]), encoding="utf-8")
    conn.execute(
        "UPDATE runs SET status='succeeded', effective_config_json=?, performance_json=?, "
        "summary_json=?, model_hash=?, finished_at=? WHERE id=?",
        (db.j(out["effective_config"]), db.j(out["performance"]), db.j(out["summary"]),
         out["performance"].get("model_hash"), time.time(), run_id))
    return {"run_id": run_id, "events_count": out["events_count"],
            "performance": out["performance"], "summary": out["summary"]}


def handle_rederive(payload: dict, job: dict, conn) -> dict:
    """仅重算事件（复用父 run 的帧观测）。"""
    from app.vision.events import compute_phase_segments, extract_events
    from app.vision.geometry import WallQuad
    from app.vision.pipeline import load_frames_window, summarize
    run_id = payload["run_id"]
    run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    parent = conn.execute("SELECT * FROM runs WHERE id=?", (run["reuses_frames_from"],)).fetchone()
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    update_job_progress(conn, job["id"], stage="events")
    frames_src = Path(parent["run_dir"]) / "frames"
    frames = load_frames_window(frames_src)
    cfg = json.loads(run["config_json"])
    from app.vision.geometry import Roi
    dims = json.loads((Path(parent["run_dir"]) / "analysis_dims.json").read_text())
    swings = None
    swings_path = Path(parent["run_dir"]) / "swings.json"
    if swings_path.exists():
        swings = json.loads(swings_path.read_text(encoding="utf-8"))["swings"]
    events = extract_events(
        frames, run_id=run_id, handedness=cfg.get("handedness", "unknown"),
        wall=WallQuad.from_dict(cfg.get("wall")), cfg_events=cfg.get("events"),
        measurements=cfg.get("measurements"), analysis_height=dims["h"],
        swings=swings)
    if payload.get("use_corrections"):
        # 人工球点修正参与：仅替换被修正帧的观测点（assisted）
        ops = [json.loads(r["op_json"]) for r in conn.execute(
            "SELECT op_json FROM corrections WHERE run_id=? ORDER BY revision", (parent["id"],)).fetchall()]
        pt_fix = {}
        for op in ops:
            if op.get("op") == "modify_ball_point" and op.get("point") is not None:
                pt_fix[op["frame_index"]] = op["point"]
        for f in frames:
            if f["frame_index"] in pt_fix and f.get("ball") and f["ball"].get("selected"):
                f["ball"]["selected"]["point"] = pt_fix[f["frame_index"]]
                f["ball"]["selected"]["raw"] = True
                f["ball"]["selected"]["status"] = "human_corrected"
    events_path = run_dir / "events.json"
    tmp = events_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"events": events, "automatic_immutable": False,
                               "assisted": bool(payload.get("use_corrections"))},
                              ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, events_path)
    summary = summarize(frames, events)
    summary["assisted"] = bool(payload.get("use_corrections"))
    conn.execute("UPDATE runs SET status='succeeded', summary_json=?, finished_at=? WHERE id=?",
                 (db.j(summary), time.time(), run_id))
    if not (run_dir / "analysis_dims.json").exists():
        (run_dir / "analysis_dims.json").write_text(json.dumps(dims), encoding="utf-8")
    return {"run_id": run_id, "events_count": len(events), "summary": summary}


def handle_export(payload: dict, job: dict, conn) -> dict:
    from app.services import corrections as corr_svc
    from app.services.exports import build_export
    export_id = payload["export_id"]
    return build_export(export_id, conn, job)


from app.services.expert import execute as handle_expert

HANDLERS = {
    "expert": handle_expert,
    "model_download": handle_model_download,
    "import": handle_import,
    "analysis": handle_analysis,
    "rederive": handle_rederive,
    "export": handle_export,
}


def run_once(job) -> None:
    payload = db.uj(job["payload_json"]) or {}
    handler = HANDLERS.get(job["type"])
    if handler is None:
        with db.connect() as conn:
            _finish(conn, job["id"], "failed", error=f"未知任务类型 {job['type']}")
        return
    log(f"start job {job['id']} type={job['type']}")
    t0 = time.time()
    with db.connect() as conn:
        update_job_progress(conn, job["id"], stage="starting")
    try:
        with db.connect() as conn:
            result = handler(payload, dict(job), conn)
        with db.connect() as conn:
            _finish(conn, job["id"], "succeeded", result=result)
        log(f"job {job['id']} succeeded in {time.time()-t0:.1f}s")
    except CancelledError as e:
        _mark_related_status(job, "cancelled")
        with db.connect() as conn:
            _finish(conn, job["id"], "cancelled", error=str(e))
        log(f"job {job['id']} cancelled")
    except Exception as e:
        _mark_related_status(job, "failed", error=str(e))
        with db.connect() as conn:
            _finish(conn, job["id"], "failed",
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
        log(f"job {job['id']} failed: {e}")
        traceback.print_exc()


def _mark_related_status(job, status: str, error: str | None = None) -> None:
    """Set the state owned by a job without corrupting unrelated resources.

    sqlite3.Row has mapping access but no ``get`` method, so normalize first.
    An export failure must not turn a previously successful analysis run into a
    failed run.
    """
    item = dict(job)
    run_id = item.get("run_id")
    export_id = item.get("export_id")
    if item.get("type") in ("analysis", "rederive") and run_id:
        with db.connect() as conn:
            conn.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
    if item.get("type") == "export" and export_id:
        with db.connect() as conn:
            conn.execute("UPDATE exports SET status=?, error=? WHERE id=?",
                         (status, error, export_id))


def main() -> int:
    db.init_db()
    n = recover_stale_jobs(WORKER_ID)
    if n:
        log(f"恢复：{n} 个失心跳任务标记为 interrupted")
    log(f"worker 启动（{WORKER_ID}），等待任务…")
    last_recovery = time.monotonic()
    while True:
        # Startup can race with the 60-second stale threshold after a fast
        # container/process restart.  Repeat recovery while the worker is idle
        # so an orphaned running job cannot remain visible forever.
        now = time.monotonic()
        if now - last_recovery >= RECOVERY_INTERVAL_S:
            n = recover_stale_jobs(WORKER_ID)
            if n:
                log(f"恢复：{n} 个失心跳任务标记为 interrupted")
            last_recovery = now
        job = claim_next_job(WORKER_ID)
        if job is None:
            time.sleep(POLL_INTERVAL_S)
            continue
        run_once(job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
