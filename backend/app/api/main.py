"""FastAPI 应用：上传、任务、run、观测、事件、复核、对比、标注、评估、导出。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app.store import db
from app.api.model_connections import router as model_connections_router
from app.services import corrections as corr_svc
from app.services import evaluations as eval_svc

app = FastAPI(title="Tennis Wall Trainer", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"])

# worker 子进程（由 API 启动；scripts/dev.sh 亦可独立启动）
_worker_proc: subprocess.Popen | None = None
_worker_id: str | None = None
_C = config.cfg()


_worker_stop = threading.Event()


def _spawn_worker() -> subprocess.Popen:
    global _worker_id
    _worker_id = f"{os.uname().nodename}-{uuid.uuid4().hex[:12]}"
    env = os.environ.copy()
    env["TENNIS_WORKER_ID"] = _worker_id
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker.main"],
        cwd=Path(__file__).resolve().parents[2], env=env)


def _worker_watchdog() -> None:
    """worker 崩溃/被杀后自动重启（新 worker 启动时会先把失心跳任务标记
    interrupted），避免任务永远停留在 running、前端无限等待。"""
    while not _worker_stop.wait(5.0):
        global _worker_proc
        proc = _worker_proc
        if proc is not None and proc.poll() is not None:
            print(f"[api] worker 已退出(code={proc.returncode})，重启 worker…", flush=True)
            if _worker_id:
                interrupted = db.interrupt_worker_jobs(
                    _worker_id,
                    reason=f"worker 异常退出（code={proc.returncode}），任务中断")
                if interrupted:
                    print(f"[api] 已中断 {interrupted} 个旧 worker 任务", flush=True)
            _worker_proc = _spawn_worker()


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    if os.environ.get("TENNIS_NO_WORKER") != "1":
        global _worker_proc
        _worker_proc = _spawn_worker()
        threading.Thread(target=_worker_watchdog, daemon=True).start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _worker_stop.set()
    if _worker_proc is not None:
        _worker_proc.terminate()


app.include_router(model_connections_router)


@app.middleware("http")
async def _guard_local_model_api(request: Request, call_next):
    from app.api.model_connections import check_origin
    if request.url.path.startswith("/api/model-connections") and \
            request.method in ("POST", "PATCH", "DELETE", "PUT"):
        denied = check_origin(request)   # 跨站/非本机写操作在任何 body 解析前拒绝
        if denied is not None:
            return denied
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    with db.connect() as conn:
        running = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='running'").fetchone()["n"]
        queued = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='queued'").fetchone()["n"]
    return {
        "status": "ok", "worker_spawned": _worker_proc is not None and _worker_proc.poll() is None,
        "jobs_running": running, "jobs_queued": queued,
        "versions": _versions(),
    }


def _versions() -> dict:
    import av, cv2, fastapi, numpy, torch, ultralytics
    try:
        import mediapipe as mp
        mediapipe_version = mp.__version__
    except ImportError:
        mediapipe_version = None
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "mediapipe": mediapipe_version,
            "opencv": cv2.__version__, "av": av.__version__, "numpy": numpy.__version__,
            "fastapi": fastapi.__version__, "code_version": config.CODE_VERSION}


# ---------------- 模型 ----------------

@app.get("/api/models")
def models_status() -> dict:
    manifest = json.loads(config.manifest_path().read_text(encoding="utf-8"))
    out = []
    for m in manifest["models"]:
        p = config.models_dir() / m["filename"]
        item = {**m, "path": str(p), "exists": p.exists()}
        if p.exists():
            item["size_bytes_actual"] = p.stat().st_size
            item["sha_match"] = (p.stat().st_size == m["size_bytes"])
        out.append(item)
    return {"models": out,
            "all_present": all(i["exists"] and i.get("sha_match") for i in out)}


@app.post("/api/models/download")
def models_download(client_request_id: Optional[str] = None) -> JSONResponse:
    with db.connect() as conn:
        job = db.create_job(conn, jtype="model_download", payload={},
                            client_request_id=client_request_id or f"mdl-{int(time.time())}")
    return JSONResponse({"job_id": job["id"]}, status_code=202)


# ---------------- 项目 ----------------

@app.post("/api/projects")
async def create_project(file: UploadFile = File(...),
                         client_request_id: Optional[str] = Form(None),
                         auto_analyze: bool = Form(False),
                         analysis_mode: str = Form("local"),
                         connection_id: Optional[str] = Form(None)) -> JSONResponse:
    from app.services.analysis_mode import snapshot
    analysis_spec = snapshot(analysis_mode, connection_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".mp4", ".mov", ".m4v"):
        raise HTTPException(415, f"仅支持 MP4/MOV（收到 {ext or '无扩展名'}）；实际可用性由容器探测决定")
    pid = db.new_id()
    proj_dir = config.projects_dir() / pid
    proj_dir.mkdir(parents=True)
    dst = proj_dir / f"original{ext}"
    max_bytes = int(_C["limits"]["max_upload_bytes"])
    h = hashlib.sha256()
    written = 0
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"文件超过上传上限 {max_bytes // (1 << 30)}GB")
            h.update(chunk)
            f.write(chunk)
    now = time.time()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, created_at, original_filename, original_path, video_hash, "
            "size_bytes, proxy_status) VALUES (?,?,?,?,?,?,?)",
            (pid, now, file.filename, str(dst), h.hexdigest(), written, "pending"))
        job = db.create_job(conn, jtype="import", payload={"project_id": pid, "auto_analyze": auto_analyze, "analysis_spec": analysis_spec},
                            project_id=pid, client_request_id=client_request_id)
    return JSONResponse({"project_id": pid, "preparation_job_id": job["id"]}, status_code=202)


@app.get("/api/projects")
def list_projects(trash: bool = False) -> dict:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM projects WHERE (id IN (SELECT project_id FROM project_trash)) = ? ORDER BY created_at DESC", (int(trash),)).fetchall()
        out = []
        for r in rows:
            latest = conn.execute(
                "SELECT id, status, created_at, mode FROM runs WHERE project_id=? "
                "ORDER BY created_at DESC LIMIT 1", (r["id"],)).fetchone()
            analysis = conn.execute("SELECT id,type,status,stage FROM jobs WHERE project_id=? AND (type='import' OR (type='expert' AND json_extract(payload_json, '$.action')='automatic')) ORDER BY created_at DESC LIMIT 1", (r["id"],)).fetchone()
            out.append({**_proj_dict(r), "latest_run": dict(latest) if latest else None,
                        "analysis": dict(analysis) if analysis else None})
    return {"projects": out}


def _proj_dict(r) -> dict:
    return {"id": r["id"], "filename": r["original_filename"], "created_at": r["created_at"],
            "size_bytes": r["size_bytes"], "video_hash": r["video_hash"][:16],
            "proxy_status": r["proxy_status"], "proxy_error": r["proxy_error"],
            "width": r["width"], "height": r["height"], "duration_ms": r["duration_ms"],
            "nominal_fps": r["nominal_fps"], "rotation": r["rotation"],
            "has_audio": bool(r["has_audio"]), "probe": db.uj(r["probe_json"])}


@app.get("/api/projects/{pid}/storage")
def project_storage(pid: str):
    from app.services.deletion import storage
    return storage(pid)


@app.post("/api/projects/{pid}/trash")
def trash_project(pid: str):
    from app.services.deletion import mutate
    return mutate(pid, "trash")


@app.post("/api/projects/{pid}/restore")
def restore_project(pid: str):
    from app.services.deletion import mutate
    return mutate(pid, "restore")


@app.delete("/api/projects/{pid}")
def purge_project(pid: str):
    from app.services.deletion import mutate
    return mutate(pid, "purge")


@app.delete("/api/projects/{pid}/runs/{rid}")
def delete_run(pid: str, rid: str):
    from app.services.deletion import mutate
    return mutate(pid, "run", rid)


@app.delete("/api/projects/{pid}/exports/{eid}")
def delete_export(pid: str, eid: str):
    from app.services.deletion import mutate
    return mutate(pid, "export", eid)


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        if r is None:
            raise HTTPException(404, "项目不存在")
        jobs = conn.execute(
            "SELECT id,type,status,stage,error FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 5",
            (pid,)).fetchall()
    return {**_proj_dict(r), "recent_jobs": [dict(j) for j in jobs]}


@app.get("/api/projects/{pid}/media")
def project_media(pid: str, request: Request):
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if r is None or r["proxy_status"] != "ready":
        raise HTTPException(404, "代理视频未就绪")
    path = Path(r["original_path"]).parent / "proxy.mp4"
    return _serve_range(path, request, media_type="video/mp4")


@app.get("/api/projects/{pid}/thumb")
def project_thumb(pid: str):
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if r is None:
        raise HTTPException(404)
    p = Path(r["original_path"]).parent / "thumb.jpg"
    if not p.exists():
        raise HTTPException(404, "缩略图不存在")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/projects/{pid}/person-frame")
def project_person_frame(pid: str, max_ms: int = Query(default=4000, ge=200, le=20000)):
    """在视频前 max_ms 内按 ~2fps 抽帧做人物检测，返回首个有人帧（JPEG+框列表）。

    供设置页显示人物框并允许用户点击选择目标；未检出时 404。
    """
    import cv2
    from app.video.decode import decode_frame_by_pts
    from app.vision.detector import Detector
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if r is None or r["proxy_status"] != "ready":
        raise HTTPException(404, "项目未就绪")
    probe = db.uj(r["probe_json"]) or {}
    det = Detector(str(config.model_path_for("yolo11n")), device="auto_mps_then_cpu",
                   imgsz=640, ball_confidence=0.15)
    t = 0
    while t <= max_ms:
        img, _, pts_ms = decode_frame_by_pts(r["original_path"], t)
        if img is None:
            break
        res = det.detect(img)
        persons = det.person_candidates(res)
        if persons:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            import io
            import base64
            return {
                "found": True, "pts_ms": pts_ms,
                "image_jpeg_base64": base64.b64encode(buf).decode(),
                "persons": [{"bbox": list(c.bbox), "confidence": c.confidence} for c in persons],
                "note": "点击选择目标人物；默认取最大框",
            }
        t += 500
    return {"found": False, "note": f"前 {max_ms}ms 未检测到人物"}


@app.get("/api/projects/{pid}/frame")
def project_frame(pid: str, index: int = Query(ge=0)):
    import cv2
    from app.video.decode import decode_exact_frame
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if r is None:
        raise HTTPException(404)
    img, fi, pts_ms = decode_exact_frame(r["original_path"], index)
    if img is None:
        raise HTTPException(404, f"无法解码第 {index} 帧")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(
        content=buf.tobytes(), media_type="image/jpeg",
        headers={"X-Frame-Index": str(fi), "X-Source-Pts-Ms": str(pts_ms),
                 "Cache-Control": "no-store"})


def _serve_range(path: Path, request: Request, media_type: str) -> Response:
    size = path.stat().st_size
    range_hdr = request.headers.get("range")
    if range_hdr:
        start = end = None
        try:
            unit, rng = range_hdr.split("=", 1)
            if unit.strip().lower() != "bytes":
                raise ValueError
            start_s, _, end_s = rng.partition("-")
            if start_s == "":  # 后缀范围 bytes=-N：最后 N 字节
                n = int(end_s)
                if n <= 0:
                    raise ValueError
                start, end = max(0, size - n), size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
        except ValueError:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if start >= size or end < start:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        end = min(end, size - 1)
        length = end - start + 1

        def _chunks():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, 4 * 1024 * 1024))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(_chunks(), status_code=206, media_type=media_type, headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes", "Content-Length": str(length)})
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})


# ---------------- 标定 ----------------

class CalibrationIn(BaseModel):
    wall: Optional[dict] = None          # {points: [[x,y]x4], real_width_m, real_height_m}
    target_zones: Optional[list[dict]] = None
    note: Optional[str] = None


@app.post("/api/projects/{pid}/calibrations")
def create_calibration(pid: str, body: CalibrationIn) -> dict:
    from app.vision.geometry import WallQuad
    errs: list[str] = []
    wall = WallQuad.from_dict(body.wall) if body.wall else None
    if wall:
        errs = wall.validate()
    if errs:
        raise HTTPException(422, {"message": "标定校验失败", "errors": errs})
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM projects WHERE id=?", (pid,)).fetchone() is None:
            raise HTTPException(404, "项目不存在")
        rev = conn.execute("SELECT COALESCE(MAX(revision),0)+1 AS r FROM calibrations WHERE project_id=?",
                           (pid,)).fetchone()["r"]
        cid = db.new_id()
        conn.execute("INSERT INTO calibrations (id, project_id, revision, data_json, created_at) VALUES (?,?,?,?,?)",
                     (cid, pid, rev, db.j({"wall": body.wall, "target_zones": body.target_zones,
                                           "note": body.note}), time.time()))
    return {"id": cid, "project_id": pid, "revision": rev}


@app.get("/api/projects/{pid}/calibrations")
def list_calibrations(pid: str) -> dict:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM calibrations WHERE project_id=? ORDER BY revision DESC",
                            (pid,)).fetchall()
    return {"calibrations": [{"id": r["id"], "revision": r["revision"],
                              "data": db.uj(r["data_json"]), "created_at": r["created_at"]} for r in rows]}


# Reviewed shot workflow (immutable worker revisions)
@app.get("/api/projects/{pid}/expert")
def expert_list(pid: str):
    from app.services.expert import listing
    return listing(pid)

@app.post("/api/projects/{pid}/expert")
def expert_start(pid: str, body: dict):
    from app.services.expert import enqueue
    return enqueue(pid, body)

@app.get("/api/projects/{pid}/expert/{jid}/files/{name}")
def expert_file(pid: str, jid: str, name: str):
    from app.services.expert import session, FILES
    if name not in FILES: raise HTTPException(404, "未知文件")
    with db.connect() as c:
        if not c.execute("SELECT 1 FROM jobs WHERE id=? AND project_id=? AND type='expert'", (jid,pid)).fetchone():
            raise HTTPException(404, "版本不存在")
    p=session(pid,jid)/name
    if not p.is_file(): raise HTTPException(404, "文件未生成")
    return FileResponse(p)

@app.get("/api/projects/{pid}/expert/{jid}/frame/{frame}")
def expert_frame(pid: str, jid: str, frame: int):
    from app.services.expert import session
    if frame < 0: raise HTTPException(422, "帧号无效")
    p=session(pid,jid)/"frames"/f"f_{frame+1:05d}.jpg"
    if not p.is_file(): raise HTTPException(404, "帧不存在")
    return FileResponse(p)

@app.get("/api/projects/{pid}/expert/{sid}/enhancement")
def expert_enhancement(pid: str, sid: str):
    """用户可读的复核摘要（脱敏：无提示词/原始响应/密钥）。"""
    from app.services.expert import session, read
    from app.store import db
    with db.connect() as c:
        if not c.execute("SELECT 1 FROM jobs WHERE id=? AND project_id=? AND type='expert'",
                         (sid, pid)).fetchone():
            raise HTTPException(404, "版本不存在")
    summary = read(session(pid, sid) / "enhancement_summary.json")
    if summary is None:
        return {"status": "not_run",
                "note": "本次分析未执行增强复核（未启用连接或无异常窗口）"}
    rounds = []
    for r in summary.get("rounds", []):
        rounds.append({k: r.get(k) for k in
                       ("kind", "question", "status", "error_code", "reason",
                        "needs_more_evidence")})
        n_acc = sum(1 for p in r.get("proposals", [])
                    if p.get("verdict", {}).get("status") == "accepted")
        n_rej = sum(1 for p in r.get("proposals", [])
                    if p.get("verdict", {}).get("status") == "rejected")
        rounds[-1].update(proposals_accepted=n_acc, proposals_rejected=n_rej)
    return {"status": summary.get("status"),
            "source_line": summary.get("source_line"),
            "requests": summary.get("requests"),
            "accepted": summary.get("accepted", 0),
            "rejected": summary.get("rejected", 0),
            "unresolved": summary.get("unresolved", 0),
            "budget": summary.get("budget"),
            "rounds": rounds,
            "note": "模型提议均需本地证据验证；未通过的仅存审计，不进入统计"}


# ---------------- Run ----------------

class RunIn(BaseModel):
    segment_start_ms: int = Field(ge=0)
    segment_end_ms: int
    handedness: str = Field(default="unknown", pattern="^(right|left|unknown)$")
    ball_search_roi: Optional[dict] = None
    wall: Optional[dict] = None
    mirror: bool = False
    model: Optional[dict] = None      # 覆盖项：{imgsz, ball_confidence, device…}
    tracking: Optional[dict] = None
    events: Optional[dict] = None
    processing: Optional[dict] = None  # {detection_mode, tiles, full_search_interval_ms, person_stride_ms}
    target_person_hint: Optional[dict] = None   # {box: [x1,y1,x2,y2]}，来自首个人物帧选择
    calibration_revision: Optional[int] = None
    client_request_id: Optional[str] = None


@app.post("/api/projects/{pid}/runs")
def create_run(pid: str, body: RunIn) -> JSONResponse:
    from app.vision.geometry import WallQuad
    with db.connect() as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        if proj is None:
            raise HTTPException(404, "项目不存在")
        if proj["proxy_status"] != "ready":
            raise HTTPException(409, "项目准备未完成（代理未就绪），无法创建 run")
        dur = proj["duration_ms"] or 0
        s, e = body.segment_start_ms, body.segment_end_ms
        if not (0 <= s < e <= dur):
            raise HTTPException(422, f"时间段无效：需 0 <= start < end <= {dur}ms（收到 {s}..{e}）")
        max_seg = int(_C["limits"]["max_segment_seconds"]) * 1000
        if e - s > max_seg:
            raise HTTPException(422, f"片段超过上限 {max_seg // 1000}s，请先裁剪")
        wall = WallQuad.from_dict(body.wall) if body.wall else None
        if wall:
            errs = wall.validate()
            if errs:
                raise HTTPException(422, {"message": "墙面四点校验失败", "errors": errs})
        proc = body.processing or {}
        mode = proc.get("detection_mode", "full_frame")
        if mode == "tracknet" and not (config.models_dir() / "tracknet_tennis.pt").is_file():
            raise HTTPException(422, "请先安装 TrackNet tennis 权重到 data/models/tracknet_tennis.pt")
        if mode not in ("full_frame", "tiled", "tiled_local", "tracknet"):
            raise HTTPException(422, f"未知检测模式 {mode}：full_frame | tiled | tiled_local")
        if body.target_person_hint and not (body.target_person_hint.get("box")
                                            and len(body.target_person_hint["box"]) == 4):
            raise HTTPException(422, "target_person_hint 需要 box: [x1,y1,x2,y2]")
        cfg = body.model_dump(exclude={"client_request_id"})
        cache_key = _run_cache_key(proj["video_hash"], cfg)
        existing = conn.execute(
            "SELECT id FROM runs WHERE cache_key=? AND status IN ('succeeded','running')",
            (cache_key,)).fetchone()
        if existing:
            return JSONResponse({"run_id": existing["id"], "reused": True}, status_code=200)
        rid = db.new_id()
        run_dir = config.projects_dir() / pid / "runs" / rid
        conn.execute(
            "INSERT INTO runs (id, project_id, mode, status, config_json, code_version, "
            "calibration_revision, segment_start_ms, segment_end_ms, cache_key, run_dir, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, pid, "automatic", "queued", db.j(cfg), config.CODE_VERSION,
             body.calibration_revision, s, e, cache_key, str(run_dir), time.time()))
        job = db.create_job(conn, jtype="analysis", payload={"run_id": rid},
                            project_id=pid, run_id=rid,
                            client_request_id=body.client_request_id)
    return JSONResponse({"run_id": rid, "job_id": job["id"]}, status_code=202)


def _run_cache_key(video_hash: str, cfg: dict) -> str:
    relevant = {k: cfg.get(k) for k in ("segment_start_ms", "segment_end_ms", "handedness",
                                        "ball_search_roi", "wall", "mirror", "model", "tracking",
                                        "processing", "target_person_hint")}
    blob = json.dumps({"v": video_hash, "c": relevant,
                       "code": config.CODE_VERSION, "pre": config.PREPROCESS_VERSION},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@app.get("/api/projects/{pid}/runs")
def list_runs(pid: str) -> dict:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC",
                            (pid,)).fetchall()
    return {"runs": [_run_dict(r) for r in rows]}


def _run_dict(r) -> dict:
    return {"id": r["id"], "project_id": r["project_id"], "parent_run_id": r["parent_run_id"],
            "mode": r["mode"], "status": r["status"],
            "segment_start_ms": r["segment_start_ms"], "segment_end_ms": r["segment_end_ms"],
            "config": db.uj(r["config_json"]), "effective_config": db.uj(r["effective_config_json"]),
            "cache_key": r["cache_key"], "model_hash": r["model_hash"],
            "code_version": r["code_version"], "calibration_revision": r["calibration_revision"],
            "performance": db.uj(r["performance_json"]), "summary": db.uj(r["summary_json"]),
            "created_at": r["created_at"], "finished_at": r["finished_at"],
            "reuses_frames_from": r["reuses_frames_from"]}


@app.get("/api/runs/{rid}")
def get_run(rid: str) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if r is None:
            raise HTTPException(404, "run 不存在")
        d = _run_dict(r)
        d["correction_revision"] = corr_svc.current_revision(conn, rid)
        jobs = conn.execute("SELECT id,status,stage,error FROM jobs WHERE run_id=? ORDER BY created_at DESC LIMIT 3",
                            (rid,)).fetchall()
        d["jobs"] = [dict(j) for j in jobs]
    return d


@app.get("/api/runs/{rid}/speed")
def run_speed(rid: str) -> dict:
    from app.vision.pipeline import load_frames_window
    from app.vision.speed import speed_series
    r = _get_run_row(rid)
    if r is None:
        raise HTTPException(404, "run 不存在")
    source = _get_run_row(r["reuses_frames_from"]) if r["reuses_frames_from"] else r
    if source is None:
        raise HTTPException(404, "观测来源不存在")
    return speed_series(load_frames_window(Path(source["run_dir"]) / "frames",
                        r["segment_start_ms"], r["segment_end_ms"]))


@app.get("/api/runs/{rid}/observations")
def run_observations(rid: str, start_ms: int = Query(default=None),
                     end_ms: int = Query(default=None)) -> dict:
    from app.vision.pipeline import load_frames_window
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    if r is None:
        raise HTTPException(404)
    frames_dir = Path(r["run_dir"]) / "frames"
    if r["reuses_frames_from"]:
        parent = _get_run_row(r["reuses_frames_from"])
        frames_dir = Path(parent["run_dir"]) / "frames"
    if not (frames_dir / "index.json").exists():
        return {"frames": [], "note": "尚无帧数据（run 未完成或失败）"}
    frames = load_frames_window(frames_dir, start_ms, end_ms)
    cap = int(_C["limits"]["observation_window_max_frames"])
    if len(frames) > cap:
        raise HTTPException(422, f"窗口内 {len(frames)} 帧超过上限 {cap}，请缩小时间范围")
    return {"frames": frames, "count": len(frames),
            "segment_start_ms": r["segment_start_ms"], "segment_end_ms": r["segment_end_ms"]}


def _get_run_row(rid: str):
    with db.connect() as conn:
        return conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()


@app.get("/api/runs/{rid}/events")
def run_events(rid: str, view: str = Query(default="automatic", pattern="^(automatic|reviewed)$")) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if r is None:
            raise HTTPException(404)
        events = _load_auto_events(r)
        if view == "reviewed":
            ops = corr_svc.list_corrections(conn, rid)
            events = corr_svc.apply_ops(events, ops)
        auto_summary = db.uj(r["summary_json"]) or {}
        reviewed = corr_svc.reviewed_summary(auto_summary, events) if view == "reviewed" else None
    return {"events": events, "view": view,
            "summary": auto_summary, "reviewed_summary": reviewed}


def _load_auto_events(r) -> list[dict]:
    p = Path(r["run_dir"]) / "events.json"
    if not p.exists():
        parent = r["reuses_frames_from"]
        if parent:
            pr = _get_run_row(parent)
            return _load_auto_events(pr)
        return []
    return json.loads(p.read_text(encoding="utf-8"))["events"]


# ---------------- 任务 ----------------

@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if r is None:
        raise HTTPException(404)
    return {"id": r["id"], "type": r["type"], "status": r["status"], "stage": r["stage"],
            "progress": db.uj(r["progress_json"]), "error": r["error"],
            "result": db.uj(r["result_json"]), "cancel_requested": bool(r["cancel_requested"]),
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "finished_at": r["finished_at"]}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if r is None:
            raise HTTPException(404)
        if r["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return {"id": jid, "status": r["status"], "note": "任务已结束，取消为幂等空操作"}
        conn.execute("UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?", (time.time(), jid))
    return {"id": jid, "status": "cancelling"}


# ---------------- 复核 ----------------

class CorrectionIn(BaseModel):
    op: str
    event_id: Optional[str] = None
    event: Optional[dict] = None
    changes: Optional[dict] = None
    reason: Optional[str] = None
    frame_index: Optional[int] = None
    point: Optional[list[float]] = None
    expected_revision: int


@app.post("/api/runs/{rid}/corrections")
def add_correction(rid: str, body: CorrectionIn) -> dict:
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM runs WHERE id=?", (rid,)).fetchone() is None:
            raise HTTPException(404)
        try:
            return corr_svc.append_correction(conn, rid, body.model_dump(exclude={"expected_revision"}),
                                              body.expected_revision)
        except corr_svc.RevisionConflict as e:
            raise HTTPException(409, {"message": "版本冲突：请刷新后重试", "current_revision": e.current})
        except ValueError as e:
            raise HTTPException(422, str(e))


class UndoIn(BaseModel):
    target_op_id: int


@app.post("/api/runs/{rid}/corrections/undo")
def undo(rid: str, body: UndoIn) -> dict:
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM runs WHERE id=?", (rid,)).fetchone() is None:
            raise HTTPException(404)
        try:
            return corr_svc.undo_correction(conn, rid, body.target_op_id)
        except ValueError as e:
            raise HTTPException(404, str(e))


@app.get("/api/runs/{rid}/corrections")
def list_corrections(rid: str) -> dict:
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM runs WHERE id=?", (rid,)).fetchone() is None:
            raise HTTPException(404)
        ops = corr_svc.list_corrections(conn, rid)
    return {"corrections": ops, "revision": ops[-1]["revision"] if ops else 0}


# ---------------- 派生 run ----------------

class DeriveIn(BaseModel):
    events: Optional[dict] = None          # 仅改事件参数 → 复用观测重算
    use_corrections: bool = False          # 使用人工球点修正 → assisted
    client_request_id: Optional[str] = None


@app.post("/api/runs/{rid}/derive")
def derive_run(rid: str, body: DeriveIn) -> JSONResponse:
    with db.connect() as conn:
        parent = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if parent is None:
            raise HTTPException(404)
        if parent["status"] != "succeeded":
            raise HTTPException(409, "父 run 未成功完成，无法派生")
        if not body.events and not body.use_corrections:
            raise HTTPException(422, "需要 events 参数或 use_corrections=true")
        mode = "assisted" if body.use_corrections else "automatic"
        cfg = db.uj(parent["config_json"]) or {}
        if body.events:
            cfg["events"] = {**(cfg.get("events") or {}), **body.events}
        if body.use_corrections:
            cfg["assisted_from_revision"] = corr_svc.current_revision(conn, rid)
        new_id = db.new_id()
        run_dir = config.projects_dir() / parent["project_id"] / "runs" / new_id
        conn.execute(
            "INSERT INTO runs (id, project_id, parent_run_id, mode, status, config_json, code_version, "
            "calibration_revision, segment_start_ms, segment_end_ms, cache_key, run_dir, "
            "reuses_frames_from, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (new_id, parent["project_id"], rid, mode, "queued", db.j(cfg), config.CODE_VERSION,
             parent["calibration_revision"], parent["segment_start_ms"], parent["segment_end_ms"],
             None, str(run_dir), rid, time.time()))
        job = db.create_job(conn, jtype="rederive", payload={"run_id": new_id,
                                                             "use_corrections": body.use_corrections},
                            project_id=parent["project_id"], run_id=new_id,
                            client_request_id=body.client_request_id)
    return JSONResponse({"run_id": new_id, "job_id": job["id"], "mode": mode}, status_code=202)


# ---------------- 独立真值 ----------------

class GTSetIn(BaseModel):
    name: str
    segment_start_ms: int
    segment_end_ms: int
    events: list[dict] = Field(default_factory=list)     # {type, source_time_ms, stroke_type, point}
    ball_points: list[dict] = Field(default_factory=list)  # {source_time_ms, point, unobservable}
    label_def: Optional[dict] = None


@app.post("/api/projects/{pid}/ground-truth-sets")
def create_gt_set(pid: str, body: GTSetIn) -> dict:
    import uuid
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM projects WHERE id=?", (pid,)).fetchone() is None:
            raise HTTPException(404)
        ver = conn.execute("SELECT COALESCE(MAX(version),0)+1 AS v FROM ground_truth_sets WHERE project_id=? AND name=?",
                           (pid, body.name)).fetchone()["v"]
        gid = uuid.uuid4().hex
        data = body.model_dump(exclude={"name"})
        data["note"] = "独立人工标注；默认隐藏自动结果后创建，不依赖自动结果"
        conn.execute("INSERT INTO ground_truth_sets (id, project_id, name, version, locked, data_json, created_at) "
                     "VALUES (?,?,?,?,1,?,?)",
                     (gid, pid, body.name, ver, db.j(data), time.time()))
    return {"id": gid, "name": body.name, "version": ver, "locked": True}


@app.get("/api/projects/{pid}/ground-truth-sets")
def list_gt_sets(pid: str) -> dict:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM ground_truth_sets WHERE project_id=? ORDER BY created_at DESC",
                            (pid,)).fetchall()
    return {"gt_sets": [{"id": r["id"], "name": r["name"], "version": r["version"],
                         "locked": bool(r["locked"]), "data": db.uj(r["data_json"]),
                         "created_at": r["created_at"]} for r in rows]}


class EvalIn(BaseModel):
    gt_set_id: str
    view: str = Field(default="automatic", pattern="^(automatic|reviewed)$")
    time_tolerance_ms: int = Field(default=100, ge=0)
    ball_position_tolerance_px: float = Field(default=40, gt=0)
    wall_unit_tolerance: float = Field(default=0.08, gt=0)


@app.post("/api/runs/{rid}/evaluations")
def create_evaluation(rid: str, body: EvalIn) -> dict:
    with db.connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if run is None:
            raise HTTPException(404)
        gt = conn.execute("SELECT * FROM ground_truth_sets WHERE id=?", (body.gt_set_id,)).fetchone()
        if gt is None:
            raise HTTPException(404, "真值集不存在")
        if not gt["locked"]:
            raise HTTPException(409, "真值集未锁定")
        if run["project_id"] != gt["project_id"]:
            raise HTTPException(422, "run 与真值集不属于同一视频")
        result = eval_svc.evaluate(conn, run, gt, body.model_dump())
        conn.execute("INSERT INTO evaluations (id, run_id, gt_set_id, params_json, result_json, created_at) "
                     "VALUES (?,?,?,?,?,?)",
                     (result["id"], rid, body.gt_set_id,
                      db.j({k: v for k, v in body.model_dump().items() if k != "gt_set_id"}),
                      db.j(result), time.time()))
    return result


@app.get("/api/runs/{rid}/evaluations")
def list_evaluations(rid: str) -> dict:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC",
                            (rid,)).fetchall()
    return {"evaluations": [db.uj(r["result_json"]) for r in rows]}


# ---------------- 导出 ----------------

class ExportIn(BaseModel):
    view: str = Field(default="automatic", pattern="^(automatic|reviewed)$")
    layers: Optional[dict] = None
    panel: bool = True
    client_request_id: Optional[str] = None


@app.post("/api/runs/{rid}/exports")
def create_export(rid: str, body: ExportIn) -> JSONResponse:
    import uuid
    with db.connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if run is None:
            raise HTTPException(404)
        if run["status"] != "succeeded":
            raise HTTPException(409, "run 未成功完成，无法导出")
        revision = corr_svc.current_revision(conn, rid)
        eid = uuid.uuid4().hex
        conn.execute("INSERT INTO exports (id, run_id, view, correction_revision, config_json, status, created_at) "
                     "VALUES (?,?,?,?,?,'queued',?)",
                     (eid, rid, body.view, revision,
                      db.j({"layers": body.layers, "panel": body.panel}), time.time()))
        job = db.create_job(conn, jtype="export", payload={"export_id": eid},
                            run_id=rid, export_id=eid, client_request_id=body.client_request_id)
    return JSONResponse({"export_id": eid, "job_id": job["id"],
                         "frozen_correction_revision": revision}, status_code=202)


@app.get("/api/exports/{eid}")
def get_export(eid: str) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM exports WHERE id=?", (eid,)).fetchone()
    if r is None:
        raise HTTPException(404)
    return {"id": r["id"], "run_id": r["run_id"], "view": r["view"],
            "correction_revision": r["correction_revision"], "status": r["status"],
            "error": r["error"], "artifacts": db.uj(r["artifacts_json"])}


@app.get("/api/exports/{eid}/files/{name}")
def export_file(eid: str, name: str, request: Request):
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM exports WHERE id=?", (eid,)).fetchone()
    if r is None:
        raise HTTPException(404)
    arts = (db.uj(r["artifacts_json"]) or {}).get("artifacts", [])
    match = [a for a in arts if a["file"] == name]
    if not match:
        raise HTTPException(404, "artifact 不存在")
    path = Path(r["dir"]) / name
    media = "video/mp4" if name.endswith(".mp4") else (
        "text/csv" if name.endswith(".csv") else "application/json")
    return _serve_range(path, request, media_type=media)


# ---------------- 环境（设置页）----------------

@app.get("/api/system/doctor")
def system_doctor() -> dict:
    import platform
    import shutil
    out = {"arch": platform.machine(), "macos": platform.mac_ver()[0],
           "chip": _chip(), "python": sys.version.split()[0],
           "ffmpeg_binary": bool(shutil.which("ffmpeg")),
           "disk_free_gb": round(_disk_free(), 1)}
    try:
        import torch
        out["mps_available"] = torch.backends.mps.is_available()
    except Exception as e:
        out["mps_available"] = False
        out["mps_error"] = str(e)[:200]
    out.update(_versions())
    out["models"] = models_status()
    return out


def _chip() -> str:
    try:
        import subprocess as sp
        return sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                      capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _disk_free() -> float:
    return os.statvfs(config.data_dir()).f_bavail * os.statvfs(config.data_dir()).f_frsize / (1 << 30)


# ---------------- 前端静态托管（正式本地模式）----------------

_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _dist.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/")
    def index():
        # index.html 不带内容哈希：必须每次回源校验，否则发版后浏览器
        # 启发式缓存会把旧页面+旧 bundle 一起续用（assets 有哈希可长缓存）
        return FileResponse(_dist / "index.html",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, f"未知 API 路径 /{full_path}")
        return FileResponse(_dist / "index.html",
                            headers={"Cache-Control": "no-cache"})
