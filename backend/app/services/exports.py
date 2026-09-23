"""导出构建：冻结 revision → 渲染 MP4 + CSV + JSON → 受控 artifact。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from app import config
from app.store import db
from app.store.db import CancelledError, check_cancel, update_job_progress
from app.services import corrections as corr_svc
from app.vision.geometry import Roi, WallQuad
from app.vision.pipeline import load_frames_window
from app.video.render import render_export_mp4, write_events_csv, write_export_json


def build_export(export_id: str, conn, job: dict) -> dict:
    exp = conn.execute("SELECT * FROM exports WHERE id=?", (export_id,)).fetchone()
    if exp is None:
        raise RuntimeError("export 不存在")
    run = conn.execute("SELECT * FROM runs WHERE id=?", (exp["run_id"],)).fetchone()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (run["project_id"],)).fetchone()
    run_dir = Path(run["run_dir"])
    out_dir = config.exports_root() / export_id
    out_dir.mkdir(parents=True, exist_ok=True)
    conn.execute("UPDATE exports SET status='running', dir=? WHERE id=?", (str(out_dir), export_id))
    conn.commit()
    update_job_progress(conn, job["id"], stage="exporting")

    view = exp["view"]  # automatic | reviewed
    cfg = json.loads(run["config_json"])
    exp_cfg = json.loads(exp["config_json"] or "{}")
    default_layers = {"skeleton": True, "ball": True, "candidates": False,
                      "racket": True, "events": True, "wall": True, "roi": False}
    # JSON requests may explicitly contain layers=null.  Treat that the same as
    # omitting the field rather than passing None into the renderer.
    layers = {**default_layers, **(exp_cfg.get("layers") or {})}
    panel = exp_cfg.get("panel", True)

    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))["events"]
    revision = int(exp["correction_revision"] or 0)
    if view == "reviewed":
        ops = [o for o in corr_svc.list_corrections(conn, run["id"])
               if o["revision"] <= revision]
        events = corr_svc.apply_ops(events, ops)
        events = [e for e in events if e.get("review_status") != "excluded"]

    wall = WallQuad.from_dict(cfg.get("wall"))
    roi = Roi.from_dict(cfg.get("ball_search_roi"),
                        **({"img_w": 10**9, "img_h": 10**9})).as_dict() if cfg.get("ball_search_roi") else None

    frames = load_frames_window(run_dir / "frames",
                                int(run["segment_start_ms"]), int(run["segment_end_ms"]))
    frames_by_index = {f["frame_index"]: f for f in frames}

    # 摘要面板（速度单位 px/s，不出现 km/h）
    hits = [e for e in events if e["type"] == "player_hit"]
    strokes = {"forehand": 0, "backhand": 0, "unknown": 0}
    for e in hits:
        strokes[e.get("stroke_type", "unknown")] = strokes.get(e.get("stroke_type", "unknown"), 0) + 1
    panel_lines = [
        f"view={view}  revision={revision}  origin={run['mode']}",
        f"hits={len(hits)}  forehand={strokes['forehand']}  backhand={strokes['backhand']}  unknown={strokes['unknown']}",
        "ball speed unit: px/s (image plane, same-camera reference only)",
        "physical km/h unavailable: no validated 3D measurement",
    ]
    if panel is False:
        panel_lines = None

    def should_continue() -> bool:
        return not db.is_cancel_requested(conn, job["id"])

    mp4_path = out_dir / f"export_{view}_rev{revision}.mp4"
    render_export_mp4(
        project=dict(proj), run=dict(run), frames_by_index=frames_by_index, events=events,
        wall=wall, roi=roi, out_path=mp4_path, layers=layers, panel_lines=panel_lines,
        source_path=proj["original_path"], start_ms=int(run["segment_start_ms"]),
        end_ms=int(run["segment_end_ms"]), rotation=json.loads(proj["probe_json"] or "{}").get("rotation", 0),
        should_continue=should_continue,
        progress=lambda d, t: update_job_progress(conn, job["id"], stage="exporting",
                                                  progress={"frames_done": d, "frames_total_est": t}))

    csv_path = out_dir / f"events_{view}_rev{revision}.csv"
    write_events_csv(events, csv_path)

    json_path = out_dir / f"export_{view}_rev{revision}.json"
    payload = {
        "export": {"id": export_id, "view": view, "correction_revision": revision,
                   "created_at": time.time()},
        "run": {"id": run["id"], "project_id": run["project_id"], "mode": run["mode"],
                "segment_start_ms": run["segment_start_ms"], "segment_end_ms": run["segment_end_ms"],
                "config": cfg, "effective_config": db.uj(run["effective_config_json"]),
                "model_hash": run["model_hash"], "code_version": run["code_version"],
                "calibration_revision": run["calibration_revision"],
                "performance": db.uj(run["performance_json"])},
        "project": {"id": proj["id"], "filename": proj["original_filename"],
                    "video_hash": proj["video_hash"],
                    "probe": db.uj(proj["probe_json"])},
        "events": events,
        "summary": db.uj(run["summary_json"]),
        "note": "坐标空间：analysis_image_px（旋转修正后）；速度单位 px/s；km/h 为 null 且给出原因",
    }
    write_export_json(json_path, payload)

    artifacts = []
    for p in (mp4_path, csv_path, json_path):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        artifacts.append({"file": p.name, "bytes": p.stat().st_size, "sha256": h})
    conn.execute("UPDATE exports SET status='succeeded', artifacts_json=? WHERE id=?",
                 (db.j({"artifacts": artifacts, "dir": str(out_dir)}), export_id))
    return {"export_id": export_id, "artifacts": artifacts}
