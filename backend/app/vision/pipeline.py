"""分析流水线：流式解码 → YOLO 球/拍检测 → MediaPipe 姿态 → 卡尔曼追踪 → 分块落盘。

内存约束：逐帧处理，帧观测按块（默认 300 帧/块）写 JSONL；落盘为崩溃恢复
与 API 读取服务。事件阶段需要的观测同时保留在内存（原本写完盘再全量读回，
内存峰值相同但多付一次 IO+解析）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from app import config
from app.store import db
from app.store.db import CancelledError, check_cancel, heartbeat, update_job_progress
from app.vision.detector import Detector
from app.vision.events import compute_phase_segments, extract_events
from app.vision.geometry import Roi, WallQuad
from app.vision.pose import PoseEstimator, PoseResult
from app.vision.tracker import BallTracker
from app.vision.person_tracker import PersonTracker
from app.video.decode import iter_frames

FRAMES_PER_CHUNK = 300


def run_dir_for(project_id: str, run_id: str) -> Path:
    d = config.projects_dir() / project_id / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames").mkdir(exist_ok=True)
    return d


def _sanitize(o: Any) -> Any:
    """NaN/Inf → None，保证 JSON 合法（任务书 §7.1）。"""
    import math
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, tuple):
        return [_sanitize(v) for v in o]
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    return o


class FrameChunkWriter:
    def __init__(self, frames_dir: Path, per_chunk: int = FRAMES_PER_CHUNK):
        self.dir = frames_dir
        self.per_chunk = per_chunk
        self.chunk_index = 0
        self.buffer: list[dict] = []
        self.index: list[dict] = []

    def add(self, rec: dict) -> dict:
        rec = _sanitize(rec)
        self.buffer.append(rec)
        if len(self.buffer) >= self.per_chunk:
            self.flush()
        return rec

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.dir / f"chunk_{self.chunk_index:05d}.jsonl"
        tmp = path.with_suffix(".tmp")
        first, last = self.buffer[0], self.buffer[-1]
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self.buffer:
                f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, path)
        self.index.append({
            "chunk": self.chunk_index, "file": path.name,
            "frame_start": first["frame_index"], "frame_end": last["frame_index"],
            "pts_start_ms": first["source_pts_ms"], "pts_end_ms": last["source_pts_ms"],
            "count": len(self.buffer),
        })
        self.chunk_index += 1
        self.buffer = []

    def finalize(self) -> list[dict]:
        self.flush()
        idx_path = self.dir / "index.json"
        tmp = idx_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.index}, f, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, idx_path)
        return self.index


def load_frames_window(frames_dir: Path, start_ms: int | None = None,
                       end_ms: int | None = None) -> list[dict]:
    """按源 PTS 时间窗读取帧观测（供 events 重算 / observations API / 前端）。"""
    idx_path = frames_dir / "index.json"
    if not idx_path.exists():
        return []
    index = json.loads(idx_path.read_text(encoding="utf-8"))["chunks"]
    out: list[dict] = []
    for ch in index:
        if start_ms is not None and ch["pts_end_ms"] < start_ms:
            continue
        if end_ms is not None and ch["pts_start_ms"] > end_ms:
            continue
        with open(frames_dir / ch["file"], "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = rec["source_pts_ms"]
                if start_ms is not None and t < start_ms:
                    continue
                if end_ms is not None and t > end_ms:
                    continue
                out.append(rec)
    return out


def summarize(frames: list[dict], events: list[dict]) -> dict:
    total = len(frames)
    selected = [((f.get("ball") or {}).get("selected") or {}) for f in frames]
    observed = sum(1 for s in selected if s.get("raw"))
    predicted = sum(1 for s in selected if s.get("status") == "predicted")
    pose_frames = sum(1 for f in frames if (f.get("pose") or {}).get("landmarks"))
    person_frames = sum(1 for f in frames if ((f.get("detections") or {}).get("person") or []))
    # 最长观测缺口（帧间 pts 差的最大值，仅统计同 track 断档）
    longest_gap_ms = 0
    last_obs_t, last_track = None, None
    for f in frames:
        sel = (f.get("ball") or {}).get("selected") or {}
        if sel.get("raw"):
            t, tid = f["source_pts_ms"], sel.get("track_id")
            if last_obs_t is not None and tid == last_track:
                longest_gap_ms = max(longest_gap_ms, t - last_obs_t)
            last_obs_t, last_track = t, tid
    hits = [e for e in events if e["type"] == "player_hit"]
    wallc = [e for e in events if e["type"] == "wall_contact_candidate"]
    # 人物目标状态统计（单目标；丢失不切换）。
    # 人物框按 person_stride_ms 步进采样：最近一次成功匹配在 500ms 内即视为覆盖，
    # 否则步进间隙会被误计为丢失。
    person_status: dict[str, int] = {}
    person_reacquired = 0
    covered_frames = 0
    last_match_ms: int | None = None
    last_matched = False
    for f in frames:
        st = ((f.get("person") or {}).get("status")) or "none"
        person_status[st] = person_status.get(st, 0) + 1
        matched = st in ("tracked", "reacquired")
        if matched:
            if not last_matched and last_match_ms is not None:
                person_reacquired += 1
            last_match_ms = f["source_pts_ms"]
        if last_match_ms is not None and f["source_pts_ms"] - last_match_ms <= 500:
            covered_frames += 1
        last_matched = matched
    return {
        "person_tracking": {
            "status_counts": person_status,
            "target_tracked_coverage": round(covered_frames / total, 4) if total else None,
            "reacquire_events": person_reacquired,
            "sampling_note": "人物框为步进采样（person_stride_ms）；覆盖=最近匹配≤500ms",
            "note": "单目标；丢失期间不切换到其他人物框",
        },
        "frames_total": total,
        "frames_ball_observed": observed,
        "frames_ball_predicted_only": predicted,
        "ball_observation_coverage": round(observed / total, 4) if total else None,
        "pose_frames": pose_frames,
        "pose_coverage": round(pose_frames / total, 4) if total else None,
        "person_detection_frames": person_frames,
        "person_detection_coverage": round(person_frames / total, 4) if total else None,
        "longest_observation_gap_ms": longest_gap_ms,
        "counts": {
            "player_hit": len(hits),
            "swing_candidate": sum(1 for e in events if e["type"] == "swing_candidate"),
            "wall_contact_candidate": len(wallc),
            "bounce_candidate": sum(1 for e in events if e["type"] == "bounce_candidate"),
        },
        "stroke_counts_auto": {
            "forehand": sum(1 for e in hits if e["stroke_type"] == "forehand"),
            "backhand": sum(1 for e in hits if e["stroke_type"] == "backhand"),
            "unknown": sum(1 for e in hits if e["stroke_type"] == "unknown"),
        },
        "phase_segments": compute_phase_segments(events),
        "note": "候选数/自动数≠准确率；coverage 是检测覆盖率，不是精度",
    }


def execute_analysis(*, job: dict, run: dict, conn_factory=db.connect) -> dict:
    """worker 调用：执行一次完整分析（人物追踪 + 检测模式调度 + 姿态 + 追踪 + 事件）。"""
    project_id, run_id = run["project_id"], run["id"]
    run_cfg = json.loads(run["config_json"])
    with conn_factory() as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if proj is None:
        raise RuntimeError("项目不存在")
    probe = json.loads(proj["probe_json"] or "{}")

    model_cfg = {**config.cfg()["model"], **(run_cfg.get("model") or {})}
    proc_cfg = {**config.cfg()["processing"], **(run_cfg.get("processing") or {})}
    track_cfg = {**config.cfg()["tracking"], **(run_cfg.get("tracking") or {})}
    ev_cfg = {**config.cfg()["events"], **(run_cfg.get("events") or {})}
    measurements = {**config.cfg()["measurements"], **(run_cfg.get("measurements") or {})}

    det_model = config.model_path_for(model_cfg["detector"])
    pose_model = config.model_path_for("pose_landmarker_full")
    missing = [str(p) for p in (det_model, pose_model) if p is None or not Path(p).exists()]
    if missing:
        raise RuntimeError(f"模型文件缺失：{missing}；请先在设置页下载官方模型")

    analysis_w, analysis_h = probe.get("rotation", 0) in (90, 270) and (probe["height"], probe["width"]) or (probe["width"], probe["height"])
    roi = Roi.from_dict(run_cfg.get("ball_search_roi"), analysis_w, analysis_h)
    wall = WallQuad.from_dict(run_cfg.get("wall"))
    handedness = run_cfg.get("handedness", "unknown")
    start_ms, end_ms = int(run["segment_start_ms"]), int(run["segment_end_ms"])

    detector = Detector(str(det_model), device=model_cfg["device"], imgsz=int(model_cfg["imgsz"]),
                        ball_confidence=float(model_cfg["ball_confidence"]),
                        racket_confidence=float(model_cfg.get("racket_confidence", 0.25)),
                        person_confidence=float(model_cfg.get("person_confidence", 0.35)))
    pose = PoseEstimator(str(pose_model))
    pose.reset()
    tracker = BallTracker(max_prediction_gap_ms=int(track_cfg["max_prediction_gap_ms"]),
                          association_max_dist_px=float(track_cfg["association_max_dist_px"]),
                          ambiguous_dist_px=float(track_cfg["ambiguous_dist_px"]),
                          ball_min_size_px=float(model_cfg.get("ball_min_size_px", 4)),
                          ball_max_size_px=float(model_cfg.get("ball_max_size_px", 160)),
                          base_search_radius_px=float(track_cfg.get("base_search_radius_px", 220)),
                          radius_per_lost_ms=float(track_cfg.get("radius_per_lost_ms", 0.8)))
    person_tracker = PersonTracker(target_hint=run_cfg.get("target_person_hint"))

    # 检测模式：full_frame（基线整图）| tiled（分块整图）| tiled_local（分块发现 + 局部跟踪）
    detection_mode = proc_cfg.get("detection_mode", "full_frame")
    temporal = None
    if detection_mode == "tracknet":
        from app.vision.tracknet import TrackNetDetector
        temporal = TrackNetDetector(config.models_dir() / "tracknet_tennis.pt", model_cfg["device"])
    tiles_cfg = proc_cfg.get("tiles", {"cols": 3, "rows": 2, "overlap": 0.25})
    full_search_interval_ms = float(proc_cfg.get("full_search_interval_ms", 300))
    discovery_interval_ms = float(proc_cfg.get("discovery_interval_ms", 150))
    person_stride_ms = float(proc_cfg.get("person_stride_ms", 300))

    run_dir = run_dir_for(project_id, run_id)
    writer = FrameChunkWriter(run_dir / "frames")

    perf = {"stage_ms": {"detector_full": 0.0, "detector_local": 0.0, "pose": 0.0,
                         "decode": 0.0, "io": 0.0},
            "devices": {}, "attempts_first": None, "frames_processed": 0,
            "detection_mode": detection_mode, "search_mode_counts": {}}
    t_stage0 = time.perf_counter()
    frames_counted = 0
    last_pose_ms: int | None = None
    pose_interval_ms = float(proc_cfg["pose_interval_ms"])
    ball_stride = max(1, int(proc_cfg.get("ball_stride", 1)))
    if temporal is not None:
        ball_stride = 1
    first_attempt_logged = False
    effective_device = None
    last_person_det_ms: int | None = None

    # 帧观测（sanitize 后）随写盘同时留在内存：事件阶段直接消费，
    # 免去刚写完的 JSONL 全量读回 + json.loads（多付一次全量 IO 与解析，
    # 且记录含 rejected[].score_detail，体积可观）
    mem_frames: list[dict] = []

    _cancel_state = {"counter": 0, "cancelled": False}

    def should_continue() -> bool:
        # 每帧开 SQLite 连接开销过大；改为每 15 帧（约 250ms@60fps）查一次取消标记，
        # 取消确认仍远低于 2s 要求。已置 cancelled 后立即返回。
        if _cancel_state["cancelled"]:
            return False
        _cancel_state["counter"] += 1
        if _cancel_state["counter"] % 15 == 1:
            with conn_factory() as conn:
                if db.is_cancel_requested(conn, job["id"]):
                    _cancel_state["cancelled"] = True
                    return False
        return True

    def run_detection(img: np.ndarray, mode_note: str):
        """按检测模式调度一次推理。返回 (DetectionResult, search_mode, search_crop)。"""
        nonlocal first_attempt_logged, effective_device
        det = None
        if temporal is not None:
            det = temporal.detect(img, img_pts_ms_holder["t"])
            smode, crop = "tracknet", None
        elif detection_mode == "full_frame":
            det = detector.detect(img, roi)
            smode, crop = "full_frame", None
        elif detection_mode == "tiled":
            det = detector.detect_tiled(img, cols=int(tiles_cfg.get("cols", 3)),
                                        rows=int(tiles_cfg.get("rows", 2)),
                                        overlap=float(tiles_cfg.get("overlap", 0.25)))
            smode, crop = "tiled", None
        else:  # tiled_local
            need_ball_full = tracker.wants_full_search(img_pts_ms_holder["t"], full_search_interval_ms,
                                                        discovery_interval_ms)
            sugg = tracker.suggest_search(img_pts_ms_holder["t"])
            if need_ball_full:
                det = detector.detect_tiled(img, cols=int(tiles_cfg.get("cols", 3)),
                                            rows=int(tiles_cfg.get("rows", 2)),
                                            overlap=float(tiles_cfg.get("overlap", 0.25)))
                tracker.note_full_search_done(img_pts_ms_holder["t"])
                smode, crop = "tiled", None
            elif sugg is not None:
                (cx, cy), radius = sugg
                det = detector.detect_local(img, (cx, cy), radius)
                smode, crop = "local", det.search_crop
            else:
                # 无轨迹且未到发现周期：本帧不做球检测（人物框仍由 stride pass 提供）
                smode, crop = "idle", None
        if det is not None:
            if not first_attempt_logged:
                perf["attempts_first"] = det.attempts
                first_attempt_logged = True
            effective_device = det.device
            perf["devices"][det.device] = perf["devices"].get(det.device, 0) + 1
            key = "detector_full" if smode != "local" else "detector_local"
            perf["stage_ms"][key] += det.elapsed_ms
        perf["search_mode_counts"][smode] = perf["search_mode_counts"].get(smode, 0) + 1
        _ = mode_note
        return det, smode, crop

    img_pts_ms_holder: dict = {"t": 0}
    last_hb = time.time()
    t0 = time.perf_counter()
    try:
        for fr in iter_frames(proj["original_path"], start_ms=start_ms, end_ms=end_ms,
                              rotation=probe.get("rotation", 0), should_continue=should_continue):
            img_pts_ms_holder["t"] = fr.pts_ms
            if time.time() - last_hb > 2.0:
                with conn_factory() as conn:
                    heartbeat(job["id"], job.get("worker_id") or "")
                    update_job_progress(conn, job["id"], stage="inference",
                                        progress={"frames_done": frames_counted,
                                                  "frames_total_est": int((end_ms - start_ms) / 1000 * (probe.get("nominal_fps") or 30)),
                                                  "device": effective_device,
                                                  "search_mode_counts": perf["search_mode_counts"]})
                last_hb = time.time()
            rec: dict[str, Any] = {"frame_index": fr.frame_index, "source_pts_ms": fr.pts_ms}
            t_img = time.perf_counter()

            # ---- 球/拍/人物检测（按模式调度；local 模式补廉价人物 pass）----
            det = None
            smode, crop = "full_frame", None
            person_det = None
            if frames_counted % ball_stride == 0:
                det, smode, crop = run_detection(fr.image, "")
            # 人物框来源：只有整图模式可直接取；分块会把大目标切碎，
            # 因此 tiled/tiled_local 一律走全图 640 廉价人物 pass（按 person_stride_ms 采样）
            if detection_mode == "full_frame":
                if det is not None:
                    last_person_det_ms = fr.pts_ms
            elif last_person_det_ms is None or fr.pts_ms - last_person_det_ms >= person_stride_ms:
                person_det = detector.detect(fr.image, None, mode="person_stride", imgsz=640)
                last_person_det_ms = fr.pts_ms
                perf["stage_ms"]["detector_full"] += person_det.elapsed_ms
                perf["devices"][person_det.device] = perf["devices"].get(person_det.device, 0) + 1

            balls = detector.ball_candidates(det) if det else []
            rackets = detector.racket_candidates(det) if det else []
            if person_det is not None:
                rackets = rackets + detector.racket_candidates(person_det)
            persons = ((detector.person_candidates(det) if det else [])
                       if detection_mode == "full_frame" else []) + \
                (detector.person_candidates(person_det) if person_det else [])

            # ---- 人物身份关联（单目标）----
            person_boxes = [c.bbox for c in persons]
            person_state = person_tracker.step(fr.pts_ms, person_boxes,
                                               [c.confidence for c in persons],
                                               sampled=(det is not None if detection_mode == "full_frame" else person_det is not None))

            # ---- 球追踪 ----
            tp = tracker.step(fr.frame_index, fr.pts_ms,
                              [{"bbox": c.bbox, "center": c.center, "confidence": c.confidence} for c in balls],
                              search_mode=smode, search_crop=crop)

            # ---- 姿态（按真实时间间隔；目标追踪时在带边距目标区域上检测）----
            pose_rec = None
            if last_pose_ms is None or (fr.pts_ms - last_pose_ms) >= pose_interval_ms:
                t_pose0 = time.perf_counter()
                if person_state.get("status") in ("tracked", "reacquired", "predicted") and person_state.get("box"):
                    b = person_state["box"]
                    crop_box = person_tracker.crop_with_margin(analysis_w, analysis_h, box=person_state["box"])
                    pres = pose.detect_on_crop(fr.image, crop_box, fr.pts_ms)
                    pose_source = "target_crop"
                else:
                    pres = PoseResult()
                    pose_source = "target_unavailable"
                perf["stage_ms"]["pose"] += (time.perf_counter() - t_pose0) * 1000
                last_pose_ms = fr.pts_ms
                pose_rec = {"landmarks": pres.landmarks if pres.person_found else None,
                            "person_found": pres.person_found,
                            "elapsed_ms": round((time.perf_counter() - t_pose0) * 1000, 2),
                            "source": pose_source,
                            "person_track_id": person_state.get("track_id"),
                            "crop_box": list(pres.crop_box) if pres.crop_box else None}

            # ---- 人体轨迹参考点（优先姿态髋部中心，退回框底中心并标注来源）----
            hip = None
            hip_source = None
            if pose_rec and pose_rec.get("landmarks"):
                lh, rh = pose_rec["landmarks"][23], pose_rec["landmarks"][24]
                if lh.get("visibility", 0) > 0.5 and rh.get("visibility", 0) > 0.5:
                    hip = [(lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2]
                    hip_source = "pose_hip_center"
            if hip is None and person_state.get("box"):
                b = person_state["box"]
                hip = [(b[0] + b[2]) / 2, b[3]]
                hip_source = "box_bottom_center"

            rec["person"] = {
                "boxes": [list(b) for b in person_boxes],
                "target": (None if not person_state.get("box") else {
                    "box": list(person_state["box"]),
                    "track_id": person_state.get("track_id"),
                    "status": person_state.get("status"),
                    "matched": person_state.get("matched"),
                }),
                "status": person_state.get("status"),
                "track_id": person_state.get("track_id"),
                "hip_center": ({"point": hip, "source": hip_source} if hip else None),
            }
            rec["ball"] = {
                "candidates": [{"bbox": list(c.bbox), "confidence": c.confidence, "cls": c.cls_name, "source": c.source} for c in balls],
                "selected": ({"point": list(tp.filtered), "raw": tp.raw is not None,
                              "status": tp.status, "track_id": tracker.current_track_id,
                              "tracking_score": tp.tracking_score,
                              "candidate_conf": tp.candidate_conf,
                              "search_mode": tp.search_mode,
                              "search_crop": list(tp.search_crop) if tp.search_crop else None}
                             if tp.status != "lost" else None),
                "rejected": tp.rejected,
            }
            rec["detections"] = {
                "racket": [list(c.bbox) for c in rackets],
                "person": [list(c.bbox) for c in persons],
            }
            if pose_rec is not None:
                rec["pose"] = pose_rec
            rec["timings"] = {"detector_ms": (det.elapsed_ms if det else None),
                              "pose_ms": (pose_rec["elapsed_ms"] if pose_rec else None)}
            mem_frames.append(writer.add(rec))
            frames_counted += 1
            det_ms = (det.elapsed_ms if det else 0)
            person_ms_used = (person_det.elapsed_ms if person_det is not None else 0)
            pose_ms_used = (pose_rec["elapsed_ms"] if pose_rec else 0)
            # 人物 stride pass 的时间已计入 detector_full；此处同样扣除，
            # 否则残差会被重复算进 decode 桶，误导性能归因
            perf["stage_ms"]["decode"] += (time.perf_counter() - t_img) * 1000 \
                - det_ms - person_ms_used - pose_ms_used

        t_loop_end = time.perf_counter()
        with conn_factory() as conn:
            update_job_progress(conn, job["id"], stage="events", progress={"frames_done": frames_counted})
        writer.finalize()
        t_writer_done = time.perf_counter()
        frames = mem_frames
        # 挥拍时间窗（P4）与事件
        from app.vision.actions import detect_swings
        swing_thr = float(ev_cfg.get("swing_wrist_speed_px_s", 600))
        swing_segs = detect_swings(frames, handedness=handedness, min_speed_px_s=swing_thr)
        swings_payload = [seg.to_dict() for seg in swing_segs]
        swings_path = run_dir / "swings.json"
        tmp = swings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"swings": swings_payload}, ensure_ascii=False,
                                  allow_nan=False), encoding="utf-8")
        os.replace(tmp, swings_path)
        events = extract_events(frames, run_id=run_id, handedness=handedness, wall=wall,
                                cfg_events=ev_cfg, measurements=measurements,
                                analysis_height=analysis_h, swings=swings_payload)
        events_path = run_dir / "events.json"
        tmp = events_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"events": events, "automatic_immutable": True},
                                  ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(tmp, events_path)
        summary = summarize(frames, events)

        t_now = time.perf_counter()
        perf["stage_ms"]["writer_finalize"] = round((t_writer_done - t_loop_end) * 1000, 1)
        perf["stage_ms"]["events_stage"] = round((t_now - t_writer_done) * 1000, 1)
        perf["stage_ms"]["io"] = round((t_now - t_stage0) * 1000
                                       - sum(v for k, v in perf["stage_ms"].items()
                                             if k not in ("io",)), 1)
        perf["stage_ms"] = {k: round(v, 1) for k, v in perf["stage_ms"].items()}
        perf["frames_processed"] = frames_counted
        perf["wall_clock_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        perf["segment_duration_ms"] = end_ms - start_ms
        perf["effective_device"] = effective_device
        perf["device_policy"] = model_cfg["device"]
        perf["model_hash"] = detector.model_hash
        perf["tracknet_model_hash"] = temporal.model_hash if temporal else None
        perf["pose_model_hash"] = pose.model_hash
        # 峰值内存（本进程 RSS，兆字节；psutil 不在依赖内）
        try:
            import resource
            raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            import sys as _sys
            if _sys.platform == "darwin":
                perf["peak_rss_mb"] = round(raw_rss / (1024.0 * 1024.0), 1)  # macOS: bytes
            else:
                perf["peak_rss_mb"] = round(raw_rss / 1024.0, 1)  # Linux: KB
        except Exception:
            pass
        summary["performance_note"] = "耗时为该次运行实测，不外推"
        summary["swing_segments"] = len(swings_payload)
        return {"performance": perf, "summary": summary, "effective_config": {
            "model": {k: model_cfg[k] for k in ("detector", "device", "imgsz", "ball_confidence") if k in model_cfg},
            "processing": {"ball_stride": ball_stride, "pose_interval_ms": pose_interval_ms,
                           "detection_mode": detection_mode, "tiles": tiles_cfg,
                           "full_search_interval_ms": full_search_interval_ms,
                           "discovery_interval_ms": discovery_interval_ms,
                           "person_stride_ms": person_stride_ms},
            "tracking": track_cfg, "events": ev_cfg,
            "ball_search_roi": roi.as_dict(), "handedness": handedness,
            "target_person_hint": run_cfg.get("target_person_hint"),
            "wall": run_cfg.get("wall"), "measurements": measurements,
        }, "analysis_dims": {"w": analysis_w, "h": analysis_h}, "events_count": len(events)}
    finally:
        pose.close()
        if tracker:
            tracker.finish()
