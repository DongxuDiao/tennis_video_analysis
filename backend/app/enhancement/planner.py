"""异常计划（方案 §7.1）：从本地产物读出仍不确定的窗口，按影响排序。

只消费 expert 会话的已有 JSON（person_quality / quality_review / auto_summary /
shots / scene_quality）；不改数据。先本地复核（quality_review 已做），这里只挑
本地仍未解决的。
"""
from __future__ import annotations

import json
from pathlib import Path

from app.skill_runtime.policy import ANOMALY_PRIORITY, ALLOWED_QUESTIONS, Budget


def _read(p: Path, default=None):
    return json.loads(p.read_text()) if p.is_file() else default


def plan_anomalies(work: Path, budget: Budget) -> list[dict]:
    """返回 [{kind, question, frames:[{frame, pts_ms, crop_xyxy}], reason}]，
    按优先级排序、截断到预算窗口数。"""
    meta = _read(work / "meta.json") or {}
    fps = float(meta.get("fps") or 30)
    n_frames = int(meta.get("n_frames") or 0)
    width = int(meta.get("width") or 1920)
    height = int(meta.get("height") or 1080)
    if not n_frames:
        return []

    items: list[tuple[int, dict]] = []

    # 1) 疑似换人：person_quality suspicious_frames 聚类成窗
    pq = _read(work / "person_quality.json") or {}
    suspicious = sorted(int(f) for f in pq.get("suspicious_frames", [])
                        if isinstance(f, (int, float)))
    for window in _cluster_frames(suspicious, gap=int(fps * 1.5)):
        items.append((ANOMALY_PRIORITY["person_switch_suspect"], {
            "kind": "person_switch_suspect",
            "frames": _sample_frames(window, fps),
            "reason": f"人物质量可疑帧 {window[0]}–{window[-1]}（共{len(window)}帧）",
        }))

    # 2) 触球归属冲突 / 正反手冲突（auto_summary 与 shots）
    shots = _read(work / "shots.json") or []
    for s in shots:
        f = int(round(float(s.get("tc", 0))))
        if s.get("attribution") == "ambiguous":
            items.append((ANOMALY_PRIORITY["contact_attribution_conflict"], {
                "kind": "contact_attribution_conflict",
                "frames": _window_around(f, fps),
                "reason": f"第{s.get('id')}拍归属冲突（{f}帧）",
                "shot_id": s.get("id"),
            }))
        if s.get("side_conflict"):
            items.append((ANOMALY_PRIORITY["side_conflict"], {
                "kind": "side_conflict",
                "frames": _window_around(f, fps),
                "reason": f"第{s.get('id')}拍正反手证据冲突（{f}帧）",
                "shot_id": s.get("id"),
            }))

    # 3) 触球附近球轨迹断裂：quality_review 窗口与触球时刻重叠且仍缺观测
    qr = _read(work / "quality_review.json") or {}
    for w in qr.get("windows", []):
        a, b = int(w.get("start", 0)), int(w.get("end", 0))
        near_contact = any(abs(int(round(float(s.get("tc", -999))) - (a + b) / 2)) <= fps
                           for s in shots)
        if near_contact:
            frames = list(range(a, b + 1))
            if frames:
                items.append((ANOMALY_PRIORITY["ball_gap_near_contact"], {
                    "kind": "ball_gap_near_contact",
                    "frames": _sample_frames(frames, fps, max_n=int(fps * 0.7)),
                    "reason": f"触球附近观测缺口 {a}–{b}",
                    "roi": w.get("roi"),
                }))

    # 4) 场地线身份待确认 / 机位不可靠（仅当存在线检测结果）
    scene = _read(work / "scene_quality.json") or {}
    if scene.get("status") in ("unreliable_camera_motion",):
        items.append((ANOMALY_PRIORITY["scene_unreliable"], {
            "kind": "scene_unreliable",
            "frames": _window_around(int(n_frames * 0.3), fps),
            "reason": "机位运动校验未通过",
        }))

    # 同类同时段合并；排序；截断
    items.sort(key=lambda t: t[0])
    merged: list[dict] = []
    for pri, item in items:
        q = ALLOWED_QUESTIONS.get(item["kind"])
        if q is None:
            continue
        item["question"] = q
        if merged and merged[-1]["kind"] == item["kind"] and \
           abs(item["frames"][0]["frame"] - merged[-1]["frames"][-1]["frame"]) <= fps * 2:
            merged[-1]["frames"] = _sample_frames(
                sorted({f["frame"] for f in merged[-1]["frames"]} |
                       {f["frame"] for f in item["frames"]}), fps)
            merged[-1]["reason"] += "；" + item["reason"]
            continue
        merged.append(item)
    return merged[:budget.max_windows]


def _cluster_frames(frames: list[int], gap: int) -> list[list[int]]:
    out: list[list[int]] = []
    for f in frames:
        if out and f - out[-1][-1] <= gap:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def _frame_meta(f: int, fps: float) -> dict:
    return {"frame": int(f), "pts_ms": round(f / fps * 1000, 2)}


def _window_around(f: int, fps: float, span: float = 1.0) -> list[dict]:
    """约 1s 窗口；锚点帧必选（触球时刻是问题中心，不能被均匀抽样丢掉）。"""
    half = max(2, int(fps * span / 2))
    frames = list(range(max(0, f - half), f + half + 1))
    out = _sample_frames(frames, fps)
    if out and all(fr["frame"] != f for fr in out):
        out.append(_frame_meta(f, fps))
        out.sort(key=lambda d: d["frame"])
    return out


def _sample_frames(frames: list[int], fps: float, max_n: int = 6) -> list[dict]:
    if not frames:
        return []
    if len(frames) <= max_n:
        return [_frame_meta(f, fps) for f in frames]
    step = (len(frames) - 1) / (max_n - 1)
    picked = sorted({frames[int(round(i * step))] for i in range(max_n)})
    return [_frame_meta(f, fps) for f in picked]
