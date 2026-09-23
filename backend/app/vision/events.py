"""事件提取（第二阶段，离线）：仅基于已落盘的观测时间序列，不重跑模型。

事件类型：
- player_hit              击球候选（球观测 + 拍/腕邻近 + 方向变化）
- swing_candidate         有挥拍证据但无球邻近，不计入确认击球
- wall_contact_candidate  撞墙候选（观测轨迹进入墙面区域且法向速度反转）
- bounce_candidate        落地反弹候选（默认排除在击球计数外）

所有事件带 evidence 明细；heuristic_score 是规则分数，不是准确率。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.vision.geometry import WallQuad, distance
from app.vision.pose import LM, PoseEstimator

EVENT_TYPES = ("player_hit", "swing_candidate", "wall_contact_candidate", "bounce_candidate")


@dataclass
class ExtractedEvent:
    source_time_ms: int
    frame_index: int
    type: str
    stroke_type: str = "unknown"
    point: tuple[float, float] | None = None
    heuristic_score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    speed_px_s: float | None = None
    track_id: int | None = None

    def to_dict(self, run_id: str, seq: int) -> dict:
        d = {
            "id": f"{run_id[:8]}-ev{seq:04d}",
            "run_id": run_id,
            "source_time_ms": self.source_time_ms,
            "frame_index": self.frame_index,
            "type": self.type,
            "stroke_type": self.stroke_type,
            "origin": "automatic",
            "review_status": "unreviewed",
            "heuristic_score": round(self.heuristic_score, 3),
            "evidence": self.evidence,
            "point": ({"x": round(self.point[0], 1), "y": round(self.point[1], 1),
                       "space": "analysis_image_px"} if self.point else None),
            "speed_px_s": (round(self.speed_px_s, 1) if self.speed_px_s is not None else None),
            "speed_kmh": None,
            "unavailable_reason": "no_validated_3d_measurement",
            "track_id": self.track_id,
        }
        return d


def _vel(p0: tuple[float, float], p1: tuple[float, float], dt_s: float) -> tuple[float, float]:
    return (p1[0] - p0[0]) / dt_s, (p1[1] - p0[1]) / dt_s


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(c))


def _pose_series(frames: list[dict]) -> tuple[list[dict], list[int]]:
    """含姿态的帧及其时间戳（升序）。

    事件阶段会按时间邻近反复查姿态帧（每个球观测/挥拍候选一次），
    预计算一次，避免逐候选重新全帧过滤。"""
    pf = [f for f in frames if (f.get("pose") or {}).get("landmarks")]
    return pf, [f["source_pts_ms"] for f in pf]


def _nearest_pose_frame(pose_frames: list[dict], pose_times: list[int], t_ms: int,
                        window_ms: int = 120) -> dict | None:
    if not pose_frames:
        return None
    import bisect
    i = bisect.bisect_left(pose_times, t_ms)
    best, bd = None, None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(pose_frames):
            d = abs(pose_times[j] - t_ms)
            if d <= window_ms and (bd is None or d < bd):
                best, bd = pose_frames[j], d
    return best


def _wrist_series(frames: list[dict], handedness: str) -> list[tuple[int, tuple[float, float]]]:
    """持拍腕 (time_ms, (x, y)) 序列（可见性 >= 0.5 的姿态帧）。"""
    lm_idx = LM["right_wrist"] if handedness == "right" else LM["left_wrist"] if handedness == "left" else None
    if lm_idx is None:
        return []
    pts = []
    for f in frames:
        pose = f.get("pose")
        if not pose or not pose.get("landmarks"):
            continue
        lm = pose["landmarks"][lm_idx]
        if lm.get("visibility", 0) < 0.5:
            continue
        pts.append((f["source_pts_ms"], (lm["x"], lm["y"])))
    return pts


def _wrist_speed_px_s(wrist_pts: list[tuple[int, tuple[float, float]]], t_ms: int,
                      window_ms: int = 200) -> float | None:
    """持拍腕在 t_ms 附近的最大速度（px/s），基于相邻姿态帧（wrist_pts
    为 _wrist_series 预提取的序列）。"""
    if not wrist_pts:
        return None
    best = 0.0
    for (t0, p0), (t1, p1) in zip(wrist_pts, wrist_pts[1:]):
        if t0 <= t_ms + window_ms and t1 >= t_ms - window_ms and t1 > t0:
            v = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / ((t1 - t0) / 1000.0)
            if t_ms - window_ms <= (t0 + t1) / 2 <= t_ms + window_ms:
                best = max(best, v)
    return best if best > 0 else None


def extract_events(frames: list[dict], *, run_id: str, handedness: str = "unknown",
                   wall: WallQuad | None = None,
                   cfg_events: dict[str, Any] | None = None,
                   measurements: dict[str, Any] | None = None,
                   analysis_height: int = 1080,
                   swings: list[dict] | None = None) -> list[dict]:
    """frames: run 的帧观测序列（时间升序）。返回事件 dict 列表（时间升序）。

    P4 规则：击球候选 = 挥拍时间窗附近存在可信球观测 + 空间接近或轨迹变化；
    挥拍候选（无球证据）单独输出，不计入确认击球。正反手用连续时间窗判定。
    """
    from app.vision import actions as actions_mod
    e = cfg_events or {}
    near_px = float(e.get("hit_near_racket_px", 220))
    dir_deg = float(e.get("hit_direction_change_deg", 35))
    wall_entry_px = float(e.get("wall_contact_entry_px", 60))
    swing_thr = float(e.get("swing_wrist_speed_px_s", 600))
    dedup_ms = int(e.get("deduplicate_window_ms", 250))
    speed_win = int((measurements or {}).get("speed_window_points", 4))

    if swings is None:
        segs = actions_mod.detect_swings(frames, handedness=handedness,
                                         min_speed_px_s=swing_thr)
        swings = [seg.to_dict() for seg in segs]

    events: list[ExtractedEvent] = []

    # 姿态/腕部序列：事件提取全程复用（见 _pose_series 注释）
    pose_frames, pose_times = _pose_series(frames)
    wrist_pts = _wrist_series(frames, handedness)

    # ---- 1. 球观测序列（按 track 分组，只用 raw 观测点）----
    track_obs: dict[int, list[dict]] = {}
    for f in frames:
        sel = (f.get("ball") or {}).get("selected") or {}
        if sel.get("raw") and sel.get("point") and sel.get("status") in ("tracked", "reacquiring", "human_corrected"):
            tid = sel.get("track_id", 0)
            track_obs.setdefault(tid, []).append(
                {"t": f["source_pts_ms"], "fi": f["frame_index"], "p": tuple(sel["point"]),
                 "tid": tid, "frame": f})
    all_obs = sorted((o for obs in track_obs.values() for o in obs), key=lambda o: o["t"])

    # 拍框/腕邻近判定素材
    def racket_boxes(frame: dict) -> list[tuple[float, float, float, float]]:
        return [tuple(b) for b in ((frame.get("detections") or {}).get("racket") or [])]

    def wrist_point(frame: dict | None) -> tuple[float, float] | None:
        if not frame:
            return None
        pose = frame.get("pose")
        if not pose or not pose.get("landmarks"):
            return None
        lm = PoseEstimator.wrist_of(pose["landmarks"], handedness)
        return (lm["x"], lm["y"]) if lm else None

    def ball_near_racket_or_wrist(p: tuple[float, float], frame: dict) -> tuple[bool, str | None, float | None]:
        best_d, how = None, None
        for (x1, y1, x2, y2) in racket_boxes(frame):
            d = _dist_to_box(p, (x1, y1, x2, y2))
            if best_d is None or d < best_d:
                best_d, how = d, "racket_bbox"
        wp = wrist_point(frame) or wrist_point(
            _nearest_pose_frame(pose_frames, pose_times, frame["source_pts_ms"]))
        if wp is not None:
            d = distance(p, wp)
            if best_d is None or d < best_d:
                best_d, how = d, "wrist"
        if best_d is None:
            return False, None, None
        return best_d <= near_px, how, best_d

    def trajectory_change_at(obs_list: list[dict], i: int) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
        """观测点 i 处的方向变化（前后 speed_win 窗口）。返回 (角度, v_in, v_out)。"""
        i0 = max(0, i - speed_win)
        i1 = min(len(obs_list) - 1, i + speed_win)
        p_i, t_i = obs_list[i]["p"], obs_list[i]["t"]
        p0, t0 = obs_list[i0]["p"], obs_list[i0]["t"]
        p1, t1 = obs_list[i1]["p"], obs_list[i1]["t"]
        if t_i - t0 < 1 or t1 - t_i < 1:
            return None
        v_in = _vel(p0, p_i, max(1e-3, (t_i - t0) / 1000.0))
        v_out = _vel(p_i, p1, max(1e-3, (t1 - t_i) / 1000.0))
        ang = _angle_between(v_in, v_out)
        return ang, v_in, v_out

    # ---- 2. 挥拍锚定击球候选 ----
    used_swing_idx: set[int] = set()
    for si, sw in enumerate(swings):
        w0, w1 = sw["start_ms"] - 150, sw["end_ms"] + 150
        win_obs = [o for o in all_obs if w0 <= o["t"] <= w1]
        prox_evidence = None     # (obs, how, dist)
        traj_evidence = None     # (obs, ang, sp_in, sp_out)
        for o in win_obs:
            near, how, d = ball_near_racket_or_wrist(o["p"], o["frame"])
            if near:
                if prox_evidence is None or d < prox_evidence[2]:
                    prox_evidence = (o, how, d)
        # 轨迹变化：在与挥拍窗口相交的观测中找最大方向变化（要求速度量级）
        for tid, obs in track_obs.items():
            for i in range(1, len(obs) - 1):
                o = obs[i]
                if not (w0 <= o["t"] <= w1):
                    continue
                tc = trajectory_change_at(obs, i)
                if tc is None:
                    continue
                ang, v_in, v_out = tc
                sp_in, sp_out = math.hypot(*v_in), math.hypot(*v_out)
                if ang >= dir_deg and min(sp_in, sp_out) >= 50:
                    if traj_evidence is None or ang > traj_evidence[1]:
                        traj_evidence = (o, ang, sp_in, sp_out)

        if prox_evidence or traj_evidence:
            used_swing_idx.add(si)
            anchor = (prox_evidence or traj_evidence)[0]
            t_ev = anchor["t"]
            frame = anchor["frame"]
            p_i = anchor["p"]
            stroke, stroke_ev = actions_mod.stroke_tendency(frames, _seg_from_dict(sw), handedness)
            ev = {
                "ball_observed_in_window": bool(win_obs),
                "near_racket": bool(prox_evidence and prox_evidence[1] == "racket_bbox"),
                "near_wrist": bool(prox_evidence and prox_evidence[1] == "wrist"),
                "nearest_distance_px": (round(prox_evidence[2], 1) if prox_evidence else None),
                "direction_change_deg": (round(traj_evidence[1], 1) if traj_evidence else None),
                "speed_in_px_s": (round(traj_evidence[2], 1) if traj_evidence else None),
                "speed_out_px_s": (round(traj_evidence[3], 1) if traj_evidence else None),
                "swing_evidence": True,
                "swing_window_ms": [sw["start_ms"], sw["end_ms"]],
                "swing_peak_speed_px_s": sw.get("peak_speed_px_s"),
                "swing_side": sw.get("side"),
                "handedness_available": handedness in ("right", "left"),
                "stroke_rule": stroke_ev,
                "detection": "swing_anchored_continuous_window",
            }
            score = 0.35
            if prox_evidence:
                score += 0.25
            if traj_evidence:
                score += 0.3
            if sw.get("peak_speed_px_s"):
                score += min(0.1, sw["peak_speed_px_s"] / 20000.0)
            events.append(ExtractedEvent(
                source_time_ms=t_ev, frame_index=frame["frame_index"], type="player_hit",
                stroke_type=stroke, point=p_i, heuristic_score=min(score, 1.0),
                evidence=ev, speed_px_s=(traj_evidence[3] if traj_evidence else None),
                track_id=anchor.get("tid")))

    # ---- 2b. 球锚定补充：无挥拍（姿态缺失/漏检）但球轨迹在拍/腕处显著反转 ----
    for tid, obs in track_obs.items():
        for i in range(1, len(obs) - 1):
            o = obs[i]
            t_i, p_i = o["t"], o["p"]
            if any(ev.type == "player_hit" and abs(ev.source_time_ms - t_i) <= 400 for ev in events):
                continue
            near, how, near_d = ball_near_racket_or_wrist(p_i, o["frame"])
            if not near:
                continue
            tc = trajectory_change_at(obs, i)
            if tc is None:
                continue
            ang, v_in, v_out = tc
            sp_in, sp_out = math.hypot(*v_in), math.hypot(*v_out)
            if ang < dir_deg or min(sp_in, sp_out) < 50:
                continue
            stroke, stroke_ev = actions_mod.stroke_tendency(
                frames, _seg_from_dict({"start_ms": t_i - 200, "end_ms": t_i + 200,
                                        "peak_ms": t_i, "peak_speed_px_s": None,
                                        "side": "unknown", "evidence": {}}), handedness)
            w_speed = _wrist_speed_px_s(wrist_pts, t_i)
            events.append(ExtractedEvent(
                source_time_ms=t_i, frame_index=o["fi"], type="player_hit",
                stroke_type=stroke, point=p_i,
                heuristic_score=min(0.3 + min(ang / 120.0, 1.0) * 0.3, 0.7),
                evidence={
                    "ball_observed_before": True, "ball_observed_after": True,
                    "near_racket": how == "racket_bbox", "near_wrist": how == "wrist",
                    "nearest_distance_px": round(near_d, 1),
                    "direction_change_deg": round(ang, 1),
                    "speed_in_px_s": round(sp_in, 1), "speed_out_px_s": round(sp_out, 1),
                    "swing_evidence": bool(w_speed and w_speed >= swing_thr),
                    "wrist_speed_px_s": round(w_speed, 1) if w_speed else None,
                    "detection": "ball_anchored_no_swing_window",
                    "stroke_rule": stroke_ev,
                },
                speed_px_s=max(sp_in, sp_out), track_id=tid))

    # ---- 3. 挥拍候选（无球证据）：不与击球重复 ----
    for si, sw in enumerate(swings):
        if si in used_swing_idx:
            continue
        t_mid = sw["peak_ms"]
        events.append(ExtractedEvent(
            source_time_ms=t_mid, frame_index=_frame_index_near(frames, t_mid),
            type="swing_candidate", point=_wrist_point_at(pose_frames, pose_times, t_mid, handedness),
            heuristic_score=min((sw.get("peak_speed_px_s") or 0) / 3000.0, 1.0),
            evidence={"swing_window_ms": [sw["start_ms"], sw["end_ms"]],
                      "peak_speed_px_s": sw.get("peak_speed_px_s"),
                      "side": sw.get("side"),
                      **(sw.get("evidence") or {}),
                      "note": "挥拍证据存在但窗口内无球邻近/轨迹变化，不计入确认击球"}))

    # ---- 4. 撞墙候选：进入墙面区域 + 法向速度反转（仅观测点）----
    if wall is not None and wall.validate() == []:
        for tid, obs in track_obs.items():
            for i in range(1, len(obs) - 1):
                p_i = obs[i]["p"]
                d_wall = wall.distance_to_polygon(p_i)
                if d_wall > wall_entry_px:
                    continue
                p0, t0 = obs[i - 1]["p"], obs[i - 1]["t"]
                p1, t1 = obs[i + 1]["p"], obs[i + 1]["t"]
                if obs[i]["t"] - t0 < 1 or t1 - obs[i]["t"] < 1:
                    continue
                v_in = _vel(p0, p_i, (obs[i]["t"] - t0) / 1000.0)
                v_out = _vel(p_i, p1, (t1 - obs[i]["t"]) / 1000.0)
                ang = _angle_between(v_in, v_out)
                if ang < 60:   # 撞墙要求显著反转
                    continue
                contact = _segment_polygon_first_intersect(p0, p1, wall)
                ev = {
                    "observed_in_and_out": True,
                    "distance_to_wall_px": round(d_wall, 1),
                    "normal_reversal_deg": round(ang, 1),
                    "contact_point_estimated": contact is not None,
                    "note": "撞墙点为候选估计，未经人工确认前不作为精确墙点",
                }
                events.append(ExtractedEvent(
                    source_time_ms=obs[i]["t"], frame_index=obs[i]["fi"],
                    type="wall_contact_candidate", stroke_type="unknown",
                    point=contact or p_i,
                    heuristic_score=min(ang / 150.0, 1.0) * 0.6 + (0.4 if d_wall == 0 else 0.2),
                    evidence=ev, track_id=tid))

    # ---- 5. 落地反弹候选：竖直速度反转且位于画面下部 ----
    for tid, obs in track_obs.items():
        for i in range(1, len(obs) - 1):
            p_i = obs[i]["p"]
            if p_i[1] < analysis_height * 0.55:
                continue
            v_in = _vel(obs[i - 1]["p"], p_i, max(1e-3, (obs[i]["t"] - obs[i - 1]["t"]) / 1000.0))
            v_out = _vel(p_i, obs[i + 1]["p"], max(1e-3, (obs[i + 1]["t"] - obs[i]["t"]) / 1000.0))
            if v_in[1] > 100 and v_out[1] < -100:   # 图像 y 向下：先向下再向上
                events.append(ExtractedEvent(
                    source_time_ms=obs[i]["t"], frame_index=obs[i]["fi"],
                    type="bounce_candidate", point=p_i, heuristic_score=0.4,
                    evidence={"vy_before": round(v_in[1], 1), "vy_after": round(v_out[1], 1),
                              "note": "疑似地面反弹，不计入击球数"},
                    track_id=tid))

    # ---- 6. 同类型去重窗口 ----
    events.sort(key=lambda ev: ev.source_time_ms)
    deduped: list[ExtractedEvent] = []
    for ev in events:
        dup = any(ev.type == prev.type and abs(ev.source_time_ms - prev.source_time_ms) <= dedup_ms
                  for prev in deduped)
        if not dup:
            deduped.append(ev)

    return [ev.to_dict(run_id, i + 1) for i, ev in enumerate(deduped)]


def t_i_diff(a: int, b: int) -> int:
    return abs(a - b)


def _dist_to_box(p: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - p[0], 0.0, p[0] - x2)
    dy = max(y1 - p[1], 0.0, p[1] - y2)
    return math.hypot(dx, dy)


def _segment_polygon_first_intersect(p0: tuple[float, float], p1: tuple[float, float],
                                      wall: WallQuad) -> tuple[float, float] | None:
    """轨迹段与墙面四边形边界第一个交点（无交点返回 None）。"""
    poly = wall.points + [wall.points[0]]
    best_t, best_pt = None, None
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        d1x, d1y = p1[0] - p0[0], p1[1] - p0[1]
        d2x, d2y = bx - ax, by - ay
        denom = d1x * d2y - d1y * d2x
        if abs(denom) < 1e-9:
            continue
        t = ((ax - p0[0]) * d2y - (ay - p0[1]) * d2x) / denom
        u = ((ax - p0[0]) * d1y - (ay - p0[1]) * d1x) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            if best_t is None or t < best_t:
                best_t = t
                best_pt = (p0[0] + t * d1x, p0[1] + t * d1y)
    return best_pt


def _seg_from_dict(d: dict):
    from app.vision.actions import SwingSegment
    return SwingSegment(start_ms=d["start_ms"], peak_ms=d.get("peak_ms", d["start_ms"]),
                        end_ms=d["end_ms"], peak_speed_px_s=d.get("peak_speed_px_s") or 0.0,
                        side=d.get("side", "unknown"))


def _frame_index_near(frames: list[dict], t_ms: int) -> int:
    best, bd = 0, None
    for f in frames:
        d = abs(f["source_pts_ms"] - t_ms)
        if bd is None or d < bd:
            best, bd = f["frame_index"], d
    return best


def _wrist_point_at(pose_frames: list[dict], pose_times: list[int], t_ms: int,
                    handedness: str) -> tuple[float, float] | None:
    fr = _nearest_pose_frame(pose_frames, pose_times, t_ms, 150)
    return wrist_point_global(fr, handedness)


def wrist_point_global(frame: dict | None, handedness: str) -> tuple[float, float] | None:
    if not frame:
        return None
    pose = frame.get("pose")
    if not pose or not pose.get("landmarks"):
        return None
    lm = PoseEstimator.wrist_of(pose["landmarks"], handedness)
    return (lm["x"], lm["y"]) if lm else None


def compute_phase_segments(events: list[dict]) -> list[dict]:
    """unknown → player_hit → outgoing → wall_contact → returning 状态段（供时间线）。"""
    segs: list[dict] = []
    state = "unknown"
    start = None
    for ev in events:
        t = ev["source_time_ms"]
        nxt = None
        if ev["type"] == "player_hit":
            nxt = "player_hit"
        elif ev["type"] == "wall_contact_candidate" and state in ("player_hit", "outgoing"):
            nxt = "wall_contact"
        if nxt is not None and nxt != state:
            if start is not None:
                segs.append({"from_ms": start, "to_ms": t, "phase": state})
            state, start = nxt, t
        elif state == "player_hit":
            state = "outgoing"; start = start if start is not None else t
        elif state == "wall_contact":
            state = "returning"
    if start is not None:
        segs.append({"from_ms": start, "to_ms": None, "phase": state})
    return segs
