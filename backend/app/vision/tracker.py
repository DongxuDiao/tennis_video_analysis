"""单球追踪：卡尔曼预测 + 候选关联。严格区分 observed / predicted / lost / ambiguous。

规则（任务书 §6.3）：
- 只允许一条活动球轨迹；多候选同近时进入 ambiguous，不任意切换。
- 预测缺口上限 max_prediction_gap_ms（默认 80ms），超时断轨。
- 预测点不得独立触发事件；保留 raw 与 filtered 两套点。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.vision.geometry import distance


class Kalman2D:
    """恒速模型的简化卡尔曼滤波（状态 [x,y,vx,vy]），逐帧标量更新。"""

    def __init__(self, x: float, y: float):
        import numpy as np
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.cov = np.diag([50.0, 50.0, 5000.0, 5000.0])
        self.F = None  # 按帧间 dt 构建
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.diag([25.0, 25.0])
        self.Q_scale = 800.0

    def predict(self, dt: float) -> tuple[float, float]:
        import numpy as np
        F = np.eye(4)
        F[0, 2] = dt; F[1, 3] = dt
        self.F = F
        self.state = F @ self.state
        Q = np.diag([self.Q_scale * dt, self.Q_scale * dt, self.Q_scale * dt * 2, self.Q_scale * dt * 2])
        self.cov = F @ self.cov @ F.T + Q
        return float(self.state[0]), float(self.state[1])

    def update(self, x: float, y: float) -> None:
        import numpy as np
        z = np.array([x, y], dtype=np.float64)
        S = self.H @ self.cov @ self.H.T + self.R
        K = self.cov @ self.H.T @ np.linalg.inv(S)
        yk = z - self.H @ self.state
        self.state = self.state + K @ yk
        I = np.eye(4)
        self.cov = (I - K @ self.H) @ self.cov

    @property
    def pos(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    def predict_position_only(self, dt: float) -> tuple[float, float]:
        """不改变滤波状态的纯预测（供局部搜索中心估计）。"""
        return (float(self.state[0] + self.state[2] * dt),
                float(self.state[1] + self.state[3] * dt))

    @property
    def vel(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])


TRACK_UNINITIALIZED = "uninitialized"
TRACK_TRACKED = "tracked"
TRACK_PREDICTED = "predicted"
TRACK_LOST = "lost"
TRACK_REACQUIRING = "reacquiring"


@dataclass
class TrackPoint:
    frame_index: int
    pts_ms: int
    raw: tuple[float, float] | None        # 观测原始点（无则 None）
    filtered: tuple[float, float]          # 滤波后位置
    status: str                            # tracked|predicted|reacquiring
    tracking_score: float                  # 关联组合分（0..1，非“准确率”）
    candidate_conf: float | None           # 选中候选的模型原始置信度
    search_mode: str = "full_frame"        # 该帧使用的检测模式
    search_crop: tuple[int, int, int, int] | None = None
    rejected: list[dict[str, Any]] = field(default_factory=list)  # 拒绝候选及原因


@dataclass
class TrackSegment:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    status: str = TRACK_UNINITIALIZED

    @property
    def start_ms(self) -> int:
        return self.points[0].pts_ms if self.points else 0

    @property
    def end_ms(self) -> int:
        return self.points[-1].pts_ms if self.points else 0

    def observed_points(self) -> list[TrackPoint]:
        return [p for p in self.points if p.raw is not None]


class BallTracker:
    """单目标球追踪：综合评分关联 + 静止误检抑制 + 自适应局部搜索半径。

    评分组成（任务书 §五C）：模型置信度、与预测点距离、尺寸及其变化、
    前后帧运动一致性、与重复静止误检的位置关系。预测点不计入真实观测。
    """

    def __init__(self, *, max_prediction_gap_ms: int = 80,
                 association_max_dist_px: float = 120.0,
                 ambiguous_dist_px: float = 40.0,
                 ball_min_size_px: float = 4.0, ball_max_size_px: float = 160.0,
                 base_search_radius_px: float = 220.0,
                 radius_per_lost_ms: float = 0.8):
        self.max_prediction_gap_ms = max_prediction_gap_ms
        self.association_max_dist_px = association_max_dist_px
        self.ambiguous_dist_px = ambiguous_dist_px
        self.ball_min_size_px = ball_min_size_px
        self.ball_max_size_px = ball_max_size_px
        self.base_search_radius_px = base_search_radius_px
        self.radius_per_lost_ms = radius_per_lost_ms

        self._kf: Kalman2D | None = None
        self._last_obs_ms: int | None = None
        self._last_any_ms: int | None = None
        self._segments: list[TrackSegment] = []
        self._cur: TrackSegment | None = None
        self._next_id = 1
        self._since_reacquire_ms: int | None = None
        self._size_ema: float | None = None          # 观测球尺寸指数平滑
        # 静止误检记忆：位置 → 出现次数（球在飞，反复原地出现的是误检）
        self._static_hits: dict[tuple[int, int], int] = {}
        self.last_track_point: TrackPoint | None = None
        # 断轨后的局部重试：记录最后已知位置/速度，供扩张局部搜索（不生成虚假轨迹点）
        self._last_known: tuple[tuple[float, float], tuple[float, float], int] | None = None
        self._miss_streak: int = 0
        self.lost_local_window_ms = 1200
        self.lost_local_retry_frames = 8

    # ---- 局部搜索建议 ----
    def suggest_search(self, now_ms: int) -> tuple[tuple[float, float], float] | None:
        """返回 (预测中心, 搜索半径 px)。无活动轨迹时返回 None（应做全画面搜索）。

        半径自适应：基础值 + 速度外推 + 丢失时长增长 + 卡尔曼不确定度。
        """
        if self._kf is None:
            if self._last_known is None:
                return None
            (px, py), (vx, vy), t0 = self._last_known
            lost = now_ms - t0
            if lost > self.lost_local_window_ms:
                return None
            dt = lost / 1000.0
            radius = (self.base_search_radius_px + lost * self.radius_per_lost_ms)
            return ((px + vx * dt, py + vy * dt), min(radius, 1600.0))
        lost_ms = max(0, now_ms - (self._last_obs_ms or now_ms))
        dt = max(1e-3, (now_ms - (self._last_any_ms or now_ms)) / 1000.0)
        px, py = self._kf.predict_position_only(dt)
        vx, vy = self._kf.vel
        speed = (vx * vx + vy * vy) ** 0.5
        horizon_s = 0.15
        radius = (self.base_search_radius_px
                  + speed * horizon_s
                  + lost_ms * self.radius_per_lost_ms)
        try:
            p_unc = (self._kf.cov[0, 0] + self._kf.cov[1, 1]) ** 0.5
            radius += max(0.0, p_unc - 30.0)
        except Exception:
            pass
        return (px, py), min(radius, 1600.0)

    def wants_full_search(self, now_ms: int, interval_ms: int = 300,
                           discovery_interval_ms: int = 150) -> bool:
        """是否应做全画面搜索。

        - 有轨迹：周期性校验（interval_ms）到期，或断轨后局部重试超限。
        - 无轨迹且无已知球位：按 discovery_interval_ms 周期发现——
          每帧全图分块会在球连续不可检测时形成死循环。
        """
        if self._kf is None:
            if self._last_known is not None and now_ms - self._last_known[2] <= self.lost_local_window_ms:
                return self._miss_streak >= self.lost_local_retry_frames
            return (self._since_reacquire_ms is None
                    or now_ms - self._since_reacquire_ms >= min(interval_ms, discovery_interval_ms))
        if self._last_obs_ms is None or now_ms - self._last_obs_ms > self.max_prediction_gap_ms:
            return self._miss_streak >= self.lost_local_retry_frames
        return (self._since_reacquire_ms is None
                or now_ms - self._since_reacquire_ms >= interval_ms)

    def note_full_search_done(self, now_ms: int) -> None:
        self._since_reacquire_ms = now_ms

    # ---- 候选预处理：尺寸过滤 + 静止误检标记 ----
    def _size_filter(self, cands: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ok, rejected = [], []
        for c in cands:
            x1, y1, x2, y2 = c["bbox"]
            w, h = x2 - x1, y2 - y1
            if w < self.ball_min_size_px or h < self.ball_min_size_px or w > self.ball_max_size_px or h > self.ball_max_size_px:
                rejected.append({**c, "reject_reason": f"尺寸越界（{w:.0f}x{h:.0f}px）"})
            else:
                ok.append(c)
        return ok, rejected

    def _static_penalty(self, c: dict[str, Any]) -> tuple[float, str | None]:
        """重复静止误检：同位置（25px 内网格）累计出现 ≥8 次的候选降权。"""
        import math
        cx, cy = c["center"]
        key = (int(cx // 25), int(cy // 25))
        n = self._static_hits.get(key, 0)
        if n >= 8:
            return 0.5, f"疑似静止误检（同位置累计 {n} 次）"
        return 0.0, None

    def _observe_static(self, chosen: dict[str, Any] | None,
                        cands: list[dict[str, Any]]) -> None:
        """更新静止误检计数：未被选中且未被关联的候选计入；选中（在飞）的清零。"""
        for c in cands:
            cx, cy = c["center"]
            key = (int(cx // 25), int(cy // 25))
            if chosen and abs(chosen["center"][0] - cx) < 25 and abs(chosen["center"][1] - cy) < 25:
                self._static_hits[key] = 0
            else:
                self._static_hits[key] = self._static_hits.get(key, 0) + 1

    def _candidate_size(self, c: dict[str, Any]) -> float:
        x1, y1, x2, y2 = c["bbox"]
        return max(1.0, (x2 - x1 + y2 - y1) / 2.0)

    def _score_candidate(self, c: dict[str, Any], pred: tuple[float, float],
                         d: float) -> tuple[float, dict[str, Any]]:
        """综合评分：置信度 + 距离 + 尺寸一致性 + 运动一致性 − 静止惩罚。"""
        import math
        v_conf = max(0.0, min(1.0, c["confidence"]))
        v_dist = max(0.0, 1.0 - d / max(1.0, self.association_max_dist_px))
        size = self._candidate_size(c)
        v_size = 1.0
        if self._size_ema:
            ratio = abs(size - self._size_ema) / self._size_ema
            v_size = max(0.0, 1.0 - ratio / 0.8)
        # 运动一致性：候选相对上一点位移方向与当前速度方向夹角
        v_motion = 1.0
        if self._kf is not None and self.last_track_point and self.last_track_point.raw:
            vx, vy = self._kf.vel
            sp = math.hypot(vx, vy)
            lx, ly = self.last_track_point.raw
            mx, my = c["center"][0] - lx, c["center"][1] - ly
            ms = math.hypot(mx, my)
            if sp > 30 and ms > 1:
                cosang = max(-1.0, min(1.0, (vx * mx + vy * my) / (sp * ms)))
                v_motion = max(0.0, cosang)
        pen, pen_reason = self._static_penalty(c)
        score = (0.30 * v_conf + 0.35 * v_dist + 0.15 * v_size + 0.20 * v_motion) * (1.0 - pen)
        detail = {"v_conf": round(v_conf, 3), "v_dist": round(v_dist, 3),
                  "v_size": round(v_size, 3), "v_motion": round(v_motion, 3),
                  "static_penalty": pen, "score": round(score, 4)}
        if pen_reason:
            detail["reason"] = pen_reason
        return score, detail

    def step(self, frame_index: int, pts_ms: int,
             ball_candidates: list[dict[str, Any]],
             search_mode: str = "full_frame",
             search_crop: tuple[int, int, int, int] | None = None) -> TrackPoint:
        """逐帧推进。ball_candidates: [{bbox, center, confidence}]（分析图像 px）。"""
        cands, rejected_size = self._size_filter(ball_candidates)

        # 无活动轨迹：若有候选则初始化新段（综合评分最高者，并要求非静止误检）
        if self._kf is None or self._cur is None:
            scored_init = []
            for c in cands:
                s, detail = self._score_candidate(c, (0.0, 0.0), 1e9)
                pen, reason = self._static_penalty(c)
                if reason:
                    continue
                scored_init.append((s, c))
            if scored_init:
                scored_init.sort(key=lambda t: -t[0])
                best = scored_init[0][1]
                self._kf = Kalman2D(*best["center"])
                self._kf.update(*best["center"])
                seg = TrackSegment(track_id=self._next_id, status=TRACK_TRACKED)
                self._next_id += 1
                self._segments.append(seg)
                self._cur = seg
                self._last_obs_ms = pts_ms
                self._last_any_ms = pts_ms
                self._size_ema = self._candidate_size(best)
                self._observe_static(best, cands)
                tp = TrackPoint(frame_index, pts_ms, raw=tuple(best["center"]),
                                filtered=self._kf.pos, status=TRACK_TRACKED,
                                tracking_score=round(best["confidence"], 4),
                                candidate_conf=best["confidence"],
                                search_mode=search_mode, search_crop=search_crop)
                tp.rejected = rejected_size + [self._rej(c, "未关联：无活动轨迹，未选中") for c in cands if c is not best]
                seg.points.append(tp)
                self.last_track_point = tp
                return tp
            self._observe_static(None, cands)
            self._last_any_ms = pts_ms
            tp = TrackPoint(frame_index, pts_ms, raw=None,
                            filtered=(float("nan"), float("nan")),
                            status=TRACK_LOST, tracking_score=0.0, candidate_conf=None,
                            search_mode=search_mode, search_crop=search_crop)
            tp.rejected = rejected_size
            self.last_track_point = tp
            return tp

        dt = max(1e-3, (pts_ms - (self._last_any_ms or pts_ms)) / 1000.0)
        pred_x, pred_y = self._kf.predict(dt)
        gap_since_obs = pts_ms - (self._last_obs_ms or pts_ms)

        # 预测缺口超限：断轨。候选（若有）作为新段重捕获起点。
        if gap_since_obs > self.max_prediction_gap_ms and self._last_obs_ms is not None:
            self._close_segment(TRACK_LOST)
            return self.step(frame_index, pts_ms, ball_candidates, search_mode, search_crop)

        # 综合评分关联
        scored = []
        for c in cands:
            d = distance(c["center"], (pred_x, pred_y))
            s, detail = self._score_candidate(c, (pred_x, pred_y), d)
            scored.append((c, d, s, detail))
        scored.sort(key=lambda t: -t[2])
        chosen = None
        ambiguous = False
        near = [(c, d, s) for c, d, s, _ in scored
                if d <= self.association_max_dist_px]
        if near:
            chosen = near[0]
            # 多候选同近：不任意切换，标记 ambiguous（仍选最高分，供复核）
            if len(near) > 1:
                best_c, best_d, best_s = near[0]
                for c2, d2, s2 in near[1:]:
                    if (abs(d2 - best_d) < self.ambiguous_dist_px
                            and abs(s2 - best_s) < 0.25):
                        ambiguous = True

        if chosen:
            c, d, s = chosen
            self._miss_streak = 0
            self._kf.update(*c["center"])
            self._last_known = ((c["center"][0], c["center"][1]), self._kf.vel, pts_ms)
            status = TRACK_REACQUIRING if ambiguous else TRACK_TRACKED
            tp = TrackPoint(frame_index, pts_ms, raw=tuple(c["center"]),
                            filtered=self._kf.pos, status=status,
                            tracking_score=round(s, 4), candidate_conf=c["confidence"],
                            search_mode=search_mode, search_crop=search_crop)
            for rc, rd, rs, rdet in scored:
                if rc is c:
                    continue
                tp.rejected.append(self._rej(rc, f"综合分 {rs:.2f}/距预测 {rd:.0f}px 未被选中", rdet))
            tp.rejected = rejected_size + tp.rejected
            self._last_obs_ms = pts_ms
            self._last_any_ms = pts_ms
            size = self._candidate_size(c)
            self._size_ema = size if self._size_ema is None else 0.8 * self._size_ema + 0.2 * size
            self._observe_static(c, cands)
            self._cur.status = TRACK_REACQUIRING if ambiguous else TRACK_TRACKED
            self._cur.points.append(tp)
            self.last_track_point = tp
            return tp

        # 无可关联候选：输出预测点（标记 predicted，不作为事件依据）
        self._miss_streak += 1
        if self._last_known is None and self._kf is not None:
            self._last_known = (self._kf.pos, self._kf.vel, self._last_obs_ms or pts_ms)
        status = TRACK_PREDICTED if gap_since_obs <= self.max_prediction_gap_ms else TRACK_LOST
        if status == TRACK_LOST:
            self._close_segment(TRACK_LOST)
            self._observe_static(None, cands)
            self._last_any_ms = pts_ms
            tp = TrackPoint(frame_index, pts_ms, raw=None,
                            filtered=(float("nan"), float("nan")),
                            status=TRACK_LOST, tracking_score=0.0, candidate_conf=None,
                            search_mode=search_mode, search_crop=search_crop)
            tp.rejected = rejected_size + [self._rej(c, "断轨后待重捕获") for c in cands]
            self.last_track_point = tp
            return tp
        tp = TrackPoint(frame_index, pts_ms, raw=None, filtered=(pred_x, pred_y),
                        status=TRACK_PREDICTED, tracking_score=0.0, candidate_conf=None,
                        search_mode=search_mode, search_crop=search_crop)
        tp.rejected = rejected_size + [self._rej(c, f"距预测点 {d:.0f}px > {self.association_max_dist_px:.0f}px") for c, d, s, _ in scored]
        self._last_any_ms = pts_ms
        self._observe_static(None, cands)
        self._cur.status = TRACK_PREDICTED
        self._cur.points.append(tp)
        self.last_track_point = tp
        return tp

    def _rej(self, c: dict, reason: str, detail: dict | None = None) -> dict:
        r = {"bbox": list(c["bbox"]), "confidence": c["confidence"], "reason": reason}
        if detail:
            r["score_detail"] = detail
        return r

    def _close_segment(self, final_status: str) -> None:
        if self._cur:
            self._cur.status = final_status
        self._kf = None
        self._cur = None
        self._last_obs_ms = None
        self._since_reacquire_ms = None

    def finish(self) -> list[TrackSegment]:
        self._close_segment(TRACK_LOST)
        return self._segments

    @property
    def current_track_id(self) -> int | None:
        return self._cur.track_id if self._cur else None
