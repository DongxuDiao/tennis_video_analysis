"""目标人物身份关联（单目标）。

关联证据（任务书 §四）：框重叠 IoU、中心距离、尺寸变化、运动预测。
策略：只维护一个目标；丢失后进入 lost（不切换到旁边的人）；
重新出现时按运动预测 + 外观尺寸做再关联，不满足阈值时保持 lost，
由用户重新选择。宁可暂停轨迹，也不串人。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_size(b: tuple[float, float, float, float]) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


PERSON_TRACKED = "tracked"
PERSON_LOST = "lost"
PERSON_REACQUIRING = "reacquiring"


@dataclass
class PersonState:
    track_id: int
    box: tuple[float, float, float, float]      # 最新关联框（分析图像 px）
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s（中心运动估计）
    last_seen_ms: int = 0
    avg_size: float = 0.0                       # 指数平滑的框对角尺寸
    status: str = PERSON_TRACKED
    switched: int = 0                           # 串人风险计数（供指标）
    history: list[dict] = field(default_factory=list)


class PersonTracker:
    """单目标人物关联器。

    target_hint: {box: [x1,y1,x2,y2]}（可选，来自用户在首个有效人物帧的点击，
    或前端默认最大框）；不提供时自动在首个有人帧锁定最大框。
    """

    def __init__(self, *, target_hint: dict | None = None,
                 iou_threshold: float = 0.25,
                 center_dist_ratio: float = 0.6,     # 允许距离 = 比例 × 框尺寸
                 size_change_ratio: float = 0.45,    # 尺寸突变容忍（相对平滑尺寸）
                 lost_keep_ms: int = 1500,           # 丢失后保留预测窗口
                 next_id: int = 1):
        self.hint = (tuple(target_hint["box"]) if target_hint and target_hint.get("box") else None)
        self.iou_threshold = iou_threshold
        self.center_dist_ratio = center_dist_ratio
        self.size_change_ratio = size_change_ratio
        self.lost_keep_ms = lost_keep_ms
        self._next_id = next_id
        self.state: PersonState | None = None

    # ---- 关联代价：四类证据合成 ----
    def _association_score(self, cand: tuple[float, float, float, float],
                           predicted_box: tuple[float, float, float, float],
                           dt_s: float) -> tuple[float, dict[str, Any]]:
        v_iou = iou(cand, predicted_box)
        c1, c2 = box_center(cand), box_center(predicted_box)
        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        max_dist = self.center_dist_ratio * max(box_size(cand), box_size(predicted_box), 1.0)
        v_dist = max(0.0, 1.0 - dist / max_dist)
        s_new = box_size(cand)
        s_ref = max(self.state.avg_size if self.state else s_new, 1.0)
        size_ratio = abs(s_new - s_ref) / s_ref
        v_size = max(0.0, 1.0 - size_ratio / max(self.size_change_ratio, 1e-6) * 0.5)
        # 运动一致性：候选位置与预测中心方向的接近程度已含在 v_dist 中
        score = 0.45 * v_iou + 0.35 * v_dist + 0.20 * v_size
        detail = {"iou": round(v_iou, 3), "center_dist_px": round(dist, 1),
                  "max_dist_px": round(max_dist, 1), "size_ratio": round(size_ratio, 3),
                  "score": round(score, 3)}
        return score, detail

    def _predict_box(self, dt_s: float) -> tuple[float, float, float, float]:
        st = self.state
        b = st.box
        dx = st.velocity[0] * dt_s
        dy = st.velocity[1] * dt_s
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def step(self, pts_ms: int, person_boxes: list[tuple[float, float, float, float]],
             confidences: list[float] | None = None, *, sampled: bool = True) -> dict[str, Any]:
        """推进一帧。返回 {status, box, track_id, matched, evidence}。"""
        if not sampled:
            st = self.state
            age = pts_ms - st.last_seen_ms if st else 0
            valid = st is not None and age <= self.lost_keep_ms and st.status != PERSON_LOST
            return {"status": "predicted" if valid else "lost", "matched": False,
                    "track_id": st.track_id if st else None,
                    "box": list(self._predict_box(age / 1000)) if valid else None,
                    "evidence": {"sampled": False, "age_ms": age}}
        confidences = confidences or [1.0] * len(person_boxes)
        # 尚未初始化：等 hint 或首个有人帧（默认取最大框）
        if self.state is None:
            if not person_boxes:
                return {"status": "none", "box": None, "track_id": None, "matched": False}
            if self.hint is not None:
                best, best_iou = None, -1.0
                for b in person_boxes:
                    v = iou(b, self.hint)
                    if v > best_iou:
                        best, best_iou = b, v
                if best is not None and best_iou >= 0.3:
                    chosen = best
                else:
                    return {"status": "lost", "box": None, "track_id": None, "matched": False}
            else:
                chosen = max(person_boxes, key=box_size)
            self.state = PersonState(track_id=self._next_id, box=chosen,
                                     last_seen_ms=pts_ms, avg_size=box_size(chosen))
            self._next_id += 1
            return {"status": PERSON_TRACKED, "box": list(chosen),
                    "track_id": self.state.track_id, "matched": True,
                    "evidence": {"init": "largest_box" if self.hint is None else "user_hint",
                                 "hint_iou": round(best_iou, 3) if self.hint is not None else None}}

        st = self.state
        dt_s = max(1e-3, (pts_ms - st.last_seen_ms) / 1000.0)
        predicted = self._predict_box(dt_s)
        best_score, best_idx, best_detail = -1.0, -1, None
        details = []
        for i, b in enumerate(person_boxes):
            s, detail = self._association_score(b, predicted, dt_s)
            detail["candidate_conf"] = round(confidences[i], 3)
            details.append(detail)
            if s > best_score:
                best_score, best_idx, best_detail = s, i, detail

        matched = best_idx >= 0 and (
            best_detail["iou"] >= self.iou_threshold or
            best_detail["score"] >= 0.55 or
            (best_detail["center_dist_px"] <= best_detail["max_dist_px"] * 0.4
             and best_detail["size_ratio"] <= 0.35))

        # 门控：隐含速度或尺寸突变不合理时拒绝匹配（碎片框/误检），
        # 宁可 coasting 也不让一次坏匹配污染速度与尺寸记忆
        if matched and best_idx >= 0:
            b_cand = person_boxes[best_idx]
            c_new, c_old = box_center(b_cand), box_center(st.box)
            implied_speed = math.hypot(c_new[0] - c_old[0], c_new[1] - c_old[1]) / dt_s
            size_ratio_gate = abs(box_size(b_cand) - st.avg_size) / max(st.avg_size, 1.0)
            if implied_speed > 1500.0 or size_ratio_gate > 0.7:
                matched = False
                best_detail = {**best_detail, "gated_out":
                               f"implied_speed {implied_speed:.0f}px/s 或尺寸比 {size_ratio_gate:.2f} 越界"}

        if matched:
            b = person_boxes[best_idx]
            c_new = box_center(b)
            c_old = box_center(st.box)
            new_vx = (c_new[0] - c_old[0]) / dt_s
            new_vy = (c_new[1] - c_old[1]) / dt_s
            # 速度平滑 + 限幅（人物不可能瞬移）
            vx = max(-1200.0, min(1200.0, 0.6 * st.velocity[0] + 0.4 * new_vx))
            vy = max(-1200.0, min(1200.0, 0.6 * st.velocity[1] + 0.4 * new_vy))
            st.velocity = (vx, vy)
            st.box = b
            st.last_seen_ms = pts_ms
            st.avg_size = 0.8 * st.avg_size + 0.2 * box_size(b)
            was_lost = st.status != PERSON_TRACKED
            st.status = PERSON_TRACKED
            st.history.append({"t_ms": pts_ms, "box": list(b)})
            return {"status": "reacquired" if was_lost else PERSON_TRACKED,
                    "box": list(b), "track_id": st.track_id, "matched": True,
                    "evidence": {"association": best_detail, "candidates": details[:6]}}

        # 未匹配：丢失窗口内保持 lost（不切换到其他框）；
        # 速度随丢失时长衰减，避免早期坏速度持续外推阻碍重关联
        lost_for = pts_ms - st.last_seen_ms
        if lost_for > 400:
            decay = 0.5
            st.velocity = (st.velocity[0] * decay, st.velocity[1] * decay)

        st.status = PERSON_LOST if lost_for > self.lost_keep_ms else st.status
        return {"status": PERSON_LOST if lost_for > 200 else "coasting",
                "box": list(predicted) if lost_for <= self.lost_keep_ms else None,
                "track_id": st.track_id, "matched": False,
                "evidence": {"lost_for_ms": lost_for,
                             "note": "目标丢失：不切换到其他人物框",
                             "candidates": details[:6]}}

    def hip_center(self) -> tuple[float, float] | None:
        """目标框的近似髋部中心（框几何中点偏下）。真实髋部来自姿态关键点时
        由调用方优先使用姿态结果并标注来源。"""
        if self.state is None:
            return None
        c = box_center(self.state.box)
        h = self.state.box[3] - self.state.box[1]
        return (c[0], c[1] + h * 0.12)

    def crop_with_margin(self, img_w: int, img_h: int, margin_ratio: float = 0.35,
                         min_size_px: int = 256, box=None) -> tuple[int, int, int, int]:
        """带边距的目标区域（保证挥拍时手腕不被裁掉），并裁剪到图像范围。"""
        b = box if box is not None else self.state.box
        w, h = b[2] - b[0], b[3] - b[1]
        mx, my = w * margin_ratio, h * margin_ratio
        x1 = max(0, int(b[0] - mx))
        y1 = max(0, int(b[1] - my))
        x2 = min(img_w, int(b[2] + mx))
        y2 = min(img_h, int(b[3] + my))
        if x2 - x1 < min_size_px:
            cx = (x1 + x2) // 2
            x1, x2 = max(0, cx - min_size_px // 2), min(img_w, cx + min_size_px // 2)
        if y2 - y1 < min_size_px:
            cy = (y1 + y2) // 2
            y1, y2 = max(0, cy - min_size_px // 2), min(img_h, cy + min_size_px // 2)
        return (x1, y1, x2, y2)
