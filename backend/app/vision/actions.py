"""连续姿态动作识别（P4）：挥拍时间窗检测，与球检测解耦。

原则（任务书 §六）：
- 没有球也要能显示人体动作；挥拍 ≠ 击球。
- 读连续时间窗，不用单帧手腕位置直接下结论。
- 输出带时间范围与可查证据；更细的准备/引拍/随挥分段放在人物与球轨迹验收之后。

输出类型：
- swing_segments: [{start_ms, peak_ms, end_ms, peak_speed_px_s, side, evidence}]
- stroke_tendency(seg): forehand/backhand tendency + evidence，或 unknown
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.vision.pose import LM


@dataclass
class SwingSegment:
    start_ms: int
    peak_ms: int
    end_ms: int
    peak_speed_px_s: float
    side: str                      # right | left | unknown（挥拍侧）
    peak_point: tuple[float, float] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"start_ms": self.start_ms, "peak_ms": self.peak_ms, "end_ms": self.end_ms,
                "peak_speed_px_s": round(self.peak_speed_px_s, 1), "side": self.side,
                "peak_point": ({"x": round(self.peak_point[0], 1), "y": round(self.peak_point[1], 1),
                                "space": "analysis_image_px"} if self.peak_point else None),
                "evidence": self.evidence}


def _wrist_series(frames: list[dict], lm_idx: int) -> list[tuple[int, tuple[float, float]]]:
    """某侧腕关键点的 (pts_ms, (x,y)) 序列，只保留可见点。"""
    out = []
    for f in frames:
        pose = f.get("pose")
        if not pose or not pose.get("landmarks"):
            continue
        lm = pose["landmarks"][lm_idx]
        if lm.get("visibility", 0) < 0.5:
            continue
        out.append((f["source_pts_ms"], (lm["x"], lm["y"])))
    return out


def _speed_profile(series: list[tuple[int, tuple[float, float]]],
                   max_gap_ms: int = 150) -> list[tuple[int, float, tuple[float, float]]]:
    """相邻可见点 → 速度样本 (中点时刻, px/s, 位置)。跨长缺口不生成样本。"""
    out = []
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > max_gap_ms:
            continue
        v = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (dt / 1000.0)
        out.append(((t0 + t1) // 2, v, p1))
    return out


def detect_swings(frames: list[dict], *, handedness: str = "unknown",
                  min_speed_px_s: float = 600.0,
                  min_duration_ms: int = 80,
                  separation_ms: int = 250,
                  elbow_boost: bool = True) -> list[SwingSegment]:
    """从连续姿态中检测挥拍时间窗。

    对持拍腕（unknown 时取双侧，取更强者并标记 side）计算速度包络，
    峰值超过阈值的连续段落构成 swing 段：起止为速度回落到峰值 35% 处。
    肘部速度作为辅助证据（同侧肘也应有运动）。
    """
    sides = {"right": [LM["right_wrist"]], "left": [LM["left_wrist"]],
             "unknown": [LM["right_wrist"], LM["left_wrist"]]}[handedness if handedness in ("right", "left") else "unknown"]
    elbow_idx = {"right": LM["right_elbow"], "left": LM["left_elbow"]}

    best_by_time: dict[int, tuple[float, str, tuple[float, float]]] = {}
    for lm_idx in sides:
        side = ("right" if lm_idx == LM["right_wrist"] else "left")
        prof = _speed_profile(_wrist_series(frames, lm_idx))
        for t, v, p in prof:
            cur = best_by_time.get(t)
            if cur is None or v > cur[0]:
                best_by_time[t] = (v, side, p)
    samples = sorted((t, v, side, p) for t, (v, side, p) in best_by_time.items())
    if not samples:
        return []

    # 肘部速度证据（按侧缓存）
    elbow_profiles: dict[str, list[tuple[int, float]]] = {}
    if elbow_boost:
        for side, idx in elbow_idx.items():
            elbow_profiles[side] = [(t, v) for t, v, _ in
                                    _speed_profile(_wrist_series(frames, idx))]

    def elbow_speed_near(side: str, t_ms: int, window_ms: int = 250) -> float:
        vals = [v for t, v in elbow_profiles.get(side, []) if abs(t - t_ms) <= window_ms]
        return max(vals) if vals else 0.0

    # 波峰提取 → 时间窗
    segments: list[SwingSegment] = []
    i = 0
    n = len(samples)
    while i < n:
        t, v, side, p = samples[i]
        if v < min_speed_px_s:
            i += 1
            continue
        # 扩展到局部极大
        j = i
        while j + 1 < n and samples[j + 1][1] >= samples[j][1] * 0.999 and samples[j + 1][0] - samples[i][0] < 600:
            j += 1
        peak_t, peak_v, peak_side, peak_p = samples[j]
        # 稳健阈值：用持续段 75 分位而非纯峰值——单帧姿态跳变的速度尖峰
        # 不应吞掉整个挥拍窗口
        import statistics
        run_vals = sorted(v for _, v, _, _ in samples[i:j + 1])
        robust_peak = run_vals[int(len(run_vals) * 0.75)] if run_vals else peak_v
        eff_peak = min(peak_v, max(robust_peak, min_speed_px_s))
        # 起止：速度回落到稳健峰值 35%（限制搜索范围）
        thresh = max(min_speed_px_s * 0.5, eff_peak * 0.35)
        s_i = j
        while s_i > 0 and samples[s_i - 1][1] > thresh and peak_t - samples[s_i - 1][0] < 800:
            s_i -= 1
        e_i = j
        while e_i + 1 < n and samples[e_i + 1][1] > thresh and samples[e_i + 1][0] - peak_t < 800:
            e_i += 1
        start_t, end_t = samples[s_i][0], samples[e_i][0]
        dur = end_t - start_t
        # 去重/合并：与上一段太近则取峰值更高者
        if segments and start_t - segments[-1].end_ms < separation_ms:
            if peak_v > segments[-1].peak_speed_px_s:
                segments.pop()
            else:
                i = e_i + 1
                continue
        if dur >= min_duration_ms:
            elbow_v = elbow_speed_near(peak_side, peak_t) if elbow_boost else 0.0
            segments.append(SwingSegment(
                start_ms=start_t, peak_ms=peak_t, end_ms=end_t,
                peak_speed_px_s=peak_v, side=peak_side if handedness != "unknown" else peak_side,
                peak_point=peak_p,
                evidence={
                    "wrist_peak_speed_px_s": round(peak_v, 1),
                    "elbow_peak_speed_px_s_near": round(elbow_v, 1),
                    "duration_ms": dur,
                    "detection": "连续姿态速度包络（非单帧位置）",
                    "note": "挥拍候选不计入确认击球；击球需球观测证据",
                }))
        i = e_i + 1
    return segments


def stroke_tendency(frames: list[dict], seg: SwingSegment,
                    handedness: str) -> tuple[str, dict[str, Any]]:
    """正反手倾向：读挥拍时间窗内的连续证据。

    证据（窗口 seg.start_ms..seg.end_ms）：
    1) 持拍腕相对躯干中线的侧向位置随时间的变化（引拍→触球侧向移动方向）；
    2) 肩线角度变化（躯干旋转）；
    3) 触球帧附近腕在中线的哪一侧。
    输出 forehand / backhand / unknown + 全部证据值。
    视角不适用（如正对背面且近轴）或持拍手未知 → unknown。
    """
    ev: dict[str, Any] = {"rule": "continuous_window_wrist_and_torso",
                          "window_ms": [seg.start_ms, seg.end_ms], "side": seg.side}
    if handedness not in ("right", "left"):
        ev["reason"] = "持拍手 unknown"
        return "unknown", ev
    if seg.side != "unknown" and seg.side != handedness:
        ev["reason"] = f"挥拍侧（{seg.side}）与持拍手（{handedness}）不一致"
        return "unknown", ev

    wrist_idx = LM[f"{handedness}_wrist"]
    midline_series, wrist_series, shoulder_series = [], [], []
    for f in frames:
        pose = f.get("pose")
        if not pose or not pose.get("landmarks"):
            continue
        t = f["source_pts_ms"]
        if not (seg.start_ms - 200 <= t <= seg.end_ms + 200):
            continue
        lms = pose["landmarks"]
        ls, rs = lms[LM["left_shoulder"]], lms[LM["right_shoulder"]]
        if ls.get("visibility", 0) > 0.5 and rs.get("visibility", 0) > 0.5:
            shoulder_series.append((t, (ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2,
                                    ls, rs))
        xs = [lms[k]["x"] for k in map(lambda n: LM[n], ("left_shoulder", "right_shoulder",
                                                          "left_hip", "right_hip"))
              if lms[k].get("visibility", 0) > 0.5]
        if xs:
            midline_series.append((t, sum(xs) / len(xs)))
        w = lms[wrist_idx]
        if w.get("visibility", 0) > 0.5:
            wrist_series.append((t, w["x"], w["y"]))

    if len(midline_series) < 2 or len(wrist_series) < 2:
        ev["reason"] = "窗口内姿态可见点不足"
        ev["points_wrist"] = len(wrist_series)
        ev["points_midline"] = len(midline_series)
        return "unknown", ev

    mid_at = lambda t: _interp(midline_series, t)
    # 腕相对中线的侧向偏移序列（>0 = 画面右侧）
    offsets = [(t, wx - (mid_at(t) if mid_at(t) is not None else wx)) for t, wx, _wy in wrist_series]
    offsets = [(t, o) for t, o in offsets]
    if len(offsets) < 2:
        ev["reason"] = "腕-中线偏移序列不足"
        return "unknown", ev
    off_start = _interp(offsets, seg.start_ms)
    off_peak = _interp(offsets, seg.peak_ms)
    # 肩线角度变化（度）：近侧肩-远侧肩连线与竖直方向夹角变化
    shoulder_delta_deg = None
    if len(shoulder_series) >= 2:
        a0 = _shoulder_angle(shoulder_series[0])
        a1 = _shoulder_angle(shoulder_series[-1])
        shoulder_delta_deg = round(a1 - a0, 1)
    ev["wrist_offset_at_start_px"] = None if off_start is None else round(off_start, 1)
    ev["wrist_offset_at_peak_px"] = None if off_peak is None else round(off_peak, 1)
    ev["shoulder_rotation_delta_deg"] = shoulder_delta_deg

    if off_peak is None or off_start is None:
        ev["reason"] = "窗口端点插值失败"
        return "unknown", ev
    # 规则：触球时刻持拍腕在持拍手一侧（右手指画面右侧）→ forehand 倾向；
    # 明显越过中线到对侧 → backhand 倾向；接近中线或证据矛盾 → unknown。
    margin = 30.0
    ev["margin_px"] = margin
    ev["camera_note"] = "规则默认人物背面拍摄视角；其他视角可能不适用，保留人工改正"
    if off_peak > margin:
        tendency = "forehand" if handedness == "right" else "backhand"
        ev["spatial"] = "腕在持拍手侧"
    elif off_peak < -margin:
        tendency = "backhand" if handedness == "right" else "forehand"
        ev["spatial"] = "腕越过中线到对侧"
    else:
        ev["reason"] = "腕接近中线，区分度不足"
        return "unknown", ev
    # 躯干旋转佐证：正手时肩线朝挥拍侧旋转（符号与画面视角相关，仅记录）
    return tendency, ev


def _interp(series: list[tuple], t: int):
    """series: [(t, value...)] 线性插值第一附加维度。"""
    if not series:
        return None
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            f = (t - t0) / (t1 - t0)
            return v0 + f * (v1 - v0)
    return None


def _shoulder_angle(sample) -> float:
    """肩线相对竖直方向的夹角（度，画面坐标系）。"""
    t, mx, my, ls, rs = sample
    dx = rs["x"] - ls["x"]
    dy = rs["y"] - ls["y"]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
