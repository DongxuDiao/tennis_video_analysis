"""导出渲染：在原始分析帧上叠加图层，编码 H.264 MP4（含音频透传）+ CSV + JSON。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np

from app.vision.geometry import WallQuad
from app.vision.pose import LM

# 图层颜色 (BGR)
COLOR_BALL = (0, 215, 255)          # 黄（观测）
COLOR_PRED = (0, 140, 255)          # 橙（预测，虚线）
COLOR_TRAIL = (200, 200, 200)
COLOR_SKELETON = (80, 220, 100)
COLOR_RACKET = (255, 120, 120)
COLOR_EVENT_HIT = (60, 60, 255)     # 红
COLOR_EVENT_WALL = (255, 80, 80)
COLOR_HUMAN = (230, 80, 230)        # 人工点
COLOR_WALL_QUAD = (180, 180, 60)
COLOR_ROI = (160, 160, 160)

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]
# 25..28 在 33 点集中存在（膝盖/脚踝），直接使用索引


def draw_frame(img: np.ndarray, frame_rec: dict | None, events: list[dict],
               wall: WallQuad | None, roi: dict | None, *, layers: dict[str, bool],
               time_label: str, panel_text: list[str] | None = None,
               show_rejected: bool = False) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(w / 1280.0, 1.0) * 0.9

    def P(p: tuple[float, float]) -> tuple[int, int]:
        return int(round(p[0])), int(round(p[1]))

    if layers.get("roi") and roi:
        cv2.rectangle(out, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), COLOR_ROI, 1)
    if layers.get("wall") and wall and wall.points:
        pts = np.array([P(p) for p in wall.points], dtype=np.int32)
        cv2.polylines(out, [pts], True, COLOR_WALL_QUAD, 2)

    if frame_rec:
        # 球候选
        if layers.get("candidates") and show_rejected:
            for c in frame_rec.get("ball", {}).get("candidates", []):
                x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)
        sel = frame_rec.get("ball", {}).get("selected")
        if layers.get("ball") and sel:
            p = tuple(sel["point"])
            color = COLOR_BALL if sel.get("raw") else COLOR_PRED
            cv2.circle(out, P(p), int(8 * scale), color, 2 if sel.get("raw") else 1)
            if not sel.get("raw"):
                cv2.circle(out, P(p), int(12 * scale), color, 1)  # 预测：外圈虚示意
            cv2.putText(out, sel.get("status", ""), (P(p)[0] + 12, P(p)[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, 1)
        # 球拍框
        if layers.get("racket"):
            for b in frame_rec.get("detections", {}).get("racket", []):
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_RACKET, 2)
        # 骨架
        if layers.get("skeleton") and (frame_rec.get("pose") or {}).get("landmarks"):
            lms = frame_rec["pose"]["landmarks"]
            for a, b in POSE_EDGES:
                la, lb = lms[a], lms[b]
                if la.get("visibility", 0) > 0.5 and lb.get("visibility", 0) > 0.5:
                    cv2.line(out, P((la["x"], la["y"])), P((lb["x"], lb["y"])), COLOR_SKELETON, 2)
            for lm in lms:
                if lm.get("visibility", 0) > 0.5:
                    cv2.circle(out, P((lm["x"], lm["y"])), 3, COLOR_SKELETON, -1)

    # 事件标记（时间上接近当前帧的）
    if layers.get("events"):
        for ev in events:
            dt = abs(ev["source_time_ms"] - frame_rec["source_pts_ms"]) if frame_rec else 9999
            if dt > 350:
                continue
            color = COLOR_HUMAN if ev.get("origin") in ("human_added", "reviewed") else (
                COLOR_EVENT_HIT if ev["type"] == "player_hit" else COLOR_EVENT_WALL)
            if ev.get("review_status") == "excluded":
                color = (90, 90, 90)
            if ev.get("point"):
                p = P((ev["point"]["x"], ev["point"]["y"]))
                cv2.circle(out, p, int(10 * scale), color, 2)
                cv2.putText(out, ev["type"][:12], (p[0] + 12, p[1] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, 1)

    # 时间码
    cv2.putText(out, time_label, (int(10 * scale), int(24 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), 2)
    # 底部面板
    if panel_text:
        ph = int(len(panel_text) * 26 * scale) + 8
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - ph), (w, h), (20, 20, 20), -1)
        out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
        for i, line in enumerate(panel_text):
            cv2.putText(out, line, (int(12 * scale), h - ph + int((i + 0.8) * 26 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (240, 240, 240), 1)
    return out


def render_export_mp4(*, project: dict, run: dict, frames_by_index: dict[int, dict],
                      events: list[dict], wall: WallQuad | None, roi: dict | None,
                      out_path: Path, layers: dict[str, bool], panel_lines: list[str],
                      source_path: str, start_ms: int, end_ms: int, rotation: int,
                      should_continue: Callable[[], bool] | None = None,
                      progress: Callable[[int, int], None] | None = None) -> dict:
    """解码原始视频片段 → 叠加图层 → 编码 H.264 yuv420p；音频尽力透传。"""
    from app.video.decode import iter_frames
    tmp = out_path.with_suffix(".tmp.mp4")
    out_h_target = 720
    from fractions import Fraction
    aout = None
    ok = False
    with av.open(source_path) as inp:
        vs = inp.streams.video[0]
        first_w, first_h = None, None
        out = None
        vstream = None
        done = 0
        rate = vs.guessed_rate or vs.average_rate or Fraction(30, 1)
        est_total = int((end_ms - start_ms) / 1000 * float(rate))
        try:
            # 姿态短时复用表：后端按 ~66ms 采样，逐帧导出时无姿态的帧
            # 复用 100ms 内最近含姿态帧（标注 reused），消除骨架闪烁；不跨更长缺口
            pose_frames = sorted(
                [(f["source_pts_ms"], f["frame_index"]) for f in frames_by_index.values()
                 if (f.get("pose") or {}).get("landmarks")])

            def pose_reuse(rec):
                if rec is None or (rec.get("pose") or {}).get("landmarks"):
                    return rec
                t0 = rec.get("source_pts_ms")
                if t0 is None or not pose_frames:
                    return rec
                best = min(pose_frames, key=lambda pf: abs(pf[0] - t0))
                if abs(best[0] - t0) > 100:
                    return rec
                donor = frames_by_index.get(best[1])
                if not donor:
                    return rec
                return {**rec, "pose": {**donor["pose"], "source": "reused",
                                        "reused_from_ms": best[0] - t0}}

            for fr in iter_frames(source_path, start_ms=start_ms, end_ms=end_ms,
                                  rotation=rotation, should_continue=should_continue):
                rec = pose_reuse(frames_by_index.get(fr.frame_index))
                t = fr.pts_ms
                near = [e for e in events if abs(e["source_time_ms"] - t) <= 350]
                drawn = draw_frame(fr.image, rec, near, wall, roi, layers=layers,
                                   time_label=f"src {t/1000:.3f}s  f{fr.frame_index}",
                                   panel_text=panel_lines)
                if first_w is None:
                    first_h, first_w = drawn.shape[:2]
                    sc = min(1.0, out_h_target / first_h)
                    ow, oh = int(first_w * sc) // 2 * 2, int(first_h * sc) // 2 * 2
                    out = av.open(str(tmp), "w", options={"movflags": "+faststart"})
                    vstream = out.add_stream("libx264", rate=rate)
                    vstream.codec_context.time_base = Fraction(1, 60000)
                    if inp.streams.audio:
                        aout = out.add_stream("aac", rate=44100)
                        aout.layout = "stereo"
                    vstream.width, vstream.height = ow, oh
                    vstream.pix_fmt = "yuv420p"
                    vstream.options = {"crf": "22", "preset": "veryfast"}
                if drawn.shape[1] != vstream.width or drawn.shape[0] != vstream.height:
                    drawn = cv2.resize(drawn, (vstream.width, vstream.height))
                vf = av.VideoFrame.from_ndarray(drawn, format="bgr24")
                vf.pts = (t - start_ms) * 60
                vf.time_base = Fraction(1, 60000)
                for pkt in vstream.encode(vf):
                    out.mux(pkt)
                done += 1
                if progress and done % 30 == 0:
                    progress(done, est_total)
            if should_continue is not None and not should_continue():
                # iter_frames 被取消时静默 return；不检查的话截断视频会被当成
                # 成功导出（tmp.replace 后标记 succeeded）
                from app.store.db import CancelledError
                raise CancelledError("导出已取消")
            if vstream is None:
                raise ValueError("所选时间段没有可导出的视频帧")
            for pkt in vstream.encode():
                out.mux(pkt)

            if aout is not None:
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=44100)
                def encode_audio(frame):
                    # Crop in sample coordinates, then rebase to the exported segment.
                    import math
                    rate = frame.sample_rate
                    first = round(float(frame.pts * frame.time_base) * rate)
                    start = math.ceil(start_ms * rate / 1000)
                    end = math.ceil(end_ms * rate / 1000)
                    lo, hi = max(0, start-first), min(frame.samples, end-first)
                    if hi <= lo:
                        return
                    samples = np.ascontiguousarray(frame.to_ndarray()[:, lo:hi])
                    clipped = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
                    clipped.sample_rate = rate
                    clipped.time_base = Fraction(1, rate)
                    clipped.pts = first + lo - start
                    for pkt in aout.encode(clipped):
                        out.mux(pkt)
                for packet in inp.demux(inp.streams.audio[0]):
                    if should_continue is not None and not should_continue():
                        from app.store.db import CancelledError
                        raise CancelledError("导出已取消")
                    for frame in packet.decode():
                        for converted in resampler.resample(frame):
                            encode_audio(converted)
                for converted in resampler.resample(None):
                    encode_audio(converted)
                for pkt in aout.encode():
                    out.mux(pkt)
            ok = True
        finally:
            if out is not None:
                try:
                    out.close()
                except Exception:
                    ok = False  # 封装失败可能产出损坏文件，不得提升为成品
            if not ok:
                tmp.unlink(missing_ok=True)
    if not ok:
        raise RuntimeError("导出文件封装失败，已清理临时文件")
    tmp.replace(out_path)
    return {"path": str(out_path)}


def write_events_csv(events: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.writer(f)
        wtr.writerow(["event_id", "source_time_ms", "frame_index", "type", "stroke_type",
                      "origin", "review_status", "point_x_px", "point_y_px", "coord_space",
                      "speed_px_s", "speed_kmh", "unavailable_reason", "heuristic_score",
                      "evidence_json"])
        for e in events:
            p = e.get("point") or {}
            wtr.writerow([e["id"], e["source_time_ms"], e.get("frame_index", ""),
                          e["type"], e.get("stroke_type", "unknown"), e.get("origin", "automatic"),
                          e.get("review_status", "unreviewed"),
                          p.get("x", ""), p.get("y", ""), p.get("space", "analysis_image_px"),
                          e.get("speed_px_s") if e.get("speed_px_s") is not None else "",
                          "", e.get("unavailable_reason", ""),
                          e.get("heuristic_score", ""),
                          json.dumps(e.get("evidence", {}), ensure_ascii=False)])
    os.replace(tmp, path)


def write_export_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
