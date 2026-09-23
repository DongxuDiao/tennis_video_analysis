"""生成合成工程测试视频（SYNTHETIC，明确标记，不用于算法准确率证明）。

用途：验证导入/解码/推理/追踪/事件/导出的工程闭环与坐标正确性。
画面：固定机位视角——左侧人体简笔（挥拍动作），右侧竖直墙线，
一个黄绿色圆球在人与墙之间往返，每次在球拍附近反弹计为一次“击球”，
在墙线处反弹计为“撞墙”。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import av
import cv2
import numpy as np

W, H = 1280, 720
FPS = 30
WALL_X = 1050          # 墙线位置（analysis px）
PLAYER_X = 260
FLOOR_Y = 640
BALL_R = 9
SYNTH_TAG = "SYNTHETIC-FIXTURE (engineering test only, not for accuracy claims)"


def draw_person(img: np.ndarray, t: float, swing_phase: float) -> tuple[int, int]:
    """简笔人体 + 挥拍手臂；返回当前持拍端（球拍）位置。"""
    cx, cy = PLAYER_X, 430
    swing = math.sin(swing_phase)  # -1..1
    # 躯干
    cv2.rectangle(img, (cx - 22, cy - 60), (cx + 22, cy + 90), (70, 60, 200), -1)
    # 头
    cv2.circle(img, (cx, cy - 95), 26, (70, 60, 200), -1)
    # 持拍臂（右臂，绕肩摆动）
    shoulder = (cx + 16, cy - 50)
    ang = -0.9 + swing * 1.5  # 挥拍角度
    arm_len = 95
    hand = (int(shoulder[0] + arm_len * math.sin(ang)), int(shoulder[1] + arm_len * math.cos(ang) * 0.6))
    cv2.line(img, shoulder, hand, (90, 80, 210), 12)
    # 球拍框
    cv2.circle(img, hand, 26, (60, 90, 230), 6)
    return hand


def make_clip(path: Path, seconds: float = 12.0, hits: int = 8) -> dict:
    frames = int(seconds * FPS)
    out = av.open(str(path), "w")
    vs = out.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"crf": "22", "preset": "veryfast"}

    ball_x, ball_y = 700.0, 500.0
    vx = 520.0  # px/s 初始向墙
    hit_period = seconds / hits
    events_meta = []

    rng = np.random.default_rng(7)
    t = 0.0
    for fi in range(frames):
        t = fi / FPS
        img = np.full((H, W, 3), (40, 44, 52), dtype=np.uint8)
        # 地面与墙
        cv2.line(img, (0, FLOOR_Y + 40), (W, FLOOR_Y + 40), (60, 70, 60), 3)
        cv2.rectangle(img, (WALL_X, 60), (W, H), (90, 95, 90), -1)
        cv2.putText(img, "WALL (synthetic)", (WALL_X + 15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        # 击球相位：0..hit_period 循环
        phase = (t % hit_period) / hit_period
        swing_phase = t / hit_period * 2 * math.pi
        hand = draw_person(img, t, swing_phase)
        # 球运动：到达墙线或球拍附近反弹
        prev_x = ball_x
        ball_x += vx / FPS
        ball_y = 500 + 40 * math.sin(t * 2.2)  # 轻微竖直起伏
        # 球拍击球：球向左运动且接近球拍手
        if vx < 0 and abs(ball_x - hand[0]) < 40 and abs(ball_y - hand[1]) < 70:
            vx = abs(vx)
            events_meta.append({"t_ms": int(t * 1000), "type": "player_hit"})
            ball_x = hand[0] + 42
        # 撞墙：球到达墙线
        if ball_x >= WALL_X - BALL_R - 2 and vx > 0:
            vx = -abs(vx)
            ball_x = WALL_X - BALL_R - 2
            events_meta.append({"t_ms": int(t * 1000), "type": "wall_contact"})
        # 落地约束（不让球出画面）
        ball_x = max(20, min(ball_x, W - 20))
        # 球（黄绿）
        cv2.circle(img, (int(ball_x), int(ball_y)), BALL_R, (40, 235, 235), -1)
        cv2.circle(img, (int(ball_x), int(ball_y)), BALL_R + 2, (255, 255, 255), 1)
        # 时间码 + 合成标记
        cv2.putText(img, f"{t:06.2f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, SYNTH_TAG, (20, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        vf = av.VideoFrame.from_ndarray(img, format="bgr24")
        for pkt in vs.encode(vf):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()
    return {"path": str(path), "duration_s": seconds, "fps": FPS,
            "synthetic_events": events_meta,
            "note": "合成素材；事件时间仅用于工程对齐检查，不构成算法准确率基准"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fixtures/synthetic_wall_practice.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--hits", type=int, default=8)
    args = ap.parse_args()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = make_clip(p, args.seconds, args.hits)
    import json
    (p.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("written:", meta["path"], "events:", len(meta["synthetic_events"]))


if __name__ == "__main__":
    main()
