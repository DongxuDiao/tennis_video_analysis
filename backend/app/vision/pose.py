"""MediaPipe Pose Landmarker（Full）封装：VIDEO 模式、严格递增时间戳、状态重置。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# MediaPipe Pose 33 关键点索引（官方常量）
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}
# 分析图像归一化坐标（0..1，相对送入帧）
VIS_THRESHOLD = 0.5


@dataclass
class PoseResult:
    landmarks: list[dict[str, float]] | None = None   # 33 个 {x,y,z,visibility}，分析图像 px（x,y 已乘宽高）
    elapsed_ms: float = 0.0
    person_found: bool = False
    crop_box: tuple[int, int, int, int] | None = None  # 目标区域裁剪来源（None=全图）


class PoseEstimator:
    def __init__(self, model_path: str, delegate: str = "CPU"):
        self.model_path = model_path
        self.delegate = delegate
        self._det = None
        self._last_ts = -1
        self.model_hash: str | None = None

    def _ensure(self) -> None:
        if self._det is not None:
            return
        import hashlib
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path, delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._det = vision.PoseLandmarker.create_from_options(opts)
        with open(self.model_path, "rb") as f:
            self.model_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    def reset(self) -> None:
        """新 run / 新时间段 / 倒序检查时必须调用，避免时序缓存污染。"""
        self._last_ts = -1

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> PoseResult:
        """timestamp_ms 必须严格递增；同帧复用最近结果时由调用方控制。"""
        self._ensure()
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        import mediapipe as mp
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        t0 = time.perf_counter()
        res = self._det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        elapsed = (time.perf_counter() - t0) * 1000
        out = PoseResult(elapsed_ms=round(elapsed, 2))
        if res.pose_landmarks:
            out.person_found = True
            lms = res.pose_landmarks[0]
            out.landmarks = [
                {"x": float(lm.x) * w, "y": float(lm.y) * h,
                 "z": float(lm.z), "visibility": float(lm.visibility)}
                for lm in lms
            ]
        return out

    def close(self) -> None:
        if self._det is not None:
            self._det.close()
            self._det = None

    def detect_on_crop(self, full_bgr: np.ndarray,
                       crop_box: tuple[int, int, int, int], timestamp_ms: int) -> PoseResult:
        """在目标人物区域（带边距裁剪）上做姿态，关键点回映到全图分析坐标。

        裁剪保证挥拍时手腕不被画面边缘裁掉；crop 失败时回退全图。
        """
        x1, y1, x2, y2 = crop_box
        h, w = full_bgr.shape[:2]
        if x2 - x1 < 64 or y2 - y1 < 64 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            res = self.detect(full_bgr, timestamp_ms)
            res.crop_box = None
            return res
        crop = full_bgr[y1:y2, x1:x2]
        res = self.detect(crop, timestamp_ms)
        if res.landmarks:
            res.landmarks = [dict(lm, x=lm["x"] + x1, y=lm["y"] + y1) for lm in res.landmarks]
        res.crop_box = (x1, y1, x2, y2)
        return res

    @staticmethod
    def wrist_of(landmarks: list[dict] | None, handedness: str) -> dict | None:
        """持拍手侧腕关键点。handedness: right/left/unknown（解剖学方向）。"""
        if not landmarks or handedness not in ("right", "left"):
            return None
        idx = LM["right_wrist"] if handedness == "right" else LM["left_wrist"]
        lm = landmarks[idx]
        return lm if lm.get("visibility", 0) >= VIS_THRESHOLD else None

    @staticmethod
    def body_midline_x(landmarks: list[dict] | None) -> float | None:
        """躯干中线 x（肩、髋可见点的均值）；不可见返回 None。"""
        if not landmarks:
            return None
        xs = []
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            lm = landmarks[LM[name]]
            if lm.get("visibility", 0) >= VIS_THRESHOLD:
                xs.append(lm["x"])
        return sum(xs) / len(xs) if xs else None
