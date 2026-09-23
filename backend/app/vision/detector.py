"""YOLO11 检测封装：整图基线 / 分块检测 / 局部高分辨率跟踪三种模式。

坐标约定：所有输出 bbox/center 均为 analysis_image_px。
分块与局部模式都会把候选回映到全图坐标，并保留来源信息供调试。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from app.vision.geometry import Roi, bbox_center_xyxy


@dataclass
class Candidate:
    cls_name: str
    confidence: float          # 模型原始置信度，不称“准确率”
    bbox: tuple[float, float, float, float]   # analysis_image_px, xyxy
    center: tuple[float, float]
    source: str = "yolo11n"    # 模型标识（统一接口，便于未来接入专用球检测器）
    tile: tuple[int, int] | None = None       # 分块来源 (row, col)；整图/局部为 None
    tile_count: int = 1                       # 分块合并时覆盖该候选的块数


@dataclass
class DetectionResult:
    candidates: list[Candidate] = field(default_factory=list)
    elapsed_ms: float = 0.0
    device: str = "cpu"
    attempts: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "full_frame"                   # full_frame | tiled | local
    tiles: list[Roi] = field(default_factory=list)
    search_crop: tuple[int, int, int, int] | None = None


BALL_NAMES = {"sports ball"}
RACKET_NAMES = {"tennis racket"}
PERSON_NAMES = {"person"}


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


@lru_cache(maxsize=16)
def plan_tiles(img_w: int, img_h: int, cols: int, rows: int,
               overlap: float) -> list[tuple[Roi, tuple[int, int]]]:
    """划分带重叠分块。返回 [(Roi, (row, col))]。

    overlap 为相邻块重叠比例（相对块尺寸，0..0.5）。
    tile_w = img_w / (cols - (cols-1)*overlap)，步进 = tile_w*(1-overlap)；
    最后一块对齐到图像边缘，避免边缘覆盖缺口。

    结果按 (w, h, cols, rows, overlap) 缓存：视频尺寸与分块参数固定时
    每帧重算是纯浪费。返回的 Roi 视为只读，调用方不得原地修改。
    """
    overlap = max(0.0, min(0.5, overlap))

    def axis_span(n: int, length: int) -> list[tuple[int, int]]:
        if n <= 1:
            return [(0, length)]
        tile_len = length / (n - (n - 1) * overlap)
        step = tile_len * (1 - overlap)
        spans = []
        for i in range(n):
            s = int(round(i * step))
            e = length if i == n - 1 else min(length, int(round(s + tile_len)))
            spans.append((s, max(s + 1, e)))
        return spans

    tiles = []
    for r, (y1, y2) in enumerate(axis_span(rows, img_h)):
        for c, (x1, x2) in enumerate(axis_span(cols, img_w)):
            tiles.append((Roi(x1, y1, x2 - x1, y2 - y1), (r, c)))
    return tiles


class Detector:
    """统一检测接口：整图 / 分块 / 局部。"""

    def __init__(self, model_path: str | None = None, *, device: str = "auto_mps_then_cpu",
                 imgsz: int = 960, ball_confidence: float = 0.15,
                 racket_confidence: float = 0.25, person_confidence: float = 0.35):
        self.model_path = model_path
        self.device_policy = device
        self.imgsz = imgsz
        self.conf_by_cls = {
            "sports ball": ball_confidence,
            "tennis racket": racket_confidence,
            "person": person_confidence,
        }
        self._model = None
        self._resolved_device: str | None = None
        self.model_hash: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        import hashlib
        self._model = YOLO(self.model_path)
        with open(self.model_path, "rb") as f:
            self.model_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    def _mps_available(self) -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def _resolve_device(self) -> str:
        if self._resolved_device:
            return self._resolved_device
        if self.device_policy == "cpu":
            self._resolved_device = "cpu"
        elif self.device_policy == "mps":
            if not self._mps_available():
                raise RuntimeError("指定 device=mps 但 torch.backends.mps 不可用")
            self._resolved_device = "mps"
        else:
            self._resolved_device = "mps" if self._mps_available() else "cpu"
        return self._resolved_device

    # ---- 底层：对单张/多张（裁剪）图推理；结果坐标 = 裁剪图坐标 ----
    def _run_predict(self, imgs):
        """执行一次 predict（imgs 可为单图或图列表），含 MPS→CPU 回退。

        返回 (results, attempts, 实际成功的设备)。"""
        self._ensure_loaded()
        attempts: list[dict] = []
        dev = self._resolve_device()
        devices = [dev] if dev == "cpu" or self.device_policy != "auto_mps_then_cpu" else [dev, "cpu"]
        out, last_err, used = None, None, None
        for d in devices:
            t0 = time.perf_counter()
            try:
                out = self._model.predict(imgs, imgsz=self.imgsz, device=d,
                                          conf=min(self.conf_by_cls.values()),
                                          verbose=False, iou=0.5)
                attempts.append({"device": d, "ok": True,
                                 "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})
                self._resolved_device = d
                used = d
                break
            except Exception as e:  # MPS 失败回退 CPU，整次重跑（不拼接）
                attempts.append({"device": d, "ok": False, "error": str(e)[:300],
                                 "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})
                last_err = e
        if out is None:
            raise RuntimeError(f"检测在所有设备上失败：{last_err}")
        return out, attempts, used

    def _parse_result(self, r) -> list[Candidate]:
        """单个 predict result → 候选列表。

        张量一次性 .cpu().numpy() 后再迭代：逐框 .item()/.tolist() 每次
        都触发一次主机-设备同步（MPS 上是热点循环里最贵的隐藏开销）。
        """
        cands: list[Candidate] = []
        boxes = r.boxes
        if boxes is None:
            return cands
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        names = self._model.names
        for i in range(len(boxes)):
            name = names.get(int(cls_arr[i]), str(cls_arr[i]))
            conf = float(conf_arr[i])
            if conf < self.conf_by_cls.get(name, 1.0):
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            cands.append(Candidate(cls_name=name, confidence=round(conf, 4),
                                   bbox=(x1, y1, x2, y2),
                                   center=bbox_center_xyxy((x1, y1, x2, y2))))
        return cands

    def _infer_crop(self, crop: np.ndarray) -> tuple[list[Candidate], list[dict], float, str]:
        out, attempts, used = self._run_predict(crop)
        elapsed = sum(a["elapsed_ms"] for a in attempts)
        return self._parse_result(out[0]), attempts, elapsed, used

    def _infer_crops(self, crops: list[np.ndarray]) -> list[tuple[list[Candidate], list[dict], float, str]]:
        """多张裁剪图推理：合并为一次批量 predict。

        分块检测的所有 tile 同尺寸，批量调用共享预处理与设备提交，
        省掉逐块调用的固定开销。返回与输入一一对应的
        (candidates, attempts, elapsed_ms, device)；attempts/elapsed 记在
        首个条目上（同一次调用）。批量路径异常时退回逐张，语义不变。
        """
        if len(crops) <= 1:
            return [self._infer_crop(crops[0])]
        try:
            out, attempts, used = self._run_predict(list(crops))
        except Exception:
            return [self._infer_crop(crop) for crop in crops]
        elapsed = sum(a["elapsed_ms"] for a in attempts)
        parsed = [self._parse_result(r) for r in out]
        return ([(parsed[0], attempts, elapsed, used)]
                + [(p, [], 0.0, used) for p in parsed[1:]])

    def detect(self, full_img: np.ndarray, roi: Roi | None = None,
               mode: str = "full_frame", imgsz: int | None = None) -> DetectionResult:
        """整图（或给定 ROI）基线检测。imgsz 可临时覆盖（如廉价人物 pass）。"""
        saved = self.imgsz
        if imgsz:
            self.imgsz = imgsz
        try:
            return self._detect_impl(full_img, roi, mode)
        finally:
            self.imgsz = saved

    def _detect_impl(self, full_img: np.ndarray, roi: Roi | None, mode: str) -> DetectionResult:
        result = DetectionResult(mode=mode)
        crop = full_img if roi is None else roi.crop(full_img)
        cands, attempts, elapsed, device = self._infer_crop(crop)
        ox, oy = (0, 0) if roi is None else (roi.x, roi.y)
        result.attempts = attempts
        result.elapsed_ms = elapsed
        result.device = device
        for c in cands:
            bb = (c.bbox[0] + ox, c.bbox[1] + oy, c.bbox[2] + ox, c.bbox[3] + oy)
            result.candidates.append(Candidate(c.cls_name, c.confidence, bb,
                                               bbox_center_xyxy(bb), c.source))
        return result

    def detect_tiled(self, full_img: np.ndarray, *, cols: int = 3, rows: int = 2,
                     overlap: float = 0.25, merge_iou: float = 0.45) -> DetectionResult:
        """分块全画面检测：所有块合并为一次批量推理，回映后跨块合并重复候选。"""
        h, w = full_img.shape[:2]
        tiles = plan_tiles(w, h, cols, rows, overlap)
        result = DetectionResult(mode="tiled")
        result.tiles = [t for t, _ in tiles]
        crops = [roi.crop(full_img) for roi, _ in tiles]
        batch = self._infer_crops(crops)
        merged: list[Candidate] = []
        for (roi, (r, c)), (cands, attempts, elapsed, device) in zip(tiles, batch):
            result.attempts.extend(attempts)
            result.elapsed_ms += elapsed
            result.device = device
            for cand in cands:
                bb = (cand.bbox[0] + roi.x, cand.bbox[1] + roi.y,
                      cand.bbox[2] + roi.x, cand.bbox[3] + roi.y)
                new = Candidate(cand.cls_name, cand.confidence, bb, bbox_center_xyxy(bb),
                                tile=(r, c))
                # 跨块合并：同类且 IoU 高 → 保留高置信度者，记录覆盖块数
                dup = None
                for m in merged:
                    if m.cls_name == new.cls_name and _iou(m.bbox, new.bbox) >= merge_iou:
                        dup = m
                        break
                if dup is None:
                    merged.append(new)
                else:
                    if new.confidence > dup.confidence:
                        new.tile_count = getattr(dup, "tile_count", 1) + 1
                        merged[merged.index(dup)] = new
                    else:
                        dup.tile_count = getattr(dup, "tile_count", 1) + 1
        result.candidates = merged
        return result

    def detect_local(self, full_img: np.ndarray, center: tuple[float, float],
                     radius: float, min_size: int = 320,
                     imgsz: int | None = None) -> DetectionResult:
        """局部高分辨率跟踪：预测点周围裁剪（放大送入模型），坐标回映全图。

        局部裁剪默认与整图同输入尺寸：小球在裁剪内被上采样后检出率显著更高
        （实测 640 输入覆盖率大幅下降），需要提速时由调用方显式传更小 imgsz。
        """
        h, w = full_img.shape[:2]
        r = max(min_size // 2, int(round(radius)))
        cx, cy = int(round(center[0])), int(round(center[1]))
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return self.detect(full_img)  # 裁剪不可行时退化为整图
        roi = Roi(x1, y1, x2 - x1, y2 - y1)
        result = self.detect(full_img, roi, mode="local", imgsz=imgsz or self.imgsz)
        result.search_crop = (x1, y1, x2, y2)
        return result

    def ball_candidates(self, r: DetectionResult) -> list[Candidate]:
        return [c for c in r.candidates if c.cls_name in BALL_NAMES]

    def racket_candidates(self, r: DetectionResult) -> list[Candidate]:
        return [c for c in r.candidates if c.cls_name in RACKET_NAMES]

    def person_candidates(self, r: DetectionResult) -> list[Candidate]:
        return [c for c in r.candidates if c.cls_name in PERSON_NAMES]
