"""几何变换：ROI / letterbox / 旋转 / 显示映射 / 墙面四点。

核心约定：
- “分析图像”= 源帧应用旋转元数据后的图像，坐标空间 analysis_image_px。
- 所有模型结果、事件坐标、ROI、墙面四点均以 analysis_image_px 记录。
- 每次裁剪/缩放往返必须可恢复（crop offset + scale 双向函数）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ---------------- ROI ----------------

@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, img_w: int, img_h: int) -> "Roi":
        x = max(0, min(self.x, img_w - 1))
        y = max(0, min(self.y, img_h - 1))
        w = max(1, min(self.w, img_w - x))
        h = max(1, min(self.h, img_h - y))
        return Roi(x, y, w, h)

    @property
    def x2(self) -> int: return self.x + self.w
    @property
    def y2(self) -> int: return self.y + self.h

    def to_full_image(self, point: tuple[float, float]) -> tuple[float, float]:
        return point[0] + self.x, point[1] + self.y

    def to_roi(self, point: tuple[float, float]) -> tuple[float, float]:
        return point[0] - self.x, point[1] - self.y

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y:self.y2, self.x:self.x2]

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @staticmethod
    def from_dict(d: dict | None, img_w: int = 10**9, img_h: int = 10**9) -> "Roi":
        if not d:
            return Roi(0, 0, img_w, img_h)
        return Roi(int(d.get("x", 0)), int(d.get("y", 0)), int(d.get("w", img_w)), int(d.get("h", img_h))).clamp(img_w, img_h)


# ---------------- letterbox（模型输入缩放）----------------

@dataclass
class Letterbox:
    """crop(Roi) 后图像 → 模型输入 imgsz 的缩放与 padding。目标坐标可无损回映。"""
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    pad_x: float
    pad_y: float

    @staticmethod
    def make(src_w: int, src_h: int, dst_w: int, dst_h: int) -> "Letterbox":
        scale = min(dst_w / src_w, dst_h / src_h)
        new_w, new_h = src_w * scale, src_h * scale
        pad_x, pad_y = (dst_w - new_w) / 2.0, (dst_h - new_h) / 2.0
        return Letterbox(src_w, src_h, dst_w, dst_h, scale, pad_x, pad_y)

    def apply(self, img: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(img, (max(1, int(round(self.src_w * self.scale))),
                                   max(1, int(round(self.src_h * self.scale)))))
        canvas = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        ox, oy = int(round(self.pad_x)), int(round(self.pad_y))
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        return canvas

    def to_src(self, point: tuple[float, float]) -> tuple[float, float]:
        """模型输入坐标 → Roi 内坐标。"""
        return (point[0] - self.pad_x) / self.scale, (point[1] - self.pad_y) / self.scale


# ---------------- 点/框往返测试辅助 ----------------

def bbox_center_xyxy(xyxy: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ---------------- 墙面四点 ----------------

@dataclass
class WallQuad:
    """四点按顺序：左上、右上、右下、左下（墙平面矩形角点在图像中的投影）。"""
    points: list[tuple[float, float]] = field(default_factory=list)
    real_width_m: float | None = None
    real_height_m: float | None = None

    def validate(self) -> list[str]:
        errs: list[str] = []
        if len(self.points) != 4:
            return ["墙面必须为 4 个点"]
        p = self.points
        # 共线/退化：任取三点面积近似 0
        for i in range(4):
            a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
            if area < 1.0:
                errs.append(f"第 {i + 1}/{(i + 1) % 4 + 1}/{(i + 2) % 4 + 1} 点几乎共线（面积 {area:.1f}px²）")
        # 自交：凸四边形对边不相交；用符号法判定
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        sgn = [math.copysign(1, cross(p[i], p[(i + 1) % 4], p[(i + 2) % 4])) for i in range(4)]
        if len(set(sgn)) > 1:
            errs.append("四点顺序构成自交四边形，请按 左上→右上→右下→左下 重新排列")
        if self.real_width_m is not None and self.real_width_m <= 0:
            errs.append("墙宽必须为正数")
        if self.real_height_m is not None and self.real_height_m <= 0:
            errs.append("墙高必须为正数")
        return errs

    def homography_to_unit(self) -> np.ndarray | None:
        """图像 → 墙面单位正方形（0..1）的单应矩阵。退化时返回 None。"""
        if self.validate():
            return None
        src = np.array(self.points, dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        H, _ = cv2_find_homography(src, dst)
        return H

    def map_point_to_wall_unit(self, p: tuple[float, float]) -> tuple[float, float] | None:
        H = self.homography_to_unit()
        if H is None:
            return None
        v = np.array([[[p[0], p[1]]]], dtype=np.float32)
        out = cv2_perspective_transform(v, H)
        x, y = float(out[0][0][0]), float(out[0][0][1])
        return (x, y)

    def polygon_contains(self, p: tuple[float, float]) -> bool:
        """点是否在四边形内（含边界附近判定另见 distance_to_polygon）。"""
        poly = [tuple(q) for q in self.points]
        n, j, inside = len(poly), len(poly) - 1, False
        for i in range(n):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > p[1]) != (yj > p[1])) and (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def distance_to_polygon(self, p: tuple[float, float]) -> float:
        """点到四边形边界/内部的最短距离（在内部为 0）。"""
        if self.polygon_contains(p):
            return 0.0
        poly = self.points + [self.points[0]]
        best = float("inf")
        for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / L2))
            px, py = x1 + t * dx, y1 + t * dy
            best = min(best, math.hypot(p[0] - px, p[1] - py))
        return best

    def to_dict(self) -> dict:
        d: dict = {"points": [list(map(float, p)) for p in self.points]}
        if self.real_width_m:
            d["real_width_m"] = float(self.real_width_m)
        if self.real_height_m:
            d["real_height_m"] = float(self.real_height_m)
        return d

    @staticmethod
    def from_dict(d: dict | None) -> "WallQuad | None":
        if not d or not d.get("points") or len(d["points"]) != 4:
            return None
        return WallQuad(
            points=[(float(p[0]), float(p[1])) for p in d["points"]],
            real_width_m=d.get("real_width_m"),
            real_height_m=d.get("real_height_m"),
        )


def cv2_find_homography(src: np.ndarray, dst: np.ndarray):
    import cv2
    return cv2.findHomography(src, dst, method=0)


def cv2_perspective_transform(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.perspectiveTransform(pts, H)


# ---------------- 显示映射（前端契约，同时供导出复用）----------------

@dataclass
class DisplayTransform:
    """analysis_image_px ↔ 显示 px（等比缩放，无 letterbox 偏移时 offset=0）。"""
    scale: float  # 显示 px / 分析 px
    offset_x: float = 0.0
    offset_y: float = 0.0

    def to_display(self, p: tuple[float, float]) -> tuple[float, float]:
        return p[0] * self.scale + self.offset_x, p[1] * self.scale + self.offset_y

    def to_analysis(self, p: tuple[float, float]) -> tuple[float, float]:
        return (p[0] - self.offset_x) / self.scale, (p[1] - self.offset_y) / self.scale

    @staticmethod
    def fit(analysis_w: int, analysis_h: int, display_w: int, display_h: int) -> "DisplayTransform":
        scale = min(display_w / analysis_w, display_h / analysis_h)
        return DisplayTransform(scale=scale,
                                offset_x=(display_w - analysis_w * scale) / 2,
                                offset_y=(display_h - analysis_h * scale) / 2)


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
