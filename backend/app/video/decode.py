"""流式解码：带 PTS 的逐帧读取、旋转归一化、片段裁剪、精确取帧。

所有输出帧均为“旋转修正后的分析图像”方向（numpy HxWx3, BGR），
坐标空间约定：分析图像像素（analysis_image_px）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import av
import numpy as np

from app.video.probe import stream_rotation


@dataclass
class DecodedFrame:
    frame_index: int      # 源视频流的帧序号（从 0 起，按解码顺序累计）
    pts_ms: int           # 源视频时间轴上的 PTS（毫秒，int）
    image: np.ndarray     # HxWx3 BGR，已应用旋转
    rotated: bool


def _rotate(img: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return np.ascontiguousarray(np.rot90(img, k=-1))   # 顺时针 90
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(img, k=2))
    if rotation == 270:
        return np.ascontiguousarray(np.rot90(img, k=1))    # 逆时针 90（即顺时针 270）
    return img


def frame_to_bgr_ndarray(frame: av.VideoFrame, rotation: int) -> np.ndarray:
    """解码帧 → BGR ndarray，并应用旋转元数据。"""
    img = frame.to_ndarray(format="bgr24")
    if rotation:
        img = _rotate(img, rotation)
    return np.ascontiguousarray(img)


def iter_frames(path: str, *, start_ms: int | None = None, end_ms: int | None = None,
                rotation: int | None = None,
                should_continue: Callable[[], bool] | None = None) -> Iterator[DecodedFrame]:
    """流式解码源视频；可选时间过滤（源时间轴，毫秒）。

    内存约束：一次只在内存保留一帧。
    """
    with av.open(path) as container:
        vs = container.streams.video[0]
        if rotation is None:
            rotation = stream_rotation(vs)
        tb_num, tb_den = vs.time_base.numerator, vs.time_base.denominator
        idx = 0
        for packet in container.demux(vs):
            if should_continue is not None and not should_continue():
                return
            for frame in packet.decode():
                pts = frame.pts
                if pts is None:
                    idx += 1
                    continue
                pts_ms = int(pts * tb_num * 1000 / tb_den)
                if start_ms is not None and pts_ms < start_ms:
                    idx += 1
                    continue
                if end_ms is not None and pts_ms > end_ms:
                    return
                img = frame_to_bgr_ndarray(frame, rotation)
                yield DecodedFrame(frame_index=idx, pts_ms=pts_ms, image=img, rotated=bool(rotation))
                idx += 1


def decode_exact_frame(path: str, frame_index: int) -> tuple[np.ndarray | None, int, int]:
    """按帧序号精确取帧。返回 (BGR 分析方向图像或 None, 实际帧序号, pts_ms)。

    后向 seek 只保证落在目标之前的关键帧；seek 之后从 0 重数帧号会整体
    偏移一个 GOP。因此用落点帧的 pts 反推其真实序号（CFR 下精确），反推
    不可信时回退为从头顺序解码计数。
    """
    with av.open(path) as container:
        vs = container.streams.video[0]
        rotation = stream_rotation(vs)
        tb = vs.time_base
        fps = (vs.average_rate.numerator / vs.average_rate.denominator
               if vs.average_rate and vs.average_rate.denominator else None)
        target_ms = int(frame_index / fps * 1000) if fps else 0
        if fps:
            try:
                container.seek(int(target_ms / 1000 / tb), stream=vs, backward=True, any_frame=False)
            except av.AVError:
                pass

        def _scan(infer: bool) -> tuple[np.ndarray | None, int, int]:
            idx = None
            for packet in container.demux(vs):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    pts_ms = int(frame.pts * tb.numerator * 1000 / tb.denominator)
                    if idx is None:
                        if infer and fps:
                            idx = max(0, round(pts_ms * fps / 1000))
                            if idx > frame_index:
                                return None, -1, -1  # 反推已越过目标：需从头重扫
                        else:
                            idx = 0
                    if idx == frame_index:
                        return frame_to_bgr_ndarray(frame, rotation), idx, pts_ms
                    idx += 1
                    if idx > frame_index + 5000:
                        return None, idx, -1
            return None, (idx if idx is not None else 0), -1

        out, idx, pts_ms = _scan(infer=True)
        if out is None and idx == -1:
            try:
                container.seek(0, stream=vs, any_frame=False)
            except av.AVError:
                pass
            out, idx, pts_ms = _scan(infer=False)
        return out, idx, pts_ms


def decode_frame_by_pts(path: str, pts_ms: int) -> tuple[np.ndarray | None, int, int]:
    """按 PTS（源毫秒）就近取帧（<= pts_ms 的最近帧）。"""
    with av.open(path) as container:
        vs = container.streams.video[0]
        rotation = stream_rotation(vs)
        tb = vs.time_base
        try:
            container.seek(int(pts_ms / 1000 / tb) - 1, stream=vs, backward=True, any_frame=False)
        except av.AVError:
            pass
        best = None
        for packet in container.demux(vs):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                f_ms = int(frame.pts * tb.numerator * 1000 / tb.denominator)
                if f_ms > pts_ms:
                    if best:
                        return frame_to_bgr_ndarray(best[0], rotation), best[1], best[2]
                    return None, -1, -1
                best = (frame, 0, f_ms)
                if pts_ms - f_ms < 1:
                    return frame_to_bgr_ndarray(frame, rotation), 0, f_ms
        if best:
            return frame_to_bgr_ndarray(best[0], rotation), 0, best[2]
        return None, -1, -1
