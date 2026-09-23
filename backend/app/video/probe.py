"""视频探测：基于 PyAV（自带 FFmpeg），不依赖外部 ffprobe 二进制。"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import av


def _rotation_from_display_matrix(data: bytes) -> int:
    """解析 FFmpeg displaymatrix 侧数据（9 个 int32，行优先）为顺时针旋转角度。"""
    if len(data) < 36:
        return 0
    m = struct.unpack("<9i", data[:36])
    # FFmpeg 约定：displaymatrix[0]*x + displaymatrix[3]*y …；常用简化判定
    a, _, _, c = m[0], m[1], m[2], m[3]
    if a == 0 and c == 65536:
        return 90
    if a == 0 and c == -65536:
        return 270
    if a == -65536 and c == 0:
        return 180
    if a == 65536 and c == 0:
        return 0
    # 通用式（度，顺时针）
    import math
    ang = math.atan2(-c, a) * 180.0 / math.pi
    return int(round(-ang)) % 360


def stream_rotation(stream: av.video.stream.VideoStream) -> int:
    sd = getattr(stream, "side_data", None)
    if sd:
        raw = None
        try:
            raw = sd.get("DISPLAYMATRIX")
        except Exception:
            raw = None
        if raw is not None:
            # PyAV 13 exposes DISPLAYMATRIX on streams as an already-decoded
            # angle (float, e.g. -180.0) for some QuickTime/iPhone files. Other
            # builds expose the original 9-int32 side-data buffer instead.
            if isinstance(raw, (int, float)):
                return int(round(float(raw))) % 360
            try:
                rot = _rotation_from_display_matrix(bytes(raw))
            except (TypeError, ValueError, OverflowError):
                rot = 0
            if rot:
                return rot
    for tag in ("rotate", "rotation"):
        v = (stream.metadata or {}).get(tag)
        if v:
            try:
                return int(v) % 360
            except ValueError:
                pass
    return 0


def probe_video(path: Path) -> dict[str, Any]:
    """返回规范化元信息；不能只信扩展名，实际打开容器检查视频轨。"""
    with av.open(str(path)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        if not vstreams:
            raise ValueError("文件中没有视频轨（video stream）")
        vs = vstreams[0]
        astreams = [s for s in container.streams if s.type == "audio"]
        duration_s = float(container.duration or 0) / 1_000_000 if container.duration else None
        if duration_s is None and vs.duration is not None:
            duration_s = float(vs.duration) * float(vs.time_base)
        if duration_s is None:
            # 兜底：duration 由 nb_frames/avg_frame_rate 估计，标记为估计值
            fps = _safe_fps(vs)
            if fps and vs.nb_frames:
                duration_s = vs.nb_frames / fps
        fps = _safe_fps(vs)
        nb_frames = int(getattr(vs, "nb_frames", 0) or 0)
        probe = {
            "container": container.format.name or "",
            "format_long": container.format.long_name or "",
            "video_codec": vs.codec_context.name if vs.codec_context else "",
            "width": int(vs.codec_context.width) if vs.codec_context else int(vs.width or 0),
            "height": int(vs.codec_context.height) if vs.codec_context else int(vs.height or 0),
            "coded_width": int(vs.width or 0),
            "coded_height": int(vs.height or 0),
            "pix_fmt": (vs.codec_context.pix_fmt if vs.codec_context and vs.codec_context.pix_fmt else ""),
            "nominal_fps": fps,
            "duration_s": duration_s,
            "duration_ms": int((duration_s or 0) * 1000),
            "rotation": stream_rotation(vs),
            "nb_frames": nb_frames,
            "has_audio": len(astreams) > 0,
            "audio_codec": astreams[0].codec_context.name if astreams and astreams[0].codec_context else None,
            "bit_rate": int(container.bit_rate) if container.bit_rate else None,
            "time_base": str(vs.time_base),
            "is_vfr_possible": bool(fps and nb_frames and fps > 0
                                     and abs((nb_frames / fps) - (duration_s or 0)) > 0.05),
        }
        if probe["width"] <= 0 or probe["height"] <= 0:
            raise ValueError("无法读取视频尺寸")
        if probe["duration_ms"] <= 0:
            raise ValueError("无法读取视频时长（duration 缺失且无法估计）")
        return probe


def _safe_fps(vs: av.video.stream.VideoStream) -> float | None:
    if vs.average_rate and vs.average_rate.denominator != 0:
        return float(vs.average_rate.numerator) / float(vs.average_rate.denominator)
    return None


def analysis_dims(probe: dict[str, Any]) -> tuple[int, int]:
    """旋转修正后的分析图像尺寸（宽, 高）。"""
    w, h, rot = probe["width"], probe["height"], probe.get("rotation", 0)
    if rot in (90, 270):
        return h, w
    return w, h


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
