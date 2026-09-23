"""代理视频转码：H.264 / yuv420p / faststart，旋转烧录，音频尽量保留。

原片保留不改写；代理仅用于浏览器播放与人工目视，模型推理不使用代理。
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import numpy as np

from app.video.decode import frame_to_bgr_ndarray
from app.video.probe import stream_rotation

PROXY_MAX_HEIGHT = 1080


def transcode_proxy(src: str, dst: Path, *, progress: Callable[[int, int], None] | None = None,
                    should_continue: Callable[[], bool] | None = None,
                    max_height: int = PROXY_MAX_HEIGHT) -> dict:
    """生成浏览器可播放代理。返回代理元信息。进度回调 (frames_done, frames_total_est)。

    实现为逐帧软件转码（解码→旋转→缩放→libx264 编码），保证旋转与缩放行为确定。
    """
    import cv2
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    with av.open(src) as inp:
        vs = inp.streams.video[0]
        rotation = stream_rotation(vs)
        src_w = vs.codec_context.width if vs.codec_context else vs.width
        src_h = vs.codec_context.height if vs.codec_context else vs.height
        if rotation in (90, 270):
            src_w, src_h = src_h, src_w
        scale = min(1.0, max_height / float(src_h))
        out_w, out_h = int(round(src_w * scale)) // 2 * 2, int(round(src_h * scale)) // 2 * 2

        astream = None
        for s in inp.streams:
            if s.type == "audio":
                astream = s
                break

        out = av.open(str(tmp), "w", options={"movflags": "+faststart"})
        try:
            # Keep the source cadence. Encoding every source frame at a fixed
            # 30 fps doubles the duration of common 60 fps phone recordings.
            # ``average_rate`` in MOV metadata can be an odd container-wide
            # estimate (the failing clip reports 19200/323). ``guessed_rate``
            # reflects its actual 60000/1001 frame cadence.
            source_rate = vs.guessed_rate or vs.base_rate or vs.average_rate or Fraction(30, 1)
            vstream = out.add_stream("libx264", rate=source_rate)
            vstream.width, vstream.height = out_w, out_h
            vstream.pix_fmt = "yuv420p"
            vstream.options = {"crf": "23", "preset": "veryfast"}
            astream_out = None
            if astream is not None:
                astream_out = out.add_stream("aac", rate=44100)
                astream_out.layout = "stereo"

            # PyAV 13 exposes the container-reported frame count as
            # ``VideoStream.frames``.  ``nb_frames`` belongs to older FFmpeg
            # bindings and raises through codec-context attribute forwarding.
            total_est = int(vs.frames or 0)
            done = 0
            import queue as _q
            for packet in inp.demux(vs):
                if should_continue is not None and not should_continue():
                    out.close()
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("转码被取消")
                for frame in packet.decode():
                    bgr = frame_to_bgr_ndarray(frame, rotation)
                    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
                        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    vf = av.VideoFrame.from_ndarray(bgr, format="bgr24")
                    for pkt in vstream.encode(vf):
                        out.mux(pkt)
                    done += 1
                    if progress and (done % 30 == 0 or done == total_est):
                        progress(done, total_est)
            for pkt in vstream.encode():
                out.mux(pkt)

            if astream is not None and astream_out is not None:
                # The video-only demux loop has advanced ``inp`` to EOF. Open
                # a fresh container for audio instead of silently producing a
                # proxy without the source soundtrack.
                with av.open(src) as audio_inp:
                    audio_src = next(s for s in audio_inp.streams if s.type == "audio")
                    resampler = av.AudioResampler(
                        format=astream_out.format, layout=astream_out.layout, rate=astream_out.rate
                    )
                    for packet in audio_inp.demux(audio_src):
                        for frame in packet.decode():
                            for audio_frame in resampler.resample(frame):
                                for pkt in astream_out.encode(audio_frame):
                                    out.mux(pkt)
                    for audio_frame in resampler.resample(None):
                        for pkt in astream_out.encode(audio_frame):
                            out.mux(pkt)
                for pkt in astream_out.encode():
                    out.mux(pkt)
        finally:
            try:
                out.close()
            except Exception:
                pass

    # 原子替换
    tmp.replace(dst)
    return {"path": str(dst), "width": out_w, "height": out_h, "rotation_burned": bool(rotation)}
