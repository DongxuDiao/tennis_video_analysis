from fractions import Fraction

import av
import numpy as np

from app.video.render import render_export_mp4


def test_export_60fps_segment_retains_audio_and_duration(tmp_path):
    src, dst = tmp_path / 'source.mp4', tmp_path / 'export.mp4'
    with av.open(str(src), 'w') as out:
        v = out.add_stream('libx264', rate=60)
        v.width, v.height, v.pix_fmt = 64, 64, 'yuv420p'
        a = out.add_stream('aac', rate=44100)
        a.layout = 'stereo'
        for i in range(120):
            f = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format='rgb24')
            f.pts, f.time_base = i, Fraction(1, 60)
            for p in v.encode(f):
                out.mux(p)
        for p in v.encode():
            out.mux(p)
        f = av.AudioFrame.from_ndarray(np.zeros((2, 88200), dtype=np.float32), format='fltp', layout='stereo')
        f.sample_rate, f.pts, f.time_base = 44100, 0, Fraction(1, 44100)
        for p in a.encode(f):
            out.mux(p)
        for p in a.encode():
            out.mux(p)
    render_export_mp4(project={}, run={}, frames_by_index={}, events=[], wall=None, roi=None,
                      out_path=dst, layers={}, panel_lines=[], source_path=str(src),
                      start_ms=500, end_ms=1500, rotation=0)
    with av.open(str(dst)) as c:
        assert len(c.streams.audio) == 1
        assert abs(c.duration / 1e6 - 1) < .05
        frames = list(c.decode(video=0))
        assert 59 <= len(frames) <= 61
        assert abs(float(frames[0].pts * frames[0].time_base)) < .02
    with av.open(str(dst)) as c:
        frames = list(c.decode(audio=0))
        assert frames
        assert abs(float(frames[0].pts * frames[0].time_base)) < .03
        assert .95 <= sum(f.samples for f in frames) / 44100 <= 1.05
