"""Step 1 - probe a video, extract frames / audio / background, or grab check frames.

Everything downstream indexes frames by a 0-based frame number, while the files on disk
are 1-based (frames/f_00001.jpg is frame 0). Keeping that in one place avoids the classic
off-by-one that silently shifts every measurement by 33 ms.

  python3 scripts/video_prep.py prep --video clip.mov --work work
  python3 scripts/video_prep.py grab --video work/annotated.mp4 --frames 0 100 540 --out work/chk
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()          # system ffmpeg is usually absent


def probe(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit('cannot open %s' % video)
    meta = dict(video=os.path.abspath(video),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=round(float(cap.get(cv2.CAP_PROP_FPS)), 4),
                n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    meta['duration'] = round(meta['n_frames'] / max(meta['fps'], 1e-6), 3)
    txt = subprocess.run([ffmpeg(), '-hide_banner', '-i', video], capture_output=True, text=True).stderr
    meta['has_audio'] = ' Audio: ' in txt
    import re
    m = re.search(r'rotation of (-?[\d.]+) degrees', txt)
    # ffmpeg applies the display matrix while decoding, so extracted frames are already
    # upright; cv2.VideoCapture may not - never mix the two sources for one analysis.
    meta['rotation_deg'] = float(m.group(1)) if m else 0.0
    return meta


def cmd_prep(a):
    os.makedirs(a.work, exist_ok=True)
    meta = probe(a.video)
    fdir = os.path.join(a.work, 'frames')
    os.makedirs(fdir, exist_ok=True)
    scale = []
    if a.width and a.width != meta['width']:
        scale = ['-vf', 'scale=%d:-2' % a.width]
        meta['scaled_from'] = [meta['width'], meta['height']]
        meta['height'] = int(round(meta['height'] * a.width / meta['width'] / 2) * 2)
        meta['width'] = a.width
    subprocess.run([ffmpeg(), '-y', '-loglevel', 'error', '-i', a.video] + scale +
                   ['-q:v', '2', '-start_number', '1', os.path.join(fdir, 'f_%05d.jpg')], check=True)
    n = len([f for f in os.listdir(fdir) if f.endswith('.jpg')])
    meta['n_frames'], meta['frames_dir'] = n, 'frames'
    meta['duration'] = round(n / max(meta['fps'], 1e-6), 3)

    if meta['has_audio'] and not a.no_audio:
        wav = os.path.join(a.work, 'audio.wav')
        subprocess.run([ffmpeg(), '-y', '-loglevel', 'error', '-i', a.video,
                        '-ac', '1', '-ar', '48000', wav], check=True)
        meta['audio'] = 'audio.wav'

    # median background: static court lines without players, used for calibration and
    # for rejecting motionless clutter in ball detection
    idx = np.linspace(0, n - 1, min(a.bg_samples, n)).astype(int)
    imgs = [cv2.imread(os.path.join(fdir, 'f_%05d.jpg' % (i + 1))) for i in idx]
    bg = np.median(np.stack([i for i in imgs if i is not None]), 0).astype(np.uint8)
    cv2.imwrite(os.path.join(a.work, 'bg_median.png'), bg)
    meta['background'] = 'bg_median.png'

    json.dump(meta, open(os.path.join(a.work, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))
    print('frames -> %s (f_00001.jpg is frame 0)' % fdir)


def cmd_grab(a):
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    want = sorted(set(a.frames))
    got = 0
    for f in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            print('frame %d unavailable' % f)
            continue
        p = os.path.join(a.out, 'f%05d.png' % f)
        cv2.imwrite(p, im)
        got += 1
        print(p, im.shape)
    cap.release()
    print('grabbed %d/%d frames' % (got, len(want)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep', help='probe + extract frames, audio, median background')
    p.add_argument('--video', required=True)
    p.add_argument('--work', default='work')
    p.add_argument('--width', type=int, default=0, help='downscale width (0 = keep original)')
    p.add_argument('--bg-samples', type=int, default=40)
    p.add_argument('--no-audio', action='store_true')
    p.set_defaults(func=cmd_prep)
    g = sub.add_parser('grab', help='save individual frames of any video as PNG for inspection')
    g.add_argument('--video', required=True)
    g.add_argument('--frames', type=int, nargs='+', required=True)
    g.add_argument('--out', default='chk')
    g.set_defaults(func=cmd_grab)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
