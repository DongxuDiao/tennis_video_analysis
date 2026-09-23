"""Step 5 - per-frame ball candidates -> ball_cand.json

A ball is a few pixels across, so no detector "just works": the only reliable signal is
colour + motion + size, minus everything that is coloured like a ball but does not move
(spare balls on the floor, green fences, logos). The output is a *candidate* list; the
racket frame, shoe logos and motion blur will still leak in, and the trajectory step is
where they get rejected.

  python3 scripts/ball_detect.py --work work --colour yellow_green
  python3 scripts/ball_detect.py --work work --colour white --min-area 3 --max-dim 20

Candidate fields: x,y (centroid px), a (area px), w,h (bbox), sat/val (mean HSV), gb (mean
G-B, a tennis-ball specific cue), n (candidates in that frame). Frames with exactly one
candidate are the ones worth trusting first.
"""
import argparse
import json
import os

import cv2
import numpy as np

PRESETS = {
    # OpenCV HSV: H in 0..179
    'yellow_green': dict(h=(22, 48), s=(55, 255), v=(80, 255)),
    'white':        dict(h=(0, 179), s=(0, 60), v=(170, 255)),
    'orange':       dict(h=(3, 22), s=(90, 255), v=(110, 255)),
    'any':          dict(h=(0, 179), s=(0, 255), v=(0, 255)),
}


def colour_mask(im, p):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((h >= p['h'][0]) & (h <= p['h'][1]) & (s >= p['s'][0]) & (s <= p['s'][1]) &
            (v >= p['v'][0]) & (v <= p['v'][1])).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work')
    ap.add_argument('--colour', '--color', dest='colour', default='yellow_green', choices=list(PRESETS))
    ap.add_argument('--min-area', type=int, default=4)
    ap.add_argument('--max-area', type=int, default=400)
    ap.add_argument('--max-dim', type=int, default=26, help='reject blobs wider/taller than this')
    ap.add_argument('--aspect', type=float, default=5.0)
    ap.add_argument('--row-min', type=int, default=0, help='ignore rows above this (background clutter)')
    ap.add_argument('--bg-thresh', type=int, default=22, help='grey difference vs the median background')
    ap.add_argument('--diff-thresh', type=int, default=18, help='grey difference vs prev/next frame')
    ap.add_argument('--dilate-static', type=int, default=7)
    ap.add_argument('--out', default='ball_cand.json')
    a = ap.parse_args()

    fdir = os.path.join(a.work, 'frames')
    n = len([f for f in os.listdir(fdir) if f.endswith('.jpg')])
    p = PRESETS[a.colour]
    bg = cv2.imread(os.path.join(a.work, 'bg_median.png'))
    bgg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)
    static = cv2.dilate(colour_mask(bg, p), np.ones((a.dilate_static,) * 2, np.uint8))
    cv2.imwrite(os.path.join(a.work, 'static_mask.png'), static * 255)
    print('static ball-coloured pixels: %d (%.2f%% of the frame)'
          % (static.sum(), 100 * static.mean()))

    out, prev = {}, None
    cur = cv2.imread(os.path.join(fdir, 'f_%05d.jpg' % 1))
    for f in range(n):
        nxt = cv2.imread(os.path.join(fdir, 'f_%05d.jpg' % (f + 2))) if f + 1 < n else None
        g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.int16)
        mov = (np.abs(g - bgg) > a.bg_thresh).astype(np.uint8)
        if prev is not None and nxt is not None:
            d1 = np.abs(g - cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.int16))
            d2 = np.abs(g - cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY).astype(np.int16))
            mov = np.maximum(mov, ((d1 > a.diff_thresh) & (d2 > a.diff_thresh)).astype(np.uint8))
        m = (colour_mask(cur, p) & mov & (1 - static)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        nl, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
        lst = []
        for i in range(1, nl):
            x, y, w, h, area = st[i]
            if not (a.min_area <= area <= a.max_area) or max(w, h) > a.max_dim:
                continue
            if max(w, h) / max(1.0, min(w, h)) > a.aspect:
                continue
            cx, cy = ce[i]
            if cy < a.row_min:
                continue
            sel = (lab[y:y + h, x:x + w] == i)
            px = cur[y:y + h, x:x + w][sel].astype(int)
            hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)[y:y + h, x:x + w][sel].astype(int)
            lst.append(dict(x=round(float(cx), 1), y=round(float(cy), 1), a=int(area),
                            w=int(w), h=int(h), gb=int(np.mean(px[:, 1] - px[:, 0])),
                            sat=int(np.mean(hsv[:, 1])), val=int(np.mean(hsv[:, 2]))))
        for d in lst:
            d['n'] = len(lst)
        out[str(f)] = lst
        prev, cur = cur, nxt
        if f % 200 == 0:
            print('  frame %d: %d candidates' % (f, len(lst)), flush=True)

    json.dump(out, open(os.path.join(a.work, a.out), 'w'))
    tot = sum(len(v) for v in out.values())
    single = sum(1 for v in out.values() if len(v) == 1)
    print('%d candidates over %d frames (%.1f per frame); %d frames have exactly one'
          % (tot, n, tot / max(n, 1), single))
    print('inspect a suspicious frame with:  python3 scripts/ball_detect.py --help '
          'then re-tune --min-area/--max-dim/--row-min')


if __name__ == '__main__':
    main()
