"""Stabilise a panning / zooming (broadcast-style) clip into one common canvas.

Why this exists
---------------
The rest of the skill assumes a *fixed* camera: one calibration is fitted once and then
reused for every frame. A broadcast or hand-held clip pans, tilts and zooms, so a single
calibration is wrong on every frame but the reference one.

The rescue is geometric: under pure rotation + zoom about a fixed centre the mapping
between two frames of the same scene is a **homography independent of depth**. Warping
every frame into one reference view therefore restores a single-camera problem, and full
3D work (calibration, ball flight, speeds) stays valid on the warped canvas.

When to use it
  * broadcast / TV clips, phone videos that follow the action, any clip where the court
    lines move between frames;
  * check first with two frames 100 apart: if a court corner moves more than a few pixels,
    you need this step.

When NOT to use it
  * a real tripod clip - warping only adds interpolation blur;
  * a clip where the camera *translates* (dolly, walking operator, drone). Then the
    frame-to-frame map is not a homography, the mosaic will look fine but the geometry is
    silently wrong. Symptom: residuals below stay small in the middle of the frame and
    blow up at the edges, and `court_calib.py verify` cannot reach 2 px on the near lines;
  * cuts / replays inside the clip - cut the segments apart first (each needs its own ref).

Practical notes / traps
  * Burned-in overlays (score bugs, logos, station idents) do not move with the scene and
    will dominate ORB matching. Exclude them with `--mask-rows TOP BOTTOM`.
  * The warped canvas is bigger than the source frame and its **principal point is no
    longer the canvas centre**. Use `calib_broadcast.py` (9-parameter fit with a free
    principal point) for calibration afterwards, not `court_calib.py fit`.
  * Frame numbers stay 0-based everywhere; on disk `frames/f_00001.jpg` is frame 0.

  python3 scripts/stabilize.py --work work --out work_stab --mask-rows 0 160 --ref auto
  python3 scripts/stabilize.py --work work --out qa/stab_dry --emit-frames false   # quality only
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


# ------------------------------------------------------------------ features
def kp_des(orb, img, mask):
    return orb.detectAndCompute(img, mask)


def match_H(bf, ref_kp, ref_des, kp, des, ransac=2.5, min_good=20):
    """Homography (this frame -> reference) plus its quality.

    Returns (H, n_inliers, residual_px). residual_px is the median reprojection error of
    the RANSAC inliers - a low inlier count with a 0.3 px residual is still a good frame,
    while 200 inliers at 3 px means the scene is not a single plane/rotation any more.
    """
    if des is None or ref_des is None or len(kp) < min_good:
        return None, 0, float('nan')
    raw = bf.knnMatch(des, ref_des, k=2)
    good = [m[0] for m in raw if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < min_good:
        return None, len(good), float('nan')
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac, maxIters=5000)
    if H is None:
        return None, 0, float('nan')
    inl = inl.ravel().astype(bool)
    if inl.sum() < 4:
        return None, int(inl.sum()), float('nan')
    proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
    res = float(np.median(np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)))
    return H, int(inl.sum()), res


def build_mask(W, H, top, bottom):
    m = np.full((H, W), 255, np.uint8)
    if top > 0:
        m[:min(top, H), :] = 0
    if bottom > 0:
        m[max(0, H - bottom):, :] = 0
    return m


def frame_path(work, f):
    return os.path.join(work, 'frames', 'f_%05d.jpg' % (f + 1))


def corners_of(W, H):
    return np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)


# ------------------------------------------------------------------ auto reference
def pick_ref(work, n, W, H, mask, orb, bf, n_samples, verbose=True):
    """Choose the reference frame whose canvas is cheapest.

    Every candidate reference is scored by warping a set of sampled frames into it and
    measuring the area of the union bounding box (1.0 = no padding at all). The frame in
    the middle of the pan/zoom range wins, which keeps the canvas - and the black borders
    - as small as possible. Cost is O(samples^2) homographies, so keep samples modest.
    """
    idx = np.unique(np.linspace(0, n - 1, min(n_samples, n)).astype(int))
    feats = {}
    for f in idx:
        img = cv2.imread(frame_path(work, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats[int(f)] = kp_des(orb, img, mask)
    cand = sorted(feats)
    C = corners_of(W, H)
    best = None
    for i in cand:
        rkp, rdes = feats[i]
        xs, ys, fails = [0.0, float(W)], [0.0, float(H)], 0
        for j in cand:
            if j == i:
                continue
            kp, des = feats[j]
            Hm, ninl, _ = match_H(bf, rkp, rdes, kp, des)
            if Hm is None or ninl < 25:
                fails += 1
                continue
            p = cv2.perspectiveTransform(C, Hm).reshape(-1, 2)
            xs += list(p[:, 0])
            ys += list(p[:, 1])
        area = (max(xs) - min(xs)) * (max(ys) - min(ys)) / float(W * H)
        cost = area + 0.5 * fails                    # a frame nobody matches is useless
        if verbose:
            print('  ref candidate f%-5d canvas x%.2f  unmatched %d' % (i, area, fails))
        if best is None or cost < best[0]:
            best = (cost, int(i), area, fails)
    if best is None:
        raise SystemExit('auto reference failed - no frame could be read')
    print('auto reference -> frame %d (canvas x%.2f of source, %d unmatched samples)'
          % (best[1], best[2], best[3]))
    return best[1]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if str(v).lower() in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError('expected true/false, got %r' % v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True, help='source work dir (frames/ + meta.json)')
    ap.add_argument('--out', default=None,
                    help='stabilized work dir (default: <work>_stab). Never the same as --work.')
    ap.add_argument('--ref', default='auto',
                    help="reference frame, 0-based, or 'auto' (default) to pick the "
                         'frame with the cheapest canvas')
    ap.add_argument('--ref-samples', type=int, default=12,
                    help='frames sampled when --ref auto scores the candidates')
    ap.add_argument('--mask-rows', type=int, nargs=2, default=(0, 0), metavar=('TOP', 'BOTTOM'),
                    help='exclude this many top and bottom rows from feature matching '
                         '(burned-in score bars, station logos, subtitles)')
    ap.add_argument('--mask-bottom', type=int, default=None,
                    help='compat alias: rows >= this index are overlay '
                         '(equivalent to --mask-rows 0 (H - value))')
    ap.add_argument('--max-pad', type=int, default=900,
                    help='max horizontal padding of the canvas, px per side')
    ap.add_argument('--max-pad-v', type=int, default=400,
                    help='max vertical padding of the canvas, px per side')
    ap.add_argument('--emit-frames', type=str2bool, default=True,
                    help='false = only estimate homographies and report quality '
                         '(no warped frames, no meta.json) - cheap way to decide first')
    ap.add_argument('--nfeatures', type=int, default=4000)
    ap.add_argument('--min-inliers', type=int, default=40,
                    help='below this the direct match to the reference is retried via the '
                         'previous frame (chained)')
    ap.add_argument('--min-inliers-chain', type=int, default=30)
    ap.add_argument('--weak-warn-frac', type=float, default=0.05,
                    help='warn loudly when this fraction of frames is weak/copied')
    ap.add_argument('--jpeg-quality', type=int, default=95)
    ap.add_argument('--bg-step', type=int, default=5, help='frame step for the median background')
    ap.add_argument('--force', action='store_true',
                    help='allow writing into an --out dir that already holds frames')
    a = ap.parse_args()

    out = a.out or (a.work.rstrip('/') + '_stab')
    if os.path.abspath(out) == os.path.abspath(a.work):
        raise SystemExit('--out must differ from --work (never warp in place)')
    if a.emit_frames and os.path.isdir(os.path.join(out, 'frames')) and not a.force:
        raise SystemExit('%s/frames already exists - pass --force to overwrite, or pick '
                         'another --out (existing work dirs are precious)' % out)

    meta = json.load(open(os.path.join(a.work, 'meta.json')))
    n = int(meta['n_frames'])
    W, Hh = int(meta['width']), int(meta['height'])
    top, bottom = int(a.mask_rows[0]), int(a.mask_rows[1])
    if a.mask_bottom is not None:                       # compat: absolute row index
        bottom = max(bottom, Hh - int(a.mask_bottom))
    mask = build_mask(W, Hh, top, bottom)
    print('source %dx%d, %d frames; feature mask excludes top %d / bottom %d rows'
          % (W, Hh, n, top, bottom))

    orb = cv2.ORB_create(nfeatures=a.nfeatures, fastThreshold=7)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    if str(a.ref).lower() == 'auto':
        ref_f = pick_ref(a.work, n, W, Hh, mask, orb, bf, a.ref_samples)
    else:
        ref_f = int(a.ref)
    if not 0 <= ref_f < n:
        raise SystemExit('--ref %d outside 0..%d' % (ref_f, n - 1))

    ref = cv2.imread(frame_path(a.work, ref_f), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise SystemExit('cannot read reference frame %d' % ref_f)
    ref_kp, ref_des = kp_des(orb, ref, mask)

    # ---------------------------------------------------------------- per-frame H
    Hs = [None] * n
    quality = [None] * n
    weak = []
    prev_img, prev_kd, prev_H = ref, (ref_kp, ref_des), np.eye(3)
    for f in range(n):
        if f == ref_f:
            Hs[f] = np.eye(3)
            quality[f] = dict(frame=f, inliers=len(ref_kp), residual_px=0.0, kind='ref')
            prev_img, prev_kd, prev_H = ref, (ref_kp, ref_des), np.eye(3)
            continue
        img = cv2.imread(frame_path(a.work, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            Hs[f] = prev_H.copy()
            quality[f] = dict(frame=f, inliers=0, residual_px=None, kind='missing')
            weak.append(quality[f])
            continue
        kd = kp_des(orb, img, mask)
        H, ninl, res = match_H(bf, ref_kp, ref_des, kd[0], kd[1])
        kind = 'direct'
        if H is None or ninl < a.min_inliers:
            # 2nd level: match to the previous frame and chain through its homography.
            # Drifts slowly, so it must stay the exception, not the rule.
            Hp, ninl2, res2 = match_H(bf, prev_kd[0], prev_kd[1], kd[0], kd[1])
            if Hp is not None and ninl2 >= a.min_inliers_chain:
                H, ninl, res, kind = prev_H @ Hp, ninl2, res2, 'chained'
        if H is None:
            # 3rd level: freeze the previous homography. Motion in this frame is lost;
            # anything measured on it is not trustworthy.
            H, ninl, res, kind = prev_H.copy(), 0, None, 'copied'
        Hs[f] = H
        q = dict(frame=f, inliers=int(ninl), kind=kind,
                 residual_px=None if res is None or np.isnan(res) else round(float(res), 3))
        quality[f] = q
        if kind != 'direct':
            weak.append(q)
        prev_img, prev_kd, prev_H = img, kd, H

    # ---------------------------------------------------------------- quality report
    inl = np.array([q['inliers'] for q in quality], float)
    rs = np.array([np.nan if q['residual_px'] is None else q['residual_px'] for q in quality], float)
    rs_ok = rs[~np.isnan(rs)]
    print('\n--- stabilisation quality ---')
    print('inliers   median %.0f   p5 %.0f   min %.0f' %
          (np.median(inl), np.percentile(inl, 5), inl.min()))
    if len(rs_ok):
        print('residual  median %.2f px   p95 %.2f px   max %.2f px' %
              (np.median(rs_ok), np.percentile(rs_ok, 95), rs_ok.max()))
    kinds = {}
    for q in quality:
        kinds[q['kind']] = kinds.get(q['kind'], 0) + 1
    print('frames by match type: %s' % kinds)
    if weak:
        print('weak frames (%d): %s' % (len(weak), ', '.join(
            '%d/%s' % (q['frame'], q['kind']) for q in weak[:40])
            + (' ...' if len(weak) > 40 else '')))
    frac = len(weak) / max(n, 1)
    if frac > a.weak_warn_frac:
        print('WARNING %.1f%% of frames are weak/copied (> %.0f%%): stabilisation is NOT '
              'reliable - do not build 3D conclusions on this canvas. Re-check the overlay '
              'mask, split the clip at cuts, or accept that the camera translates.'
              % (100 * frac, 100 * a.weak_warn_frac))
    if len(rs_ok) and np.percentile(rs_ok, 95) > 3.0:
        print('WARNING p95 reprojection residual > 3 px - the frame-to-frame map is not a '
              'pure homography (camera translation? rolling shutter?)')

    # ---------------------------------------------------------------- canvas
    C = corners_of(W, Hh)
    xs, ys = [], []
    for H in Hs:
        p = cv2.perspectiveTransform(C, H).reshape(-1, 2)
        xs += list(p[:, 0])
        ys += list(p[:, 1])
    rx0, rx1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    ry0, ry1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    x0, x1 = max(rx0, -a.max_pad), min(rx1, W + a.max_pad)
    y0, y1 = max(ry0, -a.max_pad_v), min(ry1, Hh + a.max_pad_v)
    CW, CH = x1 - x0, y1 - y0
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    print('\ncanvas %dx%d  offset (%d, %d)  ref frame %d' % (CW, CH, -x0, -y0, ref_f))
    if (rx0, rx1, ry0, ry1) != (x0, x1, y0, y1):
        print('canvas clamped by --max-pad %d / --max-pad-v %d (unclamped would be '
              'x[%d,%d] y[%d,%d]) - frames beyond the clamp lose content at the border'
              % (a.max_pad, a.max_pad_v, rx0, rx1, ry0, ry1))
        cut = []
        for f, H in enumerate(Hs):
            p = cv2.perspectiveTransform(C, H).reshape(-1, 2)
            if (p[:, 0].min() < x0 - .5 or p[:, 0].max() > x1 + .5
                    or p[:, 1].min() < y0 - .5 or p[:, 1].max() > y1 + .5):
                cut.append(f)
        if cut:
            print('  %d frames are cropped / will show black borders: %s%s'
                  % (len(cut), cut[:30], ' ...' if len(cut) > 30 else ''))

    os.makedirs(out, exist_ok=True)
    rep = dict(ref=ref_f, ref_mode='auto' if str(a.ref).lower() == 'auto' else 'given',
               offset=[-x0, -y0], canvas=[CW, CH], source_size=[W, Hh],
               mask_rows=[top, bottom], emitted_frames=bool(a.emit_frames),
               n_frames=n, n_weak=len(weak), weak_frac=round(frac, 4),
               inliers_median=float(np.median(inl)),
               residual_median_px=None if not len(rs_ok) else round(float(np.median(rs_ok)), 3),
               residual_p95_px=None if not len(rs_ok) else round(float(np.percentile(rs_ok, 95)), 3),
               quality=quality, weak=weak,
               H=[np.asarray(h).tolist() for h in Hs])
    json.dump(rep, open(os.path.join(out, 'stabilize.json'), 'w'))
    print('wrote %s' % os.path.join(out, 'stabilize.json'))

    if not a.emit_frames:
        print('--emit-frames false: no warped frames and no meta.json written '
              '(nothing else in %s was touched)' % out)
        return

    # ---------------------------------------------------------------- warp
    os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
    acc = []
    for f in range(n):
        img = cv2.imread(frame_path(a.work, f))
        if img is None:
            continue
        w = cv2.warpPerspective(img, T @ Hs[f], (CW, CH), flags=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out, 'frames', 'f_%05d.jpg' % (f + 1)), w,
                    [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        if f % max(1, a.bg_step) == 0:
            acc.append(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY))
        if f % 100 == 0:
            print('  warped %d/%d' % (f, n), flush=True)
    if acc:
        bg = np.median(np.stack(acc), axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(out, 'bg_median.png'), cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR))

    src_audio = os.path.join(a.work, meta.get('audio') or 'audio.wav')
    meta2 = dict(meta)
    meta2.update({'width': CW, 'height': CH, 'frames_dir': 'frames',
                  'background': 'bg_median.png', 'stabilized': True,
                  'stabilize_ref': ref_f, 'canvas_offset': [-x0, -y0],
                  'source_size': [W, Hh]})
    json.dump(meta2, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('wrote %s (canvas %dx%d, ref %d)' % (os.path.join(out, 'meta.json'), CW, CH, ref_f))
    if os.path.exists(src_audio) and not os.path.exists(os.path.join(out, 'audio.wav')):
        print('note: copy the audio yourself if you need it: cp %s %s/'
              % (src_audio, out))
    print('next: calibrate with calib_broadcast.py (the principal point of this canvas is '
          'NOT its centre)')


if __name__ == '__main__':
    sys.exit(main())
