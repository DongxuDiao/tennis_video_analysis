"""Step 4 - court calibration: detect the painted lines, fit a physical camera, verify it.

Calibration is the step that decides whether every later number is physics or fiction, so
it is split into detect / fit / verify and the middle step needs a human (or agent) to say
which detected line is which world line - automatic labelling silently mixes up the
service line with the far baseline and then produces plausible-looking nonsense.

  python3 scripts/court_calib.py detect --work work
      -> court_lines.json + court_lines.png (indexed overlay; look at it, then write ident.json)

  cat ident.json
  {"sport": "tennis",
   "lines": {"0": "near_baseline", "3": "near_service", "4": "left_sideline", "1": "right_sideline"},
   "player_height": [{"frame": 235, "ankle": [308, 513], "nose": [347, 281], "height": 1.60}]}

  python3 scripts/court_calib.py fit --work work --ident ident.json
      -> calib.json (7 physical parameters + ground homography + rms) and calib_overlay.png

  python3 scripts/court_calib.py verify --work work
      -> numeric row comparison + calib_zoom.png; trust numbers, not the overlay picture

Named world lines: near/far_baseline, near/far_service, left/right_sideline (singles),
left/right_sideline_doubles, net_base, centre_service. Any index may instead carry an
explicit {"world": [[x1,y1],[x2,y2]]} pair in metres.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera, build_pose, court  # noqa: E402


# ------------------------------------------------------------------ line detect
def line_mask(im, sat=70, val=150, tophat=16):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    white = (hsv[:, :, 1] < sat) & (hsv[:, :, 2] > val)
    th_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 5)))
    th_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31)))
    m = white | (th_h > tophat) | (th_v > tophat)
    return (m.astype(np.uint8)) * 255


def refine(mask, p1, p2, band=6, iters=4):
    """Intensity-weighted refit of a line, returns sampled centre points along it."""
    H, W = mask.shape
    horiz = abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1])
    (x1, y1), (x2, y2) = p1, p2
    pts = []
    for _ in range(iters):
        pts = []
        if horiz:
            m = (y2 - y1) / (x2 - x1 + 1e-9)
            b = y1 - m * x1
            for x in range(int(min(x1, x2)), int(max(x1, x2))):
                y0 = m * x + b
                lo, hi = int(y0 - band), int(y0 + band) + 1
                if lo < 0 or hi > H:
                    continue
                col = mask[lo:hi, x].astype(float)
                if col.sum() < 3 * 255:
                    continue
                pts.append((x, float((col * np.arange(lo, hi)).sum() / col.sum())))
            if len(pts) < 20:
                return None
            P = np.array(pts)
            A = np.stack([P[:, 0], np.ones(len(P))], 1)
            m, b = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
            y1, y2 = m * x1 + b, m * x2 + b
        else:
            m = (x2 - x1) / (y2 - y1 + 1e-9)
            b = x1 - m * y1
            for y in range(int(min(y1, y2)), int(max(y1, y2))):
                x0 = m * y + b
                lo, hi = int(x0 - band), int(x0 + band) + 1
                if lo < 0 or hi > W:
                    continue
                row = mask[y, lo:hi].astype(float)
                if row.sum() < 3 * 255:
                    continue
                pts.append((float((row * np.arange(lo, hi)).sum() / row.sum()), y))
            if len(pts) < 20:
                return None
            P = np.array(pts)
            A = np.stack([P[:, 1], np.ones(len(P))], 1)
            m, b = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
            x1, x2 = m * y1 + b, m * y2 + b
        band = max(3, band - 1)
    return np.array(pts)


def cmd_detect(a):
    bg = cv2.imread(os.path.join(a.work, a.image))
    if bg is None:
        raise SystemExit('missing %s' % os.path.join(a.work, a.image))
    H, W = bg.shape[:2]
    mask = line_mask(bg, a.sat, a.val, a.tophat)
    if a.skip_top:
        mask[:int(a.skip_top * H), :] = 0
    cv2.imwrite(os.path.join(a.work, 'line_mask.png'), mask)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                           minLineLength=max(40, W // 12), maxLineGap=12)
    if segs is None:
        raise SystemExit('no line segments found - relax --sat/--val/--tophat')
    segs = segs.reshape(-1, 4)
    groups = []
    for x1, y1, x2, y2 in segs:
        d = np.array([x2 - x1, y2 - y1], float)
        L = np.linalg.norm(d)
        d /= L
        n = np.array([-d[1], d[0]])
        c = float(n @ np.array([x1, y1]))
        if c < 0:
            n, c = -n, -c
        ang = np.degrees(np.arctan2(n[1], n[0])) % 180
        hit = None
        for g in groups:
            if min(abs(g['ang'] - ang), 180 - abs(g['ang'] - ang)) < 3 and abs(g['c'] - c) < 8:
                hit = g
                break
        if hit is None:
            groups.append(dict(ang=ang, c=c, segs=[(x1, y1, x2, y2)], L=L))
        else:
            hit['segs'].append((x1, y1, x2, y2))
            hit['L'] += L
            w = L / hit['L']
            hit['ang'] = hit['ang'] * (1 - w) + ang * w
            hit['c'] = hit['c'] * (1 - w) + c * w
    out, vis = [], bg.copy()
    groups.sort(key=lambda g: -g['L'])
    for g in groups[:a.max_lines]:
        pp = np.array(g['segs']).reshape(-1, 2)
        i0, i1 = np.argmin(pp[:, 0] + pp[:, 1]), np.argmax(pp[:, 0] + pp[:, 1])
        pts = refine(mask, tuple(pp[i0]), tuple(pp[i1]), band=a.band)
        if pts is None or len(pts) < 25:
            continue
        e0, e1 = pts[0], pts[-1]
        ang = float(np.degrees(np.arctan2(e1[1] - e0[1], e1[0] - e0[0])))
        out.append(dict(idx=len(out), n_pts=len(pts), angle=round(ang, 2),
                        p0=[round(float(e0[0]), 1), round(float(e0[1]), 1)],
                        p1=[round(float(e1[0]), 1), round(float(e1[1]), 1)],
                        length=round(float(np.hypot(*(e1 - e0))), 1),
                        pts=[[round(float(x), 2), round(float(y), 2)] for x, y in pts]))
        col = (0, 0, 255) if abs(ang) < 45 else (255, 90, 0)
        cv2.line(vis, tuple(np.int32(e0)), tuple(np.int32(e1)), col, 2, cv2.LINE_AA)
        cv2.putText(vis, str(out[-1]['idx']), tuple(np.int32((e0 + e1) / 2) + [4, -4]),
                    0, 0.7, (0, 255, 255), 2)
    json.dump(dict(width=W, height=H, lines=out), open(os.path.join(a.work, 'court_lines.json'), 'w'))
    cv2.imwrite(os.path.join(a.work, 'court_lines.png'), vis)
    print('%d candidate lines -> court_lines.json / court_lines.png' % len(out))
    for L in out:
        print('  [%2d] %4d pts  angle %+7.2f  %s -> %s  len %.0f'
              % (L['idx'], L['n_pts'], L['angle'], L['p0'], L['p1'], L['length']))
    print('open court_lines.png, decide which index is which world line, write ident.json')


# --------------------------------------------------------------------- fit
def world_line(name, c):
    H, DB, S, SV = c['half'], c['dbl'], c['sgl'], c['svc']
    table = {
        'near_baseline': ((-S, H), (S, H)), 'far_baseline': ((-S, -H), (S, -H)),
        'near_service': ((-S, SV), (S, SV)), 'far_service': ((-S, -SV), (S, -SV)),
        'left_sideline': ((-S, H), (-S, -H)), 'right_sideline': ((S, H), (S, -H)),
        'left_sideline_doubles': ((-DB, H), (-DB, -H)),
        'right_sideline_doubles': ((DB, H), (DB, -H)),
        'net_base': ((-DB, 0), (DB, 0)), 'centre_service': ((0, SV), (0, -SV)),
    }
    if name not in table:
        raise SystemExit('unknown world line %r (known: %s)' % (name, ', '.join(table)))
    return np.array(table[name], float)


def mask_of(work, a):
    m = cv2.imread(os.path.join(work, 'line_mask.png'), 0)
    if m is None:
        m = line_mask(cv2.imread(os.path.join(work, 'bg_median.png')))
    return m


def cmd_fit(a):
    ident = json.load(open(a.ident))
    dp = os.path.join(a.work, 'court_lines.json')
    det = json.load(open(dp)) if os.path.exists(dp) else None
    if det is None:
        bg = cv2.imread(os.path.join(a.work, 'bg_median.png'))
        det = dict(width=bg.shape[1], height=bg.shape[0], lines=[])
    W, H = det['width'], det['height']
    c = court(ident.get('court') or ident.get('sport') or 'tennis')
    data = []
    # manual hints: {"near_baseline": [[215,431],[955,456]], ...} - two rough pixel points per
    # line, refined against the white-line mask. Indispensable in cluttered indoor scenes
    # where auto-detection is swamped by beams, railings and window frames.
    for name, hint in (ident.get('manual') or {}).items():
        pts = refine(mask_of(a.work, a), tuple(hint[0]), tuple(hint[1]), band=a.band)
        if pts is None:
            raise SystemExit('manual hint for %s did not lock onto a line - adjust the points' % name)
        if len(pts) > a.max_pts:
            pts = pts[np.linspace(0, len(pts) - 1, a.max_pts).astype(int)]
        data.append((name, pts, world_line(name, c)))
        print('manual %-24s %4d pts  refined %s -> %s'
              % (name, len(pts), np.round(pts[0], 1).tolist(), np.round(pts[-1], 1).tolist()))
    lines = {int(k): v for k, v in (ident.get('lines') or {}).items()}
    for idx, spec in lines.items():
        L = [q for q in det['lines'] if q['idx'] == idx]
        if not L:
            raise SystemExit('line index %d not in court_lines.json' % idx)
        pts = np.array(L[0]['pts'], float)
        if len(pts) > a.max_pts:
            pts = pts[np.linspace(0, len(pts) - 1, a.max_pts).astype(int)]
        wl = np.array(spec['world'], float) if isinstance(spec, dict) else world_line(spec, c)
        name = spec.get('name', 'custom') if isinstance(spec, dict) else spec
        data.append((name, pts, wl))
        print('line %-24s %4d pts  world %s' % (name, len(pts), wl.tolist()))
    players = ident.get('player_height') or []

    def proj(p, Pw):
        R, C, f = build_pose(p)
        Xc = (R @ (np.atleast_2d(np.asarray(Pw, float)) - C).T).T
        z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
        return np.stack([f * Xc[:, 0] / z + W / 2, f * Xc[:, 1] / z + H / 2], 1), Xc[:, 2]

    def resid(p):
        r = []
        for _, pts, wl in data:
            P3 = np.array([[wl[0, 0], wl[0, 1], 0], [wl[1, 0], wl[1, 1], 0]])
            uv, z = proj(p, P3)
            if np.any(z < 0.2):
                return np.full(4000, 1e3)
            d = uv[1] - uv[0]
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            r.append((pts - uv[0]) @ n)                     # point-to-line distance, px
        for q in players:
            R, C, f = build_pose(p)
            d = np.stack([[(q['ankle'][0] - W / 2) / f, (q['ankle'][1] - H / 2) / f, 1.0]], 0)
            dw = (R.T @ d.T).T[0]
            dw /= np.linalg.norm(dw)
            if dw[2] > -1e-3:
                return np.full(4000, 1e3)
            G = C + dw * ((q.get('ankle_height', 0.09) - C[2]) / dw[2])
            uv, _ = proj(p, [[G[0], G[1], q['height']]])
            r.append(a.player_weight * (uv[0] - np.array(q['nose'], float)))
        return np.concatenate([np.atleast_1d(x).ravel() for x in r])

    from scipy.optimize import least_squares
    lo = [-c['dbl'] * 2, c['half'] * 1.02, 0.4, -0.5, -0.5, -0.3, 0.15 * W]
    hi = [c['dbl'] * 2, c['half'] + a.cam_back_max, 8.0, 0.5, 0.7, 0.3, 5.0 * W]
    rng = np.random.default_rng(1)
    best = None
    for _ in range(a.trials):
        x0 = np.array([rng.uniform(-c['dbl'], c['dbl']), rng.uniform(c['half'] * 1.05, c['half'] + 8),
                       rng.uniform(0.8, 3.5), rng.uniform(-.15, .15), rng.uniform(-.15, .3),
                       rng.uniform(-.06, .06), rng.uniform(0.3 * W, 1.5 * W)])
        try:
            s = least_squares(resid, x0, method='trf', bounds=(lo, hi), max_nfev=600)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(s.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, s.x)
            print('  rms=%.3f px  cam=(%.2f, %.2f, %.2f) yaw/pitch/roll=%.1f/%.1f/%.1f f=%.1f'
                  % ((rms,) + tuple(s.x[:3]) + tuple(np.degrees(s.x[3:6])) + (s.x[6],)))
    rms, p = best
    cam_json = dict(width=W, height=H, sport=ident.get('sport', 'tennis'), court=c,
                    cam_params=[float(v) for v in p], rms=round(rms, 3),
                    lines_used={n: len(pts) for n, pts, _ in data},
                    n_player_constraints=len(players))
    src = np.array([[-c['sgl'], c['half']], [c['sgl'], c['half']],
                    [c['sgl'], -c['half']], [-c['sgl'], -c['half']]], np.float32)
    uv, _ = proj(p, np.concatenate([src, np.zeros((4, 1))], 1))
    Hg = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    cam_json['Hground'] = [float(v) for v in Hg.reshape(-1)]
    json.dump(cam_json, open(os.path.join(a.work, 'calib.json'), 'w'), indent=1)

    cam = Camera(a.work)
    print('\nFINAL rms=%.3f px  |  %s' % (rms, cam.describe()))
    per = []
    for name, pts, wl in data:
        P3 = np.array([[wl[0, 0], wl[0, 1], 0], [wl[1, 0], wl[1, 1], 0]])
        uv, _ = cam.project(P3)
        d = uv[1] - uv[0]
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        e = np.abs((pts - uv[0]) @ n)
        per.append((name, e.mean(), e.max()))
        print('  %-24s mean %.2f px  max %.2f px' % (name, e.mean(), e.max()))
    if rms > a.rms_warn:
        print('WARNING rms above %.1f px - re-check the line identification before continuing'
              % a.rms_warn)
    overlay(a.work, cam, a.overlay_frame)


def overlay(work, cam, frame=None):
    src = None
    if frame is not None:
        src = cv2.imread(os.path.join(work, 'frames', 'f_%05d.jpg' % (frame + 1)))
    if src is None:
        src = cv2.imread(os.path.join(work, 'bg_median.png'))
    vis = src.copy()
    colours = dict(baseline=(0, 0, 255), sideline=(0, 255, 0), service=(255, 0, 255),
                   centre=(255, 255, 0), net_tape=(0, 200, 255), net_mesh=(70, 130, 160))
    for pts, kind in cam.court_polylines():
        pts = np.array(pts, float)
        dense = np.concatenate([np.linspace(pts[i], pts[i + 1], 60) for i in range(len(pts) - 1)])
        uv, z = cam.project(dense)
        prev = None
        for (u, v), zz in zip(uv, z):
            ok = zz > 0.3 and -3000 < u < 3000 and -3000 < v < 3000
            cur = (int(round(u)), int(round(v))) if ok else None
            if prev and cur:
                cv2.line(vis, prev, cur, colours.get(kind, (200, 200, 200)),
                         2 if kind in ('baseline', 'net_tape') else 1, cv2.LINE_AA)
            prev = cur
    p = os.path.join(work, 'calib_overlay.png')
    cv2.imwrite(p, vis)
    print('overlay -> %s' % p)


# ------------------------------------------------------------------- verify
def cmd_verify(a):
    cam = Camera(a.work)
    bg = cv2.imread(os.path.join(a.work, 'bg_median.png'))
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    bands = a.columns or [[int(W * .45), int(W * .55)], [int(W * .70), int(W * .85)],
                          [int(W * .18), int(W * .30)]]
    named = [('far_baseline', [0, -cam.HALF, 0]), ('net_base', [0, 0, 0]),
             ('net_top', [0, 0, cam.net_height(0)]), ('near_baseline', [0, cam.HALF, 0])]
    if cam.SVC:
        named += [('far_service', [0, -cam.SVC, 0]), ('near_service', [0, cam.SVC, 0])]
    print('%s\n' % cam.describe())
    worst = 0.0
    for u0, u1 in bands:
        prof = g[:, u0:u1].mean(1)
        local = cv2.blur(prof.reshape(-1, 1), (1, 25)).ravel()
        resp = prof - local
        rows = [v for v in range(a.row_min, H - 2)
                if resp[v] > a.row_thresh and resp[v] == max(resp[max(0, v - 3):v + 4])]
        print('columns %d-%d: white-line rows %s' % (u0, u1, rows))
        uc = (u0 + u1) / 2
        for name, P in named:
            uv, z = cam.project([P])
            if z[0] <= 0:
                continue
            # predict the row of that world line at this column
            xs = np.linspace(-cam.DBL, cam.DBL, 400)
            pts = np.stack([xs, np.full_like(xs, P[1]), np.full_like(xs, P[2])], 1)
            puv, pz = cam.project(pts)
            ok = pz > 0.3
            if ok.sum() < 5:
                continue
            row = float(np.interp(uc, puv[ok][:, 0], puv[ok][:, 1]))
            near = min(rows, key=lambda r: abs(r - row)) if rows else None
            d = abs(near - row) if near is not None else float('nan')
            worst = max(worst, 0 if np.isnan(d) else d)
            print('    %-14s predicted row %6.1f   nearest white row %s   delta %s'
                  % (name, row, near, 'n/a' if near is None else '%.1f px' % d))
    print('\nworst |predicted - detected| = %.1f px  (>%.0f px means recheck identification)'
          % (worst, a.tol))
    for v in (int(H * .55), int(H * .75), int(H * .95)):
        uv = [[W / 2, v]]
        print('  row %4d: ground %s m, expected ball diameter %.1f px'
              % (v, np.round(cam.img2ground(uv)[0], 2), cam.apparent_diam_px(uv)[0]))
    zoom(a.work, cam, a.frame)


def zoom(work, cam, frame):
    src = cv2.imread(os.path.join(work, 'frames', 'f_%05d.jpg' % (frame + 1)))
    if src is None:
        src = cv2.imread(os.path.join(work, 'bg_median.png'))
    vis = src.copy()
    H, W = vis.shape[:2]
    for pts, kind in cam.court_polylines():
        if kind == 'net_mesh':
            continue
        pts = np.array(pts, float)
        dense = np.concatenate([np.linspace(pts[i], pts[i + 1], 60) for i in range(len(pts) - 1)])
        uv, z = cam.project(dense)
        prev = None
        for (u, v), zz in zip(uv, z):
            cur = (int(round(u)), int(round(v))) if (zz > .3 and -3000 < u < 3000 and -3000 < v < 3000) else None
            if prev and cur:
                cv2.line(vis, prev, cur, (255, 0, 255), 1, cv2.LINE_AA)
            prev = cur
    tiles, cw, ch = [], W // 3, H // 5
    for name, (x, y) in [('bottom-left', (0, int(H * .70))), ('bottom-right', (W - cw, int(H * .70))),
                         ('mid-centre', ((W - cw) // 2, int(H * .45)))]:
        c = vis[y:y + ch, x:x + cw]
        c = cv2.resize(c, (cw * 3, ch * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, name, (6, 22), 0, .6, (255, 255, 255), 2)
        tiles.append(c)
    hh = min(t.shape[0] for t in tiles)
    out = np.hstack([t[:hh] for t in tiles])
    cv2.imwrite(os.path.join(work, 'calib_zoom.png'), out)
    print('zoom crops -> %s' % os.path.join(work, 'calib_zoom.png'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('detect')
    d.add_argument('--work', default='work')
    d.add_argument('--image', default='bg_median.png')
    d.add_argument('--sat', type=int, default=70)
    d.add_argument('--val', type=int, default=150)
    d.add_argument('--tophat', type=int, default=16)
    d.add_argument('--band', type=int, default=6)
    d.add_argument('--skip-top', type=float, default=0.0, help='blank this fraction of the top rows')
    d.add_argument('--max-lines', type=int, default=14)
    d.set_defaults(func=cmd_detect)
    f = sub.add_parser('fit')
    f.add_argument('--work', default='work')
    f.add_argument('--ident', required=True)
    f.add_argument('--trials', type=int, default=300)
    f.add_argument('--max-pts', type=int, default=400)
    f.add_argument('--player-weight', type=float, default=1.0)
    f.add_argument('--cam-back-max', type=float, default=18.0, help='max metres behind the near baseline')
    f.add_argument('--rms-warn', type=float, default=3.0)
    f.add_argument('--band', type=int, default=6, help='refine band for manual hints')
    f.add_argument('--overlay-frame', type=int, default=None)
    f.set_defaults(func=cmd_fit)
    v = sub.add_parser('verify')
    v.add_argument('--work', default='work')
    v.add_argument('--frame', type=int, default=0)
    v.add_argument('--columns', type=int, nargs='+', action='append')
    v.add_argument('--row-min', type=int, default=100)
    v.add_argument('--row-thresh', type=float, default=6.0)
    v.add_argument('--tol', type=float, default=8.0)
    v.set_defaults(func=cmd_verify)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
