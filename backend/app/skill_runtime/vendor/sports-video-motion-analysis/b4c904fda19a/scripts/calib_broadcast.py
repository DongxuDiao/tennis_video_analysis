"""Calibration for cropped / stabilised (broadcast) footage: 9 parameters, free principal point.

Why this exists
---------------
`court_calib.py fit` solves 7 parameters (camera position, yaw/pitch/roll, focal length) and
*assumes the principal point is the image centre*. That holds for a raw camera frame. It is
wrong for

  * a vertical (9:16) crop of a 16:9 broadcast frame - the optical centre is cropped away;
  * a stabilised pan/zoom mosaic (`stabilize.py`) - the reference frame's optical centre
    lands wherever the canvas padding put it, not at the canvas centre;
  * any digitally zoomed / letterboxed / re-framed clip.

Forcing the principal point to the centre does not produce a large residual: the solver
absorbs the error into yaw/pitch/focal and still reaches 1-2 px on the lines, while the
**depth scale drifts systematically** - and depth scale is exactly what turns pixels into
km/h. The symptom is a calibration that looks fine near the camera and is 10-20 px off on
the far baseline, plus speeds that are consistently 10-20 % off.

So here (cu, cv) are solved too: 9 unknowns = pose 6 + focal 1 + principal point 2.

The price: a free principal point aggravates the classic focal-depth coupling (a longer
focal with the camera further back reprojects the same ground lines). Ground lines live on
a plane and can never fix this - a plane gives 8 DOF, we need 9. **At least one independent
off-plane scale constraint is mandatory:**

  * the **net tape** - a curve of known height (0.914 m at the centre strap, 1.07 m at the
    posts) spanning the whole width at a known depth; extracted sub-pixel from the median
    background (`--net-weight`);
  * **player height** hints in `ident.json` (`player_height`: ankle pixel -> ground, nose
    pixel at a known standing height) (`--player-weight`);
  * plus a soft prior on the principal point itself (`principal_prior` in ident.json,
    `--pp-weight` / `--pp-range`) - for a stabilised canvas the prior is simply
    `source_centre + canvas_offset`, which is usually good to a few tens of pixels.

Without any of these the fit will happily return rms < 0.5 px on a wrong depth scale.

Two traps that cost hours here, both fixed below:
  1. the residual vector must have the **same length in the "bad solution" branch** as in
     the normal branch. `least_squares` either raises or (worse) converges to the constant
     1e3 vector and reports rms ~ 1000. Hence the residual length is warmed up once before
     the random restarts and every early return is padded to it.
  2. the pitch start range must cover **steeply overlooking** broadcast cameras (up to
     ~45 deg down). A 0-15 deg start range never finds the true basin and lands on a
     mirrored solution below the court.

  python3 scripts/calib_broadcast.py fit    --work work_stab --ident work_stab/ident.json
  python3 scripts/calib_broadcast.py verify --work work_stab

ident.json (same file as court_calib.py, plus two extra blocks):
  {"sport": "tennis",
   "manual": {"near_baseline": [[122,800],[876,796]], ...},
   "net_tape": {"y_range": [382, 416], "x_range": [124, 872]},
   "player_height": [{"frame": 7, "ankle": [415,804], "nose": [415,627], "height": 1.70}],
   "principal_prior": [822, 525]}
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import court_calib as CC                              # noqa: E402
from camera import Camera, build_pose, court          # noqa: E402


# ------------------------------------------------------------------ net tape
def net_tape(work, ident, image='bg_median.png', dbg=None, min_val=140, min_w=60):
    """Sub-pixel row of the net tape per image column, as a smooth quadratic v(u).

    The tape is the brightest thing inside the given window in each column; a 7-px
    intensity centroid gives sub-pixel rows, and 4 robust re-fits drop the columns where a
    court line, a player or the far fence crosses the window.
    """
    bg = cv2.imread(os.path.join(work, image))
    if bg is None:
        raise SystemExit('missing %s' % os.path.join(work, image))
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(float)
    spec = ident.get('net_tape')
    if not spec:
        raise SystemExit('ident.json needs a "net_tape": {"y_range": [..], "x_range": [..]} '
                         'window - the net tape is the scale constraint that makes a free '
                         'principal point solvable')
    y0, y1 = [int(v) for v in spec['y_range']]
    x0, x1 = [int(v) for v in spec['x_range']]
    us, vs = [], []
    for u in range(x0, x1):
        col = g[y0:y1, u]
        k = int(np.argmax(col))
        if col[k] < min_val:
            continue
        lo, hi = max(0, k - 3), min(len(col), k + 4)
        w = np.clip(col[lo:hi] - 100, 0, None)
        if w.sum() < min_w:
            continue
        us.append(u)
        vs.append(y0 + float((w * np.arange(lo, hi)).sum() / w.sum()))
    us, vs = np.array(us, float), np.array(vs, float)
    if len(us) < 40:
        raise SystemExit('net tape: only %d columns found - widen net_tape.y_range or lower '
                         '--tape-min-val' % len(us))
    keep = np.ones(len(us), bool)
    c = np.polyfit(us, vs, 2)
    for _ in range(4):
        c = np.polyfit(us[keep], vs[keep], 2)
        r = vs - np.polyval(c, us)
        keep = np.abs(r) < max(1.5, 2.5 * np.std(r[keep]))
    print('net tape: %d cols, %d kept, rms %.2f px, sag %.1f px'
          % (len(us), keep.sum(), np.sqrt(np.mean((vs[keep] - np.polyval(c, us[keep])) ** 2)),
             np.polyval(c, (x0 + x1) / 2) - np.polyval(c, x0)))
    if dbg:
        vis = bg.copy()
        for u, v in zip(us[keep], vs[keep]):
            cv2.circle(vis, (int(u), int(round(v))), 1, (0, 255, 255), -1)
        uu = np.arange(x0, x1)
        for u, v in zip(uu, np.polyval(c, uu)):
            cv2.circle(vis, (int(u), int(round(v))), 1, (0, 0, 255), -1)
        cv2.imwrite(dbg, vis)
    return c, (x0, x1)


def net_height(x, c, shape='quad'):
    """Tape height above ground: net_c at the centre strap, net_p at the posts.

    `linear` is what camera.py uses; `quad` follows the sag of a real tape more closely and
    is what the broadcast clip this was validated on preferred (0.04 m difference at
    mid-court, ~1 px in the image - enough to move the rms).
    """
    mid = c.get('net_mid', c['net_c'])
    post = c.get('net_post', c['net_p'])
    span = max(c.get('post_x', c['dbl']), 1e-6)
    t = np.minimum(np.abs(np.asarray(x, float)) / span, 1.0)
    return mid + (post - mid) * (t ** 2 if shape == 'quad' else t)


def project(p, Pw, cu, cv):
    R, C, f = build_pose(p)
    Xc = (R @ (np.atleast_2d(np.asarray(Pw, float)) - C).T).T
    z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
    return np.stack([f * Xc[:, 0] / z + cu, f * Xc[:, 1] / z + cv], 1), Xc[:, 2]


# ------------------------------------------------------------------ fit
def cmd_fit(a):
    work = a.work
    ident = json.load(open(a.ident or os.path.join(work, 'ident.json')))
    c = court(ident.get('court') or ident.get('sport') or 'tennis')
    bg = cv2.imread(os.path.join(work, a.image))
    if bg is None:
        raise SystemExit('missing %s' % os.path.join(work, a.image))
    H, W = bg.shape[:2]
    mask = cv2.imread(os.path.join(work, 'line_mask.png'), 0)
    if mask is None:
        mask = CC.line_mask(bg)

    hints = ident.get('manual') or {}
    if not hints:
        raise SystemExit('ident.json needs "manual" line hints (two rough pixel points per '
                         'world line) - auto detection indices are not enough for a 9-parameter fit')
    data = []
    for name, hint in hints.items():
        pts = CC.refine(mask, tuple(hint[0]), tuple(hint[1]), band=a.band)
        if pts is None:
            raise SystemExit('hint %s did not lock onto a line - adjust its two points' % name)
        if len(pts) > a.max_pts:
            pts = pts[np.linspace(0, len(pts) - 1, a.max_pts).astype(int)]
        data.append((name, pts, CC.world_line(name, c)))
        print('manual %-16s %4d pts  %s -> %s' % (name, len(pts),
              np.round(pts[0], 1).tolist(), np.round(pts[-1], 1).tolist()))

    tape, (tx0, tx1) = net_tape(work, ident, a.image, os.path.join(work, 'net_tape.png'),
                               a.tape_min_val)
    net_x = np.linspace(-c['dbl'] * 0.96, c['dbl'] * 0.96, 49)
    net_P = np.stack([net_x, np.zeros_like(net_x), net_height(net_x, c, a.net_shape)], 1)
    players = ident.get('player_height') or []
    if not players and a.net_weight <= 0:
        print('WARNING no player_height hints and --net-weight 0: nothing constrains the '
              'depth scale, the result will be geometrically plausible and metrically wrong')
    cu0, cv0 = ident.get('principal_prior', [W / 2.0, H / 2.0])
    print('principal prior (%.0f, %.0f), canvas centre (%.0f, %.0f)' % (cu0, cv0, W / 2, H / 2))

    n_res = [None]

    def resid(q):
        p, cu, cv = q[:7], q[7], q[8]
        # TRAP 1: every early ("bad solution") return must have the SAME length as the
        # normal residual, otherwise least_squares raises or settles on the constant.
        bad = np.full(n_res[0], 1e3) if n_res[0] else None
        r = []
        for _, pts, wl in data:
            P3 = np.array([[wl[0, 0], wl[0, 1], 0], [wl[1, 0], wl[1, 1], 0]])
            uv, z = project(p, P3, cu, cv)
            if np.any(z < 0.2) and bad is not None:
                return bad
            d = uv[1] - uv[0]
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            r.append((pts - uv[0]) @ n)                    # point-to-line distance, px
        if a.net_weight > 0:
            uv, z = project(p, net_P, cu, cv)
            if np.any(z < 0.2) and bad is not None:
                return bad
            ok = (uv[:, 0] > tx0) & (uv[:, 0] < tx1)
            r.append(a.net_weight * (uv[ok, 1] - np.polyval(tape, uv[ok, 0])))
        R, C, f = build_pose(p)
        for pl in players:
            d = np.array([(pl['ankle'][0] - cu) / f, (pl['ankle'][1] - cv) / f, 1.0])
            dw = R.T @ d
            dw /= np.linalg.norm(dw)
            if dw[2] > -1e-3:
                if bad is not None:
                    return bad
                dw = np.array([dw[0], dw[1], -1e-3])
            G = C + dw * ((pl.get('ankle_height', 0.09) - C[2]) / dw[2])
            uv, _ = project(p, [[G[0], G[1], pl['height']]], cu, cv)
            r.append(a.player_weight * (uv[0] - np.array(pl['nose'], float)))
        r.append(a.pp_weight * np.array([cu - cu0, cv - cv0]))
        out = np.concatenate([np.atleast_1d(x).ravel() for x in r])
        if n_res[0] is None:
            n_res[0] = len(out)
        if len(out) != n_res[0]:                           # net tape visibility can vary
            out = np.resize(out, n_res[0])
        return out

    from scipy.optimize import least_squares
    lo = [-c['dbl'] * 2, c['half'] + a.cam_back[0], a.cam_h[0], -0.7, -0.3, -0.35,
          0.2 * W, cu0 - a.pp_range[0], cv0 - a.pp_range[1]]
    hi = [c['dbl'] * 2, c['half'] + a.cam_back[1], a.cam_h[1], 0.7, 1.0, 0.35,
          9.0 * W, cu0 + a.pp_range[0], cv0 + a.pp_range[1]]
    # warm-up call: fixes the residual length before the random restarts (TRAP 1)
    resid(np.array([0.0, c['half'] + 16, 12.0, 0.0, 0.35, 0.0, 900.0, cu0, cv0]))
    print('residual length %d' % n_res[0])
    rng = np.random.default_rng(a.seed)
    best = None
    for _ in range(a.trials):
        # TRAP 2: pitch (index 4) must start anywhere between ~10 and ~45 deg down - a
        # broadcast camera high in the stand looks steeply onto the court.
        x0 = np.array([rng.uniform(-3, 3), c['half'] + rng.uniform(2, 25),
                       rng.uniform(a.cam_h[0], min(a.cam_h[1], 15)),
                       rng.uniform(-.25, .25), rng.uniform(.18, .75), rng.uniform(-.05, .05),
                       rng.uniform(0.5 * W, 3.0 * W), cu0, cv0])
        try:
            s = least_squares(resid, x0, method='trf', bounds=(lo, hi), max_nfev=900)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(s.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, s.x)
            print('  rms=%.3f  cam=(%.2f,%.2f,%.2f) ypr=%.1f/%.1f/%.1f f=%.0f pp=(%.0f,%.0f)'
                  % ((rms,) + tuple(s.x[:3]) + tuple(np.degrees(s.x[3:6]))
                     + (s.x[6], s.x[7], s.x[8])))
    if best is None:
        raise SystemExit('no trial converged - loosen --cam-h / --cam-back / --pp-range')
    rms, q = best
    p, cu, cv = q[:7], float(q[7]), float(q[8])

    print('\n--- per-line residuals (px) ---')
    per = {}
    for name, pts, wl in data:
        P3 = np.array([[wl[0, 0], wl[0, 1], 0], [wl[1, 0], wl[1, 1], 0]])
        uv, _ = project(p, P3, cu, cv)
        d = uv[1] - uv[0]
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        e = np.abs((pts - uv[0]) @ n)
        per[name] = [round(float(e.mean()), 2), round(float(e.max()), 2)]
        print('  %-16s mean %.2f  max %.2f' % (name, e.mean(), e.max()))
    uv, _ = project(p, net_P, cu, cv)
    ok = (uv[:, 0] > tx0) & (uv[:, 0] < tx1)
    e = np.abs(uv[ok, 1] - np.polyval(tape, uv[ok, 0]))
    per['net_tape'] = [round(float(e.mean()), 2), round(float(e.max()), 2)]
    print('  %-16s mean %.2f  max %.2f' % ('net_tape', e.mean(), e.max()))
    R, C, f = build_pose(p)
    pl_res = []
    for i, pl in enumerate(players):
        d = np.array([(pl['ankle'][0] - cu) / f, (pl['ankle'][1] - cv) / f, 1.0])
        dw = R.T @ d
        dw /= np.linalg.norm(dw)
        G = C + dw * ((pl.get('ankle_height', 0.09) - C[2]) / dw[2])
        uv1, _ = project(p, [[G[0], G[1], pl['height']]], cu, cv)
        err = uv1[0] - np.array(pl['nose'], float)
        pl_res.append([round(float(err[0]), 1), round(float(err[1]), 1)])
        print('  player[%d] f%-4d ground=(%.2f, %.2f)  nose err (%.1f, %.1f) px'
              % (i, pl.get('frame', -1), G[0], G[1], err[0], err[1]))

    cj = dict(width=W, height=H, sport=ident.get('sport', 'tennis'), court=c,
              cam_params=[float(v) for v in p], pp=[cu, cv], rms=round(rms, 3),
              per_line=per, player_residual_px=pl_res,
              net_shape=a.net_shape, principal_prior=[float(cu0), float(cv0)],
              lines_used={n: len(pts) for n, pts, _ in data},
              solver='calib_broadcast.py (9 params, free principal point)',
              note='principal point is NOT the image/canvas centre - readers must honour "pp"')
    src = np.array([[-c['sgl'], c['half']], [c['sgl'], c['half']],
                    [c['sgl'], -c['half']], [-c['sgl'], -c['half']]], np.float32)
    uvg, _ = project(p, np.concatenate([src, np.zeros((4, 1))], 1), cu, cv)
    cj['Hground'] = [float(v) for v in
                     cv2.getPerspectiveTransform(src, uvg.astype(np.float32)).reshape(-1)]
    for name in dict.fromkeys([a.out, 'calib_freepp.json']):
        json.dump(cj, open(os.path.join(work, name), 'w'), indent=1)
        print('wrote %s' % os.path.join(work, name))

    cam = Camera(work, os.path.join(work, a.out))
    print('\nrms %.3f px | %s' % (rms, cam.describe()))
    print('camera world position  x=%.2f  y=%.2f  z=%.2f m  (%.2f m behind the near baseline)'
          % (p[0], p[1], p[2], p[1] - c['half']))
    print('focal %.0f px, hfov %.1f deg, vfov %.1f deg'
          % (p[6], np.degrees(2 * np.arctan(W / 2 / p[6])),
             np.degrees(2 * np.arctan(H / 2 / p[6]))))
    print('principal point (%.1f, %.1f) = canvas centre + (%+.1f, %+.1f) px  '
          '[prior offset (%+.1f, %+.1f)]'
          % (cu, cv, cu - W / 2, cv - H / 2, cu - cu0, cv - cv0))
    for lbl, v, v0, rng in (('cu', cu, cu0, a.pp_range[0]), ('cv', cv, cv0, a.pp_range[1])):
        if abs(v - v0) > 0.98 * rng:
            print('WARNING %s sits on the --pp-range bound (%.0f px from the prior): the fit '
                  'is constrained, not converged. Either principal_prior is wrong (for a crop '
                  'of a broadcast frame the optical centre is NOT the crop centre) or the '
                  'range is too tight - widen --pp-range and check whether rms and the camera '
                  'distance move.' % (lbl, v - v0))
    if rms > a.rms_warn:
        print('WARNING rms %.2f px above --rms-warn %.2f - re-check the line identification'
              % (rms, a.rms_warn))
    CC.overlay(work, cam, a.overlay_frame)
    print('now run:  python3 %s verify --work %s' % (os.path.basename(__file__), work))


# ------------------------------------------------------------------ verify
NEAR_LINES = ('near_baseline', 'near_service')


def cmd_verify(a):
    """Numeric line-by-line check, same idea as court_calib.py verify.

    Near lines must land within --tol-near px (default 2), the net tape and everything on
    the far half within --tol-far (default 8). An overlay picture that "looks right" is not
    evidence; these numbers are.
    """
    calib = a.calib or os.path.join(a.work, 'calib.json')
    cam = Camera(a.work, calib)
    print('%s\npp=(%.1f, %.1f)  canvas centre=(%.1f, %.1f)  offset (%+.1f, %+.1f) px\n'
          % (cam.describe(), cam.CU, cam.CV, cam.W / 2, cam.H / 2,
             cam.CU - cam.W / 2, cam.CV - cam.H / 2))
    bg = cv2.imread(os.path.join(a.work, a.image))
    if bg is None:
        raise SystemExit('missing %s' % os.path.join(a.work, a.image))
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    bands = a.columns or [[int(W * .45), int(W * .55)], [int(W * .70), int(W * .85)],
                          [int(W * .18), int(W * .30)]]
    named = [('near_baseline', [0, cam.HALF, 0], a.tol_near)]
    if cam.SVC:
        named.append(('near_service', [0, cam.SVC, 0], a.tol_near))
        named.append(('far_service', [0, -cam.SVC, 0], a.tol_far))
    # net_base is informational only (tol None): in tennis there is no painted line at y=0,
    # so "nearest white row" there is whatever else is around and would fail spuriously.
    named += [('net_base', [0, 0, 0], None),
              ('net_tape', [0, 0, cam.net_height(0)], a.tol_far),
              ('far_baseline', [0, -cam.HALF, 0], a.tol_far)]
    rows_fail, rows_nocheck, table = 0, 0, []
    print('%-14s %-10s %8s %8s %8s %6s  %s'
          % ('line', 'cols', 'pred', 'white', 'delta', 'tol', 'verdict'))
    for u0, u1 in bands:
        prof = g[:, u0:u1].mean(1)
        local = cv2.blur(prof.reshape(-1, 1), (1, 25)).ravel()
        resp = prof - local
        rows = [v for v in range(a.row_min, H - 2)
                if resp[v] > a.row_thresh and resp[v] == max(resp[max(0, v - 3):v + 4])]
        uc = (u0 + u1) / 2.0
        for name, P, tol in named:
            xs = np.linspace(-cam.DBL, cam.DBL, 400)
            pts = np.stack([xs, np.full_like(xs, P[1]),
                            np.full_like(xs, P[2] if name != 'net_tape' else 0.0)], 1)
            if name == 'net_tape':
                pts[:, 2] = [cam.net_height(x) for x in xs]
            puv, pz = cam.project(pts)
            ok = pz > 0.3
            if ok.sum() < 5:
                continue
            row = float(np.interp(uc, puv[ok][:, 0], puv[ok][:, 1]))
            if not (0 <= row < H) or not rows:
                print('%-14s %4d-%-5d %8.1f %8s %8s %6s  skip (outside frame)'
                      % (name, u0, u1, row, '-', '-', '-' if tol is None else '%.1f' % tol))
                continue
            near = min(rows, key=lambda r: abs(r - row))
            d = abs(near - row)
            if tol is None:
                verd = 'info'
            elif d > a.max_search:
                # no white line anywhere near: the line is occluded / not painted in this
                # column band, so the check could not be made. Not a pass and not a fail.
                verd = 'no-line'
                rows_nocheck += 1
            elif d > tol:
                verd = 'FAIL'
                rows_fail += 1
            else:
                verd = 'ok'
            table.append(dict(line=name, cols=[u0, u1], predicted=round(row, 1),
                              white_row=near, delta=round(d, 1), tol=tol, verdict=verd))
            print('%-14s %4d-%-5d %8.1f %8d %8.1f %6s  %s'
                  % (name, u0, u1, row, near, d, '-' if tol is None else '%.1f' % tol, verd))
    print('\n%d/%d checks above tolerance, %d could not be checked (no white line within '
          '%.0f px)' % (rows_fail, len(table), rows_nocheck, a.max_search))
    if rows_fail:
        print('FAIL - the calibration or the line identification is wrong; do not use the '
              'derived speeds. Re-check ident.json, the net_tape window and the player '
              'height hints, then re-fit.')
    else:
        print('PASS - all measured lines inside tolerance (near %.1f px / far %.1f px)'
              % (a.tol_near, a.tol_far))
    for v in (int(H * .55), int(H * .75), int(H * .95)):
        uv = [[cam.W / 2, v]]
        print('  row %4d: ground %s m, expected ball diameter %.1f px'
              % (v, np.round(cam.img2ground(uv)[0], 2), cam.apparent_diam_px(uv)[0]))
    json.dump(dict(calib=calib, checks=table, n_fail=rows_fail, n_nocheck=rows_nocheck,
                   tol_near=a.tol_near, tol_far=a.tol_far),
              open(os.path.join(a.work, 'calib_verify.json'), 'w'), indent=1)
    print('wrote %s' % os.path.join(a.work, 'calib_verify.json'))
    if a.zoom:
        CC.zoom(a.work, cam, a.frame)
    return 1 if rows_fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('fit', help='9-parameter fit with a free principal point')
    f.add_argument('--work', required=True)
    f.add_argument('--ident', default=None, help='default $WORK/ident.json')
    f.add_argument('--image', default='bg_median.png')
    f.add_argument('--out', default='calib.json',
                   help='written inside $WORK; a copy always goes to calib_freepp.json '
                        'so it can be diffed against a centred-pp calib.json')
    f.add_argument('--trials', type=int, default=200, help='random restarts')
    f.add_argument('--seed', type=int, default=3)
    f.add_argument('--band', type=int, default=6, help='refine band for the manual hints')
    f.add_argument('--max-pts', type=int, default=120, help='points kept per line')
    f.add_argument('--player-weight', type=float, default=1.0)
    f.add_argument('--net-weight', type=float, default=1.5, help='0 disables the net tape')
    f.add_argument('--pp-weight', type=float, default=0.05, help='soft prior on (cu, cv)')
    f.add_argument('--pp-range', type=float, nargs=2, default=(260, 220),
                   metavar=('DU', 'DV'), help='hard box around the principal prior')
    f.add_argument('--cam-h', type=float, nargs=2, default=(3.0, 20.0), metavar=('LO', 'HI'),
                   help='camera height search range, m')
    f.add_argument('--cam-back', type=float, nargs=2, default=(1.0, 35.0), metavar=('LO', 'HI'),
                   help='metres behind the near baseline, search range')
    f.add_argument('--net-shape', choices=['quad', 'linear'], default='quad',
                   help='tape height model between centre and posts')
    f.add_argument('--tape-min-val', type=int, default=140)
    f.add_argument('--rms-warn', type=float, default=1.0)
    f.add_argument('--overlay-frame', type=int, default=None)
    f.set_defaults(func=cmd_fit)

    v = sub.add_parser('verify', help='numeric row comparison against the detected white lines')
    v.add_argument('--work', required=True)
    v.add_argument('--calib', default=None, help='default $WORK/calib.json')
    v.add_argument('--image', default='bg_median.png')
    v.add_argument('--frame', type=int, default=0)
    v.add_argument('--columns', type=int, nargs=2, action='append', metavar=('U0', 'U1'))
    v.add_argument('--row-min', type=int, default=100)
    v.add_argument('--row-thresh', type=float, default=6.0)
    v.add_argument('--tol-near', type=float, default=2.0)
    v.add_argument('--tol-far', type=float, default=8.0)
    v.add_argument('--max-search', type=float, default=20.0,
                   help='if the nearest detected white row is further than this, the line is '
                        'treated as not visible in that column band instead of as a failure')
    v.add_argument('--zoom', action='store_true', help='also write calib_zoom.png')
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    return a.func(a)


if __name__ == '__main__':
    sys.exit(main() or 0)
