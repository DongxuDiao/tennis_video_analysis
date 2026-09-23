"""Step 6 - ball flight: 3D reconstruction, launch speed, landing band -> shots.json

Physics: point mass with quadratic drag and an optional Magnus (topspin) term, integrated
at 900 Hz. Fitting a single camera is only possible because two extra constraints are
added: the contact point (from the incoming fit or given explicitly) anchors the start,
and the *apparent ball diameter* pins the depth that a bare pixel position cannot.

Two honesty rules are wired into the output and must survive into the report:
  * speed comes with a bootstrap interval and a confidence tier - a 1-2 detection arc is
    a hypothesis, not a measurement;
  * the landing point is an extrapolation. Spin cannot be measured from a 0.2 s arc, and
    spin is what decides where the ball lands, so every shot is evaluated over a spin band
    (0 / 1000 / 2000 / 3000 rpm) and flagged uncertain when the outcome flips inside it.

  python3 scripts/flight_fit.py --work work --input shots_input.json

shots_input.json:
  {"fps": 30,
   "shots": [{"id": 1, "side": "FH", "tc": 64.8,
              "incoming": [[44, 442, 369], [45, 437, 361]],
              "outgoing": [[66, 460, 276], [67, 468, 259]],
              "contact": [0.9, 12.1, 1.05]}]}
`tc` is the contact frame (fractional is fine - take it from audio_impacts.py).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera  # noqa: E402

G = 9.81


class Physics:
    def __init__(self, ball_d=0.067, mass=0.057, cd=0.55, rho=1.2):
        self.d, self.m, self.cd, self.rho = ball_d, mass, cd, rho
        self.area = np.pi * (ball_d / 2) ** 2
        self.kd = 0.5 * rho * cd * self.area / mass

    def cl_of_rpm(self, rpm):
        """Lift coefficient for a spinning ball (Cross & Lindsey style fit)."""
        return 0.0 if rpm <= 0 else float(min(0.05 + 1.4e-4 * rpm, 0.42))

    def traj(self, P0, V0, T, dt=1 / 900., cl=0.0):
        """Integrate to time T. Positive cl = topspin (extra downward force)."""
        P, V = np.array(P0, float), np.array(V0, float)
        ts, ps = [0.0], [P.copy()]
        for _ in range(int(T / dt)):
            s = np.linalg.norm(V)
            acc = np.array([0., 0., -G]) - self.kd * s * V
            if cl:
                vh = V / max(s, 1e-9)
                ax = np.array([-vh[1], vh[0], 0.0])
                na = np.linalg.norm(ax)
                if na > 1e-6:
                    ax /= na
                    acc -= (0.5 * self.rho * self.area * cl * s * s / self.m) * np.cross(vh, ax)
            V = V + acc * dt
            P = P + V * dt
            ts.append(ts[-1] + dt)
            ps.append(P.copy())
        return np.array(ts), np.array(ps)

    def sample_at(self, P0, V0, tq, cl=0.0):
        tq = np.atleast_1d(np.asarray(tq, float))
        ts, ps = self.traj(P0, V0, float(tq.max()) + 1e-3, cl=cl)
        return np.stack([np.interp(tq, ts, ps[:, i]) for i in range(3)], 1)


def blob_size(cand, f, u, v, tol=2.5):
    for q in cand.get(str(int(f)), []):
        if abs(q['x'] - u) < tol and abs(q['y'] - v) < tol:
            return min(q['w'], q['h'])
    return np.nan


def fit_incoming(cam, ph, obs, sizes, fps, size_w=1.5):
    """Ballistic (drag-free is enough over ~0.5 s) fit of the incoming ball."""
    from scipy.optimize import least_squares
    fr = np.array([o[0] for o in obs], float)
    uv = np.array([[o[1], o[2]] for o in obs], float)
    t = (fr - fr[0]) / fps
    g = np.array([0, 0, -G])

    def res(p):
        P = p[None, :3] + p[None, 3:6] * t[:, None] + 0.5 * g[None, :] * (t ** 2)[:, None]
        pr, z = cam.project(P)
        r = (pr - uv).ravel()
        ok = ~np.isnan(sizes)
        if ok.any():
            d = np.linalg.norm(P - cam.C[None, :], axis=1)
            r = np.concatenate([r, (cam.F * cam.BALL_D / np.maximum(d, .3) - sizes)[ok] / size_w])
        return r + (1e3 if np.any(z < 0.2) else 0.0)

    best = None
    for d0 in (1.5, 2.5, 4.0, 6.0, 9.0, 13.0):
        P0 = cam.C + cam.ray(uv[:1])[0] * d0
        for vg in ([0, 5, 5], [-6, 2, 0], [-9, 1, -1], [-12, 3, 2], [-4, 6, 1], [2, 6, 4], [0, 9, 2]):
            try:
                s = least_squares(res, np.concatenate([P0, vg]), method='lm', max_nfev=6000)
            except Exception:
                continue
            c = float(np.sqrt(np.mean(s.fun ** 2)))
            if best is None or c < best[0]:
                best = (c, s.x)
    return best


def fit_outgoing(cam, ph, obs, sizes, P0, tc, fps, noise=None, size_w=1.5):
    from scipy.optimize import least_squares
    fr = np.array([o[0] for o in obs], float)
    uv = np.array([[o[1], o[2]] for o in obs], float)
    sz = sizes.copy()
    if noise is not None:
        uv = uv + noise[0]
        sz = sz + noise[1]
    t = (fr - tc) / fps

    def res(v):
        P = ph.sample_at(P0, v, t)
        pr, z = cam.project(P)
        r = (pr - uv).ravel()
        ok = ~np.isnan(sz)
        if ok.any():
            d = np.linalg.norm(P - cam.C[None, :], axis=1)
            r = np.concatenate([r, (cam.F * cam.BALL_D / np.maximum(d, .3) - sz)[ok] / size_w])
        return r + (1e3 if np.any(z < 0.2) else 0.0)

    best = None
    for vg in ([0, -15, 5], [2, -22, 8], [-2, -18, 4], [0, -10, 8], [4, -28, 12],
               [-4, -28, 12], [8, -20, 6], [-8, -20, 6], [0, -32, 4]):
        try:
            s = least_squares(res, vg, method='lm', max_nfev=3000, diff_step=1e-3)
        except Exception:
            continue
        c = float(np.sqrt(np.mean(s.fun ** 2)))
        if best is None or c < best[0]:
            best = (c, s.x)
    return best


def analyse_flight(cam, ph, Pc, V, cl, horizon=4.0):
    """Where does this launch land? -> dict(net_clear, land, depth, wide, result)."""
    ts, ps = ph.traj(Pc, V, horizon, cl=cl)
    n = np.arange(len(ps))
    cross = np.where(ps[:, 1] <= 0)[0]                       # reaches the net plane
    ground_before = np.where((ps[:, 2] <= 0) & (n < (cross[0] if len(cross) else len(ps))))[0]
    out = {}
    if len(ground_before):
        j = ground_before[0]
        return dict(result='NET', net_clear=None, land=[float(ps[j, 0]), float(ps[j, 1])],
                    t_land=float(ts[j]))
    if not len(cross):
        return dict(result='SHORT', net_clear=None)
    i = cross[0]
    out['net_clear'] = float(ps[i, 2] - cam.net_height(ps[i, 0]))
    out['net_x'] = float(ps[i, 0])
    if out['net_clear'] < 0:
        out['result'] = 'NET'
        return out
    after = np.where((ps[:, 2] <= 0) & (n > i))[0]
    if not len(after):
        out['result'] = 'LONG?'
        return out
    j = after[0]
    x, y = float(ps[j, 0]), float(ps[j, 1])
    out.update(land=[x, y], t_land=float(ts[j]),
               depth=-y - cam.HALF,             # >0 = past the far baseline
               wide=abs(x) - cam.SGL)           # >0 = outside the singles sideline
    if out['depth'] > 0:
        out['result'] = 'OUT_LONG'
    elif out['wide'] > 0:
        out['result'] = 'OUT_WIDE'
    else:
        out['result'] = 'IN'
    return out


SIMPLE = {'OUT_LONG': 'OUT', 'OUT_WIDE': 'OUT', 'NET': 'NET', 'IN': 'IN', 'SHORT': 'UNK', 'LONG?': 'UNK'}
CN = {'OUT': '出界', 'NET': '下网', 'IN': '界内', 'UNK': '未判定'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work')
    ap.add_argument('--input', default='shots_input.json')
    ap.add_argument('--cand', default='ball_cand.json')
    ap.add_argument('--spins', type=float, nargs='+', default=[0, 1000, 1500, 2000, 3000])
    ap.add_argument('--nominal-spin', type=float, default=1500,
                    help='spin used for the headline verdict; 0 rpm exaggerates depth and '
                         'therefore over-reports OUT')
    ap.add_argument('--bootstrap', type=int, default=40)
    ap.add_argument('--pixel-noise', type=float, default=1.0)
    ap.add_argument('--size-noise', type=float, default=1.2)
    ap.add_argument('--mass', type=float, default=0.057)
    ap.add_argument('--cd', type=float, default=0.55)
    ap.add_argument('--out', default='shots.json')
    a = ap.parse_args()

    cam = Camera(a.work)
    ph = Physics(cam.BALL_D, a.mass, a.cd)
    spec = json.load(open(os.path.join(a.work, a.input)))
    fps = spec.get('fps') or json.load(open(os.path.join(a.work, 'meta.json')))['fps']
    cp = os.path.join(a.work, spec.get('ball_cand', a.cand))
    cand = json.load(open(cp)) if os.path.exists(cp) else {}
    rng = np.random.default_rng(7)
    res = []

    for sh in spec['shots']:
        tc = float(sh['tc'])
        rec = dict(id=sh['id'], side=sh.get('side'), tc=tc, t_s=round(tc / fps, 2))
        inc = sh.get('incoming') or []
        if sh.get('contact'):
            Pc = np.array(sh['contact'], float)
            rec['contact_src'] = 'given'
        elif len(inc) >= 4:
            sizes = np.array([blob_size(cand, *o) for o in inc], float)
            ci, xi = fit_incoming(cam, ph, inc, sizes, fps)
            if xi is None:
                res.append(dict(rec, kmh=None, conf='none', result='UNK', result_cn=CN['UNK'], result_note='来球拟合失败，无法确定触球点'))
                continue
            P0, V0 = xi[:3], xi[3:6]
            dt = (tc - inc[0][0]) / fps
            Pc = P0 + V0 * dt + 0.5 * np.array([0, 0, -G]) * dt ** 2
            Vc = V0 + np.array([0, 0, -G]) * dt
            rec.update(contact_src='incoming_fit', in_rms=round(ci, 2), n_in=len(inc),
                       feed_kmh=round(float(np.linalg.norm(Vc)) * 3.6, 1))
        else:
            print('S%s: no contact point (need "contact" or >=4 incoming obs)' % sh['id'])
            res.append(dict(rec, result='UNK', result_cn=CN['UNK'], conf='none'))
            continue
        rec['contact'] = [round(float(v), 2) for v in Pc]
        rec['contact_h'] = round(float(Pc[2]), 2)
        rec['behind_baseline'] = round(float(Pc[1] - cam.HALF), 2)

        obs = sh.get('outgoing') or []
        rec['n_out'] = len(obs)
        if len(obs) < 3:
            rec.update(kmh=None, conf='none', result='UNK', result_cn=CN['UNK'],
                       result_note='出球段未检出可信球点，速度与落点均不可测')
            res.append(rec)
            print('S%s %s: outgoing not measurable' % (sh['id'], rec.get('side')))
            continue
        sizes = np.array([blob_size(cand, *o) for o in obs], float)
        fit = fit_outgoing(cam, ph, obs, sizes, Pc, tc, fps)
        if fit is None:
            res.append(dict(rec, kmh=None, conf='none', result='UNK', result_cn='未判定'))
            continue
        co, V = fit
        sp = float(np.linalg.norm(V))
        rec.update(out_rms=round(co, 2), V0=[round(float(v), 2) for v in V], kmh=round(sp * 3.6, 1),
                   elev=round(float(np.degrees(np.arcsin(V[2] / sp))), 1),
                   azim=round(float(np.degrees(np.arctan2(V[0], -V[1]))), 1))
        boots = []
        for _ in range(a.bootstrap if len(obs) >= 2 else 0):
            nz = (rng.normal(0, a.pixel_noise, (len(obs), 2)), rng.normal(0, a.size_noise, len(obs)))
            try:
                c2, V2 = fit_outgoing(cam, ph, obs, sizes, Pc, tc, fps, noise=nz)
                if c2 < co * 4:
                    boots.append(float(np.linalg.norm(V2)) * 3.6)
            except Exception:
                pass
        if boots:
            rec['kmh_lo'] = round(float(np.percentile(boots, 16)), 1)
            rec['kmh_hi'] = round(float(np.percentile(boots, 84)), 1)
        width = (rec.get('kmh_hi', sp * 3.6) - rec.get('kmh_lo', sp * 3.6)) / max(sp * 3.6, 1e-6)
        rec['conf'] = ('high' if len(obs) >= 4 and co < 2.5 and width < 0.2 and len(boots) >= 20 else
                       'medium' if len(obs) >= 3 and co < 4 and len(boots) >= 10 else 'low')

        band = []
        fl_by_rpm = {}
        for rpm in a.spins:
            fl = analyse_flight(cam, ph, Pc, V, ph.cl_of_rpm(rpm))
            fl_by_rpm[rpm] = fl
            band.append(dict(rpm=rpm, result=SIMPLE[fl['result']],
                             net_clear=None if fl.get('net_clear') is None else round(fl['net_clear'], 2),
                             land=[round(v, 2) for v in fl['land']] if fl.get('land') else None,
                             depth=None if fl.get('depth') is None else round(fl['depth'], 2),
                             wide=None if fl.get('wide') is None else round(fl['wide'], 2)))
        rec['spin_band'] = band
        nom = min(band, key=lambda b: abs(b['rpm'] - a.nominal_spin))
        kinds = sorted({b['result'] for b in band})
        rec['result'] = nom['result'] if rec['conf'] in ('high', 'medium') and len(kinds) == 1 else 'UNK'
        rec['nominal_result'] = nom['result']
        rec['interval_note'] = '条件区间：仅像素与尺寸噪声，不含标定、触球时刻、旋转或模型误差'
        rec['result_firm'] = len(kinds) == 1
        rec['net_clear'] = nom['net_clear']
        rec['land'] = nom['land']
        rec['result_cn'] = CN.get(rec['result'], rec['result'])
        note = '按 %d rpm 上旋推算为%s' % (nom['rpm'], CN.get(nom['result'], nom['result']))
        if len(kinds) > 1:
            note += '；旋转带内结果会翻转（%s），落点属外推' % '/'.join(CN.get(k, k) for k in kinds)
        else:
            note += '；0-3000 rpm 全带一致，判定稳健'
        if rec['conf'] == 'low':
            note += '；出球观测点过少，速度为低置信'
        rec['result_note'] = note

        # polylines for the renderer
        rec['t_land'] = round(float(fl_by_rpm[nom['rpm']].get('t_land') or 1.4), 2)
        t_end = min(rec['t_land'] + 0.05, 2.2)
        ts, ps = ph.traj(Pc, V, t_end, cl=ph.cl_of_rpm(nom['rpm']))
        uv, z = cam.project(ps)
        rec['out_path'] = [[round(float(u), 1), round(float(v), 1), round(float(t), 3)]
                           for (u, v), t, zz in zip(uv, ts, z) if zz > .3][::5]
        rec['world_path'] = [[round(float(q), 2) for q in p] for p in ps[::15]]
        res.append(rec)

    json.dump(res, open(os.path.join(a.work, a.out), 'w'), indent=1, ensure_ascii=False)
    print('\n%-4s %-4s %-8s %-22s %-6s %s' % ('id', 'side', 'km/h', 'interval / conf', 'verdict', 'note'))
    for r in res:
        print('%-4s %-4s %-8s %-22s %-6s %s'
              % (r['id'], r.get('side', '-'), r.get('kmh') or '-',
                 '%s-%s %s' % (r.get('kmh_lo', '?'), r.get('kmh_hi', '?'), r.get('conf', '-')),
                 r.get('result', '-'), r.get('result_note', '')))
    # low-confidence speeds are hypotheses, not measurements - they stay out of the averages
    ok = [r['kmh'] for r in res if r.get('kmh') and r['conf'] in ('high', 'medium')]
    if ok:
        print('\nmeasured %d/%d shots  mean %.1f  max %.0f km/h'
              % (len(ok), len(res), np.mean(ok), max(ok)))
        hi = [r['kmh'] for r in res if r.get('kmh') and r['conf'] == 'high']
        if hi:
            print('high-confidence only: mean %.1f km/h (n=%d)' % (np.mean(hi), len(hi)))
    weak = [r['id'] for r in res if r.get('kmh') and r['conf'] == 'low']
    if weak:
        print('excluded from the averages (low confidence, report as "suspected"): shots %s'
              % ', '.join(str(i) for i in weak))
    none = [r['id'] for r in res if not r.get('kmh')]
    if none:
        print('not measurable (report as such, never as 0): shots %s'
              % ', '.join(str(i) for i in none))


if __name__ == '__main__':
    main()
