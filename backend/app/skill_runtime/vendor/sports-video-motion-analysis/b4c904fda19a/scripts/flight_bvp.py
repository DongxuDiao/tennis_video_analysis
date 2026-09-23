"""Ball flight from contact-to-contact boundary values - speeds without tracking the ball.

Why this exists
---------------
`flight_fit.py` fits a trajectory to *observed ball pixels*. That route dies on broadcast
footage: in the clip this was built for the ball spans 2-4 px, and

  * a global yellow-blob detector produced candidates on only 40 of 440 rally frames that a
    physical trajectory could actually pass through - the rest were racket edges, shoes,
    line paint and compression noise;
  * a guided detector seeded with the expected path found 32 points with a 28 px median
    offset from the fitted arc, i.e. the "detections" were not the ball.

With 0-2 unreliable points per shot a least-squares arc fit is fiction. But two things are
solid even then: **where and when the racket met the ball**. A contact point is a 3D point
(wrist/racket ray x a plausible contact height, or the pose-based contact estimate) and the
contact frame comes from the audio impulse cross-checked with the swing peak
(`swing_peaks.py`).

So the flight is solved as a two-point boundary value problem instead:

    given  A = contact i (3D),  B = contact i+1 (3D),  dt = (frame_{i+1} - frame_i) / fps
    find   V0  such that forward integration of
             gravity + quadratic drag + Magnus (topspin) + one hard-court bounce
           starting at A with V0 is at B after exactly dt seconds.

Three unknowns, three equations. Speed, launch angle, apex, net clearance, bounce point and
in/out are then *consequences* of that trajectory, not separate estimates.

Spin is the one quantity a single camera cannot see, and it changes where the ball lands, so
every shot is solved for a band of topspin values and the branch is selected by physical
legality (cleared the net, landed in the opponent's court since the rally demonstrably
continued, service box for a serve) plus the audio bounce impulse. The remaining spread over
the spin band is reported as the speed interval - it is an honest uncertainty, not noise.

When to use it
  * broadcast / long-range footage where the ball is a few pixels;
  * any rally where you can label contacts but not track the ball;
  * cross-checking `flight_fit.py`: the two must agree within the reported bands.

When NOT to use it
  * you cannot pin the contact frames to ~1 frame: a 1-frame error on a 0.7 s flight is
    ~5 km/h, on a 0.3 s flight ~10 km/h. Check `--dt-sens` in the output;
  * the contact *points* are guesses: a 0.5 m error in A or B goes straight into the speed;
  * shots that hit the net cord / frame / bounce twice before the next contact - the model
    allows one bounce between contacts, a second one is reported but is usually a sign that
    the contact labelling is wrong;
  * the last shot of a rally, unless you give `final_landing` (see below): with no next
    contact there is no boundary condition and nothing can be solved.

  python3 scripts/flight_bvp.py --work work --contacts work/contacts.json
  python3 scripts/flight_bvp.py --selftest        # physics sanity check, no data needed

contacts.json - everything human in one file:
  {"fps": 30,
   "players": {"N": {"label": "近端", "hand": "left",  "facing": "away"},
               "F": {"label": "远端", "hand": "right", "facing": "toward"}},
   "contacts": [{"id": 1, "frame": 34, "who": "N", "side": "SERVE",
                 "point": [-0.8, 10.41, 2.62]},
                {"id": 2, "frame": 55, "who": "F", "side": "BH",
                 "point": [5.35, -13.49, 1.06]}],
   "bounce_audio_candidates": {"1": [43.3], "3": [109.8]},
   "final_landing": {"point": [4.35, 8.5], "frame": 550,
                     "source": "rolling ball visually located at f549-557"}}
`frame` is 0-based (fractional is fine), `point` is [x, y, z] in metres in the world frame of
camera.py (near baseline y=+half, net y=0), `bounce_audio_candidates` is keyed by contact id
and holds candidate bounce frames from audio_impacts.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import court  # noqa: E402

G = 9.81


# ------------------------------------------------------------------ physics
class Physics:
    """Point mass, quadratic drag, Magnus lift, restitution bounce on a flat surface."""

    def __init__(self, ball_d=0.067, mass=0.057, cd=0.55, rho=1.2,
                 cor_v=0.75, cor_h=0.85, sim_dt=1 / 600.):
        self.d, self.m, self.cd, self.rho = ball_d, mass, cd, rho
        self.area = np.pi * (ball_d / 2) ** 2
        self.kd = 0.5 * rho * cd * self.area / mass
        self.cor_v, self.cor_h, self.sim_dt = cor_v, cor_h, sim_dt

    def cl_of_rpm(self, rpm):
        """Lift coefficient of a spinning ball (Cross & Lindsey style fit)."""
        return 0.0 if rpm <= 0 else float(min(0.05 + 1.4e-4 * rpm, 0.42))

    def accel(self, V, cl):
        s = float(np.linalg.norm(V))
        acc = np.array([0.0, 0.0, -G]) - self.kd * s * V
        if cl and s > 1e-6:
            vh = V / s
            ax = np.array([-vh[1], vh[0], 0.0])          # spin axis of pure topspin:
            na = np.linalg.norm(ax)                      # horizontal, perpendicular to v
            if na > 1e-6:
                ax /= na
                # SIGN: a -= (0.5*rho*A*Cl*v^2/m) * (v_hat x axis).  With this sign topspin
                # pushes the ball DOWN (it dips and lands shorter); flipping it produces the
                # classic bug "more topspin flies further".  selftest() asserts it.
                acc -= (0.5 * self.rho * self.area * cl * s * s / self.m) * np.cross(vh, ax)
        return acc

    def fly(self, P0, V0, T, rpm=0.0, ground=0.0):
        """Semi-implicit Euler to time T. Returns (ts, ps, bounces[(t, P), ...])."""
        cl = self.cl_of_rpm(rpm)
        P, V = np.array(P0, float), np.array(V0, float)
        dt = self.sim_dt
        ts, ps, bounces = [0.0], [P.copy()], []
        for _ in range(max(int(T / dt), 1)):
            V = V + self.accel(V, cl) * dt
            Pn = P + V * dt
            if Pn[2] <= ground < P[2]:
                f = (P[2] - ground) / max(P[2] - Pn[2], 1e-9)
                Pb = P + (Pn - P) * f
                bounces.append((ts[-1] + dt * f, Pb.copy()))
                V = np.array([V[0] * self.cor_h, V[1] * self.cor_h, -V[2] * self.cor_v])
                Pn = Pb + V * dt * (1 - f)
            P = Pn
            ts.append(ts[-1] + dt)
            ps.append(P.copy())
        return np.array(ts), np.array(ps), bounces


def at(ts, ps, t):
    return ps[int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))]


def solve_bvp(ph, A, B, dt, rpm, seed=None):
    """Launch velocity that carries the ball from A to B in dt seconds."""
    from scipy.optimize import least_squares
    A = np.array(A, float)
    B = np.array(B, float)
    if seed is None:
        d = B - A
        seed = np.array([d[0] / dt, d[1] / dt, d[2] / dt + 0.5 * G * dt])

    def res(V):
        ts, ps, _ = ph.fly(A, V, dt * 1.02, rpm)
        return at(ts, ps, dt) - B

    r = least_squares(res, seed, method='lm', max_nfev=120, xtol=1e-9, ftol=1e-9)
    ts, ps, bo = ph.fly(A, r.x, dt * 1.02, rpm)
    err = float(np.linalg.norm(at(ts, ps, dt) - B))
    return r.x, err, ts, ps, bo


# ------------------------------------------------------------------ court helpers
def net_post_x(c, sport, override=None):
    """Half-distance between the net posts. Tennis posts stand 0.914 m outside the doubles
    sideline; in most other sports they sit on the sideline itself."""
    if override:
        return float(override)
    if c.get('post_x'):
        return float(c['post_x'])
    return c['dbl'] + 0.914 if sport == 'tennis' else c['dbl']


def net_height_at(x, c, post_x):
    return c['net_c'] + (c['net_p'] - c['net_c']) * min(abs(float(x)) / max(post_x, 1e-6), 1.0)


def net_cross(ts, ps):
    """First crossing of the net plane y = 0."""
    y = ps[:, 1]
    for i in range(1, len(y)):
        if y[i - 1] * y[i] < 0:
            f = y[i - 1] / (y[i - 1] - y[i])
            return ps[i - 1] + (ps[i] - ps[i - 1]) * f, ts[i - 1]
    return None, None


# ------------------------------------------------------------------ input checks
def load_contacts(path, fps_meta=None):
    if not os.path.exists(path):
        raise SystemExit('missing %s - write the contact list first (see the docstring)' % path)
    try:
        D = json.load(open(path))
    except Exception as e:
        raise SystemExit('%s is not valid JSON: %s' % (path, e))
    if not isinstance(D, dict) or 'contacts' not in D:
        raise SystemExit('%s must be an object with a "contacts" list' % path)
    fps = D.get('fps') or fps_meta
    if not fps:
        raise SystemExit('no fps: put "fps" in %s or a meta.json in --work' % path)
    cs = D['contacts']
    if len(cs) < 2 and not D.get('final_landing'):
        raise SystemExit('need at least 2 contacts, or 1 contact plus "final_landing"')
    seen = set()
    for i, s in enumerate(cs):
        for k in ('frame', 'point'):
            if k not in s:
                raise SystemExit('contact #%d is missing "%s": %s' % (i, k, s))
        if len(s['point']) != 3:
            raise SystemExit('contact #%d "point" must be [x, y, z] in metres, got %s'
                             % (i, s['point']))
        s.setdefault('id', i + 1)
        s.setdefault('who', '?')
        s.setdefault('side', '?')
        if s['id'] in seen:
            raise SystemExit('duplicate contact id %s' % s['id'])
        seen.add(s['id'])
        if i and float(s['frame']) <= float(cs[i - 1]['frame']):
            raise SystemExit('contacts must be sorted by increasing frame (#%d f%s <= #%d f%s)'
                             % (i, s['frame'], i - 1, cs[i - 1]['frame']))
    for i in range(1, len(cs)):
        y0, y1 = cs[i - 1]['point'][1], cs[i]['point'][1]
        if y0 * y1 > 0:
            print('WARNING contacts %s and %s are on the same side of the net (y=%.1f, %.1f) - '
                  'a shot cannot start and end on one side; check the labelling'
                  % (cs[i - 1]['id'], cs[i]['id'], y0, y1))
    fl = D.get('final_landing')
    if fl and ('point' not in fl or 'frame' not in fl):
        raise SystemExit('"final_landing" needs {"point": [x, y], "frame": f}')
    D['fps'] = float(fps)
    return D


# ------------------------------------------------------------------ one flight
def candidates(ph, A, B, dt, spins, angles, fa, fps, c, post_x, audio):
    """Solve every (spin, launch-angle) branch and describe the resulting trajectory."""
    out = []
    d = np.array(B, float) - np.array(A, float)
    dh = float(np.hypot(d[0], d[1]))
    uh = d[:2] / max(dh, 1e-6)
    for rpm in spins:
        for ang in angles:
            ca = max(np.cos(np.radians(ang)), 0.2)
            vh = dh / dt / ca * 1.05
            seed = np.array([uh[0] * vh, uh[1] * vh, vh * np.tan(np.radians(ang))])
            try:
                V, err, ts, ps, bo = solve_bvp(ph, A, B, dt, rpm, seed)
            except Exception:
                continue
            npos, nt = net_cross(ts, ps)
            rec = dict(rpm=float(rpm), angle0=float(ang), err_m=err,
                       V0=[float(x) for x in V],
                       v0_kmh=float(np.linalg.norm(V)) * 3.6,
                       angle_deg=float(np.degrees(np.arctan2(V[2], np.hypot(V[0], V[1])))),
                       apex_m=float(np.max(ps[:, 2])),
                       net_z=None if npos is None else float(npos[2]),
                       net_x=None if npos is None else float(npos[0]),
                       net_clear=None if npos is None else
                       float(npos[2] - net_height_at(npos[0], c, post_x)),
                       n_bounce=len(bo), bounce=None, bounce_frame=None, bounce_in=None,
                       dt_bounce=None)
            if bo:
                t_b, Pb = bo[0]
                rec['bounce'] = [float(Pb[0]), float(Pb[1])]
                rec['bounce_frame'] = float(fa + t_b * fps)
                rec['bounce_in'] = bool(abs(Pb[0]) <= c['sgl'] and abs(Pb[1]) <= c['half'])
                if audio:
                    rec['dt_bounce'] = float(min(abs(rec['bounce_frame'] - x) for x in audio))
            out.append(rec)
    return out


def pick(cands, side_sign, is_serve, require_in, c, a):
    """Select the physically admissible branch, and the best branch per spin value.

    Scoring, in words (the weights only encode this order):
      1. hard prefilter on the endpoint residual (--err-max): a branch that does not reach B
         is not a solution of the problem we posed;
      2. +100 fully legal: cleared the net by --net-min, bounced on the opponent's side, and
         - when the rally provably continued - inside the court (+ the service box for a
         serve). +40 for "at least it bounced in the court" so a fallback exists;
      3. up to +30 for agreeing with an audio bounce impulse, minus 3 per frame of
         disagreement: the bounce is the loudest event in the flight and the only
         independent timing evidence we have about the spin;
      4. a *strong* penalty (--err-weight, default 100 per metre) on the endpoint residual.
         This matters more than it looks: the residual is not smooth in V0 (the number of
         bounces and the bounce time jump), so a fair number of branches stall at 0.1-0.9 m
         instead of converging. Those are not solutions of the boundary value problem, and a
         stalled branch that happens to match the audio must not beat a converged one;
      5. a very small pull towards --nominal-spin so ties resolve to an ordinary topspin
         instead of the extreme end of the band.

    Returns (best, by_spin) where by_spin[rpm] is the best branch at that spin - the spin
    band used for the speed interval and for the IN/OUT flip check.
    """
    ok = [q for q in cands if q['err_m'] < a.err_max] or list(cands)

    def legal(q):
        if q['net_clear'] is None or q['net_clear'] < a.net_min:
            return False
        if q['bounce'] is None:
            return False
        if np.sign(q['bounce'][1]) != side_sign:
            return False
        if require_in and not q['bounce_in']:
            return False
        if is_serve and abs(q['bounce'][1]) > c['svc'] * (1 + a.svc_tol):
            return False
        return True

    def score(q):
        s = 0.0
        if legal(q):
            s += 100.0
        elif q.get('bounce_in'):
            s += 40.0
        if q['dt_bounce'] is not None and q['dt_bounce'] <= a.audio_tol:
            s += 30.0 - 3.0 * q['dt_bounce']
        s -= a.err_weight * max(0.0, q['err_m'] - 0.02)
        s -= 0.002 * abs(q['rpm'] - a.nominal_spin)
        return s

    for q in ok:
        q['legal'] = legal(q)
        q['score'] = score(q)
    best = max(ok, key=lambda q: q['score'])
    by_spin = {}
    for q in ok:
        cur = by_spin.get(q['rpm'])
        if cur is None or q['score'] > cur['score']:
            by_spin[q['rpm']] = q
    return best, by_spin


def verdict(rec):
    if rec.get('net_clear') is not None and rec['net_clear'] < 0:
        return 'NET'
    if rec.get('bounce_in') is None:
        return 'UNK'
    return 'IN' if rec['bounce_in'] else 'OUT'


CN = {'IN': '界内', 'OUT': '出界', 'NET': '下网', 'UNK': '未判定'}


# ------------------------------------------------------------------ selftest
def selftest():
    print('--- flight_bvp selftest ---')
    ph = Physics(cd=0.0, rho=1.2)                          # drag off -> analytic projectile
    P0, V0 = np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 10.0])
    ts, ps, _ = ph.fly(P0, V0, 0.5, rpm=0)
    num = at(ts, ps, 0.5)
    ana = P0 + V0 * 0.5 + 0.5 * np.array([0, 0, -G]) * 0.25
    err = float(np.linalg.norm(num - ana))
    print('drag-free projectile at t=0.5 s: numeric %s vs analytic %s -> %.4f m'
          % (np.round(num, 4).tolist(), np.round(ana, 4).tolist(), err))
    assert err < 0.02, 'integrator disagrees with the analytic projectile'

    ph = Physics()
    a0 = ph.accel(np.array([0.0, -25.0, 3.0]), ph.cl_of_rpm(2400))
    print('topspin vertical accel %.2f m/s^2 (must be < -%.2f: the ball is pushed down)'
          % (a0[2], G))
    assert a0[2] < -G, 'Magnus sign is flipped - topspin must add downward force'

    y0, land = 11.0, {}
    for rpm in (0, 2400):
        ts, ps, bo = ph.fly([0, y0, 1.0], [0, -26.0, 2.0], 2.5, rpm=rpm)
        assert bo, 'no bounce in the reference shot'
        land[rpm] = y0 - float(bo[0][1][1])              # distance travelled down-court, m
        print('  %4d rpm -> first bounce %.2f m from the launch point (y=%.2f)'
              % (rpm, land[rpm], bo[0][1][1]))
    assert land[2400] < land[0], ('topspin must land SHORTER than no spin - if it lands '
                                  'further the Magnus sign is flipped')
    print('topspin lands %.2f m shorter than flat, as it must' % (land[0] - land[2400]))

    A, Vtrue, dt = np.array([0.5, 11.5, 1.0]), np.array([-1.0, -30.0, 4.0]), 0.7
    ts, ps, _ = ph.fly(A, Vtrue, dt * 1.02, rpm=1600)
    B = at(ts, ps, dt)
    V, err_m, _, _, _ = solve_bvp(ph, A, B, dt, 1600)
    print('BVP round trip: recovered V0 %s vs true %s (endpoint err %.4f m)'
          % (np.round(V, 3).tolist(), Vtrue.tolist(), err_m))
    assert err_m < 0.02 and np.linalg.norm(V - Vtrue) < 0.3, 'BVP solver does not invert fly()'
    print('OK - integrator, Magnus sign, spin effect and BVP inversion all consistent')
    return 0


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default=None, help='work dir (meta.json supplies fps)')
    ap.add_argument('--contacts', default=None, help='default $WORK/contacts.json')
    ap.add_argument('--out', default=None, help='default $WORK/flights.json')
    ap.add_argument('--fps', type=float, default=None, help='override fps')
    ap.add_argument('--sport', default=None, help='override the court preset')
    ap.add_argument('--spins', default='0,800,1600,2400,3200',
                    help='candidate topspin values, rpm')
    ap.add_argument('--angles', default='-8,0,8,16,24',
                    help='launch elevation branches used as solver starts, deg')
    ap.add_argument('--nominal-spin', type=float, default=2400,
                    help='tie-break spin when the audio cannot decide')
    ap.add_argument('--err-max', type=float, default=0.25,
                    help='max endpoint residual (m) for a branch to count as a solution')
    ap.add_argument('--err-good', type=float, default=0.15,
                    help='endpoint residual (m) needed for high confidence')
    ap.add_argument('--err-weight', type=float, default=100.0,
                    help='score penalty per metre of endpoint residual; high on purpose, a '
                         'stalled branch is not a solution of the boundary value problem')
    ap.add_argument('--net-min', type=float, default=0.03,
                    help='minimum clearance over the tape (m) for a legal shot')
    ap.add_argument('--svc-tol', type=float, default=0.03,
                    help='relative tolerance on the service line for serves')
    ap.add_argument('--audio-tol', type=float, default=8.0,
                    help='frames a predicted bounce may differ from an audio peak')
    ap.add_argument('--band-max-frac', type=float, default=0.35,
                    help='relative width of the speed band above which the shot is called '
                         'low confidence and kept out of the averages')
    ap.add_argument('--dt-sens', type=float, default=2.0,
                    help='+/- this many frames of contact-timing error folded into the band')
    ap.add_argument('--landing-z', type=float, default=0.02,
                    help='height (m) assigned to a visually measured landing point')
    ap.add_argument('--mass', type=float, default=0.057)
    ap.add_argument('--cd', type=float, default=0.55)
    ap.add_argument('--rho', type=float, default=1.2)
    ap.add_argument('--cor-v', type=float, default=0.75, help='vertical restitution (hard court)')
    ap.add_argument('--cor-h', type=float, default=0.85, help='horizontal restitution')
    ap.add_argument('--sim-dt', type=float, default=1 / 600., help='integration step, s')
    ap.add_argument('--net-post-x', type=float, default=None,
                    help='half distance between the net posts, m (default: from the court preset)')
    ap.add_argument('--selftest', action='store_true',
                    help='run the analytic physics checks and exit (no data needed)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.work:
        ap.error('--work is required (or use --selftest)')

    meta_p = os.path.join(a.work, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    cpath = a.contacts or os.path.join(a.work, 'contacts.json')
    D = load_contacts(cpath, a.fps or meta.get('fps'))
    fps = a.fps or D['fps']
    sport = a.sport or D.get('sport') or meta.get('sport') or 'tennis'
    c = court(D.get('court') or sport)
    post_x = net_post_x(c, sport, a.net_post_x)
    spins = [float(x) for x in str(a.spins).replace(' ', '').split(',') if x != '']
    angles = [float(x) for x in str(a.angles).replace(' ', '').split(',') if x != '']
    ph = Physics(c['ball_d'], a.mass, a.cd, a.rho, a.cor_v, a.cor_h, a.sim_dt)
    S = D['contacts']
    audio_all = {str(k): list(v) for k, v in (D.get('bounce_audio_candidates') or {}).items()}
    players = D.get('players') or {}
    print('%d contacts, fps %.3f, sport %s, net posts at |x|=%.2f m, spins %s, angles %s'
          % (len(S), fps, sport, post_x, spins, angles))

    tasks = []
    for i in range(len(S) - 1):
        tasks.append((S[i], np.array(S[i + 1]['point'], float), float(S[i + 1]['frame']),
                      'next_contact', None))
    fl = D.get('final_landing')
    if fl:
        B = np.array([fl['point'][0], fl['point'][1],
                      fl['point'][2] if len(fl['point']) > 2 else a.landing_z], float)
        tasks.append((S[-1], B, float(fl['frame']), 'visual_landing', fl.get('source')))
    else:
        print('note: the last contact (id %s) has no next contact and no "final_landing" - '
              'that shot cannot be solved and is skipped. Locate the landing frame visually '
              'and add "final_landing" to recover it.' % S[-1]['id'])

    res = []
    for s, B, fb, src, note in tasks:
        A = np.array(s['point'], float)
        dt = (fb - float(s['frame'])) / fps
        if dt <= 0:
            print('contact %s: non-positive flight time, skipped' % s['id'])
            continue
        audio = audio_all.get(str(s['id'])) or s.get('bounce_audio_cand')
        cands = candidates(ph, A, B, dt, spins, angles, float(s['frame']), fps, c, post_x, audio)
        if not cands:
            print('contact %s: no branch converged' % s['id'])
            continue
        is_serve = str(s.get('side', '')).upper().startswith('SERVE')
        best, by_spin = pick(cands, -np.sign(A[1]), is_serve, src == 'next_contact', c, a)
        keep = [by_spin[k] for k in sorted(by_spin)]

        # timing sensitivity at the selected spin: a contact frame is good to about +-1 frame
        sens = []
        for df in (-a.dt_sens, a.dt_sens):
            dt2 = dt + df / fps
            if dt2 <= 0.05:
                continue
            try:
                V2, _, _, _, _ = solve_bvp(ph, A, B, dt2, best['rpm'])
                sens.append(float(np.linalg.norm(V2)) * 3.6)
            except Exception:
                pass
        band = [q['v0_kmh'] for q in keep] + sens

        # Confidence. Two things can make a speed untrustworthy: the boundary value problem
        # was not really solved (endpoint residual), or the answer depends on the spin we
        # cannot measure. The second one needs care: when the rally demonstrably continued,
        # "the ball landed in play" is an OBSERVATION, so a spin whose branch lands out is
        # refuted by that observation - not a contradiction in our conclusion. A flip only
        # matters among branches that no independent evidence rules out (typically the last
        # shot of a rally, where the outcome is exactly what we are trying to decide).
        audio_ok = best['dt_bounce'] is not None and best['dt_bounce'] <= a.audio_tol
        pool = [q for q in keep if q['bounce_in'] is not None and q['err_m'] < a.err_max]
        live = [q for q in pool if q['bounce_in']] if src == 'next_contact' else pool
        flips = sorted({bool(q['bounce_in']) for q in live})
        spread = (max(band) - min(band)) / max(best['v0_kmh'], 1e-6)
        flip_reason = None
        if best['err_m'] > a.err_max:
            conf = 'low'
            flip_reason = ('endpoint residual %.2f m > %.2f m: the contact points or the '
                           'contact frames are inconsistent' % (best['err_m'], a.err_max))
        elif src == 'next_contact' and pool and not live:
            conf = 'low'
            flip_reason = ('no spin in the band %s rpm lands the ball in play, yet the rally '
                           'continued - a contact point or a contact frame must be wrong'
                           % sorted(q['rpm'] for q in pool))
        elif len(flips) > 1 and not audio_ok:
            conf = 'low'
            ins = sorted(q['rpm'] for q in live if q['bounce_in'])
            outs = sorted(q['rpm'] for q in live if q['bounce_in'] is False)
            flip_reason = ('verdict flips inside the spin band (IN at %s rpm, OUT at %s rpm) '
                           'and no audio bounce peak to arbitrate' % (ins, outs))
        elif spread > a.band_max_frac:
            conf = 'low'
            flip_reason = ('speed spans %.0f-%.0f km/h (%.0f%% of the value) over the spin '
                           'band and the +/-%.0f frame timing check - too wide to average'
                           % (min(band), max(band), 100 * spread, a.dt_sens))
        elif best['err_m'] < a.err_good and best['legal'] and audio_ok:
            conf = 'high'
        else:
            conf = 'medium'

        rec = dict(id=s['id'], fa=float(s['frame']), fb=fb, dt=round(dt, 3),
                   who=s.get('who'), side=s.get('side'),
                   player=(players.get(s.get('who')) or {}).get('label'),
                   contact=[round(float(x), 2) for x in A],
                   target=[round(float(x), 2) for x in B],
                   boundary_source=src,
                   rpm=best['rpm'], V0=[round(x, 2) for x in best['V0']],
                   v0_kmh=round(best['v0_kmh'], 1),
                   v0_lo=round(min(band), 1), v0_hi=round(max(band), 1),
                   angle_deg=round(best['angle_deg'], 1),
                   apex_m=round(best['apex_m'], 2),
                   net_clear=None if best['net_clear'] is None else round(best['net_clear'], 2),
                   net_x=None if best['net_x'] is None else round(best['net_x'], 2),
                   err_m=round(best['err_m'], 3),
                   bounce_audio_cand=audio, dt_bounce=None if best['dt_bounce'] is None
                   else round(best['dt_bounce'], 1),
                   n_branches=len(cands), conf=conf, flip_reason=flip_reason,
                   spin_band=[dict(rpm=q['rpm'], v0_kmh=round(q['v0_kmh'], 1),
                                   angle_deg=round(q['angle_deg'], 1),
                                   bounce=None if q['bounce'] is None else
                                   [round(q['bounce'][0], 2), round(q['bounce'][1], 2)],
                                   bounce_in=q['bounce_in'], err_m=round(q['err_m'], 3),
                                   dt_bounce=None if q['dt_bounce'] is None
                                   else round(q['dt_bounce'], 1),
                                   legal=q['legal'])
                              for q in keep])
        if src == 'visual_landing':
            # the landing point IS the measurement here - report it, and keep the simulated
            # bounce alongside so a disagreement is visible instead of hidden
            rec.update(bounce=[round(float(B[0]), 2), round(float(B[1]), 2)],
                       bounce_frame=round(fb, 1),
                       bounce_in=bool(abs(B[0]) <= c['sgl'] and abs(B[1]) <= c['half']),
                       bounce_sim=None if best['bounce'] is None else
                       [round(best['bounce'][0], 2), round(best['bounce'][1], 2)],
                       note=note or 'landing point measured visually')
        else:
            rec.update(bounce=None if best['bounce'] is None else
                       [round(best['bounce'][0], 2), round(best['bounce'][1], 2)],
                       bounce_frame=None if best['bounce_frame'] is None
                       else round(best['bounce_frame'], 1),
                       bounce_in=best['bounce_in'])
        rec['result'] = verdict(rec)
        rec['result_cn'] = CN[rec['result']]
        res.append(rec)
        print('%3s %-2s %-5s f%-4.0f->f%-4.0f dt=%.2f v0=%5.1f [%5.1f-%5.1f] rpm=%4.0f '
              'ang=%+5.1f apex=%4.2f net=%s bnc=%s %-3s dtb=%s err=%.3f %-6s %s'
              % (rec['id'], rec['who'], rec['side'], rec['fa'], rec['fb'], dt, rec['v0_kmh'],
                 rec['v0_lo'], rec['v0_hi'], rec['rpm'], rec['angle_deg'], rec['apex_m'],
                 rec['net_clear'], rec['bounce'], rec['result'], rec['dt_bounce'],
                 rec['err_m'], rec['conf'], rec['boundary_source']), flush=True)
        if flip_reason:
            print('      low confidence: %s' % flip_reason)

    # ---------------------------------------------------------------- summary
    good = [r for r in res if r['conf'] in ('high', 'medium')]
    def stats(rows):
        v = [r['v0_kmh'] for r in rows]
        return dict(n=len(rows), mean_kmh=None if not v else round(float(np.mean(v)), 1),
                    max_kmh=None if not v else round(float(max(v)), 1))
    by_player, by_side = {}, {}
    for r in good:
        by_player.setdefault(r['who'], []).append(r)
        by_side.setdefault(str(r['side']).upper(), []).append(r)
    summary = dict(n_flights=len(res), n_in_average=len(good),
                   excluded_low_conf=[r['id'] for r in res if r['conf'] == 'low'],
                   overall=stats(good),
                   by_player={k: dict(stats(v), label=(players.get(k) or {}).get('label'))
                              for k, v in sorted(by_player.items())},
                   by_side={k: stats(v) for k, v in sorted(by_side.items())},
                   results={k: sum(1 for r in res if r['result'] == k)
                            for k in ('IN', 'OUT', 'NET', 'UNK')},
                   conf={k: sum(1 for r in res if r['conf'] == k)
                         for k in ('high', 'medium', 'low')})
    out = a.out or os.path.join(a.work, 'flights.json')
    json.dump(dict(fps=fps, sport=sport, source=os.path.basename(cpath),
                   method='contact-to-contact boundary value problem (flight_bvp.py)',
                   summary=summary, flights=res), open(out, 'w'), indent=1, ensure_ascii=False)

    print('\n--- summary ---')
    print('%d flights, %d in the averages (low confidence excluded: %s)'
          % (summary['n_flights'], summary['n_in_average'],
             summary['excluded_low_conf'] or 'none'))
    if summary['overall']['mean_kmh'] is not None:
        print('overall  mean %.1f  max %.1f km/h'
              % (summary['overall']['mean_kmh'], summary['overall']['max_kmh']))
    for k, v in summary['by_player'].items():
        print('player %-3s %-6s n=%-3d mean %s  max %s km/h'
              % (k, v['label'] or '', v['n'], v['mean_kmh'], v['max_kmh']))
    for k, v in summary['by_side'].items():
        print('side   %-6s n=%-3d mean %s  max %s km/h' % (k, v['n'], v['mean_kmh'], v['max_kmh']))
    print('verdicts %s   confidence %s' % (summary['results'], summary['conf']))
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
