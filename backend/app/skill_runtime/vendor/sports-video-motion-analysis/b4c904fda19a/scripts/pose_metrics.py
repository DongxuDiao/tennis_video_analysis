"""Step 7 - pose-derived technique metrics per shot -> technique.json

With one camera, absolute joint angles in 3D are not recoverable, but *relative* quantities
are, and those are what coaching feedback is built on:

  torso rotation   how far the chest is turned away from the net at contact, and how much of
                   that turn is released into the ball (`torso_unwind_deg`). Two independent
                   routes, see "torso metrics" below - every torso field carries a `source`
                   so a number can always be traced back to the route that produced it
  coil timing      frames between peak shoulder turn and contact; too short = arm-only swing
  X-factor         shoulder rotation minus hip rotation at peak coil (torso separation)
  contact geometry contact height (m) and lateral offset from the body centre, expressed in
                   shoulder widths, which is scale-free
  stability        head displacement and hip vertical travel through the swing

Report differences and trends, not absolute degrees, and say so in the report.

Racket hand and which way the player faces (--hand / --facing)
--------------------------------------------------------------
Forehand or backhand is decided from **which image side of the torso mid line the contact
point sits on**, and the mapping between "image side" and "forehand" needs both the racket
hand and the body orientation:

  * a player with the **back to the camera** shares the viewer's left/right, so the forehand
    is on the image side of the racket hand (right hand -> image right);
  * a player **facing the camera** is mirrored, so the forehand is on the *opposite* image
    side (right hand -> image left).

    fh_on_image_right = (hand == 'right') == (facing == 'away')

Verified on the two-player broadcast clip this was extended for: the near player (left hand,
back to the camera) hits forehands on the image LEFT of his torso mid line, the far player
(right hand, facing the camera) also on the image LEFT - both exactly as the rule predicts
(13/13 non-serve contacts agree). The defaults `--hand right --facing away` reproduce the
historical hard-coded rule ("camera behind a right-handed player"), so single-player clips
are unaffected. `side_suggest` / `side_conflict` always report the rule that was applied.

Torso metrics: 3D shoulder/hip lines, with a projected-ratio fallback
--------------------------------------------------------------------
  shoulder_hip_3d    each shoulder (hip) keypoint is back-projected onto the horizontal
                     plane at its own height, giving the azimuth of the shoulder (hip) line
                     on the court: 0 deg = square to the net, 90 deg = fully sideways. Needs
                     a big, sharp figure - the depth of a back-projected point is very
                     sensitive to a pixel of error.
  torso_ratio_proxy  image shoulder width / torso length. The ratio shrinks as the chest
                     turns away from the camera axis, and `arccos(ratio / ratio_max)` with
                     `ratio_max` taken from the whole clip turns it into a closing angle.
                     Scale-free, so it survives a 100 px tall far-court player.

The 3D route is used when it is trustworthy and the proxy otherwise; the decision is printed
with its reason and stored per row in `torso_source`. It degrades when the angles are
degenerate (all ~0 / no variance - what happens when both shoulders get projected onto one
depth plane), when the figure is too small (`--rot3d-min-box-h`), when the shoulder/hip
keypoints are not confident enough, or when the frame-to-frame jitter shows the angle is
noise (`--rot3d-max-jitter`). Force either route with `--torso-source`.

Shot input: both formats
------------------------
  --shots shots.json     list (or {"shots": [...]}) from flight_fit.py: `tc`, `side`, `contact`
  --shots contacts.json  {"fps", "players": {"N": {"hand", "facing", "label"}}, "contacts":
                         [{"id", "frame", "who", "side", "point"}]} - the human-labelled file
                         that flight_bvp.py also reads. `who` then selects the pose file via
                         --pose-map and the hand/facing from `players`.

  python3 scripts/pose_metrics.py --work work --shots shots.json
  python3 scripts/pose_metrics.py --work work_stab --shots contacts.json \
      --pose-map N=pose_near.json,F=pose_far.json --hand N=left,F=right \
      --facing N=away,F=toward
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera  # noqa: E402

def resolve(work, name):
    """A bare file name is always looked up inside --work (so 'pose.json' never picks up a
    same-named file in the current directory); anything with a path separator is used as
    given when it exists."""
    if os.path.isabs(name):
        return name
    if os.sep in name or (os.altsep and os.altsep in name):
        return name if os.path.exists(name) else os.path.join(work, name)
    return os.path.join(work, name)


NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
SRC_3D, SRC_PROXY = 'shoulder_hip_3d', 'torso_ratio_proxy'


# --------------------------------------------------------------------------- pose container
class Pose:
    def __init__(self, work, file='pose.json'):
        path = resolve(work, file)
        P = json.load(open(path))
        P = P.get('frames', P)
        # pose_extract.py stores its QC block under non-numeric top-level keys
        self.P = {k: v for k, v in P.items() if str(k).isdigit()}
        self.path = path
        self.frames = sorted(int(k) for k in self.P)

    def k(self, f):
        d = self.P.get(str(int(f)))
        return np.array(d['kpts'], float) if d else None

    def box(self, f):
        d = self.P.get(str(int(f)))
        return d['box'] if d else None

    def width(self, f, a, b, cmin=0.35):
        k = self.k(f)
        if k is None or k[a][2] < cmin or k[b][2] < cmin:
            return np.nan
        return float(abs(k[a][0] - k[b][0]))

    def seg(self, f, a, b, cmin=0.35):
        k = self.k(f)
        if k is None or k[a][2] < cmin or k[b][2] < cmin:
            return np.nan
        return float(np.hypot(k[a][0] - k[b][0], k[a][1] - k[b][1]))

    def angle(self, f, a, b, c, cmin=0.3):
        k = self.k(f)
        if k is None or min(k[a][2], k[b][2], k[c][2]) < cmin:
            return np.nan
        v1, v2 = k[a][:2] - k[b][:2], k[c][:2] - k[b][:2]
        cs = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        return float(np.degrees(np.arccos(np.clip(cs, -1, 1))))

    def wrist_speed(self, f, w):
        a, b = self.k(f - 1), self.k(f + 1)
        if a is None or b is None or min(a[w][2], b[w][2]) < 0.3:
            return np.nan
        return float(np.hypot(a[w][0] - b[w][0], a[w][1] - b[w][1]) / 2)

    def box_h_median(self):
        h = [d['box'][3] - d['box'][1] for d in self.P.values()]
        return float(np.median(h)) if h else 0.0


# --------------------------------------------------------------------- geometry helpers
def at_depth(cam, uv, y):
    """Back-project an image point onto the vertical plane y = const (the player's depth)."""
    dr = cam.ray([uv])[0]
    t = (y - cam.C[1]) / dr[1]
    return cam.C + t * dr


def ground_of(cam, k, cmin=0.25):
    """Court position of the player from the visible ankles."""
    an = [k[i, :2] for i in (LAN, RAN) if k[i, 2] > cmin]
    if not an:
        return None
    return cam.img2ground([np.mean(an, axis=0)])[0]


def torso_ratio(k, cmin=0.2, min_len=5.0):
    """apparent shoulder width / torso length in the image: shrinks as the torso turns away
    from the camera axis, which is exactly the 'closed vs open stance' information."""
    if min(k[LSH, 2], k[RSH, 2], k[LHIP, 2], k[RHIP, 2]) < cmin:
        return None
    w = float(np.linalg.norm(k[LSH, :2] - k[RSH, :2]))
    t = float(np.linalg.norm((k[LSH, :2] + k[RSH, :2]) / 2 - (k[LHIP, :2] + k[RHIP, :2]) / 2))
    return None if t < min_len else w / t


def rot_scale(po, pct=92.0):
    """Calibrate the proxy: the widest projected torso in the clip is 'square to the camera'.
    A percentile, not the max, so one bad frame cannot set the scale."""
    r = [torso_ratio(po.k(f)) for f in po.frames]
    r = [x for x in r if x]
    return float(np.percentile(r, pct)) if r else None


def proxy_deg(ratio, rmax):
    if ratio is None or not rmax:
        return None
    return float(np.degrees(np.arccos(np.clip(ratio / rmax, 0, 1))))


def line_azimuth_3d(cam, k, a, b, cmin=0.25, z_lo=0.2, z_hi=2.6):
    """Closing angle of the a-b body line from 3D back-projection (deg, 0 = square to net).

    Both endpoints are lifted onto the horizontal plane at the height the depth plane gives
    them, so the line gets a real court azimuth instead of an image slope.
    """
    if min(k[a, 2], k[b, 2]) < cmin:
        return None
    g = ground_of(cam, k)
    if g is None:
        return None
    zm = (at_depth(cam, k[a, :2], g[1])[2] + at_depth(cam, k[b, :2], g[1])[2]) / 2
    if not (z_lo < zm < z_hi):
        return None
    A = cam.img2plane([k[a, :2]], zm)[0]
    B = cam.img2plane([k[b, :2]], zm)[0]
    d = B[:2] - A[:2]
    if np.linalg.norm(d) < 1e-6:
        return None
    ang = abs(float(np.degrees(np.arctan2(d[1], d[0]))))
    return 180.0 - ang if ang > 90 else ang


def rot3d_quality(cam, po, a):
    """Is the 3D shoulder/hip route usable on this pose file? Returns (ok, info)."""
    sh, hip, conf = {}, {}, []
    for f in po.frames:
        k = po.k(f)
        conf.append(min(k[LSH, 2], k[RSH, 2], k[LHIP, 2], k[RHIP, 2]))
        v = line_azimuth_3d(cam, k, LSH, RSH)
        if v is not None:
            sh[f] = v
            h = line_azimuth_3d(cam, k, LHIP, RHIP)
            if h is not None:
                hip[f] = h
    vals = np.array([sh[f] for f in sorted(sh)], float)
    dif = [abs(sh[f + 1] - sh[f]) for f in sorted(sh) if f + 1 in sh]
    info = dict(n_ok=len(sh), n_frames=len(po.frames),
                frac=round(len(sh) / max(len(po.frames), 1), 3),
                std_deg=round(float(vals.std()), 2) if len(vals) else 0.0,
                jitter_deg=round(float(np.median(dif)), 2) if dif else None,
                box_h_px=round(po.box_h_median(), 1),
                kpt_conf_median=round(float(np.median(conf)), 3) if conf else 0.0)
    why = []
    if info['frac'] < a.rot3d_min_frac:
        why.append('3D angles available on only %.0f%% of frames (< %.0f%%)'
                   % (100 * info['frac'], 100 * a.rot3d_min_frac))
    if info['std_deg'] < a.rot3d_min_std:
        why.append('angle spread %.2f deg is degenerate (all ~0: the keypoints collapse onto '
                   'one depth plane)' % info['std_deg'])
    if info['box_h_px'] < a.rot3d_min_box_h:
        why.append('player only %.0f px tall (< %.0f): back-projected depth is too noisy'
                   % (info['box_h_px'], a.rot3d_min_box_h))
    if info['jitter_deg'] is not None and info['jitter_deg'] > a.rot3d_max_jitter:
        why.append('frame-to-frame jitter %.1f deg (> %.1f): the angle is noise, not motion'
                   % (info['jitter_deg'], a.rot3d_max_jitter))
    if info['kpt_conf_median'] < a.rot3d_min_conf:
        why.append('median shoulder/hip confidence %.2f (< %.2f)'
                   % (info['kpt_conf_median'], a.rot3d_min_conf))
    info['reasons'] = why
    return (not why), info, sh, hip


# --------------------------------------------------------------------------- inputs
def parse_map(spec, default):
    """'N=left,F=right' -> {'N': 'left', 'F': 'right'};  'left' -> {} with default 'left'."""
    if spec is None:
        return {}, default
    if '=' not in str(spec):
        return {}, str(spec)
    out = {}
    for part in str(spec).replace(';', ',').split(','):
        if not part.strip():
            continue
        k, _, v = part.partition('=')
        out[k.strip()] = v.strip()
    return out, default


def load_shots(path, work):
    """Normalise both shot formats -> (rows, players, fps).

    rows carry `frame` (contact frame), `side`, `who`, `contact` (3D or None) plus the whole
    original record, so config-driven consumers keep seeing their own field names.
    """
    p = resolve(work, path)
    D = json.load(open(p))
    players, fps = {}, None
    if isinstance(D, dict) and 'contacts' in D:                  # contacts.json (flight_bvp)
        players, fps = D.get('players') or {}, D.get('fps')
        raw, key_f, key_p, kind = D['contacts'], 'frame', 'point', 'contacts'
    else:                                                        # shots.json (flight_fit)
        raw = D['shots'] if isinstance(D, dict) and 'shots' in D else D
        fps = D.get('fps') if isinstance(D, dict) else None
        key_f, key_p, kind = 'tc', 'contact', 'shots'
    rows = []
    for i, s in enumerate(raw):
        r = dict(s)
        r['id'] = s.get('id', i + 1)
        r['frame'] = float(s[key_f] if key_f in s else s.get('frame', s.get('tc')))
        r['side'] = s.get('side') or '?'
        r['who'] = s.get('who')
        r['contact'] = s.get(key_p) or s.get('contact')
        rows.append(r)
    return rows, players, fps, kind, os.path.basename(p)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='work')
    ap.add_argument('--shots', default='shots.json',
                    help='shots.json (flight_fit) or contacts.json (flight_bvp) - detected')
    ap.add_argument('--pose', default='pose.json', help='pose file used when a shot has no --pose-map entry')
    ap.add_argument('--pose-map', default=None,
                    help="player -> pose file, e.g. 'N=pose_near.json,F=pose_far.json'")
    ap.add_argument('--hand', default='right',
                    help="racket hand: 'right' / 'left', or per player 'N=left,F=right'. "
                         "contacts.json players[*].hand wins over this")
    ap.add_argument('--facing', default='away',
                    help="body orientation towards the camera: 'away' (camera behind the "
                         "player, the historical assumption) / 'toward', or per player "
                         "'N=away,F=toward'. contacts.json players[*].facing wins over this")
    ap.add_argument('--calib', default='calib.json',
                    help='calibration for the metric quantities; without it the wrist speed '
                         'is reported in px/frame')
    ap.add_argument('--prep-window', type=int, default=20, help='frames before contact scanned for peak coil')
    ap.add_argument('--torso-source', default='auto',
                    choices=['auto', SRC_3D, SRC_PROXY], help='force the torso metric route')
    ap.add_argument('--torso-pct', type=float, default=92.0,
                    help='percentile of the shoulder/torso ratio taken as "square to camera"')
    ap.add_argument('--torso-window', type=int, nargs=2, default=(-12, 6), metavar=('A', 'B'),
                    help='frames around contact scanned for the peak closing angle')
    ap.add_argument('--rot3d-min-frac', type=float, default=0.5)
    ap.add_argument('--rot3d-min-std', type=float, default=1.0, help='deg; below this the 3D angle is degenerate')
    ap.add_argument('--rot3d-min-box-h', type=float, default=240.0, help='px; smaller figures use the proxy')
    ap.add_argument('--rot3d-max-jitter', type=float, default=3.0, help='deg per frame, median')
    ap.add_argument('--rot3d-min-conf', type=float, default=0.5)
    ap.add_argument('--speed-window', type=int, default=14, help='frames each side of contact for the wrist profile')
    ap.add_argument('--backswing-look', type=int, default=10,
                    help='frames before contact searched for the slowest wrist frame '
                         '(= end of the take-back)')
    ap.add_argument('--racket-offset-z', type=float, default=0.08,
                    help='metres the ball sits above the wrist at contact')
    ap.add_argument('--fps', type=float, default=None, help='default: contacts.json / meta.json')
    ap.add_argument('--side-tol', type=float, default=0.2,
                    help='|lateral offset| in shoulder widths below which FH/BH is not suggested')
    ap.add_argument('--out', default='technique.json')
    a = ap.parse_args()

    # ------------------------------------------------------------------ camera / fps
    cam = None
    cpath = resolve(a.work, a.calib)
    if os.path.exists(cpath):
        cam = Camera(a.work, cpath)
        print(cam.describe())
    else:
        print('no calibration at %s - metric quantities (contact height, m/s wrist speed, '
              'court position) are skipped and the wrist speed stays in px/frame' % cpath)

    shots, players, fps_in, kind, shots_name = load_shots(a.shots, a.work)
    from_contacts = (kind == 'contacts')
    meta_p = os.path.join(a.work, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    fps = a.fps or fps_in or meta.get('fps')
    print('%d shots from %s (%s format), fps %s' % (len(shots), shots_name, kind, fps))

    # ------------------------------------------------------------------ pose files per player
    pose_map, _ = parse_map(a.pose_map, None)
    hand_map, hand_def = parse_map(a.hand, 'right')
    face_map, face_def = parse_map(a.facing, 'away')

    def hand_of(who):
        h = (players.get(who) or {}).get('hand') or hand_map.get(who) or hand_def
        return 'left' if str(h).lower().startswith('l') else 'right'

    def facing_of(who):
        f = (players.get(who) or {}).get('facing') or face_map.get(who) or face_def
        return 'toward' if str(f).lower().startswith('t') else 'away'

    cache = {}

    def pose_of(who):
        f = pose_map.get(who, a.pose) if pose_map else a.pose
        if f not in cache:
            po = Pose(a.work, f)
            info = dict(path=po.path, n=len(po.frames), rmax=rot_scale(po, a.torso_pct),
                        box_h=po.box_h_median())
            sw = np.array([po.width(f_, LSH, RSH) for f_ in po.frames], float)
            hw = np.array([po.width(f_, LHIP, RHIP) for f_ in po.frames], float)
            sl = np.array([po.seg(f_, LSH, RSH) for f_ in po.frames], float)
            info['SW'], info['HW'], info['SL'] = (float(np.nanpercentile(x, 96)) for x in (sw, hw, sl))
            src, q3, sh3, hip3 = SRC_PROXY, None, {}, {}
            if cam is not None:
                ok, q3, sh3, hip3 = rot3d_quality(cam, po, a)
                src = SRC_3D if ok else SRC_PROXY
            if a.torso_source != 'auto':
                src = a.torso_source
            info.update(source=src, rot3d=q3, sh3=sh3, hip3=hip3, pose=po)
            print('pose %s: %d frames, bbox h %.0f px, shoulders %.1f px, hips %.1f px, '
                  'torso-ratio scale %.3f' % (po.path, len(po.frames), info['box_h'],
                                              info['SW'], info['HW'], info['rmax'] or float('nan')))
            if q3 is not None:
                print('  3D shoulder/hip route: %s (frames %.0f%%, spread %.1f deg, jitter %s '
                      'deg/frame, bbox %.0f px, kpt conf %.2f)'
                      % ('usable' if not q3['reasons'] else 'NOT usable', 100 * q3['frac'],
                         q3['std_deg'], q3['jitter_deg'], q3['box_h_px'], q3['kpt_conf_median']))
                for w in q3['reasons']:
                    print('    - %s' % w)
            print('  torso metrics source -> %s%s'
                  % (src, '' if a.torso_source == 'auto' else ' (forced by --torso-source)'))
            if src == SRC_PROXY and cam is None:
                print('    (no calibration, so the 3D route could not even be tried)')
            cache[f] = info
        return cache[f]

    # ------------------------------------------------------------------ per shot
    rows = []
    for s in shots:
        who = s.get('who')
        P = pose_of(who)
        po, SW, HW, SL = P['pose'], P['SW'], P['HW'], P['SL']
        hand, facing = hand_of(who), facing_of(who)
        fh_right = (hand == 'right') == (facing == 'away')
        rule = 'hand=%s,facing=%s -> FH on image %s' % (hand, facing, 'right' if fh_right else 'left')
        f0 = int(round(s['frame']))
        side = s.get('side') or '?'
        fmax = max(po.frames) if po.frames else f0

        def rot(f, a_, b_, wmax):
            w = po.width(f, a_, b_)
            return np.nan if np.isnan(w) else float(np.degrees(np.arccos(np.clip(w / wmax, 0, 1))))

        # ---- legacy image-width rotation block (unchanged formulas)
        hand_w = RWR if (side == 'FH') == (hand == 'right') else LWR
        cand = [(rot(f, LSH, RSH, SW), f) for f in range(f0 - a.prep_window, f0 - 2)]
        cand = [(v, f) for v, f in cand if not np.isnan(v)]
        coil, f_coil = max(cand) if cand else (np.nan, f0)
        hip = rot(f_coil, LHIP, RHIP, HW)
        r = dict(id=s['id'], side=side, tc=s.get('tc', s['frame']), frame=f0, who=who,
                 hand=hand, facing=facing, side_rule=rule, f_coil=f_coil,
                 pose_file=os.path.basename(po.path),
                 shoulder_rot_prep=None if np.isnan(coil) else round(coil, 1),
                 hip_rot_prep=None if np.isnan(hip) else round(hip, 1),
                 x_factor=None if (np.isnan(coil) or np.isnan(hip)) else round(coil - hip, 1),
                 coil_to_contact_frames=f0 - f_coil,
                 coil_to_contact_s=round((f0 - f_coil) / fps, 3) if fps else None,
                 shoulder_rot_contact=round(rot(f0, LSH, RSH, SW), 1),
                 shoulder_rot_finish=round(float(np.nanmean([rot(f, LSH, RSH, SW)
                                                             for f in range(f0 + 6, f0 + 12)])), 1),
                 elbow_angle_contact=round(po.angle(f0, RSH if hand_w == RWR else LSH,
                                                    REL if hand_w == RWR else LEL, hand_w), 1),
                 wrist_speed_px=round(float(np.nanmax([po.wrist_speed(f, hand_w)
                                                       for f in range(f0 - 2, f0 + 3)])), 1),
                 contact_h=s.get('contact_h'))
        if r['contact_h'] is not None:
            r['contact_h_source'] = 'shot_field'
        elif from_contacts and s.get('contact') and len(s['contact']) > 2:
            # contacts.json carries the labelled 3D contact point instead of a contact_h field
            r['contact_h'] = round(float(s['contact'][2]), 2)
            r['contact_h_source'] = 'contact_point'
        # the headline coaching number: how much the torso actually unwinds into the ball.
        # absolute rotation is scale-dependent, this difference is not.
        for k1, k2, name in (('shoulder_rot_prep', 'shoulder_rot_contact', 'unwind_to_contact'),
                             ('shoulder_rot_contact', 'shoulder_rot_finish', 'unwind_to_finish')):
            v1, v2 = r.get(k1), r.get(k2)
            r[name] = None if (v1 is None or v2 is None or np.isnan(v1) or np.isnan(v2)) else round(v1 - v2, 1)

        # ---- torso closing / unwinding, 3D route or proxy
        w0, w1 = a.torso_window
        r['torso_source'] = P['source']
        if P['source'] == SRC_3D:
            def closed(f):
                k = po.k(f)
                return None if k is None else line_azimuth_3d(cam, k, LSH, RSH)
            prof = [closed(f) for f in range(f0 + w0, f0 + w1 + 1)]
            prof = [v for v in prof if v is not None]
            c0 = closed(f0)
            k0 = po.k(f0)
            hip3 = None if k0 is None else line_azimuth_3d(cam, k0, LHIP, RHIP)
            r['hip_closed_deg'] = None if hip3 is None else round(hip3, 1)
            if c0 is not None and hip3 is not None:
                r['torso_separation_deg'] = round(c0 - hip3, 1)
        else:
            k0 = po.k(f0)
            c0 = proxy_deg(None if k0 is None else torso_ratio(k0), P['rmax'])
            prof = []
            for f in range(f0 + w0, f0 + w1 + 1):
                k = po.k(f)
                v = proxy_deg(None if k is None else torso_ratio(k), P['rmax'])
                if v is not None:
                    prof.append(v)
        r['torso_closed_deg'] = None if c0 is None else round(c0, 1)
        if prof:
            r['torso_closed_max_deg'] = round(max(prof), 1)
            # from the rounded values on purpose: this is the number that gets quoted, and it
            # must equal what a reader computes from the two printed angles
            r['torso_unwind_deg'] = None if c0 is None else \
                round(r['torso_closed_max_deg'] - r['torso_closed_deg'], 1)

        # ---- metric (metre) quantities from the racket wrist, needs a calibration
        wi = LWR if hand == 'left' else RWR
        k0 = po.k(f0)
        g = None if (cam is None or k0 is None) else ground_of(cam, k0)
        if g is not None:
            W = at_depth(cam, k0[wi, :2], g[1])
            LS, RS = at_depth(cam, k0[LSH, :2], g[1]), at_depth(cam, k0[RSH, :2], g[1])
            LH, RH = at_depth(cam, k0[LHIP, :2], g[1]), at_depth(cam, k0[RHIP, :2], g[1])
            r['contact_h_pose'] = round(float(W[2] + a.racket_offset_z), 2)
            r['lat_from_body_m'] = round(float(abs(W[0] - (LS[0] + RS[0]) / 2)), 2)
            r['hip_h'] = round(float((LH[2] + RH[2]) / 2), 2)
            r['ground'] = [round(float(x), 2) for x in g[:2]]
            r['behind_baseline'] = round(float(abs(g[1]) - cam.HALF), 2)
            if r.get('contact_h') is None and from_contacts:
                r['contact_h'] = r['contact_h_pose']
                r['contact_h_source'] = 'pose_wrist'
            # wrist speed profile: px/frame -> m/s with the metres-per-pixel scale at the
            # player's depth. Only the component parallel to the image plane is observable.
            win = a.speed_window
            m_per_px = max(abs(g[1] - cam.C[1]), 1e-3) / cam.F
            sp, hs = [], []
            for ff in range(f0 - win, f0 + win + 1):
                kk, k2 = po.k(ff), po.k(ff - 1)
                if kk is None or k2 is None:
                    sp.append(np.nan)
                    hs.append(np.nan)
                    continue
                sp.append(float(np.linalg.norm(kk[wi, :2] - k2[wi, :2])) * m_per_px * (fps or 30.0))
                hs.append(float(at_depth(cam, kk[wi, :2], g[1])[2]))
            sp, hs = np.array(sp), np.array(hs)
            if np.any(~np.isnan(sp[win - 3:win + 3])):
                r['wrist_speed_ms'] = round(float(np.nanmax(sp[win - 3:win + 3])), 1)
            look = min(a.backswing_look, win)
            pre = sp[win - look:win]
            if np.any(~np.isnan(pre)):
                r['backswing_to_contact_f'] = int(look - int(np.nanargmin(pre)))
            if np.any(~np.isnan(hs[win:win + 10])):
                r['followthrough_h'] = round(float(np.nanmax(hs[win:win + 10])), 2)
            if np.any(~np.isnan(hs[max(0, win - 10):win])):
                r['prep_h'] = round(float(np.nanmin(hs[max(0, win - 10):win])), 2)
        elif cam is None:
            r['wrist_speed_unit'] = 'px/frame'

        # ---- contact geometry and the FH/BH cross-check
        if s.get('contact') and cam is not None:
            uv, _ = cam.project([s['contact']])
            if k0 is not None:
                bc = (k0[LSH][0] + k0[RSH][0]) / 2
                r['contact_lateral_sw'] = round(float((uv[0][0] - bc) / SL), 2)
                r['contact_vs_shoulder_px'] = round(float(uv[0][1] - (k0[LSH][1] + k0[RSH][1]) / 2), 1)
                lat = r['contact_lateral_sw']
                r['side_suggest'] = ('FH' if (lat > 0) == fh_right else 'BH') \
                    if abs(lat) > a.side_tol else '?'
                if side in ('FH', 'BH') and r['side_suggest'] not in ('?', side):
                    r['side_conflict'] = True
            # contact position on court relative to the player's feet
            if k0 is not None and min(k0[LAN][2], k0[RAN][2]) > 0.3:
                ank = [[(k0[LAN][0] + k0[RAN][0]) / 2, (k0[LAN][1] + k0[RAN][1]) / 2]]
                gg = cam.img2plane(ank, 0.05)[0]
                r['contact_ahead_m'] = round(float(gg[1] - s['contact'][1]), 2)
        hy = [(f, (po.k(f)[LHIP][1] + po.k(f)[RHIP][1]) / 2) for f in range(f0 - 18, min(f0 + 10, fmax))
              if po.k(f) is not None and min(po.k(f)[LHIP][2], po.k(f)[RHIP][2]) > 0.4]
        if len(hy) > 6:
            ys = np.array([q[1] for q in hy])
            r['hip_y_range_px'] = round(float(ys.max() - ys.min()), 1)
        n1, n2 = po.k(f_coil), po.k(min(fmax, f0 + 8))
        if n1 is not None and n2 is not None and min(n1[NOSE][2], n2[NOSE][2]) > 0.2:
            r['head_move_px'] = round(float(np.hypot(n1[NOSE][0] - n2[NOSE][0],
                                                     n1[NOSE][1] - n2[NOSE][1])), 1)
        rows.append(r)

    json.dump(rows, open(os.path.join(a.work, a.out), 'w'), indent=1, ensure_ascii=False)

    # ------------------------------------------------------------------ report
    cols = ['id', 'who', 'side', 'side_sugg', 'torso_clo', 'torso_max', 'torso_unw',
            'unwind_to', 'x_factor', 'coil_to_co', 'wrist_ms', 'wrist_px', 'contact_h',
            'contact_hp', 'contact_la', 'bswing_f', 'src']
    keys = ['id', 'who', 'side', 'side_suggest', 'torso_closed_deg', 'torso_closed_max_deg',
            'torso_unwind_deg', 'unwind_to_contact', 'x_factor', 'coil_to_contact_frames',
            'wrist_speed_ms', 'wrist_speed_px', 'contact_h', 'contact_h_pose',
            'contact_lateral_sw', 'backswing_to_contact_f', 'torso_source']
    print(' '.join('%-10s' % c[:10] for c in cols))
    for r in rows:
        v = [r.get(k, '-') for k in keys]
        v[-1] = '3d' if v[-1] == SRC_3D else 'proxy'
        print(' '.join('%-10s' % str(x)[:10] for x in v))
    for grp in sorted({str(r['side']) for r in rows}):
        g = [r for r in rows if str(r['side']) == grp]

        def mean(key):
            v = [r[key] for r in g if r.get(key) is not None and not np.isnan(r[key])]
            return np.mean(v) if v else float('nan')
        print('%-6s n=%-2d mean: torso unwind %.1f deg (the number to quote), coil->contact '
              '%.1f frames, X-factor %.1f deg, contact height %.2f m, lateral %.2f m'
              % (grp, len(g), mean('torso_unwind_deg'), mean('coil_to_contact_frames'),
                 mean('x_factor'), mean('contact_h'), mean('lat_from_body_m')))
    bad = [r['id'] for r in rows if r.get('side_conflict')]
    if bad:
        print('WARNING contact geometry disagrees with the FH/BH label on shots %s '
              '(rule in use: %s) - re-check those frames, or the hand/facing of that player'
              % (bad, '; '.join(sorted({r['side_rule'] for r in rows}))))
    print('wrote %s' % os.path.join(a.work, a.out))


if __name__ == '__main__':
    main()
