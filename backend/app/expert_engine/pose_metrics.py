"""Step 7 - pose-derived technique metrics per shot -> technique.json

With one camera behind the player, absolute joint angles in 3D are not recoverable, but
*relative* quantities are, and those are what coaching feedback is built on:

  torso rotation   the projected shoulder / hip line shortens as the player turns away, so
                   rot = arccos(width / width_when_square) - noisy in absolute terms but
                   reliable when comparing preparation vs contact vs finish, or forehand
                   vs backhand within the same clip
  coil timing      frames between peak shoulder turn and contact; too short = arm-only swing
  X-factor         shoulder rotation minus hip rotation at peak coil (torso separation)
  contact geometry contact height (m, from the 3D fit) and lateral offset from the body
                   centre expressed in shoulder widths, which is scale-free
  stability        head displacement and hip vertical travel through the swing

Report differences and trends, not absolute degrees, and say so in the report.

  python3 scripts/pose_metrics.py --work work --shots shots.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera  # noqa: E402

NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16


class Pose:
    def __init__(self, work, file='pose.json'):
        self.P = json.load(open(os.path.join(work, file)))
        self.frames = sorted(int(k) for k in self.P)

    def k(self, f):
        d = self.P.get(str(int(f)))
        return np.array(d['kpts'], float) if d else None

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work')
    ap.add_argument('--shots', default='shots.json')
    ap.add_argument('--hand', default='right', choices=['right', 'left'])
    ap.add_argument('--prep-window', type=int, default=20, help='frames before contact scanned for peak coil')
    ap.add_argument('--out', default='technique.json')
    a = ap.parse_args()

    cam = Camera(a.work)
    po = Pose(a.work)
    shots = json.load(open(os.path.join(a.work, a.shots)))
    if not po.frames:
        json.dump([], open(os.path.join(a.work, a.out), 'w'))
        return
    fmax = max(po.frames)

    sw = np.array([po.width(f, LSH, RSH) for f in po.frames], float)
    hw = np.array([po.width(f, LHIP, RHIP) for f in po.frames], float)
    sl = np.array([po.seg(f, LSH, RSH) for f in po.frames], float)
    SW, HW, SL = (np.nanpercentile(x, 96) for x in (sw, hw, sl))
    print('reference widths: shoulders %.1f px, hips %.1f px, shoulder length %.1f px' % (SW, HW, SL))

    def rot(f, a_, b_, wmax):
        w = po.width(f, a_, b_)
        return np.nan if np.isnan(w) else float(np.degrees(np.arccos(np.clip(w / wmax, 0, 1))))

    rows = []
    for s in shots:
        f0 = int(round(s['tc']))
        side = s.get('side') or '?'
        hand_w = RWR if (side == 'FH') == (a.hand == 'right') else LWR
        cand = [(rot(f, LSH, RSH, SW), f) for f in range(f0 - a.prep_window, f0 - 2)]
        cand = [(v, f) for v, f in cand if not np.isnan(v)]
        coil, f_coil = max(cand) if cand else (np.nan, f0)
        hip = rot(f_coil, LHIP, RHIP, HW)
        r = dict(id=s['id'], side=side, tc=s['tc'], f_coil=f_coil,
                 shoulder_rot_prep=None if np.isnan(coil) else round(coil, 1),
                 hip_rot_prep=None if np.isnan(hip) else round(hip, 1),
                 x_factor=None if (np.isnan(coil) or np.isnan(hip)) else round(coil - hip, 1),
                 coil_to_contact_frames=f0 - f_coil,
                 coil_to_contact_s=round((f0 - f_coil) / (json.load(open(os.path.join(a.work, 'meta.json')))['fps']), 3)
                 if os.path.exists(os.path.join(a.work, 'meta.json')) else None,
                 shoulder_rot_contact=round(rot(f0, LSH, RSH, SW), 1),
                 shoulder_rot_finish=round(float(np.nanmean([rot(f, LSH, RSH, SW)
                                                             for f in range(f0 + 6, f0 + 12)])), 1),
                 elbow_angle_contact=round(po.angle(f0, RSH if hand_w == RWR else LSH,
                                                    REL if hand_w == RWR else LEL, hand_w), 1),
                 wrist_speed_px=round(float(np.nanmax([po.wrist_speed(f, hand_w)
                                                       for f in range(f0 - 2, f0 + 3)])), 1),
                 contact_h=s.get('contact_h'))
        # the headline coaching number: how much the torso actually unwinds into the ball.
        # absolute rotation is scale-dependent, this difference is not.
        for k1, k2, name in (('shoulder_rot_prep', 'shoulder_rot_contact', 'unwind_to_contact'),
                             ('shoulder_rot_contact', 'shoulder_rot_finish', 'unwind_to_finish')):
            v1, v2 = r.get(k1), r.get(k2)
            r[name] = None if (v1 is None or v2 is None or np.isnan(v1) or np.isnan(v2)) else round(v1 - v2, 1)
        if s.get('contact'):
            uv, _ = cam.project([s['contact']])
            k = po.k(f0)
            if k is not None:
                bc = (k[LSH][0] + k[RSH][0]) / 2
                r['contact_lateral_sw'] = round(float((uv[0][0] - bc) / SL), 2)
                r['contact_vs_shoulder_px'] = round(float(uv[0][1] - (k[LSH][1] + k[RSH][1]) / 2), 1)
                lat = r['contact_lateral_sw']
                fh_right = (a.hand == 'right')
                r['side_suggest'] = ('FH' if (lat > 0) == fh_right else 'BH') if abs(lat) > 0.2 else '?'
                if side in ('FH', 'BH') and r['side_suggest'] not in ('?', side):
                    r['side_conflict'] = True
            # contact position on court relative to the player's feet
            if k is not None and min(k[LAN][2], k[RAN][2]) > 0.3:
                ank = [[(k[LAN][0] + k[RAN][0]) / 2, (k[LAN][1] + k[RAN][1]) / 2]]
                g = cam.img2plane(ank, 0.05)[0]
                r['contact_ahead_m'] = round(float(g[1] - s['contact'][1]), 2)
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
    cols = ['id', 'side', 'side_suggest', 'shoulder_rot_prep', 'shoulder_rot_contact',
            'unwind_to_contact', 'unwind_to_finish', 'x_factor', 'coil_to_contact_frames',
            'elbow_angle_contact', 'wrist_speed_px', 'contact_h', 'contact_lateral_sw',
            'hip_y_range_px', 'head_move_px']
    print(' '.join('%-10s' % c[:10] for c in cols))
    for r in rows:
        print(' '.join('%-10s' % str(r.get(c, '-'))[:10] for c in cols))
    for grp in ('FH', 'BH'):
        g = [r for r in rows if r['side'] == grp]
        if g:
            def mean(key):
                v = [r[key] for r in g if r.get(key) is not None and not np.isnan(r[key])]
                return np.mean(v) if v else float('nan')
            print('%s mean: unwind prep->contact %.1f deg (the number to quote), '
                  'coil->contact %.1f frames, X-factor %.1f deg, contact height %.2f m, '
                  'lateral %.2f shoulder-widths'
                  % (grp, mean('unwind_to_contact'), mean('coil_to_contact_frames'),
                     mean('x_factor'), mean('contact_h'), mean('contact_lateral_sw')))
    bad = [r['id'] for r in rows if r.get('side_conflict')]
    if bad:
        print('WARNING contact geometry disagrees with the FH/BH label on shots %s - '
              're-check those frames' % bad)


if __name__ == '__main__':
    main()
