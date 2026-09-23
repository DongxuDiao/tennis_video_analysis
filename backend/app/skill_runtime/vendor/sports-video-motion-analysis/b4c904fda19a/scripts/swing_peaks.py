"""Swing peaks from pose, and who actually hit the ball - matches audio impacts to players.

Why this exists
---------------
`audio_impacts.py` finds *when* something was struck, to a fraction of a frame, but it has
no idea *who* struck it. With one player on court that is fine. With two players rallying it
is the single biggest source of wrong labelling: an impulse at frame 458 belongs to the far
player, the next one at 491 to the near player, and if you swap them every contact point,
every speed and every FH/BH label downstream is wrong.

The pose side has the complementary information: the racket hand accelerates into contact and
decelerates right after, so the dominant-wrist speed has a sharp peak within a frame or two
of the real contact. Neither signal alone is reliable (a bounce, a footstep or a grunt also
makes an impulse; a big non-hitting swing also makes a wrist peak), but **their intersection
is**: audio peak + a wrist peak of exactly one player inside a small window = that player hit
the ball at that frame.

Outputs, per pose file and per frame:
  * `spd`  - racket-wrist speed. Normalised (shoulder widths / frame) so it survives the
             player walking towards or away from the camera; `spd_px` is the raw pixel speed
             and `spd_ms` an approximate m/s if `--calib` is given;
  * `lat`  - wrist offset from the shoulder mid line, in **shoulder widths** (+ = image
             right). Never use raw pixels: the same 40 px means different things at the two
             ends of the court;
  * `hgt`  - wrist height above the hip line in **torso lengths**;
  * `peaks`- detected swing peaks with the frame where the backswing started.

When NOT to trust it
  * frames where the pose tracker swapped the two players (check `box` continuity first) -
    the wrist "jumps" and fakes a peak;
  * `--hand auto` when the two wrists move similarly: it decides by median wrist activity and
    on the clip this was built for it got the near player right and the far player wrong. It
    prints the margin and warns - when warned, look at one contact frame and pass --hand;
  * serves and overheads: the wrist peak can precede the audio impulse by 3-5 frames, widen
    `--window`;
  * doubles or occluded players: a missing wrist yields NaN, those frames simply cannot vote.

  python3 scripts/swing_peaks.py --work work --pose work/pose_near.json work/pose_far.json \
      --hand auto --match-impacts work/impacts.json --window 4 --truth 34 55 85
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# COCO-17 keypoint indices, as produced by pose_extract.py
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12


def series(path, hand, kp_conf=0.3, wr_conf=0.2, min_sw=8.0, hand_margin=0.15):
    """Per-frame swing signals from a pose json ({frame: {box, conf, kpts}})."""
    P = json.load(open(path))
    fr = P.get('frames', P)
    fs = sorted(int(k) for k in fr if str(k).isdigit())
    if not fs:
        raise SystemExit('%s holds no frame keys' % path)
    n = max(fs) + 1
    wr = np.full((n, 2), np.nan)
    sw = np.full(n, np.nan)                      # shoulder width, px
    tl = np.full(n, np.nan)                      # torso length, px
    lat = np.full(n, np.nan)
    hgt = np.full(n, np.nan)
    anchor = np.full((n, 2), np.nan)             # ankle/hip mid, for the depth scale
    used = {'left': 0, 'right': 0}
    activity = {'left': [], 'right': []}
    if hand == 'auto':
        # decide by median wrist activity over the whole clip: the racket hand moves more
        for side, wi in (('left', L_WR), ('right', R_WR)):
            prev, acc = None, []
            for f in fs:
                k = np.array(fr[str(f)]['kpts'], float)
                w = k[wi]
                if w[2] < wr_conf:
                    prev = None
                    continue
                if prev is not None:
                    s = float(np.hypot(*(k[L_SH][:2] - k[R_SH][:2])))
                    acc.append(float(np.hypot(*(w[:2] - prev))) / max(s, min_sw))
                prev = w[:2]
            activity[side] = acc
        med = {s: (float(np.median(v)) if v else 0.0) for s, v in activity.items()}
        hand = 'left' if med['left'] > med['right'] else 'right'
        margin = abs(med['left'] - med['right']) / max(med['left'], med['right'], 1e-9)
        print('  --hand auto -> %s (median normalised wrist motion L=%.4f R=%.4f, margin %.0f%%)'
              % (hand, med['left'], med['right'], 100 * margin))
        if margin < hand_margin:
            # Both arms move in a two-handed sport; in long-range footage the pose noise on
            # the non-racket wrist is of the same order as the racket wrist's motion. On the
            # broadcast clip this was built for, `auto` got the near player right and the far
            # player wrong. A small margin means: go look at one contact frame and pass
            # --hand explicitly. Do not let a coin flip decide the FH/BH labels.
            print('  WARNING handedness margin is only %.0f%% - `auto` is a coin flip here. '
                  'Look at one contact frame and pass --hand left|right explicitly.'
                  % (100 * margin))
    wi = L_WR if hand == 'left' else R_WR
    for f in fs:
        k = np.array(fr[str(f)]['kpts'], float)
        ls, rs, lh, rh, w = k[L_SH], k[R_SH], k[L_HIP], k[R_HIP], k[wi]
        if min(ls[2], rs[2]) < kp_conf or w[2] < wr_conf:
            continue
        s = max(float(np.hypot(*(ls[:2] - rs[:2]))), min_sw)
        sh_mid = (ls[:2] + rs[:2]) / 2.0
        hip_mid = (lh[:2] + rh[:2]) / 2.0
        sw[f] = s
        tl[f] = max(float(np.hypot(*(sh_mid - hip_mid))), min_sw)
        wr[f] = w[:2]
        lat[f] = (w[0] - sh_mid[0]) / s
        hgt[f] = (hip_mid[1] - w[1]) / tl[f]
        anchor[f] = hip_mid
        used[hand] += 1
    spd = np.full(n, np.nan)                     # shoulder widths / frame
    spd_px = np.full(n, np.nan)
    for f in range(1, n - 1):
        if np.isnan(wr[f - 1, 0]) or np.isnan(wr[f + 1, 0]) or np.isnan(sw[f]):
            continue
        d = float(np.hypot(*(wr[f + 1] - wr[f - 1]))) / 2.0   # central difference
        spd_px[f] = d
        spd[f] = d / sw[f]
    return dict(hand=hand, n=n, spd=spd, spd_px=spd_px, lat=lat, hgt=hgt, sw=sw, tl=tl,
                wr=wr, anchor=anchor, n_used=used[hand])


def to_ms(S, cam, fps):
    """Rough pixel/frame -> m/s conversion.

    A single camera cannot measure the radial component of a wrist motion, so this is only
    the component parallel to the image plane: metres per pixel at the player's depth
    (depth / focal), times fps. Good enough to say "the racket hand reaches ~15 m/s", not
    good enough for a racket-head-speed claim.
    """
    out = np.full(S['n'], np.nan)
    for f in range(S['n']):
        if np.isnan(S['spd_px'][f]) or np.isnan(S['anchor'][f, 0]):
            continue
        g = cam.img2ground([S['anchor'][f]])[0]
        depth = float(np.linalg.norm(np.array([g[0], g[1], 0.9]) - cam.C))
        out[f] = S['spd_px'][f] * depth / cam.F * fps
    return out


def peaks(sig, thr, minsep, win=3):
    out = []
    n = len(sig)
    for f in range(2, n - 2):
        v = sig[f]
        if np.isnan(v) or v < thr:
            continue
        w = sig[max(0, f - win):f + win + 1]
        if v < np.nanmax(w) - 1e-9:
            continue
        if out and f - out[-1] < minsep:
            if v > sig[out[-1]]:
                out[-1] = f
            continue
        out.append(f)
    return out


def backswing_start(sig, pk, frac=0.35, look=14):
    """Frame where the forward swing started: last frame before the peak whose speed is
    below `frac` of the peak (i.e. the end of the take-back)."""
    v = sig[pk]
    for f in range(pk - 1, max(pk - look, 0) - 1, -1):
        if np.isnan(sig[f]):
            continue
        if sig[f] < frac * v:
            return f
    return None


def match_impacts(impacts, per_file, window, min_peak=None):
    """Cross audio impulses with the per-player swing peaks."""
    peaks_audio = impacts.get('peaks', impacts if isinstance(impacts, list) else [])
    rows = []
    for p in peaks_audio:
        fa = float(p['frame'] if isinstance(p, dict) else p)
        hits = []
        for name, S in per_file.items():
            cand = [(abs(f - fa), f) for f in S['peaks']
                    if abs(f - fa) <= window and (min_peak is None or S['spd'][f] >= min_peak)]
            if cand:
                d, f = min(cand)
                hits.append(dict(player=name, swing_frame=f, delta_frames=round(f - fa, 1),
                                 peak=round(float(S['spd'][f]), 3)))
        row = dict(audio_frame=round(fa, 1), score=p.get('score') if isinstance(p, dict) else None,
                   best_player=None, delta_frames=None, ambiguous=False,
                   candidates=sorted(hits, key=lambda h: abs(h['delta_frames'])))
        if hits:
            b = min(hits, key=lambda h: (abs(h['delta_frames']), -h['peak']))
            row.update(best_player=b['player'], delta_frames=b['delta_frames'],
                       swing_frame=b['swing_frame'],
                       ambiguous=len(hits) > 1)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--pose', nargs='+', required=True,
                    help='one or more pose json files (one per player), e.g. '
                         '$WORK/pose_near.json $WORK/pose_far.json')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='names for the pose files (default: file basename without pose_/.json)')
    ap.add_argument('--hand', nargs='+', default=['auto'],
                    help='left|right|auto per pose file (one value = same for all)')
    ap.add_argument('--hand-margin', type=float, default=0.15,
                    help='relative margin below which --hand auto warns that it is guessing')
    ap.add_argument('--thr', type=float, default=0.55,
                    help='peak threshold in shoulder widths per frame')
    ap.add_argument('--minsep', type=int, default=8, help='minimum frames between peaks')
    ap.add_argument('--out', default=None,
                    help='output json; default $WORK/swing_<basename>.json per pose file')
    ap.add_argument('--calib', default=None,
                    help='calib.json - adds an approximate m/s wrist speed')
    ap.add_argument('--fps', type=float, default=None, help='default: from meta.json')
    ap.add_argument('--match-impacts', default=None,
                    help='impacts.json from audio_impacts.py - cross audio impulses with the '
                         'swing peaks of every pose file')
    ap.add_argument('--window', type=float, default=4.0,
                    help='matching window in frames (+/-)')
    ap.add_argument('--match-out', default=None, help='default $WORK/swing_match.json')
    ap.add_argument('--truth', nargs='+', default=None,
                    help='known contact frames; prints recall of the matching')
    ap.add_argument('--truth-tol', type=float, default=4.0,
                    help='frames a matched audio impulse may differ from a truth contact; '
                         'audio_impacts.py peaks tend to sit 2-3 frames after the visual '
                         'contact, so do not set this below 3')
    a = ap.parse_args()

    meta_p = os.path.join(a.work, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    fps = a.fps or meta.get('fps') or 30.0
    cam = None
    if a.calib:
        from camera import Camera
        cam = Camera(a.work, a.calib if os.path.isabs(a.calib) or os.path.exists(a.calib)
                     else os.path.join(a.work, a.calib))

    labels = a.labels or [os.path.basename(p).replace('pose_', '').replace('pose', 'main')
                          .replace('.json', '') or 'main' for p in a.pose]
    if len(labels) != len(a.pose):
        raise SystemExit('--labels must have one entry per --pose file')
    hands = a.hand if len(a.hand) == len(a.pose) else [a.hand[0]] * len(a.pose)

    per_file = {}
    for path, label, hand in zip(a.pose, labels, hands):
        if not os.path.exists(path):
            raise SystemExit('missing pose file %s' % path)
        print('=== %s  (label %s, hand %s)' % (path, label, hand))
        S = series(path, hand, hand_margin=a.hand_margin)
        pk = peaks(S['spd'], a.thr, a.minsep)
        ms = to_ms(S, cam, fps) if cam is not None else None
        rows = []
        for f in pk:
            bs = backswing_start(S['spd'], f)
            rows.append(dict(frame=int(f), peak=round(float(S['spd'][f]), 3),
                             peak_px=round(float(S['spd_px'][f]), 2),
                             peak_ms=None if ms is None or np.isnan(ms[f]) else round(float(ms[f]), 1),
                             backswing_start=None if bs is None else int(bs),
                             lat=round(float(S['lat'][f]), 3), hgt=round(float(S['hgt'][f]), 3)))
        print('  %d frames with a usable wrist, %d swing peaks (thr %.2f, minsep %d)'
              % (S['n_used'], len(pk), a.thr, a.minsep))
        print('  %-6s %-7s %-8s %-8s %-7s %-7s %s'
              % ('frame', 'peak', 'peak_px', 'peak_m/s', 'lat_sw', 'hgt_tl', 'backswing'))
        for r in rows:
            print('  %-6d %-7.2f %-8.1f %-8s %-+7.2f %-+7.2f %s'
                  % (r['frame'], r['peak'], r['peak_px'],
                     '-' if r['peak_ms'] is None else '%.1f' % r['peak_ms'],
                     r['lat'], r['hgt'], r['backswing_start']))
        out = a.out if (a.out and len(a.pose) == 1) else \
            os.path.join(a.work, 'swing_%s.json' % label)
        rec = dict(pose=path, label=label, hand=S['hand'], fps=fps, n=S['n'],
                   thr=a.thr, minsep=a.minsep, n_used=S['n_used'], peaks=rows,
                   spd=[None if np.isnan(v) else round(float(v), 4) for v in S['spd']],
                   spd_px=[None if np.isnan(v) else round(float(v), 2) for v in S['spd_px']],
                   lat=[None if np.isnan(v) else round(float(v), 4) for v in S['lat']],
                   hgt=[None if np.isnan(v) else round(float(v), 4) for v in S['hgt']])
        if ms is not None:
            rec['spd_ms'] = [None if np.isnan(v) else round(float(v), 2) for v in ms]
        json.dump(rec, open(out, 'w'), indent=1, ensure_ascii=False)
        print('  wrote %s' % out)
        per_file[label] = dict(S, peaks=pk)

    if not a.match_impacts:
        return 0

    # ------------------------------------------------------------ audio x pose
    if not os.path.exists(a.match_impacts):
        raise SystemExit('missing %s' % a.match_impacts)
    impacts = json.load(open(a.match_impacts))
    rows = match_impacts(impacts, per_file, a.window)
    print('\n=== audio impulses x swing peaks (window +/-%.1f frames) ===' % a.window)
    print('%-12s %-8s %-10s %-6s %s' % ('audio_frame', 'player', 'delta', 'ambig', 'candidates'))
    for r in rows:
        print('%-12.1f %-8s %-10s %-6s %s'
              % (r['audio_frame'], r['best_player'] or '-',
                 '-' if r['delta_frames'] is None else '%+.1f' % r['delta_frames'],
                 'YES' if r['ambiguous'] else '',
                 ', '.join('%s@%d(%+.1f)' % (h['player'], h['swing_frame'], h['delta_frames'])
                           for h in r['candidates'])))
    matched = [r for r in rows if r['best_player']]
    amb = [r for r in matched if r['ambiguous']]
    print('%d/%d audio impulses attributed to a player, %d ambiguous (both players swung '
          'inside the window - look at those frames before trusting the label)'
          % (len(matched), len(rows), len(amb)))
    if len(matched) < 0.6 * max(len(rows), 1):
        print('hint: few impulses could be attributed. Many audio peaks are bounces and '
              'footsteps and *should* stay unmatched, but if real contacts are missing, '
              'lower --thr (0.40-0.45) and widen --window (5-6): on the clip this was built '
              'for that moved the contact recall from 57% to 86%.')
    summary = dict(window=a.window, n_audio=len(rows), n_matched=len(matched),
                   n_ambiguous=len(amb),
                   per_player={k: sum(1 for r in matched if r['best_player'] == k)
                               for k in per_file})
    if a.truth:
        # accept both "--truth 34 55 85" and "--truth 34,55,85"
        a.truth = [float(t) for chunk in a.truth for t in str(chunk).split(',') if t.strip()]
        hit, miss = [], []
        for t in a.truth:
            near = [r for r in matched if abs(r['audio_frame'] - t) <= a.truth_tol]
            (hit if near else miss).append(t)
        summary['truth'] = dict(n=len(a.truth), recalled=len(hit),
                                recall=round(len(hit) / max(len(a.truth), 1), 3),
                                missed=miss, tol=a.truth_tol)
        print('truth check: %d/%d known contacts have an attributed audio impulse within '
              '%.0f frames -> recall %.0f%%   missed %s'
              % (len(hit), len(a.truth), a.truth_tol,
                 100 * len(hit) / max(len(a.truth), 1), miss or 'none'))
    mout = a.match_out or os.path.join(a.work, 'swing_match.json')
    json.dump(dict(summary=summary, matches=rows), open(mout, 'w'), indent=1, ensure_ascii=False)
    print('wrote %s' % mout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
