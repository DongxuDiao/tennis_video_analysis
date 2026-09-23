"""Step 8 - render the annotated video (and mux the original sound back in).

Reads the artefacts produced by the earlier steps and draws, per frame: court + net
wireframe from the calibration, every athlete's skeleton with the current swing phase, the
verified ball detections with a fading trail (or the physically reconstructed flight), a
contact flash, a per-shot info card, a top-down minimap with the landing spin band, a
running tally and a shot timeline.

Iterate on a handful of frames before rendering everything - a full clip costs minutes and
layout collisions are only visible on real frames:

  python3 scripts/render_annotated.py --work work --frames 60 72 --out check.mp4
  python3 scripts/render_annotated.py --work work --out annotated.mp4

Panel text and colours come from --config (see references/render-config.md for the keys),
so the same renderer serves other sports and languages without touching the code.

Stabilised (panning / zooming) clips
------------------------------------
When the calibration was fitted on a stabilised canvas (stabilize.py), the geometry is only
valid in *canvas* coordinates - drawing it straight onto the original frame puts the court
lines metres away from the court. With

  --stabilize $WORK/stabilize.json --src-frames work/frames

every geometric overlay (court, net, ball, skeletons, contact flash) is drawn into a
transparent canvas-sized layer, warped back into each original frame with the inverse of
that frame's homography, and alpha-composited onto the untouched original frame, so the
video keeps its own sharpness. Without --stabilize the behaviour is exactly as before.

Two traps that cost hours:
  * a projected point behind the camera, or thousands of pixels off-canvas, produces a line
    that slashes across the whole frame. Everything goes through `safe_poly()`, which drops
    segments with a non-positive depth or an endpoint outside the canvas margin;
  * on a portrait clip (H > W) there is no room for a side panel. `--panel auto` puts the
    panel *below* the video for portrait input and to the right for landscape.

Two players
-----------
  --pose $W/pose_near.json $W/pose_far.json --pose-who N F \
      --pose-label "近端 · 纳达尔=255,190,90" "远端 · 费德勒=170,255,170"

Each player gets one skeleton colour, a name tag, a dot on the minimap (near side at the
BOTTOM of the minimap - the direction was wrong once and nobody noticed for a day) and its
own column in the tally. With a single --pose file the historical per-limb colouring is kept.

Ball flights from flight_bvp.py
-------------------------------
  --flights $WORK/flights_final.json

replays each flight by integrating its own contact point, V0 **and its own rpm** (a single
global spin looks right for one shot and wrong for the rest) and states in the panel that
the path is a physical reconstruction, not a per-frame detection (`ball_source_note`).
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera  # noqa: E402

EDGES = [(5, 7, 'L'), (7, 9, 'L'), (6, 8, 'R'), (8, 10, 'R'), (5, 6, 'T'), (5, 11, 'T'),
         (6, 12, 'T'), (11, 12, 'T'), (11, 13, 'G'), (13, 15, 'G'), (12, 14, 'G'), (14, 16, 'G'),
         (0, 5, 'T'), (0, 6, 'T')]
LIMB = {'L': (255, 210, 60), 'R': (60, 140, 255), 'T': (120, 255, 120), 'G': (255, 120, 200)}
CY, YE, RD, GR = (255, 220, 90), (60, 230, 255), (60, 60, 255), (90, 230, 120)
RESULT_COL = {'OUT': (255, 120, 120), 'NET': (255, 175, 80), 'IN': (130, 240, 160), 'UNK': (200, 200, 200)}
PALETTE = [(255, 190, 90), (170, 255, 170), (120, 200, 255), (210, 160, 255)]
DEFAULT = dict(
    title='动作 / 球路逐帧分析',
    # [name, start_ref, start_offset, end_ref, end_offset, BGR] with ref in {coil, contact}
    phases=[['准备 / 迎球', 'coil', -12, 'coil', 0, [150, 150, 150]],
            ['引拍 / 转体', 'coil', 0, 'contact', -3, [0, 200, 255]],
            ['击球瞬间', 'contact', -3, 'contact', 2, [60, 60, 255]],
            ['随挥 / 收拍', 'contact', 2, 'contact', 16, [90, 230, 120]]],
    idle_phase='移动 / 还原',
    panel_fields=[['出球速度', 'kmh', '%s km/h'], ['速度区间', '_range', '%s'],
                  ['出球仰角', 'elev', '%.1f°'], ['过网余量', 'net_clear', '%+.2f m'],
                  ['击球点高度', 'contact_h', '%.2f m'], ['来球速度', 'feed_kmh', '%.0f km/h']],
    tally=[['正手', 'side', 'FH'], ['反手', 'side', 'BH'], ['出界', 'result', 'OUT'],
           ['下网', 'result', 'NET'], ['界内', 'result', 'IN']],
    legend=['青色=来球  黄色=出球  虚线=模型预测', '×=落点推算（不同上旋）  红=出界 绿=界内'],
    side_names={'FH': '正手', 'BH': '反手', 'SERVE': '发球'},
    result_names={'IN': '界内', 'OUT': '出界', 'NET': '下网', 'UNK': '未判定'},
    unmeasurable='无法测量',
    # --- keys used by the bottom (portrait) panel and by the multi-player overlays
    waiting='等待下一拍…',
    shot_fmt='第 %s 拍 · %s%s',
    stat_title='累计统计',
    minimap_title='俯视图',
    half_labels={'near': '近端半场', 'far': '远端半场'},
    landing_title='落点判定',
    landing_fmt='x=%+.1f m   y=%+.1f m',
    verdict_fmt='判定：%s',
    speed_stat=['平均球速 %.0f km/h', '最快球速 %.0f km/h'],
    ball_source_note='球路：接触点物理重建（非逐帧检测）',
    timeline_note='刻度 = 每次击球      白线 = 当前帧',
    player_names={},
    player_tally=[['正手', 'side', 'FH'], ['反手', 'side', 'BH']],
)


def resolve(work, name):
    """A bare file name always means "inside --work".

    Resolving it against the current directory first looked harmless until a stray pose.json
    in the shell's cwd silently replaced the one in the work dir and the skeleton landed in
    the corner of the frame.
    """
    return name if os.sep in str(name) else os.path.join(work, name)


def load_cfg(path):
    cfg = dict(DEFAULT)
    if path:
        cfg.update(json.load(open(path)))
    return cfg


# --------------------------------------------------------------------------- safe drawing
def safe_poly(img, uv, col, w=2, depth=None, min_depth=0.3, margin=4000):
    """Draw a polyline, dropping every segment that cannot be trusted.

    A point behind the camera (depth <= 0) or far outside the canvas comes back from the
    projection as a huge or sign-flipped coordinate, and cv2.line happily draws a streak
    right across the picture - the classic "dirty line" that ruined a whole render pass.
    Clip per segment instead of per polyline, so a court line that leaves the frame keeps
    the part that is really visible.
    """
    uv = np.asarray(uv, float).reshape(-1, 2)
    H, W = img.shape[:2]
    ok = np.isfinite(uv).all(1) & (uv[:, 0] > -margin) & (uv[:, 0] < W + margin) \
        & (uv[:, 1] > -margin) & (uv[:, 1] < H + margin)
    if depth is not None:
        ok &= np.asarray(depth, float) > min_depth
    for i in range(len(uv) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, (int(round(uv[i][0])), int(round(uv[i][1]))),
                     (int(round(uv[i + 1][0])), int(round(uv[i + 1][1]))), col, w, cv2.LINE_AA)


def dense(a, b, n=40):
    """Sample a straight world line so it can be clipped (and curved by the lens) properly."""
    a, b = np.array(a, float), np.array(b, float)
    t = np.linspace(0, 1, n)[:, None]
    return a[None, :] + (b - a)[None, :] * t


# --------------------------------------------------------------------------- stabilisation
class Stab:
    """Per-frame homography canvas <-> original frame, from stabilize.py."""

    def __init__(self, path, meta):
        d = json.load(open(path))
        self.H = [np.array(h, float) for h in d['H']]
        off = d.get('offset') or meta.get('canvas_offset') or [0, 0]
        self.T = np.array([[1, 0, -off[0]], [0, 1, -off[1]], [0, 0, 1]], float)
        self.src_w, self.src_h = [int(v) for v in (d.get('source_size')
                                                   or meta.get('source_size') or [0, 0])]

    def canvas2orig(self, f):
        """Matrix mapping canvas pixels into the original frame f."""
        return np.linalg.inv(self.H[min(f, len(self.H) - 1)]) @ self.T

    def compose(self, frame, overlay, f, alpha=0.75):
        M = self.canvas2orig(f)
        w = cv2.warpPerspective(overlay, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        m = w.sum(2) > 12
        out = frame.copy()
        out[m] = ((1 - alpha) * out[m] + alpha * w[m]).astype(np.uint8)
        return out


# --------------------------------------------------------------------------- shot / flight IO
def load_shots(path):
    """shots.json (flight_fit) or contacts.json (flight_bvp) -> one list with `tc`."""
    D = json.load(open(path))
    if isinstance(D, dict) and 'contacts' in D:
        raw, kf, kp = D['contacts'], 'frame', 'point'
    else:
        raw = D['shots'] if isinstance(D, dict) and 'shots' in D else D
        kf, kp = 'tc', 'contact'
    out = []
    for i, s in enumerate(raw):
        r = dict(s)
        r['id'] = s.get('id', i + 1)
        r['tc'] = float(s[kf] if kf in s else s.get('frame', s.get('tc')))
        r['contact'] = s.get(kp) or s.get('contact')
        if kp == 'point' and r.get('contact') and r.get('contact_h') is None \
                and len(r['contact']) > 2:
            # contacts.json carries no derived fields; shots.json keeps whatever flight_fit wrote
            r['contact_h'] = round(float(r['contact'][2]), 2)
        out.append(r)
    return out


def merge_flights(shots, path, fps, sim_dt=1 / 600.):
    """Attach the flight_bvp result to its shot and replay the trajectory frame by frame.

    Each flight is integrated with **its own** rpm: the spin decides how much the ball dips,
    so one global value draws a path that visibly misses the bounce on most shots.
    """
    import flight_bvp
    D = json.load(open(path))
    flights = D['flights'] if isinstance(D, dict) else D
    ph = flight_bvp.Physics(sim_dt=sim_dt)
    ball, bounces = {}, []
    by_id = {f.get('id'): f for f in flights}
    for fl in flights:
        A, V = np.array(fl['contact'], float), np.array(fl['V0'], float)
        dt = float(fl['dt'])
        ts, ps, bo = ph.fly(A, V, dt * 1.15, float(fl.get('rpm') or 0.0))
        fa, fb = float(fl['fa']), float(fl['fb'])
        for f in range(int(round(fa)), int(round(fb)) + 1):
            t = (f - fa) / fps
            ball[f] = ps[int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))]
        if bo:
            bounces.append(dict(f=fa + bo[0][0] * fps, p=[float(bo[0][1][0]), float(bo[0][1][1])],
                                inside=bool(fl.get('bounce_in', True)), id=fl.get('id')))
        elif fl.get('bounce') and fl.get('bounce_frame'):
            bounces.append(dict(f=float(fl['bounce_frame']), p=list(fl['bounce'][:2]),
                                inside=bool(fl.get('bounce_in', True)), id=fl.get('id')))
    # panel fields keep their historical names, so a --config written for flight_fit.py works
    alias = dict(kmh='v0_kmh', kmh_lo='v0_lo', kmh_hi='v0_hi', elev='angle_deg',
                 net_clear='net_clear', apex='apex_m', f_land='bounce_frame', land='bounce')
    for s in shots:
        fl = by_id.get(s['id'])
        if not fl:
            continue
        for k, src in alias.items():
            if s.get(k) is None and fl.get(src) is not None:
                s[k] = fl[src]
        for k in ('result', 'result_cn', 'conf', 'rpm', 'v0_kmh', 'bounce', 'bounce_in',
                  'flip_reason', 'dt'):
            if fl.get(k) is not None:
                s.setdefault(k, fl[k])
        if s.get('result') is None and fl.get('bounce_in') is not None:
            s['result'] = 'IN' if fl['bounce_in'] else 'OUT'
        s['flight'] = fl
    return ball, bounces


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='work')
    ap.add_argument('--shots', default='shots.json',
                    help='shots.json (flight_fit) or contacts.json (flight_bvp)')
    ap.add_argument('--input', default='shots_input.json', help='source of the verified ball observations')
    ap.add_argument('--flights', default=None,
                    help='flights.json from flight_bvp.py - the ball path is then physically '
                         'reconstructed per shot instead of taken from detections')
    ap.add_argument('--technique', default='technique.json')
    ap.add_argument('--pose', nargs='+', default=['pose.json'],
                    help='one pose json per player (canvas coordinates when --stabilize)')
    ap.add_argument('--pose-label', nargs='+', default=None,
                    help='display name per --pose file, optionally with a BGR colour: '
                         '"近端 · 纳达尔=255,190,90"')
    ap.add_argument('--pose-who', nargs='+', default=None,
                    help="player key per --pose file, matching `who` in the shots (e.g. N F)")
    ap.add_argument('--stabilize', default=None,
                    help='stabilize.json - draw the geometry on the stabilised canvas and '
                         'warp it back onto the original frames')
    ap.add_argument('--src-frames', default=None,
                    help='original frames dir used with --stabilize (default $WORK/frames)')
    ap.add_argument('--overlay-alpha', type=float, default=0.75,
                    help='opacity of the warped overlay over the original frame')
    ap.add_argument('--config', default=None)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--panel', default='auto',
                    help="panel placement: auto (bottom for portrait, right for landscape) | "
                         "right | bottom. A plain number is the historical panel size in px")
    ap.add_argument('--panel-size', type=int, default=300,
                    help='panel width (right) or height (bottom), px')
    ap.add_argument('--frames', type=int, nargs=2, default=None, help='render only [a,b) - use while iterating')
    ap.add_argument('--font', default='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    ap.add_argument('--fps', type=float, default=0)
    ap.add_argument('--quality', type=int, default=7)
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--out', default='annotated.mp4')
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    cam = Camera(a.work)
    meta = json.load(open(os.path.join(a.work, 'meta.json')))
    fps = a.fps or meta['fps']
    SC = a.scale

    # ---------------------------------------------------------------- geometry / layout
    st = Stab(resolve(a.work, a.stabilize), meta) if a.stabilize else None
    if st is not None:
        # geometry lives on the canvas (= the calibrated image), pictures come from the source
        GW, GH = cam.W, cam.H                       # canvas, where every overlay is drawn
        SW0 = st.src_w or int(meta.get('source_size', [cam.W, cam.H])[0])
        SH0 = st.src_h or int(meta.get('source_size', [cam.W, cam.H])[1])
        src_dir = a.src_frames or os.path.join(a.work, 'frames')
        SCG = 1.0                                   # overlays are drawn at canvas scale
    else:
        GW, GH = cam.W, cam.H
        SW0, SH0 = cam.W, cam.H
        src_dir = a.src_frames or os.path.join(a.work, 'frames')
        SCG = SC                                    # ... straight onto the up-scaled frame
    W, H = int(SW0 * SC), int(SH0 * SC)             # video area of the output
    panel = str(a.panel).strip().lower()
    PW = a.panel_size
    if panel not in ('auto', 'right', 'bottom'):
        PW = int(float(panel))                      # historical `--panel 300`
        panel = 'auto'
    if panel == 'auto':
        panel = 'bottom' if H > W else 'right'
    # snap the canvas to a multiple of 16 so the encoder does not silently rescale it
    if panel == 'bottom':
        OW, OH = int(np.ceil(W / 16) * 16), int(np.ceil((H + PW) / 16) * 16)
        PX0, PY0 = 0, H                             # panel origin
        PWID, PHGT = OW, OH - H
    else:
        OW, OH = int(np.ceil((W + PW) / 16) * 16), int(np.ceil(H / 16) * 16)
        PX0, PY0 = W, 0
        PWID, PHGT = OW - W, OH
    print('layout %s panel %dx%d, video %dx%d -> output %dx%d%s'
          % (panel, PWID, PHGT, W, H, OW, OH,
             ', geometry on a %dx%d stabilised canvas' % (GW, GH) if st is not None else ''))

    shots = load_shots(resolve(a.work, a.shots))
    tech = {}
    tp = resolve(a.work, a.technique)
    if os.path.exists(tp):
        tech = {r['id']: r for r in json.load(open(tp))}

    # ---------------------------------------------------------------- players
    poses, labels, whos, cols = [], [], [], []
    for i, p in enumerate(a.pose):
        path = resolve(a.work, p)
        P = json.load(open(path))
        poses.append(P.get('frames', P))
        base = os.path.basename(path).replace('pose_', '').replace('pose', 'main').replace('.json', '')
        who = a.pose_who[i] if a.pose_who and i < len(a.pose_who) else base
        lab, col = cfg['player_names'].get(who, base), PALETTE[i % len(PALETTE)]
        if a.pose_label and i < len(a.pose_label):
            spec = a.pose_label[i]
            if '=' in spec:
                lab, _, c = spec.partition('=')
                try:
                    col = tuple(int(v) for v in c.split(',')[:3])
                except ValueError:
                    lab = spec
            else:
                lab = spec
        whos.append(who)
        labels.append(lab)
        cols.append(col)
    multi = len(poses) > 1
    if not multi and not a.pose_label:
        cols[0] = (80, 160, 255)       # historical single-player dot colour
    print('players: %s' % ', '.join('%s(%s)' % (l, w) for l, w in zip(labels, whos)))

    # ---------------------------------------------------------------- ball sources
    ball_obs, ball3d, bounces, ball_note = {}, {}, [], None
    if a.flights:
        fp = resolve(a.work, a.flights)
        ball3d, bounces = merge_flights(shots, fp, fps)
        ball_note = cfg.get('ball_source_note')
        print('ball path from %s: %d frames reconstructed, %d bounces (own rpm per shot)'
              % (os.path.basename(fp), len(ball3d), len(bounces)))
    ip = resolve(a.work, a.input)
    if os.path.exists(ip) and not a.flights:
        for s in json.load(open(ip))['shots']:
            for kind in ('incoming', 'outgoing'):
                for f, u, v in (s.get(kind) or []):
                    ball_obs[int(f)] = dict(u=u, v=v, kind='in' if kind == 'incoming' else 'out',
                                            shot=s['id'])

    F = {sz: ImageFont.truetype(a.font, sz) for sz in (13, 14, 15, 17, 20, 24, 25, 28, 30)}
    n_all = meta['n_frames']
    f0, f1 = a.frames if a.frames else (0, n_all)

    def P(p):
        uv, z = cam.project(np.atleast_2d(p))
        return (int(round(uv[0][0] * SCG)), int(round(uv[0][1] * SCG))), z[0]

    def to_out(uv, f):
        """geometry pixel -> output pixel (through the warp when stabilised)."""
        if st is None:
            return uv[0] * SCG, uv[1] * SCG
        q = st.canvas2orig(f) @ np.array([uv[0], uv[1], 1.0])
        return q[0] / q[2] * SC, q[1] / q[2] * SC

    def draw_court(im):
        cols_ = dict(baseline=(80, 200, 120), sideline=(70, 180, 110), service=(70, 170, 110),
                     centre=(70, 170, 110), net_tape=(0, 200, 255), net_mesh=(0, 165, 255))
        for pts, kind in cam.court_polylines():
            col = cols_.get(kind, (150, 150, 150))
            th = 2 if kind in ('baseline', 'net_tape') else 1
            for p, q in zip(pts, pts[1:]):
                seg = dense(p, q, 24)
                uv, z = cam.project(seg)
                safe_poly(im, uv * SCG, col, th, depth=z)

    def skeleton(im, pose, f, col, per_limb):
        d = pose.get(str(f))
        if not d:
            return None
        k = np.array(d['kpts'], float)
        for i, j, t in EDGES:
            if k[i][2] > .3 and k[j][2] > .3:
                cv2.line(im, (int(k[i][0] * SCG), int(k[i][1] * SCG)),
                         (int(k[j][0] * SCG), int(k[j][1] * SCG)),
                         LIMB[t] if per_limb else col, 3 if per_limb else 2, cv2.LINE_AA)
        for x, y, c in k:
            if c > .3:
                cv2.circle(im, (int(x * SCG), int(y * SCG)), 4 if per_limb else 3,
                           (255, 255, 255) if per_limb else col, -1, cv2.LINE_AA)
        b = d['box']
        cv2.rectangle(im, (int(b[0] * SCG), int(b[1] * SCG)), (int(b[2] * SCG), int(b[3] * SCG)),
                      col, 2, cv2.LINE_AA)
        return k

    def player_ground(k):
        """court position of a player: hip line on the 1 m plane, ankles as a fallback."""
        if k is None:
            return None
        if min(k[11][2], k[12][2]) > .35:
            g = cam.img2plane([[(k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2]], 1.0)[0]
            return float(g[0]), float(g[1])
        an = [k[i][:2] for i in (15, 16) if k[i][2] > .3]
        if an:
            g = cam.img2ground([np.mean(an, axis=0)])[0]
            return float(g[0]), float(g[1])
        return None

    def minimap(canvas, x0, y0, shot, players, height=None, ball_xy=None, marks=()):
        pad = 1.5
        X0, X1 = -cam.DBL - pad, cam.DBL + pad
        Y0, Y1 = -cam.HALF - 4.0, cam.HALF + 3.0
        s = max(6.0, 120.0 / cam.HALF) if height is None else height / (Y1 - Y0)
        mw, mh = int((X1 - X0) * s), int((Y1 - Y0) * s)
        m = np.full((mh, mw, 3), 28, np.uint8)

        def q(x, y):
            # y = +HALF is the near baseline and must end up at the BOTTOM of the minimap
            return (int((x - X0) * s), int((y - Y0) * s))
        cv2.rectangle(m, q(-cam.DBL, cam.HALF), q(cam.DBL, -cam.HALF), (46, 74, 52), -1)
        for pts, kind in cam.court_polylines():
            if kind in ('net_tape', 'net_mesh'):
                continue
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(m, q(p1[0], p1[1]), q(p2[0], p2[1]), (150, 220, 170), 1)
        cv2.line(m, q(-cam.DBL - .4, 0), q(cam.DBL + .4, 0), (0, 200, 255), 3)
        for xy, col in players:
            if xy is not None:
                # a player standing well behind the baseline would fall off the map - clamp,
                # otherwise the dot silently disappears and the map looks like a bug
                px, py = q(float(np.clip(xy[0], X0, X1)), float(np.clip(xy[1], Y0, Y1)))
                cv2.circle(m, (int(np.clip(px, 5, mw - 5)), int(np.clip(py, 5, mh - 5))),
                           5, col, -1, cv2.LINE_AA)
        if ball_xy is not None:
            cv2.circle(m, q(ball_xy[0], ball_xy[1]), 3, (80, 255, 255), -1, cv2.LINE_AA)
        for p, inside in marks:
            col = GR if inside else RD
            cv2.drawMarker(m, q(p[0], p[1]), col, cv2.MARKER_TILTED_CROSS, 11, 2)
        if shot and shot.get('world_path'):
            wp = [q(p[0], p[1]) for p in shot['world_path']]
            for p1, p2 in zip(wp, wp[1:]):
                cv2.line(m, p1, p2, YE, 2, cv2.LINE_AA)
            band = [b for b in (shot.get('spin_band') or []) if b.get('land')]
            pts = [q(b['land'][0], b['land'][1]) for b in band]
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(m, p1, p2, (120, 120, 235), 1, cv2.LINE_AA)
            for i, b in enumerate(band):
                col = GR if b['result'] == 'IN' else RD
                cv2.drawMarker(m, pts[i], col, cv2.MARKER_TILTED_CROSS, 11 if i == 0 else 7,
                               2 if i == 0 else 1)
        mh, mw = min(mh, canvas.shape[0] - y0), min(mw, canvas.shape[1] - x0)
        canvas[y0:y0 + mh, x0:x0 + mw] = m[:mh, :mw]
        cv2.rectangle(canvas, (x0, y0), (x0 + mw, y0 + mh), (90, 90, 90), 1)
        return mw, mh

    def phase_of(f):
        for s in shots:
            ref = dict(contact=s['tc'], coil=tech.get(s['id'], {}).get('f_coil', s['tc'] - 12))
            for name, r0, o0, r1, o1, col in cfg['phases']:
                if ref[r0] + o0 <= f <= ref[r1] + o1:
                    return name, tuple(col), s
        return cfg['idle_phase'], (110, 110, 110), None

    def result_cn(s):
        return s.get('result_cn') or cfg['result_names'].get(s.get('result'), s.get('result'))

    def side_cn(s):
        return cfg['side_names'].get(s.get('side'), s.get('side') or '')

    def player_cn(s):
        w = s.get('who')
        if w is None:
            return ''
        for who, lab in zip(whos, labels):
            if who == w:
                return lab + ' · '
        return cfg['player_names'].get(w, str(w)) + ' · '

    def tally_of(f, key, val):
        return sum(s.get(key) == val for s in shots if s['tc'] <= f)

    tmp = os.path.join(a.work, '_video_noaudio.mp4')
    import imageio_ffmpeg as iio
    writer = iio.write_frames(tmp, (OW, OH), fps=fps, quality=a.quality, macro_block_size=1)
    writer.send(None)
    trail = []
    for f in range(f0, f1):
        src = cv2.imread(os.path.join(src_dir, 'f_%05d.jpg' % (f + 1)))
        if src is None:
            break
        if st is not None:
            geo = np.zeros((GH, GW, 3), np.uint8)   # transparent-by-blackness overlay layer
        else:
            geo = cv2.resize(src, (W, H), interpolation=cv2.INTER_CUBIC)
        draw_court(geo)
        phase, pcol, shot_ctx = phase_of(f)
        ks = [skeleton(geo, pose, f, col if multi else pcol, not multi)
              for pose, col in zip(poses, cols)]
        # ---- ball: detections + fading trail, or the reconstructed 3D flight
        b = ball_obs.get(f)
        if b:
            trail.append((b['u'] * SCG, b['v'] * SCG, b['kind']))
        elif ball3d.get(f) is not None:
            uv, z = cam.project([ball3d[f]])
            if z[0] > .3:
                trail.append((uv[0][0] * SCG, uv[0][1] * SCG, 'out'))
        trail = trail[-14:]
        for i, (u, v, kd) in enumerate(trail):
            if abs(u) < 3 * OW and abs(v) < 3 * OH:
                cv2.circle(geo, (int(u), int(v)), 2 + int(4 * i / max(len(trail) - 1, 1)),
                           CY if kd == 'in' else YE, -1, cv2.LINE_AA)
        if b:
            cv2.circle(geo, (int(b['u'] * SCG), int(b['v'] * SCG)), 15,
                       CY if b['kind'] == 'in' else YE, 2, cv2.LINE_AA)
        elif ball3d.get(f) is not None:
            uv, z = cam.project([ball3d[f]])
            if z[0] > .3:
                rr = max(4, int(cam.F * cam.BALL_D / max(z[0], 1e-3) * SCG * 1.6))
                cv2.circle(geo, (int(uv[0][0] * SCG), int(uv[0][1] * SCG)), rr + 3, YE, 2, cv2.LINE_AA)
        act = None
        for s in shots:
            f_end = max(s['tc'] + 45, (s.get('f_land') or 0) + 18)
            if s['tc'] - 1 <= f <= f_end:
                act = s
        if act and act.get('out_path'):
            el = (f - act['tc']) / fps
            past = [(int(u * SCG), int(v * SCG)) for u, v, t in act['out_path'] if t <= el + 1e-3]
            fut = [(int(u * SCG), int(v * SCG)) for u, v, t in act['out_path'] if t > el]
            safe_poly(geo, np.array(past, float).reshape(-1, 2), YE, 2)
            for i in range(0, max(len(fut) - 2, 0), 3):
                safe_poly(geo, np.array([fut[i], fut[min(i + 1, len(fut) - 1)]], float),
                          (120, 190, 210), 1)
            if past:
                cv2.circle(geo, past[-1], 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(geo, past[-1], 6, YE, 2, cv2.LINE_AA)
        # bounce markers of the reconstructed flights
        for bo in bounces:
            age = f - bo['f']
            if -1 <= age <= 26:
                col = GR if bo['inside'] else RD
                uv, z = cam.project([[bo['p'][0], bo['p'][1], 0.0]])
                if z[0] > .3:
                    c2 = (int(uv[0][0] * SCG), int(uv[0][1] * SCG))
                    cv2.circle(geo, c2, int(6 + max(0, age) * 1.1), col, 2, cv2.LINE_AA)
                    cv2.drawMarker(geo, c2, col, cv2.MARKER_CROSS, 14, 2)
        for s in shots:
            if abs(f - s['tc']) <= 2.5 and s.get('contact'):
                (cx, cy), z = P(s['contact'])
                if z > .3:
                    rr = int(20 + 26 * abs(f - s['tc']) / 2.5)
                    cv2.circle(geo, (cx, cy), rr, (80, 80, 255), 3, cv2.LINE_AA)
                    cv2.line(geo, (cx - 34, cy), (cx + 34, cy), (80, 80, 255), 1)
                    cv2.line(geo, (cx, cy - 34), (cx, cy + 34), (80, 80, 255), 1)

        # ---- compose the video area
        if st is not None:
            im = st.compose(src, geo, f, a.overlay_alpha)
            if (W, H) != (im.shape[1], im.shape[0]):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            im = geo
        canvas = np.full((OH, OW, 3), 24, np.uint8)
        canvas[:H, :W] = im

        players_xy = [(player_ground(k), col) for k, col in zip(ks, cols)]
        ball_xy = None if ball3d.get(f) is None else (ball3d[f][0], ball3d[f][1])
        marks = [(bo['p'], bo['inside']) for bo in bounces if -1 <= f - bo['f'] <= 40]

        # ---------------------------------------------------------- panel geometry
        if panel == 'bottom':
            mm_h = PHGT - 84
            mx, my = PX0 + 14, PY0 + 30
            mw, mh = minimap(canvas, mx, my, act, players_xy, height=mm_h,
                             ball_xy=ball_xy, marks=marks)
            col1 = mx + mw + 20
            col2 = PX0 + int(PWID * 0.62)
            ty = PY0 + PHGT - 38
        else:
            mx, my = W + 10, 336
            mw, mh = minimap(canvas, mx, my, act, players_xy, ball_xy=ball_xy, marks=marks)
            col1 = col2 = W + 16
            ty = OH - 26
            roi = canvas[ty - 14:OH, 0:W].astype(np.float32)
            canvas[ty - 14:OH, 0:W] = (roi * .22 + np.float32([10, 10, 10]) * .78).astype(np.uint8)
        # ---------------------------------------------------------- timeline
        tl_x0 = (PX0 + 14) if panel == 'bottom' else 12
        tl_x1 = (PX0 + PWID - 14) if panel == 'bottom' else (W - 12)
        cv2.rectangle(canvas, (tl_x0, ty), (tl_x1, ty + 12), (55, 55, 55), -1)
        for s in shots:
            x = int(tl_x0 + (tl_x1 - tl_x0) * s['tc'] / max(n_all, 1))
            if multi and s.get('who') is not None:
                col = next((c for w, c in zip(whos, cols) if w == s['who']), (140, 140, 140))
            else:
                col = {'OUT': RD, 'NET': (80, 175, 255), 'IN': GR}.get(s.get('result'), (140, 140, 140))
            cv2.line(canvas, (x, ty - 5), (x, ty + 17), col, 3)
        cx = int(tl_x0 + (tl_x1 - tl_x0) * f / max(n_all, 1))
        cv2.line(canvas, (cx, ty - 9), (cx, ty + 21), (255, 255, 255), 2)

        # ---------------------------------------------------------- text (PIL, CJK-safe)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert('RGBA')
        ov = Image.new('RGBA', pil.size, (0, 0, 0, 0))
        ad = ImageDraw.Draw(ov)
        ad.rectangle([8, 8, min(320, W - 8), 84], fill=(12, 12, 16, 175))
        if panel == 'right':
            ad.rectangle([W + 6, 8, OW - 6, 330], fill=(12, 12, 16, 190))
        pil = Image.alpha_composite(pil, ov).convert('RGB')
        dr = ImageDraw.Draw(pil)
        tally = '   '.join('%s %d' % (lab, tally_of(f, key, val)) for lab, key, val in cfg['tally'])
        dr.text((18, 12), '帧 %04d / %d    %05.2f s' % (f, n_all, f / fps), font=F[17], fill=(230, 230, 230))
        dr.text((18, 36), phase, font=F[20], fill=tuple(pcol[::-1]))
        dr.text((18, 62), tally, font=F[14], fill=(210, 210, 210))
        if multi:   # which half of the minimap belongs to whom (far = top, near = bottom)
            dr.text((mx + 4, my + 2), cfg['half_labels']['far'], font=F[13], fill=(170, 200, 180))
            dr.text((mx + 4, my + mh - 19), cfg['half_labels']['near'], font=F[13],
                    fill=(170, 200, 180))
        # player name tags next to each skeleton
        if multi:
            for pose, lab, col in zip(poses, labels, cols):
                d = pose.get(str(f))
                if not d:
                    continue
                bx = d['box']
                px, py = to_out(((bx[0] + bx[2]) / 2, bx[1] - 8), f)
                if 0 < px < W - 40 and 20 < py < H:
                    dr.text((px - 30, py - 22), lab, font=F[15], fill=tuple(col[::-1]))

        if panel == 'right':
            y = 16
            dr.text((col1, y), cfg['title'], font=F[20], fill=(255, 230, 120))
            y += 34
            if act:
                dr.text((col1, y), cfg['shot_fmt'] % (act['id'], player_cn(act), side_cn(act)),
                        font=F[25], fill=(255, 255, 255))
                y += 34
                for lab, key, fmt in cfg['panel_fields']:
                    if key == '_range':
                        if act.get('kmh_lo') is None:
                            continue
                        txt = '%d–%d km/h%s' % (act['kmh_lo'], act['kmh_hi'],
                                                '（低置信）' if act.get('conf') == 'low' else '')
                    else:
                        v = act.get(key)
                        if v is None:
                            txt = cfg['unmeasurable'] if key == 'kmh' else None
                        else:
                            txt = fmt % v
                        if txt is None:
                            continue
                    dr.text((col1, y), '%s  %s' % (lab, txt), font=F[17], fill=(220, 220, 220))
                    y += 24
                col = RESULT_COL.get(act.get('result'), (200, 200, 200))
                dr.text((col1, y), cfg['verdict_fmt'] % (result_cn(act) or '-'), font=F[20], fill=col)
                y += 28
                for ln in textwrap.wrap(act.get('result_note', ''), 19)[:4]:
                    dr.text((col1, y), ln, font=F[14], fill=(165, 175, 185))
                    y += 19
            else:
                dr.text((col1, y), cfg['waiting'], font=F[20], fill=(170, 170, 170))
            notes = list(cfg['legend']) + ([ball_note] if ball_note else [])
            for i, ln in enumerate(notes):
                dr.text((col1, 336 + mh + 8 + 20 * i), ln, font=F[14], fill=(170, 170, 170))
        else:
            # ------------------------------------------------ bottom panel (portrait)
            dr.text((PX0 + 14, PY0 + 8), '帧 %04d / %d    %05.2f s   %s'
                    % (f, n_all, f / fps, cfg['title']), font=F[15], fill=(225, 228, 232))
            dr.text((PX0 + 14, PY0 + 30 + mh + 3), cfg['minimap_title'], font=F[13],
                    fill=(150, 160, 172))
            y = PY0 + 30
            if act:
                dr.text((col1, y), cfg['shot_fmt'] % (act['id'], player_cn(act), side_cn(act)),
                        font=F[24], fill=(255, 255, 255) if abs(f - act['tc']) <= 8
                        else (200, 205, 212))
                y += 32
                for lab, key, fmt in cfg['panel_fields']:
                    if key == '_range':
                        if act.get('kmh_lo') is None:
                            continue
                        txt = '%d–%d km/h%s' % (act['kmh_lo'], act['kmh_hi'],
                                                '（低置信）' if act.get('conf') == 'low' else '')
                    else:
                        v = act.get(key)
                        if v is None:
                            txt = cfg['unmeasurable'] if key == 'kmh' else None
                        else:
                            txt = fmt % v
                        if txt is None:
                            continue
                    big = key in ('kmh', '_range')
                    dr.text((col1, y), '%s  %s' % (lab, txt), font=F[20] if big else F[15],
                            fill=(120, 240, 255) if big else (205, 212, 220))
                    y += 26 if big else 21
                verdict = result_cn(act)
                if verdict:
                    dr.text((col1, y), cfg['verdict_fmt'] % verdict, font=F[17],
                            fill=tuple(RESULT_COL.get(act.get('result'), (200, 200, 200))[::-1]))
                    y += 22
                if act.get('land') or act.get('bounce'):
                    L = act.get('land') or act.get('bounce')
                    inside = act.get('bounce_in', act.get('result') == 'IN')
                    dr.text((col1, y), '%s %s' % (cfg['landing_title'],
                                                  cfg['landing_fmt'] % (L[0], L[1])),
                            font=F[15], fill=(120, 240, 150) if inside else (255, 130, 130))
                    y += 21
            else:
                dr.text((col1, y), cfg['waiting'], font=F[20], fill=(170, 170, 170))
            ys = PY0 + 30
            dr.text((col2, ys), cfg['stat_title'], font=F[17], fill=(255, 225, 130))
            ys += 24
            for who, lab, col in zip(whos, labels, cols):
                txt = '  '.join('%s %d' % (t[0], sum(1 for s in shots if s['tc'] <= f
                                                     and s.get('who') == who
                                                     and s.get(t[1]) == t[2]))
                                for t in cfg['player_tally'])
                dr.text((col2, ys), '%s  %s' % (lab, txt), font=F[14], fill=tuple(col[::-1]))
                ys += 20
            sp = [s['v0_kmh'] for s in shots if s['tc'] <= f and s.get('v0_kmh')] or \
                 [s['kmh'] for s in shots if s['tc'] <= f and s.get('kmh')]
            if sp:
                for i, fmt in enumerate(cfg['speed_stat']):
                    dr.text((col2, ys), fmt % (sum(sp) / len(sp) if i == 0 else max(sp)),
                            font=F[15], fill=(120, 240, 255))
                    ys += 21
            dr.text((col2, ys), tally, font=F[14], fill=(210, 210, 210))
            ys += 22
            for ln in ([ball_note] if ball_note else []) + list(cfg['legend'])[:1]:
                for part in textwrap.wrap(ln, 18)[:2]:
                    dr.text((col2, ys), part, font=F[13], fill=(140, 150, 160))
                    ys += 17
            dr.text((PX0 + 14, ty + 16), cfg['timeline_note'], font=F[13], fill=(150, 160, 172))
        writer.send(np.ascontiguousarray(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)[:, :, ::-1]))
        if (f - f0) % 150 == 0:
            print('  frame %d' % f, flush=True)
    writer.close()

    out = os.path.join(a.work, a.out)
    wav = os.path.join(a.work, 'audio.wav')
    if os.path.exists(wav) and not a.no_audio:
        ss = f0 / fps
        subprocess.run([iio.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', tmp,
                        '-ss', '%.3f' % ss, '-i', wav, '-map', '0:v:0', '-map', '1:a:0',
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest', out], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, out)
    print('rendered %d frames -> %s (%.1f MB)'
          % (f1 - f0, out, os.path.getsize(out) / 1e6))
    print('now check a few frames:  python3 scripts/video_prep.py grab --video %s --frames ...' % out)


if __name__ == '__main__':
    main()
