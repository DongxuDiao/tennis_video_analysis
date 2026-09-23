"""Step 8 - render the annotated video (and mux the original sound back in).

Reads the artefacts produced by the earlier steps and draws, per frame: court + net
wireframe from the calibration, the athlete's skeleton with the current swing phase, the
verified ball detections with a fading trail, the fitted flight path (solid up to now,
dashed ahead), a contact flash, a per-shot info card, a top-down minimap with the landing
spin band, a running tally and a shot timeline.

Iterate on a handful of frames before rendering everything - a full clip costs minutes and
layout collisions are only visible on real frames:

  python3 scripts/render_annotated.py --work work --frames 60 72 --out check.mp4
  python3 scripts/render_annotated.py --work work --out annotated.mp4

Panel text and colours come from --config (see references/render-config.md for the keys),
so the same renderer serves other sports and languages without touching the code.
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
    side_names={'FH': '正手', 'BH': '反手'},
    unmeasurable='无法测量',
)


def load_cfg(path):
    cfg = dict(DEFAULT)
    if path:
        cfg.update(json.load(open(path)))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work')
    ap.add_argument('--shots', default='shots.json')
    ap.add_argument('--input', default='shots_input.json', help='source of the verified ball observations')
    ap.add_argument('--technique', default='technique.json')
    ap.add_argument('--config', default=None)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--panel', type=int, default=300)
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
    W, H = int(cam.W * SC), int(cam.H * SC)
    # snap the canvas to a multiple of 16 so the encoder does not silently rescale it
    OH = int(np.ceil(H / 16) * 16)
    OW = int(np.ceil((W + a.panel) / 16) * 16)
    shots = json.load(open(os.path.join(a.work, a.shots)))
    pose = json.load(open(os.path.join(a.work, 'pose.json')))
    tech = {}
    tp = os.path.join(a.work, a.technique)
    if os.path.exists(tp):
        tech = {r['id']: r for r in json.load(open(tp))}
    ball = {}
    ip = os.path.join(a.work, a.input)
    if os.path.exists(ip):
        for s in json.load(open(ip))['shots']:
            for kind in ('incoming', 'outgoing'):
                for f, u, v in (s.get(kind) or []):
                    ball[int(f)] = dict(u=u, v=v, kind='in' if kind == 'incoming' else 'out', shot=s['id'])
    F = {sz: ImageFont.truetype(a.font, sz) for sz in (14, 17, 20, 25, 30)}
    n_all = meta['n_frames']
    f0, f1 = a.frames if a.frames else (0, n_all)

    def P(p):
        uv, z = cam.project(np.atleast_2d(p))
        return (int(round(uv[0][0] * SC)), int(round(uv[0][1] * SC))), z[0]

    def line3(im, p, q, col, th=1):
        (x1, y1), z1 = P(p)
        (x2, y2), z2 = P(q)
        if z1 > .3 and z2 > .3:
            cv2.line(im, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)

    def draw_court(im):
        cols = dict(baseline=(80, 200, 120), sideline=(70, 180, 110), service=(70, 170, 110),
                    centre=(70, 170, 110), net_tape=(0, 200, 255), net_mesh=(0, 165, 255))
        for pts, kind in cam.court_polylines():
            for p, q in zip(pts, pts[1:]):
                line3(im, p, q, cols.get(kind, (150, 150, 150)),
                      2 if kind in ('baseline', 'net_tape') else 1)

    def skeleton(im, f, col):
        d = pose.get(str(f))
        if not d:
            return None
        k = np.array(d['kpts'], float)
        for i, j, t in EDGES:
            if k[i][2] > .3 and k[j][2] > .3:
                cv2.line(im, (int(k[i][0] * SC), int(k[i][1] * SC)), (int(k[j][0] * SC), int(k[j][1] * SC)),
                         LIMB[t], 3, cv2.LINE_AA)
        for x, y, c in k:
            if c > .3:
                cv2.circle(im, (int(x * SC), int(y * SC)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        b = d['box']
        cv2.rectangle(im, (int(b[0] * SC), int(b[1] * SC)), (int(b[2] * SC), int(b[3] * SC)),
                      col, 2, cv2.LINE_AA)
        return k

    def minimap(canvas, x0, y0, shot, player_xy):
        s, pad = max(6.0, 120.0 / cam.HALF), 1.5
        X0, X1 = -cam.DBL - pad, cam.DBL + pad
        Y0, Y1 = -cam.HALF - 4.0, cam.HALF + 3.0
        mw, mh = int((X1 - X0) * s), int((Y1 - Y0) * s)
        m = np.full((mh, mw, 3), 28, np.uint8)

        def q(x, y):
            return (int((x - X0) * s), int((Y1 - y) * s))     # near side at the bottom
        cv2.rectangle(m, q(-cam.DBL, cam.HALF), q(cam.DBL, -cam.HALF), (46, 74, 52), -1)
        for pts, kind in cam.court_polylines():
            if kind in ('net_tape', 'net_mesh'):
                continue
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(m, q(p1[0], p1[1]), q(p2[0], p2[1]), (150, 220, 170), 1)
        cv2.line(m, q(-cam.DBL - .4, 0), q(cam.DBL + .4, 0), (0, 200, 255), 3)
        if player_xy is not None:
            cv2.circle(m, q(*player_xy), 5, (80, 160, 255), -1, cv2.LINE_AA)
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
        canvas[y0:y0 + mh, x0:x0 + mw] = m
        cv2.rectangle(canvas, (x0, y0), (x0 + mw, y0 + mh), (90, 90, 90), 1)
        return mh

    def phase_of(f):
        for s in shots:
            ref = dict(contact=s['tc'], coil=tech.get(s['id'], {}).get('f_coil', s['tc'] - 12))
            for name, r0, o0, r1, o1, col in cfg['phases']:
                if ref[r0] + o0 <= f <= ref[r1] + o1:
                    return name, tuple(col), s
        return cfg['idle_phase'], (110, 110, 110), None

    tmp = os.path.join(a.work, '_video_noaudio.mp4')
    import imageio_ffmpeg as iio
    writer = iio.write_frames(tmp, (OW, OH), fps=fps, quality=a.quality, macro_block_size=1)
    writer.send(None)
    trail = []
    for f in range(f0, f1):
        src = cv2.imread(os.path.join(a.work, 'frames', 'f_%05d.jpg' % (f + 1)))
        if src is None:
            break
        im = cv2.resize(src, (W, H), interpolation=cv2.INTER_CUBIC)
        draw_court(im)
        phase, pcol, shot_ctx = phase_of(f)
        k = skeleton(im, f, pcol)
        b = ball.get(f)
        if b:
            trail.append((b['u'] * SC, b['v'] * SC, b['kind']))
        trail = trail[-14:]
        for i, (u, v, kd) in enumerate(trail):
            cv2.circle(im, (int(u), int(v)), 2 + int(4 * i / max(len(trail) - 1, 1)),
                       CY if kd == 'in' else YE, -1, cv2.LINE_AA)
        if b:
            cv2.circle(im, (int(b['u'] * SC), int(b['v'] * SC)), 15,
                       CY if b['kind'] == 'in' else YE, 2, cv2.LINE_AA)
        act = None
        for s in shots:
            if s['tc'] - 1 <= f <= s['tc'] + 45:
                act = s
        if act and act.get('out_path'):
            el = (f - act['tc']) / fps
            past = [(int(u * SC), int(v * SC)) for u, v, t in act['out_path'] if t <= el + 1e-3]
            fut = [(int(u * SC), int(v * SC)) for u, v, t in act['out_path'] if t > el]
            for p1, p2 in zip(past, past[1:]):
                cv2.line(im, p1, p2, YE, 2, cv2.LINE_AA)
            for i in range(0, max(len(fut) - 2, 0), 3):
                cv2.line(im, fut[i], fut[min(i + 1, len(fut) - 1)], (120, 190, 210), 1, cv2.LINE_AA)
            if past:
                cv2.circle(im, past[-1], 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(im, past[-1], 6, YE, 2, cv2.LINE_AA)
        for s in shots:
            if abs(f - s['tc']) <= 2.5 and s.get('contact'):
                (cx, cy), _ = P(s['contact'])
                rr = int(20 + 26 * abs(f - s['tc']) / 2.5)
                cv2.circle(im, (cx, cy), rr, (80, 80, 255), 3, cv2.LINE_AA)
                cv2.line(im, (cx - 34, cy), (cx + 34, cy), (80, 80, 255), 1)
                cv2.line(im, (cx, cy - 34), (cx, cy + 34), (80, 80, 255), 1)

        canvas = np.full((OH, OW, 3), 24, np.uint8)
        canvas[:H, :W] = im
        player_xy = None
        if k is not None and min(k[11][2], k[12][2]) > .35:
            hip = [[(k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2]]
            g = cam.img2plane(hip, 1.0)[0]
            player_xy = (g[0], g[1])
        mh = minimap(canvas, W + 10, 336, act, player_xy)
        ty = OH - 26
        roi = canvas[ty - 14:OH, 0:W].astype(np.float32)
        canvas[ty - 14:OH, 0:W] = (roi * .22 + np.float32([10, 10, 10]) * .78).astype(np.uint8)
        cv2.rectangle(canvas, (12, ty), (W - 12, ty + 12), (55, 55, 55), -1)
        for s in shots:
            x = int(12 + (W - 24) * s['tc'] / n_all)
            cv2.line(canvas, (x, ty - 5), (x, ty + 17),
                     {'OUT': RD, 'NET': (80, 175, 255), 'IN': GR}.get(s.get('result'), (140, 140, 140)), 3)
        cx = int(12 + (W - 24) * f / n_all)
        cv2.line(canvas, (cx, ty - 9), (cx, ty + 21), (255, 255, 255), 2)

        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert('RGBA')
        ov = Image.new('RGBA', pil.size, (0, 0, 0, 0))
        ad = ImageDraw.Draw(ov)
        ad.rectangle([8, 8, 320, 84], fill=(12, 12, 16, 175))
        ad.rectangle([W + 6, 8, OW - 6, 330], fill=(12, 12, 16, 190))
        pil = Image.alpha_composite(pil, ov).convert('RGB')
        dr = ImageDraw.Draw(pil)
        done = [s for s in shots if s['tc'] <= f]
        tally = '   '.join('%s %d' % (lab, sum(s.get(key) == val for s in done))
                           for lab, key, val in cfg['tally'])
        dr.text((18, 12), '帧 %04d / %d    %05.2f s' % (f, n_all, f / fps), font=F[17], fill=(230, 230, 230))
        dr.text((18, 36), phase, font=F[20], fill=tuple(pcol[::-1]))
        dr.text((18, 62), tally, font=F[14], fill=(210, 210, 210))
        y = 16
        dr.text((W + 16, y), cfg['title'], font=F[20], fill=(255, 230, 120))
        y += 34
        if act:
            side = cfg['side_names'].get(act.get('side'), act.get('side') or '')
            dr.text((W + 16, y), '第 %s 拍 · %s' % (act['id'], side), font=F[25], fill=(255, 255, 255))
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
                        if key == 'kmh' and act.get('conf') not in ('high', 'medium'):
                            txt = '疑似 ' + txt
                    if txt is None:
                        continue
                dr.text((W + 16, y), '%s  %s' % (lab, txt), font=F[17], fill=(220, 220, 220))
                y += 24
            col = RESULT_COL.get(act.get('result'), (200, 200, 200))
            dr.text((W + 16, y), '判定：%s' % act.get('result_cn', act.get('result', '-')),
                    font=F[20], fill=col)
            y += 28
            for ln in textwrap.wrap(act.get('result_note', ''), 19)[:max(0, (330-y)//19)]:
                dr.text((W + 16, y), ln, font=F[14], fill=(165, 175, 185))
                y += 19
        else:
            dr.text((W + 16, y), '等待下一拍…', font=F[20], fill=(170, 170, 170))
        for i, ln in enumerate(cfg['legend']):
            dr.text((W + 16, 336 + mh + 8 + 20 * i), ln, font=F[14], fill=(170, 170, 170))
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
