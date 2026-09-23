"""Pinhole camera + playing-surface model shared by every script in this skill.

World frame (right-handed, metres):
  X : lateral, positive to the right as seen from the camera
  Y : along the court length; near baseline at y=+HALF, net at y=0, far baseline y=-HALF
  Z : up, ground plane z=0
The camera normally stands behind the near baseline, i.e. Cy > HALF.

Camera is stored as 7 physical parameters [X0, Y0, Z0, yaw, pitch, roll, f_px] so that
every quantity has a physical meaning and can be sanity-checked (a 20 m high or 3000 px
focal solution is a bug, not a result).
"""
import json
import os

import numpy as np

# ---------------------------------------------------------------- court presets
# half   : baseline distance from the net (m)
# dbl    : outer sideline half-width; sgl: inner (singles) sideline half-width
# svc    : service line distance from the net (0 if the sport has none)
# net_c / net_p : net height at the centre / at the posts (m)
# ball_d : ball diameter (m), used as an apparent-size depth cue
COURTS = {
    'tennis':       dict(half=11.885, dbl=5.485, sgl=4.115, svc=6.40, net_c=0.914, net_p=1.07, ball_d=0.067),
    'badminton':    dict(half=6.70,   dbl=3.05,  sgl=2.59,  svc=1.98, net_c=1.524, net_p=1.55, ball_d=0.058),
    'table_tennis': dict(half=1.37,   dbl=0.7625, sgl=0.7625, svc=0.0, net_c=0.1525, net_p=0.1525, ball_d=0.040),
    'pickleball':   dict(half=6.70,   dbl=3.05,  sgl=3.05,  svc=2.13, net_c=0.864, net_p=0.914, ball_d=0.074),
    'volleyball':   dict(half=9.00,   dbl=4.50,  sgl=4.50,  svc=3.00, net_c=2.43,  net_p=2.43,  ball_d=0.21),
}


def court(name_or_dict):
    if isinstance(name_or_dict, dict):
        c = dict(COURTS['tennis'])
        c.update(name_or_dict)
        return c
    return dict(COURTS[name_or_dict])


def build_pose(p):
    """7 params -> (R, C, f). R maps world vectors into camera coords."""
    X0, Y0, Z0, yaw, pitch, roll, f = [float(v) for v in p]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    view = np.array([sy * cp, -cy * cp, -sp])       # optical axis, pointing down-court (-Y)
    right = np.array([cy, sy, 0.0])
    down = np.cross(right, view)
    R = np.stack([right, down, view])
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ R
    return R, np.array([X0, Y0, Z0]), f


class Camera:
    """Loads calib.json produced by court_calib.py."""

    def __init__(self, work='.', calib_file=None):
        self.work = work
        path = calib_file or os.path.join(work, 'calib.json')
        d = json.load(open(path))
        self.raw = d
        self.W, self.H = int(d['width']), int(d['height'])
        self.params = np.array(d['cam_params'], float)
        self.R, self.C, self.F = build_pose(self.params)
        self.court = court(d.get('court') or d.get('sport') or 'tennis')
        self.HALF = self.court['half']
        self.DBL, self.SGL, self.SVC = self.court['dbl'], self.court['sgl'], self.court['svc']
        self.BALL_D = self.court['ball_d']
        if d.get('Hground'):
            self.Hg = np.array(d['Hground'], float).reshape(3, 3)
        else:
            self.Hg = self._ground_homography()
        self.Hi = np.linalg.inv(self.Hg)

    # ------------------------------------------------------------- projections
    def project(self, Pw):
        """world (N,3) -> (image (N,2), depth (N,)). depth<=0 means behind the camera."""
        Pw = np.atleast_2d(np.asarray(Pw, float))
        Xc = (self.R @ (Pw - self.C).T).T
        z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
        return np.stack([self.F * Xc[:, 0] / z + self.W / 2,
                         self.F * Xc[:, 1] / z + self.H / 2], 1), Xc[:, 2]

    def ray(self, uv):
        """image (N,2) -> unit viewing directions in world coords (N,3)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        d = np.stack([(uv[:, 0] - self.W / 2) / self.F,
                      (uv[:, 1] - self.H / 2) / self.F, np.ones(len(uv))], 1)
        dw = (self.R.T @ d.T).T
        return dw / np.linalg.norm(dw, axis=1, keepdims=True)

    def img2plane(self, uv, z=0.0):
        """intersect viewing rays with the horizontal plane at height z -> world (N,3)."""
        d = self.ray(uv)
        t = (z - self.C[2]) / d[:, 2]
        return self.C[None, :] + d * t[:, None]

    def img2ground(self, uv):
        return self.img2plane(uv, 0.0)[:, :2]

    def ground2img(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        return self.project(np.concatenate([xy, np.zeros((len(xy), 1))], 1))[0]

    def _ground_homography(self):
        s = np.array([[-self.SGL, self.HALF], [self.SGL, self.HALF],
                      [self.SGL, -self.HALF], [-self.SGL, -self.HALF]], np.float32)
        uv, _ = self.project(np.concatenate([s, np.zeros((4, 1))], 1))
        import cv2
        return cv2.getPerspectiveTransform(s, uv.astype(np.float32))

    # ------------------------------------------------------------- sanity cues
    def apparent_diam_px(self, uv, height=None):
        """expected image diameter (px) of the ball at image point uv, assuming it flies
        at `height` metres. Comparing this with the measured blob size is the only depth
        cue a single camera has - always use it before trusting a speed."""
        h = self.court['ball_d'] * 12 if height is None else height
        Pw = self.img2plane(uv, h)
        d = np.linalg.norm(Pw - self.C[None, :], axis=1)
        return self.F * self.BALL_D / np.maximum(d, 0.3)

    def net_height(self, x):
        c, p = self.court['net_c'], self.court['net_p']
        return c + (p - c) * min(abs(float(x)) / max(self.DBL, 1e-6), 1.0)

    def net_top_image_y(self, u):
        xs = np.linspace(-self.DBL, self.DBL, 200)
        hs = [self.net_height(x) for x in xs]
        uv, _ = self.project(np.stack([xs, np.zeros_like(xs), hs], 1))
        return float(np.interp(u, uv[:, 0], uv[:, 1]))

    def court_polylines(self):
        """world polylines of the court + net, for overlays: list of (pts, kind)."""
        H, DB, S, SV = self.HALF, self.DBL, self.SGL, self.SVC
        out = [([(-DB, H, 0), (DB, H, 0)], 'baseline'), ([(-DB, -H, 0), (DB, -H, 0)], 'baseline')]
        for x in (-DB, DB, -S, S):
            out.append(([(x, H, 0), (x, -H, 0)], 'sideline'))
        if SV:
            out.append(([(-S, SV, 0), (S, SV, 0)], 'service'))
            out.append(([(-S, -SV, 0), (S, -SV, 0)], 'service'))
            out.append(([(0, SV, 0), (0, -SV, 0)], 'centre'))
        tape = [(x, 0.0, self.net_height(x)) for x in np.linspace(-DB, DB, 24)]
        out.append((tape, 'net_tape'))
        for x in np.linspace(-DB, DB, 24):
            out.append(([(x, 0, 0), (x, 0, self.net_height(x))], 'net_mesh'))
        return out

    def describe(self):
        fov = 2 * np.degrees(np.arctan(self.W / 2 / self.F))
        return ('camera at x=%.2f y=%.2f z=%.2f m (%.2f m behind the near baseline), '
                'f=%.1f px, hfov=%.1f deg, rms=%s px'
                % (self.C[0], self.C[1], self.C[2], self.C[1] - self.HALF, self.F, fov,
                   self.raw.get('rms')))


if __name__ == '__main__':
    import sys
    cam = Camera(sys.argv[1] if len(sys.argv) > 1 else '.')
    print(cam.describe())
    for name, P in [('near baseline centre', [0, cam.HALF, 0]), ('net centre ground', [0, 0, 0]),
                    ('net centre top', [0, 0, cam.net_height(0)]),
                    ('far baseline centre', [0, -cam.HALF, 0])]:
        uv, z = cam.project([P])
        print('  %-22s -> u=%6.1f v=%6.1f  depth=%.2f m' % (name, uv[0][0], uv[0][1], z[0]))
    for v in (300, 400, 500):
        uv = [[cam.W / 2, v]]
        print('  image row %d: ground xy=%s  expected ball diam=%.1f px'
              % (v, np.round(cam.img2ground(uv)[0], 2), cam.apparent_diam_px(uv)[0]))
