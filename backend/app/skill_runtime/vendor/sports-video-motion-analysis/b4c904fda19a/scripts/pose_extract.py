"""Step 2 - per-frame human pose for one athlete -> pose.json (one file per player)

Output schema (0-based frame index as string key):
  {"137": {"box": [x1,y1,x2,y2], "conf": 0.93, "kpts": [[x,y,conf] * 17]},
   "qc": {...}, "meta": {...}}
Keypoint order is COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles (left index first).
`qc` / `meta` are the only non-numeric top-level keys - every reader must skip keys that
are not digits (pose_metrics.py, render_annotated.py and swing_peaks.py already do).

One file per player
-------------------
The picker keeps exactly one detection per frame, so a rally with two players needs one run
per player. Three knobs decide who is kept, and they stack:

  --roi X0 Y0 X1 Y1   only detections whose centre falls in this pixel box (the coordinate
                      system of the frames being read - the stabilised canvas if you run on
                      a stabilised work dir). This is by far the strongest tool against
                      subject switching: on the broadcast clip this was built for, the far
                      player picked up ball kids and line judges until `--roi 60 200 1060
                      400` was added, which took the cross-person jumps from 34 to 0.
  --subject near|far  near = the bigger / lower figure, far = the smaller / higher one
                      (auto = the largest detection, the historical behaviour).
  --min-area          plain pixel-area floor; raise it for the near player, lower it
                      (2000-3000) for a far player 100 px tall.

  --label far -> $WORK/pose_far.json, so the two players cannot overwrite each other.
  Without --label the file stays $WORK/pose.json (single-player pipelines are unaffected).

Always read the QC block that is printed at the end (and stored in the json):
  * coverage below ~90 % of frames makes the swing-phase metrics noisy - usually a too-high
    --conf or a too-large --min-area;
  * **jumps > 0 means the "subject" is probably not the same person on every frame.** A jump
    is a frame-to-frame bbox centre shift beyond --jump-disp box widths or an area change
    beyond --jump-area x. Fix it with --roi / --subject and re-run; do not build metrics on
    a jumpy track, every swing number downstream will be garbage.

  python3 scripts/pose_extract.py --work work --model yolo11m-pose.pt
  python3 scripts/pose_extract.py --work work_stab --label near --subject near
  python3 scripts/pose_extract.py --work work_stab --label far --subject far \
      --roi 60 200 1060 400 --min-area 2500
  python3 scripts/pose_extract.py --qc-only work_stab/pose_far.json     # QC an existing file
"""
import argparse
import glob
import json
import os

import numpy as np

KPT_GROUPS = [('shoulders', (5, 6)), ('wrists', (9, 10)), ('hips', (11, 12)), ('ankles', (15, 16))]


def frames_of(pose):
    """detected frame indices, skipping the qc / meta keys."""
    return sorted(int(k) for k in pose if str(k).isdigit())


def missing_runs(det, f0, f1):
    """[[a, b], ...] closed intervals of frames in [f0, f1) without a detection."""
    runs, start = [], None
    for f in range(f0, f1):
        if f in det:
            if start is not None:
                runs.append([start, f - 1])
                start = None
        elif start is None:
            start = f
    if start is not None:
        runs.append([start, f1 - 1])
    return runs


def jump_check(pose, jump_disp=0.5, jump_area=2.0, max_gap=1):
    """Frames where the kept subject probably switched to a different person.

    Two symptoms, both scale-free so they work for a 450 px near player and a 100 px far
    one: the bbox centre moves more than `jump_disp` box widths between neighbouring
    frames, or the bbox area changes by more than `jump_area` x. Only frame pairs at most
    `max_gap` apart are compared - across a long gap a real player can legitimately have
    moved far.
    """
    det = {f: pose[str(f)] for f in frames_of(pose)}
    fs = sorted(det)
    jumps = []
    for i, f in enumerate(fs[:-1]):
        g = fs[i + 1]
        if g - f > max_gap:
            continue
        b1, b2 = np.array(det[f]['box'], float), np.array(det[g]['box'], float)
        w1 = max(b1[2] - b1[0], 1e-6)
        a1 = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1e-6)
        a2 = max((b2[2] - b2[0]) * (b2[3] - b2[1]), 1e-6)
        disp = float(np.hypot((b2[0] + b2[2] - b1[0] - b1[2]) / 2, (b2[1] + b2[3] - b1[1] - b1[3]) / 2))
        ratio = max(a1 / a2, a2 / a1)
        if disp > jump_disp * w1 or ratio > jump_area:
            jumps.append(dict(frame=g, disp_px=round(disp, 1), disp_boxw=round(disp / w1, 2),
                              area_ratio=round(ratio, 2)))
    return jumps


def qc_report(pose, n_frames, f0=0, f1=None, jump_disp=0.5, jump_area=2.0, max_gap=1,
              kpt_conf=0.35, min_cov=0.9, quiet=False):
    """Coverage + keypoint confidence + cross-person jump self-check. Printed and stored."""
    f1 = n_frames if f1 is None else f1
    fs = frames_of(pose)
    det = set(fs)
    span = max(f1 - f0, 1)
    cov = len(det) / span
    kpt = {}
    for name, idx in KPT_GROUPS:
        vals = [min(pose[str(f)]['kpts'][i][2] for i in idx) > kpt_conf for f in fs]
        kpt[name] = round(float(np.mean(vals)) if vals else 0.0, 3)
    jumps = jump_check(pose, jump_disp, jump_area, max_gap)
    gaps = missing_runs(det, f0, f1)
    boxh = [pose[str(f)]['box'][3] - pose[str(f)]['box'][1] for f in fs]
    qc = dict(frames_range=[f0, f1], n_detected=len(det), coverage=round(cov, 4),
              kpt_confident_frac=kpt, box_h_median_px=round(float(np.median(boxh)), 1) if boxh else None,
              jump_disp=jump_disp, jump_area=jump_area, jump_max_gap=max_gap,
              n_jumps=len(jumps), jump_frames=[j['frame'] for j in jumps],
              jumps=jumps[:200], missing_runs=gaps,
              n_missing=sum(b - a + 1 for a, b in gaps))
    if quiet:
        return qc
    print('pose coverage %.1f%% (%d/%d frames), bbox height median %s px'
          % (cov * 100, len(det), span, qc['box_h_median_px']))
    for name, _ in KPT_GROUPS:
        print('  %-10s confident in %.1f%% of detected frames' % (name, 100 * kpt[name]))
    print('  cross-person jumps: %d (centre shift > %.2f box widths or area change > %.1fx)'
          % (len(jumps), jump_disp, jump_area))
    if jumps:
        shown = [j['frame'] for j in jumps[:20]]
        print('    jump frames %s%s' % (shown, ' ... (%d more)' % (len(jumps) - 20)
                                        if len(jumps) > 20 else ''))
        worst = max(jumps, key=lambda j: j['disp_boxw'])
        print('    worst: frame %d moved %.2f box widths (%.0f px), area x%.2f'
              % (worst['frame'], worst['disp_boxw'], worst['disp_px'], worst['area_ratio']))
        print('WARNING the subject may be switching between several people - restrict the '
              'search with --roi X0 Y0 X1 Y1 and/or --subject near|far and re-run. Metrics '
              'computed on a jumpy track are meaningless.')
    if gaps:
        print('  missing frames: %d in %d runs, longest %d frames %s%s'
              % (qc['n_missing'], len(gaps), max(b - a + 1 for a, b in gaps),
                 [[a, b] for a, b in gaps[:12]], ' ...' if len(gaps) > 12 else ''))
    if cov < min_cov:
        print('WARNING: low coverage - lower --conf / --min-area, or use a larger model')
    return qc


def pick(boxes, confs, kp, args, roi, frame_h):
    """One detection per frame. `auto` keeps the largest box (the historical behaviour);
    `near` / `far` break ties by where the figure sits in the image, which is what
    distinguishes the two ends of a court once --roi has removed the bystanders."""
    cand = []
    for j, (x1, y1, x2, y2) in enumerate(boxes):
        area = float((x2 - x1) * (y2 - y1))
        if area < args.min_area:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if roi and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            continue
        cand.append((area, float(y2), j))
    if not cand:
        return None
    amax = max(c[0] for c in cand)
    hmax = max(frame_h, 1.0)
    if args.subject == 'auto':
        best = max(cand, key=lambda c: c[0])
    elif args.subject == 'near':
        best = max(cand, key=lambda c: c[0] / amax + c[1] / hmax)
    else:                                        # far: smaller and higher up the image
        best = min(cand, key=lambda c: c[0] / amax + c[1] / hmax)
    j = best[2]
    return (boxes[j].tolist(), float(confs[j]), kp[j].tolist() if kp is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='work')
    ap.add_argument('--model', default='yolo11m-pose.pt', help='yolo11n/s/m/l-pose.pt (n = fastest, m = good default)')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--min-area', type=float, default=6000, help='px^2, rejects far-side players and spectators')
    ap.add_argument('--roi', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                    default=None, help='keep detections whose centre falls in this pixel box')
    ap.add_argument('--subject', default='auto', choices=['auto', 'near', 'far'],
                    help='auto = largest detection (default, unchanged behaviour); '
                         'near = bigger / lower in frame; far = smaller / higher in frame')
    ap.add_argument('--label', default='main',
                    help="name of this subject; output becomes pose_<label>.json "
                         "(the default 'main' keeps the historical pose.json)")
    ap.add_argument('--frames', type=int, nargs=2, default=None, metavar=('A', 'B'),
                    help='only process frames [A, B) - use while tuning --roi / --subject')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--jump-disp', type=float, default=0.5,
                    help='QC: bbox centre shift, in box widths, that counts as a jump')
    ap.add_argument('--jump-area', type=float, default=2.0,
                    help='QC: bbox area ratio between neighbouring frames that counts as a jump')
    ap.add_argument('--jump-gap', type=int, default=1,
                    help='QC: max frame distance between the two frames compared')
    ap.add_argument('--min-cov', type=float, default=0.9, help='QC: warn below this coverage')
    ap.add_argument('--qc-only', default=None,
                    help='run only the QC block on an existing pose json and exit '
                         '(no model, no frames needed)')
    ap.add_argument('--qc-out', default=None, help='with --qc-only: write the qc dict here')
    ap.add_argument('--out', default=None,
                    help='output file name inside --work (default from --label)')
    a = ap.parse_args()

    # ---------------------------------------------------------------- QC of an existing file
    if a.qc_only:
        # a bare file name means "inside --work" (a stray pose.json in the shell's cwd
        # otherwise gets checked instead of the one that was just written)
        qp = a.qc_only if os.sep in a.qc_only else os.path.join(a.work, a.qc_only)
        pose = json.load(open(qp))
        pose = pose.get('frames', pose)
        fs = frames_of(pose)
        n = (a.frames[1] if a.frames else max(fs) + 1)
        f0 = a.frames[0] if a.frames else 0
        print('=== QC %s' % qp)
        qc = qc_report(pose, n, f0, n, a.jump_disp, a.jump_area, a.jump_gap,
                       min_cov=a.min_cov)
        if a.qc_out:
            json.dump(qc, open(a.qc_out, 'w'), indent=1)
            print('wrote %s' % a.qc_out)
        return

    out = a.out or ('pose.json' if a.label in (None, '', 'main') else 'pose_%s.json' % a.label)
    from ultralytics import YOLO
    frames = sorted(glob.glob(os.path.join(a.work, 'frames', '*.jpg')))
    if not frames:
        raise SystemExit('no frames in %s/frames - run video_prep.py first' % a.work)
    n_all = len(frames)
    f0, f1 = (max(0, a.frames[0]), min(n_all, a.frames[1])) if a.frames else (0, n_all)
    if f1 <= f0:
        raise SystemExit('--frames %s selects nothing (%d frames on disk)' % (a.frames, n_all))
    sel = frames[f0:f1]
    frame_h = 0.0
    model = YOLO(a.model)
    print('subject=%s label=%s roi=%s min_area=%.0f frames [%d, %d) of %d -> %s'
          % (a.subject, a.label, a.roi, a.min_area, f0, f1, n_all, os.path.join(a.work, out)))
    res = {}
    for s in range(0, len(sel), a.batch):
        batch = sel[s:s + a.batch]
        for i, r in enumerate(model.predict(batch, imgsz=a.imgsz, conf=a.conf,
                                            verbose=False, device=a.device)):
            fi = f0 + s + i
            if r.orig_shape is not None:
                frame_h = float(r.orig_shape[0])
            if r.boxes is None or not len(r.boxes):
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            best = pick(boxes, confs, kp, a, a.roi, frame_h)
            if best:
                res[str(fi)] = dict(box=[round(v, 1) for v in best[0]], conf=round(best[1], 3),
                                    kpts=[[round(v, 1), round(w, 1), round(c, 3)] for v, w, c in best[2]])
        if s % (a.batch * 10) == 0:
            print('  frame %d/%d' % (f0 + s, f1), flush=True)

    qc = qc_report(res, n_all, f0, f1, a.jump_disp, a.jump_area, a.jump_gap, min_cov=a.min_cov)
    res['qc'] = qc
    res['meta'] = dict(label=a.label, subject=a.subject, roi=list(a.roi) if a.roi else None,
                       min_area=a.min_area, conf=a.conf, model=os.path.basename(a.model),
                       imgsz=a.imgsz, frames=[f0, f1], n_frames_dir=n_all)
    json.dump(res, open(os.path.join(a.work, out), 'w'))
    print('wrote %s' % os.path.join(a.work, out))


if __name__ == '__main__':
    main()
