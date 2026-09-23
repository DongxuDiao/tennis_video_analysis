"""Step 2 - per-frame human pose for the main athlete -> pose.json

Output schema (0-based frame index as string key):
  {"137": {"box": [x1,y1,x2,y2], "conf": 0.93, "kpts": [[x,y,conf] * 17]}}
Keypoint order is COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles (left index first).

The main athlete is the largest detection, optionally restricted to a region of interest,
because in court sports the near-side player is much larger than opponents and bystanders.
Always check the reported coverage: below ~90 % of frames the swing-phase metrics get
noisy and the missing frames are usually a too-high --conf or a too-large --min-area.

  python3 scripts/pose_extract.py --work work --model yolo11m-pose.pt
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work')
    ap.add_argument('--model', default='yolo11m-pose.pt', help='yolo11n/s/m/l-pose.pt (n = fastest, m = good default)')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--min-area', type=float, default=6000, help='px^2, rejects far-side players and spectators')
    ap.add_argument('--roi', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                    default=None, help='keep detections whose centre falls in this pixel box')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='pose.json')
    a = ap.parse_args()

    from ultralytics import YOLO
    frames = sorted(glob.glob(os.path.join(a.work, 'frames', '*.jpg')))
    if not frames:
        raise SystemExit('no frames in %s/frames - run video_prep.py first' % a.work)
    model = YOLO(a.model)
    from person_quality import select, inspect, VERSION
    res = {}; people={}; previous=None; missing=0
    for s in range(0, len(frames), a.batch):
        batch = frames[s:s + a.batch]
        for i, r in enumerate(model.predict(batch, imgsz=a.imgsz, conf=a.conf,
                                            verbose=False, device=a.device)):
            fi = s + i
            if r.boxes is None or not len(r.boxes):
                missing+=1
                if missing>30:previous=None
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            candidates=[]
            for j, (x1,y1,x2,y2) in enumerate(boxes):
                if kp is None or (x2-x1)*(y2-y1)<max(800,a.min_area/4):continue
                if a.roi and not (a.roi[0]<=(x1+x2)/2<=a.roi[2] and a.roi[1]<=(y1+y2)/2<=a.roi[3]):continue
                candidates.append(dict(box=boxes[j].tolist(),conf=float(confs[j]),kpts=kp[j].tolist()))
            people[str(fi)]=candidates
            eligible=[p for p in candidates if (p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1])>=a.min_area]
            best=select(eligible,previous)
            if best:
                res[str(fi)]=best;previous=best;missing=0
            else:
                missing+=1
                # Do not silently choose a different player after a brief occlusion.
                if missing>30:previous=None
        if s % (a.batch * 10) == 0:
            print('  frame %d/%d' % (s, len(frames)), flush=True)

    res,qc=inspect(res,len(frames))
    json.dump(people,open(os.path.join(a.work,'people.json'),'w'))
    json.dump(qc,open(os.path.join(a.work,'person_quality.json'),'w'))
    json.dump(res, open(os.path.join(a.work, a.out), 'w'))
    cov = len(res) / len(frames)
    print('pose coverage %.1f%% (%d/%d frames)' % (cov * 100, len(res), len(frames)))
    for name, idx in [('shoulders', (5, 6)), ('wrists', (9, 10)), ('hips', (11, 12)), ('ankles', (15, 16))]:
        ok = np.mean([min(d['kpts'][i][2] for i in idx) > 0.35 for d in res.values()]) if res else 0
        print('  %-10s confident in %.1f%% of detected frames' % (name, ok * 100))
    if cov < 0.9:
        print('WARNING: low coverage - lower --conf / --min-area, or use a larger model')


if __name__ == '__main__':
    main()
