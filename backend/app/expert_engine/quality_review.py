"""Bounded, local second pass. Predictions select ROIs; only new detections fill gaps."""
import math
import cv2

VERSION = 'anchored-crop-v1'


def plan_review(tracks, meta, max_windows=6, max_frames=72):
    observed = sorted(int(f) for f,p in tracks.items() if p['status']=='observed')
    windows=[]; used=0
    for a,b in zip(observed,observed[1:]):
        gap=b-a-1
        if not 1 <= gap <= min(12,round(meta['fps']*.4)):continue
        left,right=tracks[str(a)],tracks[str(b)]
        # A large jump is not enough evidence to bridge two unrelated trajectories.
        if math.hypot(right['x']-left['x'],right['y']-left['y']) > 35*meta['width']/960*(b-a):continue
        if len(windows)>=max_windows or used+gap+2>max_frames:break
        used+=gap+2
        windows.append(dict(start=a,end=b,reason='anchored_ball_gap'))
    return windows


def recover(work, cache, tracks, poses, meta, model, associate, filter_candidates, main_player_tracks):
    import copy
    windows=plan_review(tracks,meta)
    audit=dict(version=VERSION,mode='local',cloud_used=False,windows=windows,
               attempted_frames=0,proposed_points=0,accepted_points=0,rejected_points=0,
               unresolved_pose_frames=meta['n_frames']-len(poses),failures=[])
    merged=copy.deepcopy(cache); proposals={}
    for window in windows:
        a,b=window['start'],window['end'];left,right=tracks[str(a)],tracks[str(b)]
        pad=max(64,round(meta['width']*.12))
        x0=max(0,int(min(left['x'],right['x'])-pad));x1=min(meta['width'],int(max(left['x'],right['x'])+pad))
        y0=max(0,int(min(left['y'],right['y'])-pad));y1=min(meta['height'],int(max(left['y'],right['y'])+pad))
        window['roi']=[x0,y0,x1,y1]
        # Constant ROI through each sequence; never mix crops in temporal input.
        model.frames.clear();model.last=None
        previous=None
        try:
            for f in range(max(0,a-1),b):
                image=cv2.imread(str(work/'frames'/f'f_{f+1:05d}.jpg'))
                if image is None:raise ValueError(f'missing frame {f}')
                crop=image[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                result=model.detect(crop,1000*f/meta['fps']);audit['attempted_frames']+=1
                if f>a:
                    t=(f-a)/(b-a);px=left['x']+(right['x']-left['x'])*t;py=left['y']+(right['y']-left['y'])*t
                    valid=[]
                    for c in result.candidates:
                        cx,cy=c.center;x,y=cx+x0,cy+y0
                        if c.confidence<.55 or math.hypot(x-px,y-py)>max(12,meta['width']*.025):continue
                        ix,iy=int(cx),int(cy)
                        patch=gray[max(0,iy-4):iy+5,max(0,ix-4):ix+5]
                        old=previous[max(0,iy-4):iy+5,max(0,ix-4):ix+5] if previous is not None else patch
                        if not patch.size or abs(patch.astype(float)-old).mean()<3:continue
                        valid.append(dict(x=x,y=y,score=float(c.confidence),source='tracknet_crop_review'))
                    if valid:proposals[str(f)]=[min(valid,key=lambda c:math.hypot(c['x']-px,c['y']-py))]
                previous=gray
        except (RuntimeError,ValueError) as exc:
            audit['failures'].append(dict(start=a,end=b,error=str(exc)[:200]))
        print(f'review {audit["attempted_frames"]}/72',flush=True)
    for f,items in proposals.items():merged.setdefault(f,[]).extend(items)
    audit['proposed_points']=len(proposals)
    verified=main_player_tracks(associate(filter_candidates(merged,poses),meta['fps'],meta['width']),poses,meta['width'])
    accepted=dict(tracks)
    for f,items in proposals.items():
        point=verified.get(f)
        if not point or point['source']!='tracknet_crop_review':continue
        if f in tracks and tracks[f]['status']=='observed':continue
        # Keep the original segment identity and all original observations unchanged.
        window=next(w for w in windows if w['start']<int(f)<w['end'])
        accepted[f]={**point,'segment':tracks[str(window['start'])]['segment']}
        audit['accepted_points']+=1
    audit['rejected_points']=audit['proposed_points']-audit['accepted_points']
    audit['remaining_unobserved_frames']=meta['n_frames']-sum(p['status']=='observed' for p in accepted.values())
    return {f:accepted[f] for f in sorted(accepted,key=int)},audit
