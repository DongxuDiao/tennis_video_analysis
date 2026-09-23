"""Automatic 2D analysis: pretrained detections + conservative temporal association.
No manual labels, guessed camera calibration, or metric speed are required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))


def associate(candidates, fps, width):
    """Offline tracklets; reject stationary/single-frame clutter, never fill long gaps.

    At most one point per frame. Prediction is used only for association, and short
    interpolations are explicitly identified rather than counted as detections.
    """
    scale = width / 960
    nodes = []
    by_frame = {}
    for f in sorted(map(int, candidates)):
        for c in candidates[str(f)]:
            p = np.array([c['x'], c['y']], float)
            best = None
            for gap in range(1, max(2, int(fps * .1) + 1)):
                for j in by_frame.get(f-gap, []):
                    prev = nodes[j]
                    delta = p-prev['p']
                    if np.linalg.norm(delta) > 60*scale*gap*30/fps:
                        continue
                    velocity = delta/gap
                    accel = np.linalg.norm(velocity-prev['v']) if prev['parent'] is not None else 0
                    if accel > 22*scale*30/fps:
                        continue
                    score = prev['score'] + float(c['score']) + .6 - .012*accel/scale - .15*(gap-1)
                    if best is None or score > best[0]:
                        best = score, j, velocity
            node = dict(f=f, p=p, c=c, score=best[0] if best else float(c['score'])+.6,
                        parent=best[1] if best else None, v=best[2] if best else np.zeros(2))
            by_frame.setdefault(f, []).append(len(nodes)); nodes.append(node)
    chosen = {}; segment = 0
    for end in sorted(range(len(nodes)), key=lambda i: nodes[i]['score'], reverse=True):
        path = []; i = end
        while i is not None:
            node = nodes[i]
            if node['f'] in chosen:
                break
            path.append(node); i = node['parent']
        path.reverse()
        if len(path) < 4:
            continue
        xy = np.array([n['p'] for n in path])
        if np.linalg.norm(np.ptp(xy, axis=0)) < 12*scale:
            continue
        # Background tracks wobble around one pixel location; reject mostly stationary paths.
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if np.median(step) < 1.2*scale:
            continue
        segment += 1
        for n in path:
            chosen[n['f']] = dict(frame=n['f'], x=float(n['p'][0]), y=float(n['p'][1]),
                                   status='observed', segment=segment, source=n['c']['source'],
                                   score=round(float(n['c']['score']), 3))
        for a, b in zip(path, path[1:]):
            gap = b['f']-a['f']
            if 1 < gap <= max(2, round(.08*fps)) and all(f not in chosen for f in range(a['f']+1,b['f'])):
                for f in range(a['f']+1,b['f']):
                    p = a['p']+(b['p']-a['p'])*(f-a['f'])/gap
                    chosen[f] = dict(frame=f, x=float(p[0]), y=float(p[1]), status='interpolated',
                                     segment=segment, source='interpolation', score=None)
    return {str(f): chosen[f] for f in sorted(chosen)}


def filter_candidates(candidates, poses):
    """Reject observations on the athlete's torso/legs before temporal association."""
    result={}
    for f,items in candidates.items():
        po=poses.get(f)
        if not po:
            result[f]=items;continue
        k=np.array(po['kpts']);x1,y1,x2,y2=po['box'];h=max(40,y2-y1)
        torso=k[[5,6,12,11]]
        polygon=cv2.convexHull(torso[:,:2].astype(np.float32)) if min(torso[:,2])>.4 else None
        hip_y=float(np.mean(k[11:13,1])) if min(k[11:13,2])>.4 else y1+h*.5
        kept=[]
        for c in items:
            x,y=c['x'],c['y']
            if x1<=x<=x2 and hip_y<=y<=y2:continue
            if polygon is not None and cv2.pointPolygonTest(polygon,(float(x),float(y)),True)>=-3:continue
            kept.append(c)
        result[f]=kept
    return result


def main_player_tracks(tracks, poses, width=960):
    """A ball segment must pass near the main player's racket reach at least once."""
    keep=set()
    for f,t in tracks.items():
        p=poses.get(f)
        if not p or t['status']!='observed':continue
        h=max(40,p['box'][3]-p['box'][1])
        for x,y,c in p['kpts'][9:11]:
            if c>=.45 and math.hypot(t['x']-x,t['y']-y)<h*.45:keep.add(t['segment'])
    supported=set()
    for segment in keep:
        points=[t for t in tracks.values() if t['segment']==segment and t['status']=='observed']
        xy=np.array([[t['x'],t['y']] for t in points]);extent=float(np.linalg.norm(xy[-1]-xy[0]))
        length=float(np.sum(np.linalg.norm(np.diff(xy,axis=0),axis=1)))
        model_points=sum('tracknet' in t['source'] for t in points)
        if model_points>=2 or (len(points)>=6 and extent>=65*width/960 and extent/max(length,1)>.7):supported.add(segment)
    return {f:t for f,t in tracks.items() if t['segment'] in supported}


def wrist_motion(poses, n, fps):
    speeds = np.zeros((n, 2))
    for f in range(1,n-1):
        a, b = poses.get(str(f-1)), poses.get(str(f+1))
        if not a or not b: continue
        ka, kb = np.array(a['kpts']), np.array(b['kpts'])
        height = max(40, a['box'][3]-a['box'][1])
        for j, k in enumerate((9,10)):
            if min(ka[k,2],kb[k,2]) >= .45:
                speeds[f,j] = np.linalg.norm(kb[k,:2]-ka[k,:2])*fps/(2*height)
    return speeds


def classify_side(poses, tc, fps, hand):
    if hand not in ('right','left'):return 'UNK'
    axes=[]
    for p in poses.values():
        k=np.array(p['kpts']);axis=k[6,:2]-k[5,:2];length=np.linalg.norm(axis)
        if min(k[5,2],k[6,2])>.6 and length>20:axes.append(axis[0]/length)
    if not axes or abs(float(np.median(axes)))<.35:return 'UNK'
    direction=float(np.sign(np.median(axes)))*(1 if hand=='right' else -1)
    values=[[],[]];wrist=10 if hand=='right' else 9
    radius=max(2,round(fps*.08))
    for f in range(round(tc)-radius,round(tc)+radius+1):
        p=poses.get(str(f))
        if not p:continue
        k=np.array(p['kpts']);height=max(40,p['box'][3]-p['box'][1]);center=(k[5,0]+k[6,0])/2
        for j,wi in enumerate((wrist,19-wrist)):
            if min(k[5,2],k[6,2],k[wi,2])>=.4:values[j].append((k[wi,0]-center)/height*direction)
    if len(values[0])>=2 and abs(float(np.median(values[0])))>.08:
        return 'FH' if np.median(values[0])>0 else 'BH'
    # Occluded dominant wrist: the other hand on the backhand side is weak support only.
    if len(values[1])>=3 and np.median(values[1])<-.12:return 'BH'
    return 'UNK'


def detect_events(poses, tracks, peaks, meta, hand='auto'):
    from scipy.signal import find_peaks
    n, fps = meta['n_frames'], meta['fps']
    speeds = wrist_motion(poses,n,fps)
    energy = np.sum(np.minimum(speeds,8)**2,axis=0)
    inferred = ('right' if energy[1]>energy[0] else 'left') if max(energy)>1.5*max(min(energy),1e-6) else 'unknown'
    hand_used = inferred if hand == 'auto' else hand
    motion = np.max(speeds,axis=1)
    smooth = np.convolve(motion,np.ones(3)/3,mode='same')[:n]
    positive = smooth[smooth>0]
    threshold = max(.75, float(np.percentile(positive,65)) if len(positive) else .75)
    ids,_ = find_peaks(smooth,height=threshold,prominence=.3,distance=max(1,round(fps*.6)))
    events=[]
    for f in ids:
        pose=poses.get(str(int(f)))
        if not pose: continue
        height=max(40,pose['box'][3]-pose['box'][1]); k=np.array(pose['kpts'])
        audio=[p for p in peaks if abs(p['frame']-f)<=fps*.14]
        near=[]
        for g in range(max(0,int(f)-round(fps*.14)),min(n,int(f)+round(fps*.14)+1)):
            t=tracks.get(str(g)); po=poses.get(str(g))
            if not t or t['status']!='observed' or not po: continue
            wrists=[v[:2] for v in po['kpts'][9:11] if v[2]>=.45]
            if wrists and min(np.linalg.norm(np.array([t['x'],t['y']])-w) for w in wrists)<height*.55:
                near.append(g)
        if not audio:
            # A passing ball is insufficient: need an observed approach/departure reversal.
            before=tracks.get(str(int(f)-3));after=tracks.get(str(int(f)+3));mid=tracks.get(str(int(f)))
            if not near or not all(t and t['status']=='observed' for t in (before,mid,after)):continue
            v1=np.array([mid['x']-before['x'],mid['y']-before['y']]);v2=np.array([after['x']-mid['x'],after['y']-mid['y']])
            if np.linalg.norm(v1)<8 or np.linalg.norm(v2)<8 or float(v1@v2)>0:continue
        tc=min(audio,key=lambda p:abs(p['frame']-f))['frame'] if audio else float(f)
        side=classify_side(poses,tc,fps,hand_used)
        events.append(dict(id=len(events)+1, tc=round(float(tc),1), t_s=round(float(tc/fps),3),
                           side=side, side_source='pose_heuristic', conf='medium' if audio and near else 'low',
                           evidence=['wrist_motion']+(['audio'] if audio else [])+(['ball_near_wrist'] if near else []),
                           wrist_speed_body_s=round(float(smooth[f]),2), kmh=None, result='UNK',
                           result_cn='未测量', result_note='自动候选；未标定，不推算真实球速及落点'))
    selected=[]
    for event in sorted(events,key=lambda e:(len(e['evidence']),e['wrist_speed_body_s']),reverse=True):
        if all(abs(event['tc']-other['tc'])>fps*.9 for other in selected):selected.append(event)
    events=sorted(selected,key=lambda e:e['tc'])
    for i,event in enumerate(events,1):event['id']=i
    return events, dict(hand=hand_used, hand_source='inferred' if hand=='auto' else 'user', event_threshold=threshold)


def summarize(poses, tracks, shots, meta, info):
    n=meta['n_frames']; observed=sum(p['status']=='observed' for p in tracks.values())
    return dict(n_frames=n, pose_frames=len(poses), observed_ball_frames=observed,
                interpolated_ball_frames=len(tracks)-observed, ball_coverage=round(observed/max(n,1),4),
                pose_coverage=round(len(poses)/max(n,1),4), shot_candidates=len(shots),
                hand=info['hand'], hand_source=info['hand_source'], calibrated=False,
                limitations=['覆盖率不是准确率；球点与击球事件为自动候选。','未做可靠三维标定，不输出 km/h 或真实落点。',
                             '正反手采用姿态规则推断，持拍手或视角不明确时显示未确认。'])


def analyze(work, model_path, hand='auto'):
    from app.vision.tracknet import TrackNetDetector
    meta=read(work/'meta.json'); poses=read(work/'pose.json',{}); colors=read(work/'ball_cand.json',{})
    old=read(work/'auto_summary.json',{})
    compatible=old.get('candidate_version')=='tracknet-color-v1' or old.get('engine_hash')=='ae6925ab206fe53f'
    if compatible and old.get('models',{}).get('ball_hash')==hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:16] and (work/'auto_candidates.json').exists():
        print('reuse compatible model observations; recompute tracks/events',flush=True)
        finalize(work,read(work/'auto_candidates.json'),meta,poses,hand,old['models'],model_path=model_path)
        return
    cache={}
    # TrackNet runs on every frame: temporal inputs must remain consecutive.
    model=TrackNetDetector(model_path,threshold=.35)
    prev=None
    for f in range(meta['n_frames']):
        im=cv2.imread(str(work/'frames'/f'f_{f+1:05d}.jpg'))
        if im is None: raise RuntimeError(f'缺失帧 {f}')
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        detection=model.detect(im,1000*f/meta['fps'])
        candidates=[]
        for c in detection.candidates:
            x,y=c.center; ix,iy=int(x),int(y)
            patch=gray[max(0,iy-5):iy+6,max(0,ix-5):ix+6]
            old=prev[max(0,iy-5):iy+6,max(0,ix-5):ix+6] if prev is not None else patch
            if not patch.size or float(np.mean(np.abs(patch.astype(float)-old)))<3: continue
            candidates.append(dict(x=float(x),y=float(y),score=float(c.confidence),source='tracknet'))
        for c in colors.get(str(f),[]):
            if any(math.hypot(c['x']-d['x'],c['y']-d['y'])<10 for d in candidates):
                for d in candidates:
                    if math.hypot(c['x']-d['x'],c['y']-d['y'])<10:
                        d.update(x=c['x'],y=c['y'],score=min(1,d['score']+.2),source='tracknet+color')
            else:
                candidates.append(dict(x=c['x'],y=c['y'],score=.45,source='color_motion'))
        cache[str(f)]=sorted(candidates,key=lambda d:d['score'],reverse=True)[:12]
        prev=gray
        if f%60==0: print(f'tracknet {f}/{meta["n_frames"]}',flush=True)
    save(work/'auto_candidates.json',cache)
    finalize(work,cache,meta,poses,hand,dict(ball='TrackNet + color/motion',ball_hash=model.model_hash,device=model.device,pose='YOLO11m-pose'),model=model)


def review_events(shots,poses,people,meta,info):
    for event in shots:
        f=round(event['tc']);main=poses.get(str(f));ambiguous=False
        if main:
            for other in people.get(str(f),[]):
                if np.linalg.norm(np.array(other['box'])-np.array(main['box']))<10:continue
                for offset in (-2,2):
                    peers=people.get(str(f+offset),[])
                    if not peers:continue
                    peer=min(peers,key=lambda p:np.linalg.norm(np.array(p['box'])-np.array(other['box'])))
                    from app.expert_engine.person_quality import select
                    if select([peer],other) is None:continue
                    h=max(40,other['box'][3]-other['box'][1])
                    for j in (9,10):
                        a,b=other['kpts'][j],peer['kpts'][j]
                        if min(a[2],b[2])>.45 and np.linalg.norm(np.array(a[:2])-b[:2])*meta['fps']/2/h>info['event_threshold']:
                            ambiguous=True
        event['attribution']='ambiguous' if ambiguous else ('main_player_candidate' if people else 'unchecked')
        # Require local view consistency, rather than a whole-video orientation assumption.
        local={str(g):poses[str(g)] for g in range(max(0,f-3),min(meta['n_frames'],f+4)) if str(g) in poses}
        local_side=classify_side(local,event['tc'],meta['fps'],info['hand'])
        event['side_conflict']=local_side!=event['side'] and local_side!='UNK' and event['side']!='UNK'
        if ambiguous or event['side_conflict'] or local_side=='UNK':event.update(side='UNK',conf='low')
    return shots


def finalize(work,cache,meta,poses,hand,models,model=None,model_path=None):
    from app.expert_engine.person_quality import inspect
    poses,person_qc=inspect(poses,meta['n_frames'])
    previous_qc=read(work/'person_quality.json',{})
    person_qc['suspicious_frames']=sorted(set(person_qc['suspicious_frames']+previous_qc.get('suspicious_frames',[])))
    tracks=main_player_tracks(associate(filter_candidates(cache,poses),meta['fps'],meta['width']),poses,meta['width'])
    from app.expert_engine.quality_review import recover, plan_review, VERSION
    before=sum(p['status']=='observed' for p in tracks.values())
    audit=dict(version=VERSION,mode='local',cloud_used=False,windows=[],accepted_points=0,
               baseline_observed_frames=before,unresolved_pose_frames=meta['n_frames']-len(poses))
    if plan_review(tracks,meta) and (model is not None or model_path is not None):
        try:
            if model is None:
                from app.vision.tracknet import TrackNetDetector
                model=TrackNetDetector(model_path,threshold=.35)
            tracks,audit=recover(work,cache,tracks,poses,meta,model,associate,filter_candidates,main_player_tracks)
            audit['baseline_observed_frames']=before
        except RuntimeError as exc:
            audit['error']=str(exc)[:200]  # Optional review failure preserves first-pass results.
    save(work/'quality_review.json',audit)
    shots,info=detect_events(poses,tracks,read(work/'impacts.json',{}).get('peaks',[]),meta,hand)
    review_events(shots,poses,read(work/'people.json',{}),meta,info)
    save(work/'person_quality.json',person_qc)

    summary=summarize(poses,tracks,shots,meta,info)
    summary['person_quality']=person_qc
    summary['scene_quality']=read(work/'scene_quality.json',{'status':'unchecked'})
    if summary['scene_quality']['status']=='unreliable_camera_motion':
        summary['limitations'].append('机位匹配未通过，保留原画面；运动轨迹与动作判断可能受相机运动影响。')
    elif summary['scene_quality']['status']=='stabilized':
        summary['limitations'].append('标注视频已稳定化，画面边缘可能有黑边；坐标属于稳定化参考画面。')
    summary['ambiguous_events']=sum(s['attribution']=='ambiguous' for s in shots)
    summary['quality_review']=audit
    summary['models']=models
    summary['candidate_version']='tracknet-color-v1'
    summary['engine_hash']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    save(work/'auto_tracks.json',tracks); save(work/'shots.json',shots); save(work/'auto_summary.json',summary)
    # Editable automatic suggestions retain provenance; never mark them human-reviewed.
    spec=read(work/'input.json',{}); spec.update(reviewed=False,shots=[])
    for s in shots:
        draft={k:s[k] for k in ('id','side','tc')}
        for key,sign in [('incoming',-1),('outgoing',1)]:
            draft[key]=[[int(f),p['x'],p['y']] for f,p in tracks.items() if p['status']=='observed' and 0<(int(f)-s['tc'])*sign<meta['fps']*.35]
        spec['shots'].append(draft)
    save(work/'input.json',spec)
    write_report(work,summary,shots)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def write_report(work, summary, shots):
    side={'FH':'疑似正手','BH':'疑似反手','UNK':'未确认'}
    lines=['# 网球自动分析报告','', '本报告由已有模型和时序规则自动生成，未经过人工逐拍复核。',
           f'总帧数：{summary["n_frames"]}；人体覆盖：{summary["pose_coverage"]:.1%}；球观测覆盖：{summary["ball_coverage"]:.1%}。',
           f'球观测帧 {summary["observed_ball_frames"]}，短缺口插值帧 {summary["interpolated_ball_frames"]}；击球候选 {len(shots)} 次。',
           '', '覆盖率不代表准确率。缺失轨迹保留断点；插值不计入观测。', '',
           '|候选|时间（秒）|动作|证据|可信度|','|---|---|---|---|---|']
    review=summary.get('quality_review',{})
    lines.insert(7, f"自动复核：检查 {len(review.get('windows',[]))} 个短缺口，新增通过校验的球观测 {review.get('accepted_points',0)} 帧。复核不等于人工确认；未调用云端模型。")
    lines.insert(7, f"人物质检：疑似跳变 {len(summary.get('person_quality',{}).get('suspicious_frames',[]))} 帧；击球归属存在歧义 {summary.get('ambiguous_events',0)} 次。")
    evidence={'wrist_motion':'挥拍动作','audio':'音频冲击','ball_near_wrist':'球接近手腕'}
    for s in shots: lines.append(f'|{s["id"]}|{s["t_s"]}|{side[s["side"]]}|'+ '、'.join(evidence[e] for e in s['evidence'])+f'|{s["conf"]}|')
    lines+=['','## 测量边界',*summary['limitations'],'','检测来源：TrackNet 时序模型、YOLO11m-pose，结合颜色及运动候选。']
    (work/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--hand',default='auto',choices=['auto','right','left'])
    a=p.parse_args();analyze(a.work,a.model,a.hand)
