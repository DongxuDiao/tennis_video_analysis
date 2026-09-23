"""Identity continuity and pose validity; uncertain frames never generate swing peaks."""
import numpy as np
VERSION='identity-v2'

def select(candidates,previous):
    if not candidates:return None
    if previous is None:return max(candidates,key=lambda p:(p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]))
    b=np.array(previous['box']);center=(b[:2]+b[2:])/2;size=np.maximum(b[2:]-b[:2],1)
    valid=[]
    for p in candidates:
        c=np.array(p['box']);ratio=np.prod(np.maximum(c[2:]-c[:2],1))/np.prod(size)
        distance=np.linalg.norm(((c[:2]+c[2:])/2-center)/size)
        if .5<ratio<2 and distance<.65:valid.append((distance+abs(np.log(ratio))*.2,p))
    return min(valid,key=lambda v:v[0])[1] if valid else None

def inspect(poses,n):
    clean={};bad=[];last=None;last_f=None
    for f in sorted(map(int,poses)):
        p=poses[str(f)];k=np.asarray(p.get('kpts',[]))
        if k.shape!=(17,3) or not np.isfinite(k).all():bad.append(f);continue
        if last is not None and f-last_f<=2 and select([p],last) is None:
            bad.append(f);continue
        clean[str(f)]=p;last=p;last_f=f
    return clean,dict(version=VERSION,suspicious_frames=bad,missing_frames=n-len(clean),
        keypoint_coverage={name:sum(min(p['kpts'][i][2] for i in indices)>.45 for p in clean.values())/max(n,1) for name,indices in [('wrists',(9,10)),('shoulders',(5,6)),('hips',(11,12))]})
