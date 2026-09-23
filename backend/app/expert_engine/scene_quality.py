"""Conservative camera-motion check and optional stabilization of analysis frames."""
import argparse,json,shutil
from pathlib import Path
import cv2
import numpy as np

VERSION='scene-pose-v2'

def estimate(reference,image):
    a=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask=np.full(a.shape,255,np.uint8);mask[:int(a.shape[0]*.15)]=0
    orb=cv2.ORB_create(nfeatures=1600)
    k1,d1=orb.detectAndCompute(b,mask);k2,d2=orb.detectAndCompute(a,mask)
    if d1 is None or d2 is None:return None
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    if len(good)<30:return None
    src=np.float32([k1[m.queryIdx].pt for m in good]);dst=np.float32([k2[m.trainIdx].pt for m in good])
    matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2)
    if matrix is None:return None
    keep=inliers.ravel().astype(bool);error=np.linalg.norm(cv2.transform(src[None],matrix)[0]-dst,axis=1)
    h,w=a.shape
    # Require spatially distributed support, not one player or a scoreboard.
    cells={(int(x>=w/2),int(y>=h/2)) for x,y in src[keep]}
    scale=np.hypot(matrix[0,0],matrix[0,1]);angle=abs(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
    if keep.mean()<.6 or len(cells)<3 or np.median(error[keep])>1.5:return None
    if not .96<scale<1.04 or angle>3 or np.linalg.norm(matrix[:,2])>.06*w:return None
    return matrix

def run(work):
    paths=sorted((work/'frames').glob('*.jpg'));ref=cv2.imread(str(paths[len(paths)//2]));h,w=ref.shape[:2]
    sampled=sorted(set([0,len(paths)//2,len(paths)-1]+list(range(0,len(paths),max(1,len(paths)//12)))))
    mats=[estimate(ref,cv2.imread(str(paths[i]))) for i in sampled]
    valid=[m for m in mats if m is not None]
    moving=any(np.linalg.norm(m[:,2])>3 or abs(np.hypot(m[0,0],m[0,1])-1)>.005 for m in valid)
    report=dict(version=VERSION,status='static',sample_count=len(sampled),valid_samples=len(valid),stabilized=False)
    if len(valid)!=len(mats):report['status']='unreliable_camera_motion'
    elif moving:
        report['status']='moving';matrices=[]
        for i,path in enumerate(paths):
            m=estimate(ref,cv2.imread(str(path)))
            if m is None:break
            matrices.append(m)
            if i%60==0:print(f'frame {i}/{len(paths)}',flush=True)
        if len(matrices)==len(paths):
            staging=work/'stabilized_frames';staging.mkdir(exist_ok=True)
            for path,m in zip(paths,matrices):
                out=cv2.warpAffine(cv2.imread(str(path)),m,(w,h))
                if not cv2.imwrite(str(staging/path.name),out):raise RuntimeError('stabilized frame write failed')
            (work/'frames').rename(work/'source_frames');staging.rename(work/'frames')
            samples=[cv2.imread(str(paths[i])) for i in sampled]
            cv2.imwrite(str(work/'bg_median.png'),np.median(samples,axis=0).astype(np.uint8))
            report.update(status='stabilized',stabilized=True,transforms=[m.tolist() for m in matrices],output_coordinates='stabilized_reference')
        else:report.update(status='unreliable_camera_motion',failed_frame=len(matrices))
    (work/'scene_quality.json').write_text(json.dumps(report))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);run(p.parse_args().work)
