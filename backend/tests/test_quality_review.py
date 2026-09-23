from collections import deque
from types import SimpleNamespace
import numpy as np
import cv2
from app.expert_engine.quality_review import plan_review, recover

def point(f,x):return dict(frame=f,x=x,y=80,status='observed',segment=1,source='tracknet',score=.8)

def test_plan_is_bounded_and_does_not_bridge_long_or_impossible_gaps():
    meta=dict(fps=30,width=960,height=540,n_frames=1000)
    tracks={str(i):point(i,100+i) for i in range(0,100,4)}
    assert len(plan_review(tracks,meta))==6
    assert not plan_review({'0':point(0,0),'20':point(20,20)},meta)
    assert not plan_review({'0':point(0,0),'2':point(2,500)},meta)
    assert not plan_review({},meta)

def test_repair_requires_detector_evidence_and_preserves_original(tmp_path):
    meta=dict(fps=30,width=320,height=180,n_frames=6)
    tracks={'1':point(1,100),'4':point(4,130)}
    (tmp_path/'frames').mkdir()
    for f in range(6):cv2.imwrite(str(tmp_path/'frames'/f'f_{f+1:05d}.jpg'),np.full((180,320,3),f*30,np.uint8))
    class Model:
        frames=deque();last=None
        def detect(self,im,pts):
            f=round(pts*30/1000)
            return SimpleNamespace(candidates=[SimpleNamespace(center=(100+(f-1)*10-36,64),confidence=.8)])
    def associate(cache,*args):return {f:{**v[0], 'frame':int(f),'status':'observed','segment':2} for f,v in cache.items()}
    identity=lambda x,*args:x
    result,audit=recover(tmp_path,{},tracks,{},meta,Model(),associate,identity,identity)
    assert audit['accepted_points']==2
    assert result['1']==tracks['1'] and result['4']==tracks['4']
    assert result['2']['source']=='tracknet_crop_review'
    assert result['2']['segment']==1
    rejected,info=recover(tmp_path,{},tracks,{},meta,Model(),lambda *a:{},identity,identity)
    assert rejected==tracks and info['accepted_points']==0 and info['rejected_points']==2
