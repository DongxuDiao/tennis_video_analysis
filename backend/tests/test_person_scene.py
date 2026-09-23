import cv2
import numpy as np
from app.expert_engine.person_quality import select,inspect
from app.expert_engine.scene_quality import estimate

def p(x):return dict(box=[x,10,x+100,210],kpts=[[x+50,100,.9]]*17)

def test_identity_rejects_switch_and_keeps_nearby_person():
    assert select([p(400)],p(10)) is None
    assert select([p(400),p(15)],p(10))==p(15)
    clean,qc=inspect({'0':p(10),'1':p(400),'2':p(15)},3)
    assert list(clean)==['0','2'] and qc['suspicious_frames']==[1]
    assert qc['keypoint_coverage']['wrists']==2/3

def test_scene_rejects_textureless_and_recovers_small_translation():
    assert estimate(np.zeros((300,500,3),np.uint8),np.zeros((300,500,3),np.uint8)) is None
    rng=np.random.default_rng(5);im=rng.integers(0,256,(300,500,3),dtype=np.uint8)
    shifted=cv2.warpAffine(im,np.float32([[1,0,5],[0,1,3]]),(500,300))
    m=estimate(im,shifted)
    assert m is not None and np.allclose(m[:,2],[-5,-3],atol=1)

def test_event_attribution_detects_other_player_swing():
    from app.expert_engine.automatic import review_events
    import copy
    main=p(10);other=p(400);later=copy.deepcopy(other)
    later['kpts'][10]=[520,100,.9]
    poses={str(f):main for f in range(7)}
    people={'3':[main,other],'5':[main,later]}
    shots=[dict(tc=3,side='FH',conf='medium')]
    review_events(shots,poses,people,dict(n_frames=7,fps=30),dict(hand='right',event_threshold=.75))
    assert shots[0]['attribution']=='ambiguous' and shots[0]['side']=='UNK'

def test_missing_people_is_not_claimed_verified():
    from app.expert_engine.automatic import review_events
    shots=[dict(tc=3,side='FH',conf='medium')]
    review_events(shots,{}, {},dict(n_frames=7,fps=30),dict(hand='unknown',event_threshold=.75))
    assert shots[0]['attribution']=='unchecked' and shots[0]['side']=='UNK'

def test_stabilization_keeps_source_and_emits_transform(tmp_path,monkeypatch):
    from app.expert_engine import scene_quality
    import json
    frames=tmp_path/'frames';frames.mkdir()
    for i in range(4):cv2.imwrite(str(frames/f'f_{i+1:05d}.jpg'),np.full((80,120,3),100,np.uint8))
    monkeypatch.setattr(scene_quality,'estimate',lambda *a:np.float64([[1,0,4],[0,1,0]]))
    report=scene_quality.run(tmp_path)
    assert report['stabilized'] and len(report['transforms'])==4
    assert len(list((tmp_path/'source_frames').glob('*.jpg')))==4
    assert cv2.imread(str(frames/'f_00001.jpg'))[:,0].max()==0
    assert (tmp_path/'bg_median.png').exists()

def test_failed_scene_match_never_warps(tmp_path,monkeypatch):
    from app.expert_engine import scene_quality
    frames=tmp_path/'frames';frames.mkdir()
    for i in range(3):cv2.imwrite(str(frames/f'f_{i+1:05d}.jpg'),np.full((80,120,3),100,np.uint8))
    monkeypatch.setattr(scene_quality,'estimate',lambda *a:None)
    report=scene_quality.run(tmp_path)
    assert not report['stabilized'] and report['status']=='unreliable_camera_motion'
    assert not (tmp_path/'source_frames').exists()
