from collections import deque
import numpy as np
import torch
from app.vision.tracknet import TrackNetDetector, heatmap_candidates


def test_heatmap_mapping_and_empty():
    heat=np.zeros((360,640),np.uint8)
    assert heatmap_candidates(heat,1280,720)==[]
    heat[100:103,200:203]=255
    c=heatmap_candidates(heat,1280,720)[0]
    assert c.center==(403,203)
    assert c.source=='tracknet'


def test_temporal_order_and_gap_reset():
    class Fake:
        def to(self, device): return self
        def __call__(self,x):
            assert torch.allclose(x[0,:,0,0],torch.tensor([30]*3+[20]*3+[10]*3)/255)
            return torch.zeros(1,256,1).expand(1,256,360*640)
    d=TrackNetDetector.__new__(TrackNetDetector)
    d.model=Fake();d.device='cpu';d.policy='cpu';d.frames=deque(maxlen=3);d.last=None;d.threshold=.5
    for t,v in [(0,10),(33,20),(66,30)]:
        r=d.detect(np.full((20,30,3),v,np.uint8),t)
        assert not r.candidates
    assert r.attempts==[{'device':'cpu','ok':True}]
    assert not d.detect(np.zeros((20,30,3),np.uint8),500).attempts
    assert len(d.frames)==1


def test_missing_weights(tmp_path):
    import pytest
    with pytest.raises(RuntimeError,match='权重缺失'):
        TrackNetDetector(tmp_path/'missing.pt')
