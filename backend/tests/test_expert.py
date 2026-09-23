import pytest
from app.services.expert import validate, report
from pathlib import Path
import json
META={'width':960,'height':540,'n_frames':1050}

def spec(points):return {'shots':[{'id':1,'side':'FH','tc':60.5,'outgoing':points}]}

def test_points_contract():
    validate(spec([[61,300,200],[62,305,180],[64,320,150]]),META)
    for points in [[[60,300,200]],[[61,300,200],[61,301,201]],[[61,960,200]],[[61,float('nan'),200]]]:
        with pytest.raises(ValueError):validate(spec(points),META)

def test_report_excludes_weak_and_missing(tmp_path):
    (tmp_path/'shots.json').write_text(json.dumps([{'id':1,'kmh':80,'conf':'medium'},{'id':2,'kmh':140,'conf':'low'},{'id':3,'kmh':None,'conf':'none'}]))
    report(tmp_path)
    text=(tmp_path/'report.md').read_text()
    assert '可测速度 1 拍' in text and '均值 80.0' in text and '最大 80.0' in text

def test_topspin_lowers_height():
    from app.expert_engine.flight_fit import Physics
    p=Physics();_,a=p.traj([0,12,1.5],[0,-22,6],.5);_,b=p.traj([0,12,1.5],[0,-22,6],.5,cl=p.cl_of_rpm(2000))
    assert b[-1,2]<a[-1,2]
