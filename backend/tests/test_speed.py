import pytest
from app.vision.speed import speed_series


def frame(t, raw=True, track=1, point=None):
    return {'frame_index':t,'source_pts_ms':t,'ball':{'selected':{'raw':raw,'track_id':track,'point':point or [t*.3,t*.4]}}}


def test_speed_uses_irregular_timestamps_without_events():
    r=speed_series([frame(t) for t in (0,17,51,90)])
    assert r['samples'][-1]['speed_px_s']==pytest.approx(500)


def test_prediction_and_identity_break_speed():
    r=speed_series([frame(0),frame(20),frame(40,False),frame(60),frame(80,track=2)])
    assert r['summary']['valid']==0
    assert r['samples'][2]['reason']=='predicted_only'
    assert r['samples'][-1]['reason']=='track_break'


def test_long_gap_and_no_ball_are_safe():
    r=speed_series([frame(0),frame(20),frame(200),{'frame_index':201,'source_pts_ms':201,'ball':{'selected':None}}])
    assert r['summary']['valid']==0
    assert r['samples'][-1]['reason']=='no_observation'


def test_unstable_positions_reject_speed():
    r=speed_series([frame(0,point=[0,0]),frame(20,point=[200,200]),frame(40,point=[1,1])])
    assert r['samples'][-1]['reason']=='unstable_fit'
