import json
import numpy as np
import pytest
from app.expert_engine.automatic import associate, detect_events, summarize
from app.services import expert
from app.store import db
from app import config


def candidate(x,y=120):return dict(x=x,y=y,score=.8,source='tracknet')


def test_stationary_and_singletons_are_not_tracks():
    assert not associate({str(i):[candidate(100)] for i in range(20)},30,960)
    assert not associate({'3':[candidate(100)]},30,960)


def test_short_interpolation_is_separate_from_observations_and_long_gaps_break():
    frames={str(i):[candidate(100+i*6)] for i in [0,1,2,4,5,6,20,21,22,23]}
    tracks=associate(frames,30,960)
    assert tracks['3']['status']=='interpolated'
    assert all(str(f) not in tracks for f in range(7,20))
    assert tracks['6']['segment']!=tracks['20']['segment']
    assert tracks['2']['status']=='observed'


def test_audio_alone_does_not_create_shots_or_metric_speed():
    meta=dict(n_frames=100,fps=30)
    events,info=detect_events({}, {}, [{'frame':50,'score':10}],meta)
    assert events==[] and info['hand']=='unknown'
    s=summarize({}, {}, events,meta,info)
    assert s['observed_ball_frames']==0 and not s['calibrated']


def test_shot_requires_motion_and_supporting_evidence():
    poses={}
    for f in range(100):
        k=np.zeros((17,3));k[:,2]=.9;k[5,:2]=[80,100];k[6,:2]=[120,100]
        k[9,:2]=[80,180];k[10,:2]=[140+60*np.tanh((f-50)/3),160]
        poses[str(f)]=dict(box=[50,60,200,260],kpts=k.tolist())
    events,_=detect_events(poses,{},[{'frame':50,'score':8}],dict(n_frames=100,fps=30),'right')
    assert len(events)==1 and abs(events[0]['tc']-50)<1
    assert events[0]['kmh'] is None and events[0]['result']=='UNK'
    assert set(events[0]['evidence'])=={'wrist_motion','audio'}
    assert not detect_events(poses,{},[],dict(n_frames=100,fps=30),'right')[0]


def test_automatic_enqueue_needs_no_review_or_source(tmp_path,monkeypatch):
    monkeypatch.setenv('TENNIS_DATA_DIR',str(tmp_path));db.init_db();pid='e'*32
    with db.connect() as c:
        c.execute("INSERT INTO projects(id,created_at,original_filename,original_path,video_hash,size_bytes,duration_ms) VALUES (?,0,'test','test.mp4','hash',1,1000)",(pid,))
    j=expert.enqueue(pid,{'action':'automatic','source':'untrusted','spec':{'reviewed':False,'shots':[{'malformed':True}]}})
    with db.connect() as c:
        row=c.execute('SELECT payload_json FROM jobs WHERE id=?',(j['job_id'],)).fetchone()
    payload=json.loads(row['payload_json'])
    assert payload=={'action':'automatic','source':None,'spec':{'hand':'auto','analysis_mode':'local'}}


def test_body_clutter_and_unrelated_tracklets_are_rejected():
    from app.expert_engine.automatic import filter_candidates,main_player_tracks
    k=np.zeros((17,3));k[:,2]=.9;k[5,:2]=[80,80];k[6,:2]=[120,80];k[11,:2]=[80,150];k[12,:2]=[120,150];k[9,:2]=[60,110];k[10,:2]=[140,110]
    poses={'0':dict(box=[60,50,140,250],kpts=k.tolist())}
    raw={'0':[candidate(100,100),candidate(100,200),candidate(180,100)]}
    assert filter_candidates(raw,poses)['0']==[candidate(180,100)]
    tracks={str(f):dict(frame=f,x=20+f,y=20,status='observed',segment=1,source='color_motion') for f in range(6)}
    assert not main_player_tracks(tracks,poses)


def test_render_keeps_all_video_frames_with_short_audio(tmp_path):
    import wave
    import cv2
    import av
    from app.expert_engine.render_automatic import render
    from app.services.expert import chinese_font
    from pathlib import Path
    (tmp_path/'frames').mkdir()
    for f in range(12):cv2.imwrite(str(tmp_path/'frames'/f'f_{f+1:05d}.jpg'),np.zeros((90,160,3),np.uint8))
    for name,data in {'meta.json':dict(width=160,height=90,n_frames=12,fps=30,audio='audio.wav'),'pose.json':{},'auto_tracks.json':{},'shots.json':[], 'auto_summary.json':dict(pose_coverage=0,ball_coverage=0)}.items():
        (tmp_path/name).write_text(json.dumps(data))
    with wave.open(str(tmp_path/'audio.wav'),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(48000);w.writeframes(bytes(4800*2))
    render(tmp_path,chinese_font())
    with av.open(str(tmp_path/'annotated.mp4')) as c:
        assert len(c.streams.audio)==1
        assert sum(1 for _ in c.decode(video=0))==12


def test_cancel_stops_descendants_after_launcher_exits(monkeypatch):
    import signal
    calls=[]
    monkeypatch.setattr(expert.os,'killpg',lambda pid,sig:calls.append((pid,sig)))
    class Process:
        pid=12345
        def poll(self):return None
        def wait(self,timeout=None):return 0
    expert.stop_process_group(Process())
    assert calls==[(12345,signal.SIGTERM),(12345,signal.SIGKILL)]
