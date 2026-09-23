import io
import json
import zipfile
from pathlib import Path
import pytest
from fastapi import HTTPException
from app.model_connections import skills, service, secret_store
from app.services import external_analysis as ext, analysis_mode, expert
from app.store import db
from app.model_providers.base import ReviewResponse


def archive(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for k,v in files.items():z.writestr(k,v)
    return stream.getvalue()


def test_skill_package_is_data_only():
    v=skills.parse_upload('s.zip',archive({'s/SKILL.md':'Observe evidence','s/references/r.md':'Rules','s/scripts/a.py':'raise RuntimeError()'}))
    assert 'Rules' in v['text'] and 'RuntimeError' not in v['text']
    assert len(v['documents'])==2
    assert len(skills.load()['documents'])==9


@pytest.mark.parametrize('files',[{'../SKILL.md':'x'},{'/SKILL.md':'x'},{'readme.md':'x'}, {'a/SKILL.md':'a','b/SKILL.md':'b'}])
def test_skill_archive_invalid(files):
    with pytest.raises(ValueError):skills.parse_upload('s.zip',archive(files))


def test_skill_size_and_path_validation():
    with pytest.raises(ValueError):skills.parse_upload('s.md',b'a'*(skills.MAX_TEXT_BYTES+1))
    with pytest.raises(ValueError):skills.load('../secrets')
    with pytest.raises(ValueError):skills.parse_upload('s.md',b'\xff')


def test_coordinates_and_frame_ids_are_validated():
    valid={'frames':[{'frame':0,'ball':[.1,.2]}]}
    assert ext.parse_result(json.dumps(valid),[0]).frames[0].ball==[.1,.2]
    for data in [{'frames':[{'frame':1}]},{'frames':[{'frame':0},{'frame':0}]},
                 {'frames':[{'frame':0,'ball':[2,.2]}]},
                 {'frames':[{'frame':0,'people':[{'label':'near','box':[.5,.2,.1,.3]}]}]},
                 {'frames':[{'frame':0,'speed_kmh':120}]}]:
        with pytest.raises(ValueError):ext.parse_result(json.dumps(data),[0])


@pytest.fixture
def environment(tmp_path,monkeypatch):
    monkeypatch.setenv('TENNIS_DATA_DIR',str(tmp_path));db.init_db()
    from app.model_providers.fake import FakeProvider
    monkeypatch.setattr(service,'_adapter_for',lambda *a:FakeProvider())
    store=secret_store.MemoryStore()
    monkeypatch.setattr(secret_store,'default_store',lambda:store)
    out=service.probe('glm','coding_plan','fake-test-key',store=store)
    cid=service.create_from_probe(out['probe_id'],enable=True,store=store)['id']
    pid='a'*32
    folder=tmp_path/'projects'/pid;folder.mkdir(parents=True)
    (folder/'proxy.mp4').write_bytes(b'test')
    with db.connect() as conn:
        conn.execute("INSERT INTO projects(id,created_at,original_filename,original_path,video_hash,size_bytes,duration_ms,proxy_status) VALUES (?,0,'test.mp4',?,'hash',1,1000,'ready')",(pid,str(folder/'original.mp4')))
    return pid,cid,store


def test_external_enqueued_without_local_source_and_skill_frozen(environment,monkeypatch):
    pid,cid,store=environment
    selected=skills.save('custom.md',b'Custom evidence rules')
    service.patch(cid,expected_revision=1,skill_id=selected['id'],store=store)
    job=expert.enqueue(pid,{'action':'automatic','spec':{'analysis_mode':'external','connection_id':cid}})
    with db.connect() as conn:
        row=conn.execute('SELECT * FROM jobs WHERE id=?',(job['job_id'],)).fetchone()
    spec=db.uj(row['payload_json'])
    assert spec['source'] is None
    assert spec['spec']['skill_hash']==selected['id']
    assert spec['spec']['connection_revision']==2
    def route(*args):return {'routed':True}
    monkeypatch.setattr(ext,'execute',route)
    with db.connect() as conn:
        assert expert.execute(spec,dict(row),conn)=={'routed':True}
    with pytest.raises(HTTPException):analysis_mode.snapshot('hybrid',cid)
    with pytest.raises(HTTPException):analysis_mode.snapshot('external','missing')
    assert analysis_mode.snapshot('local',cid)=={'analysis_mode':'local'}


def test_external_end_to_end_without_local_detectors(environment,monkeypatch,tmp_path):
    pid,cid,store=environment
    selected=skills.save('custom.md',b'Custom evidence rules')
    service.patch(cid,expected_revision=1,skill_id=selected['id'],store=store)
    jid=expert.enqueue(pid,{'action':'automatic','spec':{'analysis_mode':'external','connection_id':cid}})['job_id']
    calls=[]
    class Provider:
        def review_images(self,key,request):
            calls.append(request)
            assert 'Custom evidence rules' in request.prompt
            return ReviewResponse(ok=True,content=json.dumps({'frames':[{'frame':int(f),'people':[{'label':'near','box':[.1,.1,.8,.9]}],'ball':[.4,.5],'hit':True} for f in request.frame_ids],'observations':'看到了球员','advice':['保持准备姿态']}))
    monkeypatch.setattr(service,'_adapter_for',lambda *a:Provider())
    def prep(work,job,script,*args):
        assert script in ('video_prep.py','render_automatic.py')
        if script=='video_prep.py':
            from PIL import Image
            (work/'frames').mkdir()
            for f in range(8):Image.new('RGB',(160,90),'green').save(work/'frames'/f'f_{f+1:05d}.jpg')
            (work/'meta.json').write_text(json.dumps({'width':160,'height':90,'n_frames':8,'fps':8,'audio':False}))
        else:
            from app.expert_engine.render_automatic import render
            render(work,expert.chinese_font())
    monkeypatch.setattr(ext,'run_script',prep)
    with db.connect() as conn:
        row=dict(conn.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone())
        result=expert.execute(db.uj(row['payload_json']),row,conn)
    work=expert.session(pid,jid)
    assert result['analysis_mode']=='external' and 'annotated.mp4' in result['files']
    assert len(calls)==1 and len(calls[0].frame_ids)==4
    assert json.loads((work/'pose.json').read_text())=={}
    assert '不可测' in (work/'report.md').read_text()
    assert not (work/'ball_cand.json').exists()
    import av
    with av.open(str(work/'annotated.mp4')) as c:
        assert len(list(c.decode(video=0)))==8


def test_changed_connection_and_cancel_stop_external(environment):
    pid,cid,store=environment
    spec=analysis_mode.snapshot('external',cid)
    service.patch(cid,expected_revision=1,enabled=False,store=store)
    with pytest.raises(RuntimeError):ext.connection(spec)
    with db.connect() as conn:
        job=dict(db.create_job(conn,jtype='expert',project_id=pid,payload={}))
        conn.execute('UPDATE jobs SET cancel_requested=1 WHERE id=?',(job['id'],))
    with pytest.raises(db.CancelledError):ext.check(job)


def test_skill_http_upload_requires_session_and_patch_revision(environment):
    from fastapi.testclient import TestClient
    from app.api.main import app
    pid,cid,_=environment
    # Do not enter lifespan: avoid spawning a worker in isolated test storage.
    client=TestClient(app,base_url='http://127.0.0.1:8765')
    token=client.get('/api/model-connections').json()['session_token']
    assert client.post('/api/model-connections/skills',files={'file':('skill.md',b'Rules')}).status_code==403
    reply=client.post('/api/model-connections/skills',headers={'X-Session-Token':token},files={'file':('skill.md',b'Rules')})
    assert reply.status_code==200
    reply=client.patch(f'/api/model-connections/{cid}',headers={'X-Session-Token':token},json={'expected_revision':1,'skill_id':reply.json()['id']})
    assert reply.status_code==200 and reply.json()['config_revision']==2
    reply=client.patch(f'/api/model-connections/{cid}',headers={'X-Session-Token':token},json={'expected_revision':1,'skill_id':'default'})
    assert reply.status_code==409


def test_external_failure_never_publishes_or_falls_back(environment,monkeypatch):
    pid,cid,_=environment
    jid=expert.enqueue(pid,{'action':'automatic','spec':{'analysis_mode':'external','connection_id':cid}})['job_id']
    class Provider:
        def review_images(self,*args):return ReviewResponse(ok=False,error_code='plan_limit_reached',error_message='额度上限')
    monkeypatch.setattr(service,'_adapter_for',lambda *a:Provider())
    def prepare(work,job,script,*args):
        assert script=='video_prep.py' # No local detector or renderer may run.
        (work/'meta.json').write_text(json.dumps({'width':160,'height':90,'n_frames':1,'fps':30}))
    monkeypatch.setattr(ext,'run_script',prepare)
    monkeypatch.setattr(ext,'make_sheet',lambda *a:'data:image/jpeg;base64,dGVzdA==')
    with db.connect() as conn:
        row=dict(conn.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone())
        with pytest.raises(RuntimeError,match='未切换到本地'):
            expert.execute(db.uj(row['payload_json']),row,conn)
    work=expert.session(pid,jid)
    assert not (work/'annotated.mp4').exists() and not (work/'auto_summary.json').exists()


def test_default_execution_prompt_is_compact_but_custom_skill_is_preserved():
    default=skills.load()
    text=skills.execution_text(default)
    assert len(text.encode()) < 6000
    assert len(text.encode()) < len(default['text'].encode()) / 20
    assert '不可测' in text and '归属' in text
    custom=skills.parse_upload('s.md',b'Custom user instructions')
    assert skills.execution_text(custom)==custom['text']
    assert ext.BATCH_SIZE==4 and ext.REQUEST_TIMEOUT_S==180


def test_equivalent_ball_object_is_normalized_without_guessing():
    result=ext.parse_result(json.dumps({'frames':[{'frame':32,'ball':{'x':.42,'y':.63}},{'frame':40,'ball':None}]}),[32,40])
    assert result.frames[0].ball==[.42,.63]
    assert result.frames[1].ball is None
    for ball in [{'x':1.5,'y':.2},{'x':.1},{'point':[.1,.2]},'unknown',{'x':.1,'y':.2,'units':'pixels'}]:
        with pytest.raises(ValueError):ext.parse_result(json.dumps({'frames':[{'frame':32,'ball':ball}]}),[32])


def test_numeric_time_echo_is_ignored_but_unknown_fields_stay_forbidden():
    result=ext.parse_result(json.dumps({'frames':[{'frame':320,'time_s':10.667,'ball':None}]}),[320])
    assert result.frames[0].frame==320
    assert 'time_s' not in result.frames[0].model_dump()
    for invalid in (-1, float('inf'), '10.667', True):
        with pytest.raises(ValueError):
            ext.parse_result(json.dumps({'frames':[{'frame':320,'time_s':invalid}]}),[320])
    with pytest.raises(ValueError):
        ext.parse_result(json.dumps({'frames':[{'frame':320,'confidence':.9}]}),[320])
