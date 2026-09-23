"""The upload-to-analysis handoff must survive closing the browser."""
import pytest
from app.store import db
from app.worker.main import handle_import
from app.video import probe, proxy

@pytest.mark.parametrize('automatic', [True, False])
def test_import_enqueues_only_after_success(tmp_path, monkeypatch, automatic):
    monkeypatch.setenv('TENNIS_DATA_DIR',str(tmp_path));db.init_db()
    monkeypatch.setattr(probe,'probe_video',lambda p:dict(width=160,height=90,duration_ms=1000,nominal_fps=30))
    monkeypatch.setattr(probe,'file_hash',lambda p:'hash')
    monkeypatch.setattr(proxy,'transcode_proxy',lambda *a,**kw:{})
    with db.connect() as c:
        c.execute("INSERT INTO projects(id,created_at,original_filename,original_path,video_hash,size_bytes) VALUES ('p',0,'test.mp4',?,'hash',1)",(str(tmp_path/'test.mp4'),))
        job=dict(db.create_job(c,jtype='import',project_id='p',payload={}))
        payload=dict(project_id='p',auto_analyze=automatic)
        result=handle_import(payload,job,c)
        # Replaying import must not spawn duplicate analyses.
        handle_import(payload,job,c)
        jobs=c.execute("SELECT * FROM jobs WHERE type='expert'").fetchall()
        assert len(jobs)==int(automatic)
        if automatic:
            assert result['analysis_job_id']==jobs[0]['id']
            assert db.uj(jobs[0]['payload_json'])==dict(action='automatic',source=None,spec={'hand':'auto','analysis_mode':'local'})


def test_failed_import_never_enqueues_analysis(tmp_path,monkeypatch):
    monkeypatch.setenv('TENNIS_DATA_DIR',str(tmp_path));db.init_db()
    def fail(*args):raise ValueError('bad media')
    monkeypatch.setattr(probe,'probe_video',fail)
    with db.connect() as c:
        c.execute("INSERT INTO projects(id,created_at,original_filename,original_path,video_hash,size_bytes) VALUES ('p',0,'bad.mp4','bad.mp4','hash',1)")
        job=dict(db.create_job(c,jtype='import',project_id='p',payload={}))
        with pytest.raises(ValueError):handle_import(dict(project_id='p',auto_analyze=True),job,c)
        assert not c.execute("SELECT 1 FROM jobs WHERE type='expert'").fetchone()
