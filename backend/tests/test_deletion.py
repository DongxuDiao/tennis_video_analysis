import sqlite3
from pathlib import Path
import pytest
from fastapi import HTTPException
from app import config
from app.store import db
from app.services import deletion


@pytest.fixture
def project(tmp_path,monkeypatch):
    monkeypatch.setenv('TENNIS_DATA_DIR',str(tmp_path))
    db.init_db()
    pid='a'*32
    d=config.projects_dir()/pid
    d.mkdir();(d/'original.mp4').write_bytes(b'original')
    with db.connect() as c:
        c.execute("INSERT INTO projects(id,created_at,original_filename,original_path,video_hash,size_bytes) VALUES (?,0,'test',?,'hash',8)",(pid,str(d/'original.mp4')))
    return pid,d


def test_trash_restore_and_purge(project):
    pid,d=project
    with pytest.raises(HTTPException):deletion.mutate(pid,'purge')
    deletion.mutate(pid,'trash')
    assert d.exists()
    with db.connect() as c:
        with pytest.raises(sqlite3.IntegrityError):
            db.create_job(c,jtype='import',payload={},project_id=pid)
    deletion.mutate(pid,'restore')
    deletion.mutate(pid,'trash');deletion.mutate(pid,'purge')
    assert not d.exists()
    with db.connect() as c:assert c.execute('select count(*) from projects').fetchone()[0]==0


def test_active_job_blocks_delete(project):
    pid,d=project
    with db.connect() as c:db.create_job(c,jtype='import',payload={},project_id=pid)
    with pytest.raises(HTTPException) as e:deletion.mutate(pid,'trash')
    assert e.value.status_code==409 and d.exists()


def test_staging_rollback_restores_files(project):
    pid,d=project
    with pytest.raises(RuntimeError):
        with deletion.cleanup_transaction():
            deletion.remove(d)
            raise RuntimeError('db failure')
    assert (d/'original.mp4').read_bytes()==b'original'


def test_run_dependencies_and_export_cleanup(project):
    pid,d=project
    r1,r2,eid='b'*32,'c'*32,'d'*32
    for rid in (r1,r2):
        p=d/'runs'/rid;p.mkdir(parents=True);(p/'frames.json').write_text('[]')
    ep=config.exports_root()/eid;ep.mkdir();(ep/'out.mp4').write_bytes(b'export')
    with db.connect() as c:
        for rid,parent in ((r1,None),(r2,r1)):
            c.execute("INSERT INTO runs(id,project_id,status,config_json,segment_start_ms,segment_end_ms,created_at,parent_run_id,reuses_frames_from) VALUES (?,?,'succeeded','{}',0,1000,0,?,?)",(rid,pid,parent,parent))
        c.execute("INSERT INTO exports(id,run_id,view,created_at) VALUES (?,?,'automatic',0)",(eid,r2))
    with pytest.raises(HTTPException):deletion.mutate(pid,'run',r1)
    deletion.mutate(pid,'run',r2)
    assert not ep.exists() and not (d/'runs'/r2).exists()
    assert (d/'original.mp4').exists()
    deletion.mutate(pid,'run',r1)
