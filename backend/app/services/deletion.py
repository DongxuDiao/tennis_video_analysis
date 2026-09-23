"""Explicit project trash and retryable file-first cleanup under a DB write lock."""
import shutil
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from fastapi import HTTPException
from app import config
from app.store import db


def project(conn, pid):
    row = conn.execute('SELECT * FROM projects WHERE id=?', (pid,)).fetchone()
    if row is None:
        raise HTTPException(404, '项目不存在')
    return row


def idle(conn, pid):
    active = conn.execute("SELECT id FROM jobs WHERE status IN ('queued','running') AND (project_id=? OR run_id IN (SELECT id FROM runs WHERE project_id=?) OR export_id IN (SELECT e.id FROM exports e JOIN runs r ON e.run_id=r.id WHERE r.project_id=?))", (pid,pid,pid)).fetchall()
    if active:
        raise HTTPException(409, '项目有运行或排队任务，请先取消并等待任务结束')


def safe_path(root, ident):
    if not ident or any(c not in '0123456789abcdef' for c in ident):
        raise HTTPException(409, '文件标识无效，已停止清理')
    p = root / ident
    if p.is_symlink() or p.resolve().parent != root.resolve():
        raise HTTPException(409, '文件路径异常，已停止清理')
    return p


_staged = ContextVar('deletion_staged', default=None)


@contextmanager
def cleanup_transaction():
    staged = []
    token = _staged.set(staged)
    try:
        yield
    except BaseException:
        for original, temporary in reversed(staged):
            if temporary.exists():
                temporary.rename(original)
        raise
    else:
        for _, temporary in staged:
            # Committed resources stay hidden even if physical cleanup fails.
            # Leave isolated leftovers available for later disk maintenance.
            try:
                shutil.rmtree(temporary)
            except OSError:
                pass
    finally:
        _staged.reset(token)


def remove(p):
    if p.exists():
        root = config.data_dir() / 'deletion_cleanup'
        root.mkdir(exist_ok=True)
        temporary = root / uuid.uuid4().hex
        p.rename(temporary)
        _staged.get().append((p, temporary))


def export_delete(c, eid):
    remove(safe_path(config.exports_root(), eid))
    c.execute('DELETE FROM jobs WHERE export_id=?', (eid,))
    c.execute('DELETE FROM exports WHERE id=?', (eid,))


def run_delete(c, rid, pid):
    if c.execute('SELECT 1 FROM runs WHERE reuses_frames_from=? OR parent_run_id=?', (rid,rid)).fetchone():
        raise HTTPException(409, '其他分析依赖此版本，请先删除依赖版本')
    for e in c.execute('SELECT id FROM exports WHERE run_id=?',(rid,)).fetchall():
        export_delete(c,e['id'])
    remove(safe_path(safe_path(config.projects_dir(),pid) / 'runs',rid))
    for table in ('corrections','evaluations','jobs'):
        c.execute(f'DELETE FROM {table} WHERE run_id=?',(rid,))
    c.execute('DELETE FROM runs WHERE id=?',(rid,))


def mutate(pid, action, ident=None):
    with cleanup_transaction(), db.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        project(c,pid)
        idle(c,pid)
        trashed = c.execute('SELECT 1 FROM project_trash WHERE project_id=?',(pid,)).fetchone()
        if action == 'trash':
            c.execute('INSERT OR IGNORE INTO project_trash VALUES (?,?)',(pid,time.time()))
        elif action == 'restore':
            c.execute('DELETE FROM project_trash WHERE project_id=?',(pid,))
        elif action == 'purge':
            if not trashed:
                raise HTTPException(409,'请先将项目移入回收站')
            for e in c.execute('SELECT e.id FROM exports e JOIN runs r ON e.run_id=r.id WHERE r.project_id=?',(pid,)).fetchall():
                export_delete(c,e['id'])
            remove(safe_path(config.projects_dir(),pid))
            for table in ('corrections','evaluations','jobs'):
                c.execute(f'DELETE FROM {table} WHERE run_id IN (SELECT id FROM runs WHERE project_id=?)',(pid,))
            c.execute('DELETE FROM evaluations WHERE gt_set_id IN (SELECT id FROM ground_truth_sets WHERE project_id=?)',(pid,))
            for table in ('jobs','runs','calibrations','ground_truth_sets'):
                c.execute(f'DELETE FROM {table} WHERE project_id=?',(pid,))
            c.execute('DELETE FROM project_trash WHERE project_id=?',(pid,))
            c.execute('DELETE FROM projects WHERE id=?',(pid,))
        elif action in ('run','export'):
            if action == 'run':
                if not c.execute('SELECT 1 FROM runs WHERE id=? AND project_id=?',(ident,pid)).fetchone():
                    raise HTTPException(404)
                run_delete(c,ident,pid)
            else:
                if not c.execute('SELECT 1 FROM exports e JOIN runs r ON e.run_id=r.id WHERE e.id=? AND r.project_id=?',(ident,pid)).fetchone():
                    raise HTTPException(404)
                export_delete(c,ident)
        else:
            raise HTTPException(400)
    cleanup = config.data_dir() / 'deletion_cleanup'
    return {'ok':True, 'cleanup_pending': cleanup.exists() and any(cleanup.iterdir())}


def storage(pid):
    with db.connect() as c:
        project(c,pid)
        runs=[dict(r) for r in c.execute('SELECT id,status,parent_run_id,reuses_frames_from FROM runs WHERE project_id=? ORDER BY created_at DESC',(pid,))]
        exports=[dict(r) for r in c.execute('SELECT e.id,e.run_id,e.status FROM exports e JOIN runs r ON e.run_id=r.id WHERE r.project_id=?',(pid,))]
        paths=[safe_path(config.projects_dir(),pid)]+[safe_path(config.exports_root(),e['id']) for e in exports]
        size=sum(f.stat().st_size for p in paths if p.exists() for f in p.rglob('*') if f.is_file() and not f.is_symlink())
        jobs=[dict(r) for r in c.execute("SELECT id,type,status FROM jobs WHERE status IN ('running','queued') AND (project_id=? OR run_id IN (SELECT id FROM runs WHERE project_id=?))",(pid,pid))]
    return dict(bytes=size,runs=runs,exports=exports,jobs=jobs)
