"""Freeze the selected analysis route before enqueueing; never silently fall back."""
from fastapi import HTTPException
from app.store import db
from app.model_connections import skills


def snapshot(mode='local', connection_id=None):
    if mode not in ('local', 'external'):
        raise HTTPException(422, '请选择本地分析或外部模型分析')
    if mode == 'local':
        return {'analysis_mode': 'local'}
    with db.connect() as conn:
        row = conn.execute('SELECT * FROM model_connections WHERE id=? AND enabled=1', (connection_id,)).fetchone()
    if row is None:
        raise HTTPException(422, '请先连接并选择可用的外部模型')
    caps = db.uj(row['capabilities_json']) or {}
    if not caps.get('vision_ok'):
        raise HTTPException(422, '所选模型尚未通过图片能力测试')
    skill_id = caps.get('skill_id', 'default')
    skill = skills.load(skill_id)
    return {'analysis_mode': 'external', 'connection_id': row['id'],
            'connection_revision': row['config_revision'], 'skill_id': skill_id, 'skill_hash': skill['id']}
