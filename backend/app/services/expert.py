"""Local reviewed-shot workflow adapted from the user supplied motion-analysis skill.
All heavy work runs in the existing cancellable worker. Each job is an immutable revision.
"""
from pathlib import Path
import json, math, os, re, shutil, signal, subprocess, sys, time
from fastapi import HTTPException
from app import config
from app.store import db
from app.services.deletion import safe_path, idle, project

ENGINE=Path(__file__).resolve().parents[1]/'expert_engine'
FILES={'external_people.json','external_requests.json','external_observations.json','scene_quality.json','person_quality.json','people.json','quality_review.json','meta.json','impacts.json','ball_cand.json','pose.json','court_lines.json','court_lines.png','bg_median.png','calib.json','calib_overlay.png','shots.json','technique.json','report.md','annotated.mp4','input.json','ident.json','shots_input.json','run.log','verification.json','auto_candidates.json','auto_tracks.json','auto_summary.json','enhancement_summary.json','evidence'}

def root(pid): return safe_path(config.projects_dir(),pid)/'expert'
def session(pid,jid): return safe_path(root(pid),jid)
def read(path,default=None): return json.loads(path.read_text()) if path.is_file() else default

def validate(spec,meta=None):
    if not isinstance(spec,dict): raise ValueError('输入必须为 JSON 对象')
    shots=spec.get('shots',[])
    if not isinstance(shots,list) or len(shots)>100: raise ValueError('最多100拍')
    ids=set()
    for s in shots:
        if not isinstance(s,dict) or not isinstance(s.get('id'),int) or s['id'] in ids: raise ValueError('每拍必须有唯一整数 id')
        ids.add(s['id'])
        if s.get('side') not in ('FH','BH','UNK'): raise ValueError('side 必须为 FH/BH/UNK')
        tc=s.get('tc')
        if not isinstance(tc,(int,float)) or not math.isfinite(tc) or tc<0 or (meta and tc>=meta['n_frames']):raise ValueError('触球帧无效')
        for key in ('incoming','outgoing'):
            points=s.get(key,[])
            if not isinstance(points,list) or len(points)>300:raise ValueError('每段最多300个点')
            prev=-1
            for p in points:
                if not isinstance(p,list) or len(p)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in p):raise ValueError('球点格式 [帧,x,y]')
                f,x,y=p
                if f!=int(f) or f<=prev or f<0:raise ValueError('球点帧号必须递增且不重复')
                if (key=='incoming' and f>=tc) or (key=='outgoing' and f<=tc):raise ValueError('来球在触球前，出球在触球后')
                if meta and not (f<meta['n_frames'] and 0<=x<meta['width'] and 0<=y<meta['height']):raise ValueError('球点越界')
                prev=f
        contact=s.get('contact')
        if contact is not None and (not isinstance(contact,list) or len(contact)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in contact) or not 0<contact[2]<4):raise ValueError('击球点为世界坐标[x,y,z]米，高度需0–4m')
    return spec


def enqueue(pid,body):
    if not isinstance(body.get('spec',{}),dict):raise HTTPException(422,'spec 必须为对象')
    from app.services.analysis_mode import snapshot
    route = snapshot(body.get('spec', {}).get('analysis_mode', 'local'), body.get('spec', {}).get('connection_id')) if body.get('action') == 'automatic' else {}
    action=body.get('action')
    if action not in ('automatic','prepare','calibrate','fit','render'):raise HTTPException(422,'未知精细分析操作')
    with db.connect() as c:
        c.execute('BEGIN IMMEDIATE');p=project(c,pid);idle(c,pid)
        if c.execute('SELECT 1 FROM project_trash WHERE project_id=?',(pid,)).fetchone():raise HTTPException(409,'先恢复项目')
        source=None if action in ('prepare','automatic') else body.get('source')
        if action=='automatic' and route['analysis_mode']=='local':
            # Reuse only immutable extraction artifacts from this project's successful jobs.
            prior=c.execute("SELECT id FROM jobs WHERE project_id=? AND type='expert' AND status='succeeded' ORDER BY created_at DESC",(pid,)).fetchall()
            source=next((r['id'] for r in prior if all((session(pid,r['id'])/n).exists() for n in ('meta.json','pose.json','ball_cand.json','impacts.json','frames','person_quality.json','scene_quality.json','people.json'))),None)
        if action not in ('prepare','automatic'):
            if not source:raise HTTPException(422,'先准备素材')
            j=c.execute("SELECT * FROM jobs WHERE id=? AND project_id=? AND type='expert' AND status='succeeded'",(source,pid)).fetchone()
            if not j:raise HTTPException(422,'来源版本不存在或未成功')
            meta=read(session(pid,source)/'meta.json')
        else:
            meta=None
            if (p['duration_ms'] or 0)>60000:raise HTTPException(422,'精细分析首版限60秒')
        if action=='automatic':
            body={**body,'spec':{'hand':body.get('spec',{}).get('hand','auto'), **route}}
            if body['spec']['hand'] not in ('auto','right','left'):raise HTTPException(422,'持拍手无效')
        try: validate(body.get('spec',{}),meta)
        except ValueError as e:raise HTTPException(422,str(e))
        if action=='fit' and not body.get('spec',{}).get('reviewed'):raise HTTPException(422,'请先逐拍确认球点与触球时刻')
        if action=='fit' and body.get('spec',{}).get('hand','right') not in ('right','left'):raise HTTPException(422,'人工拟合请明确选择右手或左手')
        row=db.create_job(c,jtype='expert',project_id=pid,payload={'action':action,'source':source,'spec':body.get('spec',{})})
        return {'job_id':row['id']}


def listing(pid):
    with db.connect() as c:
        project(c,pid)
        rows=c.execute("SELECT id,status,stage,error,result_json,progress_json,created_at FROM jobs WHERE project_id=? AND type='expert' ORDER BY created_at DESC",(pid,)).fetchall()
    return {'sessions':[{**dict(r),'result':db.uj(r['result_json']),'progress':db.uj(r['progress_json'])} for r in rows]}


def report(work):
    shots=read(work/'shots.json',[]);tech=read(work/'technique.json',[])
    ok=[s['kmh'] for s in shots if s.get('kmh') and s.get('conf') in ('high','medium')]
    lines=['# 网球精细分析报告','', '逐拍球点经复核；速度为标定及物理假设下的拟合值。条件区间不含标定/触球时刻/旋转误差。落点为外推，非实测。', '',f'共 {len(shots)} 拍；可测速度 {len(ok)} 拍；'+(f'均值 {sum(ok)/len(ok):.1f} km/h，最大 {max(ok):.1f} km/h。' if ok else '暂无可测球速。'),'', '|拍|正反手|时间s|速度km/h|置信等级|旋转带判定|说明|','|---|---|---|---|---|---|---|']
    for s in shots:lines.append(f"|{s['id']}|{s.get('side')}|{s.get('t_s')}|{s.get('kmh') if s.get('conf') in ('high','medium') else '不可测/疑似'}|{s.get('conf')}|{s.get('result_cn','未判定')}|{s.get('result_note','')}|")
    lines+=['','## 动作指标','肩线角度为投影代理量，不能解释为真实三维转体角度；视角变化、遮挡影响结果。','```json',json.dumps(tech,ensure_ascii=False,indent=2),'```']
    (work/'report.md').write_text('\n'.join(lines))


def execute(payload,job,conn):
    pid,jid=job['project_id'],job['id'];work=session(pid,jid);work.mkdir(parents=True,exist_ok=True)
    action=payload['action'];spec=payload.get('spec',{})
    if action == 'automatic' and spec.get('analysis_mode') == 'external':
        from app.services.external_analysis import execute as execute_external
        return execute_external(payload, job, conn, work)
    if payload.get('source'):
        src=session(pid,payload['source'])
        for p in src.iterdir():
            if p.is_dir():
                # 稳定化会把原 frames 挪进 source_frames/；evidence/ 为增强证据目录。
                # copy2 碰到目录会抛 IsADirectoryError，目录一律走 copytree（硬链接不占空间）。
                if p.name in ('frames','source_frames','evidence'):shutil.copytree(p,work/p.name,copy_function=os.link,dirs_exist_ok=True)
            elif action=='automatic':
                if p.name in ('scene_quality.json','person_quality.json','people.json','meta.json','pose.json','ball_cand.json','impacts.json','bg_median.png','audio.wav','auto_candidates.json','auto_summary.json'):shutil.copy2(p,work/p.name)
            elif p.name not in ('run.log','input.json','annotated.mp4'):shutil.copy2(p,work/p.name)
    (work/'input.json').write_text(json.dumps(spec,ensure_ascii=False))
    def run(script,*args):
        with (work/'run.log').open('a') as log:
            command=[sys.executable,'-u',str(ENGINE/script),*map(str,args),'--work',str(work)]
            popen_options={'stdout':log,'stderr':log}
            if sys.platform.startswith('linux') and shutil.which('setsid'):
                # Docker Desktop's amd64/Rosetta path can SIGBUS the parent
                # after fork() when the child starts PyTorch.  Let Popen use
                # posix_spawn(), while the external setsid wrapper still gives
                # cancellation a dedicated process group.
                command=[shutil.which('setsid'),*command]
                popen_options['close_fds']=False
            else:
                popen_options['start_new_session']=True
            proc=subprocess.Popen(command,**popen_options)
            try:
                with db.connect() as cc:
                    db.update_job_progress(cc,jid,stage=script,progress={})
                simple_wait = os.environ.get('TENNIS_SIMPLE_SUBPROCESS_WAIT') == '1' and \
                    script in ('pose_extract.py','automatic.py')
                if simple_wait:
                    # Docker Desktop bind mounts have crashed the polling
                    # worker with SIGBUS while a CPU model process is active.
                    # Avoid concurrent SQLite/log access for these long stages.
                    proc.wait()
                else:
                    while proc.poll() is None:
                        with db.connect() as cc:
                            db.check_cancel(cc,jid)
                            with (work/'run.log').open('rb') as stream:
                                stream.seek(max(0,stream.seek(0,2)-1000))
                                matches=re.findall(r'(?:tracknet|render|frame|review) (\d+)/(\d+)',stream.read().decode(errors='replace'))
                            progress={'frames_done':int(matches[-1][0]),'frames_total_est':int(matches[-1][1])} if matches else {}
                            db.update_job_progress(cc,jid,stage=script,progress=progress)
                        db.heartbeat(jid,job.get('worker_id') or '')
                        time.sleep(.5)
                if proc.returncode:raise RuntimeError(f'{script} 失败：'+(work/'run.log').read_text()[-1800:])
            finally:
                stop_process_group(proc)
    if action=='prepare' or (action=='automatic' and not payload.get('source')):
        p=project(conn,pid)
        from app.video.probe import probe_video
        probe=probe_video(Path(p['original_path']))
        if probe.get('is_vfr_possible'):raise RuntimeError('精细拟合暂只支持恒定帧率，请先转换视频')
        model=config.models_dir()/'yolo11m-pose.pt'
        if not model.is_file():raise RuntimeError('缺少 data/models/yolo11m-pose.pt')
        run('video_prep.py','prep','--video',p['original_path'])
        if action=='automatic':run('scene_quality.py')
        # Keep PyTorch out of the long-lived worker on Linux.  Importing it in
        # the amd64 worker before spawning the inference process can crash the
        # worker under Apple Silicon's Docker/Rosetta emulation (SIGBUS).  The
        # model script owns the heavy runtime; only macOS needs an MPS probe.
        device = 'cpu'
        if sys.platform == 'darwin':
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
        run('pose_extract.py','--model',model,'--device',device,'--batch',1)
        meta=read(work/'meta.json')
        if meta.get('audio'):run('audio_impacts.py','--top',25)
        else:(work/'impacts.json').write_text('{"peaks":[]}')
        run('court_calib.py','detect','--skip-top',.5)
        run('ball_detect.py','--row-min',int(meta['height']*.25))
    if action=='automatic':
        model=config.models_dir()/'tracknet_tennis.pt'
        if not model.exists():raise RuntimeError('缺少 TrackNet 权重 data/models/tracknet_tennis.pt')
        run('automatic.py','--model',model,'--hand',spec.get('hand','auto'))
        (work/'enhancement_summary.json').write_text(json.dumps({'status':'disabled','source_line':'本地分析'},ensure_ascii=False))
        run('render_automatic.py','--font',chinese_font())
    elif action=='calibrate':
        ident=spec.get('ident',{})
        if len(ident.get('manual',{}))<4:raise RuntimeError('请标记至少4条场地线')
        for name, pts in ident['manual'].items():
            if not isinstance(pts,list) or len(pts)!=2 or any(not isinstance(p,list) or len(p)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in p) for p in pts):raise RuntimeError('标定线必须包含两个有限坐标端点')
        for name in ('shots.json','technique.json','report.md','shots_input.json','auto_summary.json','auto_tracks.json','auto_candidates.json'):
            (work/name).unlink(missing_ok=True)
        (work/'ident.json').write_text(json.dumps(ident))
        run('court_calib.py','fit','--ident',work/'ident.json','--trials',80)
        run('court_calib.py','verify')
        # Human line identity verification is still necessary; RMS alone is insufficient.
    elif action=='fit':
        if not (work/'calib.json').exists():raise RuntimeError('先完成球场标定')
        if not spec.get('calibration_verified') or not spec.get('scale_constraint'):raise RuntimeError('需确认标定叠加、独立尺度约束和镜头稳定性')
        if not spec.get('camera_static'):raise RuntimeError('镜头移动时需按镜头段单独标定，当前不支持移动机位三维反演')
        calib=read(work/'calib.json');ident=read(work/'ident.json',{})
        if not math.isfinite(calib.get('rms',float('inf'))) or calib['rms']>3:raise RuntimeError('标定 RMS 超过3px，先重新标定')
        if 'far_baseline' not in ident.get('manual',{}) and not ident.get('player_height'):raise RuntimeError('需标记远端底线或提供已知身高约束，不能仅勾选尺度确认')
        meta=read(work/'meta.json');validate(spec,meta)
        clean={'fps':meta['fps'],'shots':spec.get('shots',[])}
        (work/'shots_input.json').write_text(json.dumps(clean))
        run('flight_fit.py','--bootstrap',20)
        run('pose_metrics.py','--hand',spec.get('hand','right'))
        # JSON must remain valid when pose landmarks are unavailable.
        from app.vision.pipeline import _sanitize
        for name in ('shots.json','technique.json'):
            (work/name).write_text(json.dumps(_sanitize(read(work/name)),ensure_ascii=False,allow_nan=False))
        report(work)
    elif action=='render':
        if not (work/'shots.json').exists():raise RuntimeError('先完成逐拍拟合')
        if (work/'auto_summary.json').exists():
            run('render_automatic.py','--font',chinese_font())
        else:
            run('render_annotated.py','--font',chinese_font(),'--scale',1.5,'--out','annotated.mp4')
    return {'session_id':jid,'action':action,'files':sorted(p.name for p in work.iterdir() if p.name in FILES)}


def chinese_font():
    font=next((p for p in ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Medium.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'] if Path(p).exists()),None)
    if not font:raise RuntimeError('缺少中文字体 PingFang / NotoSansCJK')
    return font


def stop_process_group(proc):
    """Reap the launcher and stop descendants even if the launcher exits on TERM."""
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()
