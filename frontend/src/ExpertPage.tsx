import {useEffect,useRef,useState} from 'react'
import {api} from './api'
import type {Project} from './types'
import './expert.css'
const stageNames:Record<string,string>={'video_prep.py':'准备视频','pose_extract.py':'识别人体姿态','audio_impacts.py':'寻找击球声音','court_calib.py':'提取场地线','ball_detect.py':'提取小球候选','automatic.py':'时序追踪与动作分析','render_automatic.py':'生成标注视频'}
const initial={ident:{sport:'tennis',manual:{}},shots:[],hand:'right',reviewed:false,calibration_verified:false,camera_static:false,scale_constraint:''}
export function ExpertPage({project,back}:{project:Project;back:()=>void}){
 const[rows,setRows]=useState<any[]>([]),[sid,setSid]=useState(''),[meta,setMeta]=useState<any>(null),[cand,setCand]=useState<any>({}),[peaks,setPeaks]=useState<any[]>([]),[frame,setFrame]=useState(0),[err,setErr]=useState(''),[text,setText]=useState(JSON.stringify(initial,null,2)),[target,setTarget]=useState('near_baseline'),[shot,setShot]=useState(0),[kind,setKind]=useState('outgoing'),[pending,setPending]=useState<number[]|null>(null),[busy,setBusy]=useState(false),[files,setFiles]=useState<string[]>([])
 const[manual,setManual]=useState(false),[hand,setHand]=useState('auto'),[summary,setSummary]=useState<any>(null),[report,setReport]=useState('')
 const followJob=useRef('')
 const canvas=useRef<HTMLCanvasElement>(null);const base=`/api/projects/${project.id}/expert`
 const load=()=>api.get(base).then(d=>setRows(d.sessions)).catch(e=>setErr(e.message))
 useEffect(()=>{load();const t=setInterval(load,2000);return()=>clearInterval(t)},[project.id])
 useEffect(()=>{if(!sid){const done=rows.find(r=>r.status==='succeeded');if(done)setSid(done.id)}},[rows,sid])
 useEffect(()=>{if(!followJob.current){const active=rows.find(r=>['running','queued'].includes(r.status));if(active)followJob.current=active.id}const r=rows.find(r=>r.id===followJob.current);if(r?.status==='succeeded'){setSid(r.id);followJob.current=''}else if(r&&['failed','cancelled','interrupted'].includes(r.status)){followJob.current=''}},[rows])
 const url=(name:string)=>`${base}/${sid}/files/${name}`
 useEffect(()=>{if(!sid)return;let alive=true
  setSummary(null);setReport('');const names=rows.find(r=>r.id===sid)?.result?.files||[]
  if(names.includes('auto_summary.json'))api.get(url('auto_summary.json')).then(d=>{if(alive)setSummary(d)}).catch(e=>setErr(e.message))
  if(names.includes('report.md'))fetch(url('report.md')).then(r=>{if(!r.ok)throw Error('报告读取失败');return r.text()}).then(t=>{if(alive)setReport(t)}).catch(e=>setErr(e.message))
  Promise.all([api.get(url('meta.json')),api.get(url('ball_cand.json')),api.get(url('impacts.json')),api.get(url('input.json'))]).then(([m,c,p,s])=>{if(alive){setMeta(m);setCand(c);setPeaks(p.peaks||[]);setText(localStorage.getItem(`tennis-expert:${project.id}:${sid}`)||JSON.stringify({...initial,...s},null,2));setFrame(0);setFiles(rows.find(r=>r.id===sid)?.result?.files||[])}}).catch(e=>setErr(e.message));return()=>{alive=false}
 },[sid])
 useEffect(()=>{if(!meta||!canvas.current)return;let active=true;const im=new Image();im.onload=()=>{if(!active)return;const c=canvas.current!,ctx=c.getContext('2d')!;c.width=meta.width;c.height=meta.height;ctx.drawImage(im,0,0);ctx.strokeStyle='#ffe049';ctx.lineWidth=2;for(const p of cand[String(frame)]||[]){ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.stroke()}
 try{const s=JSON.parse(text);ctx.strokeStyle='#57d7ff';for(const pts of Object.values(s.ident?.manual||{}) as any[]){if(pts.length===2){ctx.beginPath();ctx.moveTo(...pts[0] as [number,number]);ctx.lineTo(...pts[1] as [number,number]);ctx.stroke()}}ctx.fillStyle='#ff5d9e';for(const sh of s.shots||[])for(const k of ['incoming','outgoing'])for(const p of sh[k]||[])if(p[0]===frame){ctx.beginPath();ctx.arc(p[1],p[2],5,0,Math.PI*2);ctx.fill()}}catch{}
 };im.src=`${base}/${sid}/frame/${frame}`;return()=>{active=false}},[sid,meta,frame,cand,text,manual])
 useEffect(()=>{if(sid&&meta)localStorage.setItem(`tennis-expert:${project.id}:${sid}`,text)},[text])
 const edit=(fn:(s:any)=>void)=>{try{const s=JSON.parse(text);fn(s);s.reviewed=false;setText(JSON.stringify(s,null,2));setErr('')}catch(e:any){setErr(e.message)}}
 const click=(e:React.MouseEvent<HTMLCanvasElement>)=>{const b=e.currentTarget.getBoundingClientRect();const p=[Math.round((e.clientX-b.left)*meta.width/b.width),Math.round((e.clientY-b.top)*meta.height/b.height)]
 if(target==='ball'){edit(s=>{if(!s.shots[shot])throw Error('先新增或选择一拍');const ps=s.shots[shot][kind]||[];s.shots[shot][kind]=[...ps.filter((v:any)=>v[0]!==frame),[frame,...p]].sort((a,b)=>a[0]-b[0])})}
 else if(pending){edit(s=>{s.ident.manual[target]=[pending,p];s.calibration_verified=false});setPending(null)}else setPending(p)
 }
 const start=async(action:string)=>{setBusy(true);setErr('');try{const r=await api.post(base,{action,source:sid||null,spec:action==='automatic'?{hand}:{...JSON.parse(text),...(hand!=='auto'?{hand}:{})}});followJob.current=r.job_id;await load()}catch(e:any){setErr(e.message)}finally{setBusy(false)}}
 const running=rows.find(r=>['running','queued'].includes(r.status));const disabled=busy||!!running
 let spec:any;try{spec=JSON.parse(text)}catch{spec=initial}
 return <section className="expert"><button className="ghost" onClick={back}>返回项目</button><h1>自动网球分析</h1><p>无需标注，自动追踪人体与小球、识别击球候选，并生成中文标注视频和报告。未可靠标定时不显示真实球速与落点。</p>
 <div className="expert-actions"><button disabled={disabled} onClick={()=>start("automatic")}>{disabled?"正在分析…":"开始全自动分析"}</button><label>持拍手（可选）<select value={hand} onChange={e=>setHand(e.target.value)}><option value="auto">自动判断，不确定时留空</option><option value="right">右手</option><option value="left">左手</option></select></label></div>
 {err&&<p role="alert">{err}</p>}{manual&&<div className="expert-actions"><button disabled={disabled} onClick={()=>start('prepare')}>1 准备素材 / 姿态 / 音频</button><button disabled={disabled||!sid} onClick={()=>start('calibrate')}>2 拟合场地</button><button disabled={disabled||!sid} onClick={()=>start('fit')}>3 拟合逐拍</button><button disabled={disabled||!files.includes('shots.json')} onClick={()=>start('render')}>4 导出中文视频</button></div>}
 {running&&<p>任务：{stageNames[running.stage]||running.stage||running.status} {running.progress?.frames_total_est>0&&<span> {running.progress.frames_done} / {running.progress.frames_total_est} 帧 </span>}<button onClick={()=>api.post(`/api/jobs/${running.id}/cancel`,{}).then(load).catch(e=>setErr(e.message))}>取消任务</button></p>}
 <label>分析版本<select value={sid} onChange={e=>setSid(e.target.value)}><option value="">选择版本</option>{rows.filter(r=>r.status==='succeeded').map(r=><option key={r.id} value={r.id}>{r.result?.action==='automatic'?'全自动':r.result?.action} · {r.id.slice(0,8)}</option>)}</select></label>
 {rows[0]?.status==='failed'&&<p role="alert">{rows[0].error}</p>}
 {summary&&<div className="auto-summary"><strong>自动分析已完成</strong><p>人体覆盖 {(summary.pose_coverage*100).toFixed(1)}% · 球观测覆盖 {(summary.ball_coverage*100).toFixed(1)}% · 击球候选 {summary.shot_candidates} 次</p><p>覆盖率不是准确率。黄色为模型观测，灰色为短缺口插值；击球与正反手为自动判断。</p></div>}
 {files.includes('annotated.mp4')&&<div className="auto-result"><video key={sid} controls preload="metadata" src={url('annotated.mp4')}/><p><a href={url('annotated.mp4')} download="网球自动分析.mp4">下载标注视频</a></p></div>}
 {report&&<details open={!manual}><summary>分析报告</summary><pre className="auto-report">{report}</pre><a href={url('report.md')} download>下载报告</a></details>}
 <button onClick={()=>setManual(!manual)}>{manual?'收起人工修正':'人工修正 / 精细标定（可选）'}</button>
 {manual&&meta&&<div className="expert-grid"><div><canvas ref={canvas} onClick={click}/><div className="expert-actions"><button onClick={()=>setFrame(Math.max(0,frame-1))}>上一帧</button><input aria-label="帧号" type="number" min={0} max={meta.n_frames-1} value={frame} onChange={e=>setFrame(Math.max(0,Math.min(meta.n_frames-1,+e.target.value)))}/><button onClick={()=>setFrame(Math.min(meta.n_frames-1,frame+1))}>下一帧</button><span>{(frame/meta.fps).toFixed(3)} 秒</span></div><input aria-label="帧时间轴" type="range" min={0} max={meta.n_frames-1} value={frame} onChange={e=>setFrame(+e.target.value)}/>
 <label>点击画面标记<select value={target} onChange={e=>{setTarget(e.target.value);setPending(null)}}><option value="ball">球中心（当前拍）</option>{[['near_baseline','近端底线'],['near_service','近端发球线'],['left_sideline','左单打边线'],['right_sideline','右单打边线'],['far_baseline','远端底线']].map(([v,l])=><option key={v} value={v}>{l}：点击两个端点</option>)}</select></label>{pending&&<p>已选第一个端点，请选择第二个。</p>}
 <h3>音频冲击候选</h3><p>可能包含脚步/弹地/其他球场声，必须看画面确认。</p><div className="expert-actions">{peaks.map((p:any,i)=><button key={i} onClick={()=>setFrame(Math.round(p.frame))}>{p.t}s</button>)}</div>
 {files.includes('calib_overlay.png')&&<details><summary>标定叠加图</summary><img src={url('calib_overlay.png')}/><a href={url('run.log')} target="_blank">查看逐线校验与拟合日志</a></details>}
 </div><div><h3>逐拍复核</h3><button onClick={()=>edit(s=>{s.shots.push({id:Math.max(0,...s.shots.map((x:any)=>x.id))+1,side:'UNK',tc:frame,incoming:[],outgoing:[]});setShot(s.shots.length-1)})}>在当前帧新增一拍</button>
 <select value={shot} onChange={e=>setShot(+e.target.value)}>{(spec.shots||[]).map((s:any,i:number)=><option key={s.id} value={i}>第 {s.id} 拍 · {s.side} · 帧 {s.tc}</option>)}</select>
 {spec.shots?.[shot]&&<><label>正反手<select value={spec.shots[shot].side} onChange={e=>edit(s=>s.shots[shot].side=e.target.value)}><option value="UNK">未确认</option><option value="FH">正手</option><option value="BH">反手</option></select></label><label>触球帧（可填音频小数帧）<input type="number" step="0.1" value={spec.shots[shot].tc} onChange={e=>edit(s=>s.shots[shot].tc=+e.target.value)}/></label><select value={kind} onChange={e=>setKind(e.target.value)}><option value="incoming">来球点</option><option value="outgoing">出球点</option></select><button onClick={()=>edit(s=>s.shots[shot][kind]=(s.shots[shot][kind]||[]).filter((p:any)=>p[0]!==frame))}>删除当前帧球点</button></>}
 <p>来球至少4点用于反演触球点，或在 JSON 给定 contact=[x,y,z] 米；出球至少3点。不知道深度时不要猜测。</p>
 <label>独立尺度依据<input value={spec.scale_constraint||''} placeholder="已核对远端底线 / 已知身高约束" onChange={e=>edit(s=>s.scale_constraint=e.target.value)}/></label>
 {(['calibration_verified','camera_static','reviewed'] as const).map((k,i)=><label key={k}><input type="checkbox" checked={!!spec[k]} onChange={e=>{const s=JSON.parse(text);s[k]=e.target.checked;setText(JSON.stringify(s,null,2))}}/>{['已核对标定叠加和逐线误差','该片段机位稳定，未平移/变焦','已逐拍核对触球帧与球点'][i]}</label>)}
 <details><summary>导入 / 编辑标定与逐拍 JSON</summary><input type="file" accept=".json" onChange={async e=>{const f=e.target.files?.[0];if(f){try{const x=JSON.parse(await f.text());setText(JSON.stringify({...initial,...x},null,2))}catch{setErr('JSON 格式错误')}}}}/><textarea aria-label="逐拍配置 JSON" value={text} onChange={e=>setText(e.target.value)}/><p>支持本页完整配置；skill 的 shots_input.json 可直接导入逐拍部分。ident.json 内容放入 ident 字段。所有点使用当前原尺寸、0-based帧号。</p></details>
 <h3>产物</h3>{files.filter(f=>['shots.json','technique.json','report.md','annotated.mp4','input.json'].includes(f)).map(f=><p key={f}><a href={url(f)} target="_blank">{f}</a></p>)}
 </div></div>}
 </section>
}
