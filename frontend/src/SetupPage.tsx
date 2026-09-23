import {useEffect,useRef,useState} from 'react'
import {ChevronLeft,CircleDot,Play,User,Users} from 'lucide-react'
import {api,poll} from './api'
import type{PersonFrame,Project,Run} from './types'

export function Setup({project,back,onRun,setError,onCompare}:{project:Project;back:()=>void;onRun:(r:Run)=>void;setError:(s:string)=>void;onCompare:()=>void}){
 const dur=project.duration_ms||0,[start,setStart]=useState(0),[end,setEnd]=useState(Math.min(dur,20000)),[hand,setHand]=useState('right')
 const[mode,setMode]=useState('tiled_local'),[imgsz,setImgsz]=useState(960),[conf,setConf]=useState(.15)
 const[cols,setCols]=useState(3),[rows,setRows]=useState(2),[overlap,setOverlap]=useState(.25)
 const[assoc,setAssoc]=useState(160),[radius,setRadius]=useState(260)
 const[running,setRunning]=useState(false),[job,setJob]=useState<any>(null),[note,setNote]=useState('')
 const[personFrame,setPersonFrame]=useState<PersonFrame|null>(null),[selBox,setSelBox]=useState<number>(-1),[personLoading,setPersonLoading]=useState(false)
 const imgW=project.width||1920,imgH=project.height||1080

 const loadPersonFrame=async()=>{setPersonLoading(true);setSelBox(-1)
  try{const r=await api.get(`/api/projects/${project.id}/person-frame?max_ms=4000`);setPersonFrame(r)
   if(r.found&&r.persons?.length){let bi=0,bs=-1;r.persons.forEach((p:any,i:number)=>{const s=(p.bbox[2]-p.bbox[0])*(p.bbox[3]-p.bbox[1]);if(s>bs){bs=s;bi=i}});setSelBox(bi)}
  }catch(e:any){setPersonFrame({found:false,note:e.message})}finally{setPersonLoading(false)}}
 useEffect(()=>{loadPersonFrame()},[project.id])

 const create=async()=>{
  setRunning(true);setNote('')
  try{
   const body:any={segment_start_ms:start,segment_end_ms:end,handedness:hand,
    model:{imgsz,ball_confidence:conf,device:'auto_mps_then_cpu'},
    processing:{detection_mode:mode,tiles:{cols,rows,overlap},full_search_interval_ms:300,person_stride_ms:300},
    tracking:{association_max_dist_px:assoc,base_search_radius_px:radius,radius_per_lost_ms:1.0,max_prediction_gap_ms:80},
    client_request_id:crypto.randomUUID()}
   if(personFrame?.found&&selBox>=0&&personFrame.persons?.[selBox])
    body.target_person_hint={box:personFrame.persons[selBox].bbox,frame_pts_ms:personFrame.pts_ms}
   const r=await api.post(`/api/projects/${project.id}/runs`,body)
   if(r.reused){onRun(await api.get(`/api/runs/${r.run_id}`));return}
   const j=await poll(r.job_id,setJob)
   if(j.status==='succeeded')onRun(await api.get(`/api/runs/${r.run_id}`))
   else{setError(j.error||`任务${j.status}`);setRunning(false)}
  }catch(e:any){setError(e.message);setRunning(false)}
 }
 const cancel=async()=>{if(job)try{await api.post(`/api/jobs/${job.id}/cancel`,{})}catch(e:any){setError(e.message)}}

 return <section><button className="back" onClick={back}><ChevronLeft/>返回项目</button>
 <div className="setup-grid"><div className="video-shell"><video controls src={`/api/projects/${project.id}/media`}/><span className="media-label">原始预览代理 · 推理读取原片</span></div>
 <div className="form-card"><p className="eyebrow">NEW ANALYSIS</p><h2>设置分析</h2>
  <label>时间段 <b>{(start/1000).toFixed(1)}s – {(end/1000).toFixed(1)}s（{(end-start)/1000|0} 秒）</b>
   <input type="range" min="0" max={Math.max(1,end-500)} step="100" value={start} onChange={e=>setStart(+e.target.value)}/>
   <input type="range" min={start+500} max={dur} step="100" value={end} onChange={e=>setEnd(+e.target.value)}/></label>
  <div className="two"><label>持拍手<select value={hand} onChange={e=>setHand(e.target.value)}><option value="right">右手</option><option value="left">左手</option><option value="unknown">未知</option></select></label>
  <label>检测输入<select disabled={mode==='tracknet'} value={imgsz} onChange={e=>setImgsz(+e.target.value)}><option value={640}>640 · 快</option><option value={960}>960 · 默认</option><option value={1280}>1280 · 小球优先</option></select></label></div>
  {mode==='tracknet'&&<p className="notice">球使用连续三帧、固定 640×360 输入，热图阈值 0.5；YOLO 球阈值和分块设置不生效。前两帧预热，人物和球拍仍由 YOLO 检测。</p>}<label>检测模式<select value={mode} onChange={e=>setMode(e.target.value)}>
   <option value="tracknet">TrackNet 三帧时序</option><option value="full_frame">整图（基线）</option>
   <option value="tiled">分块整图</option>
   <option value="tiled_local">分块发现 + 局部跟踪（推荐）</option></select></label>
  {(mode==='tiled'||mode==='tiled_local')&&<div className="three"><label>列<input type="number" min="1" max="6" value={cols} onChange={e=>setCols(+e.target.value)}/></label>
   <label>行<input type="number" min="1" max="6" value={rows} onChange={e=>setRows(+e.target.value)}/></label>
   <label>重叠<input type="number" min="0" max="0.5" step="0.05" value={overlap} onChange={e=>setOverlap(+e.target.value)}/></label></div>}
  <div className="two"><label>关联距离 px<input type="number" min="40" max="600" step="10" value={assoc} onChange={e=>setAssoc(+e.target.value)}/></label>
  <label>局部基础半径 px<input type="number" min="100" max="800" step="20" value={radius} onChange={e=>setRadius(+e.target.value)}/></label></div>
  <label>球候选阈值<input type="number" min="0.01" max="0.9" step="0.01" value={conf} onChange={e=>setConf(+e.target.value)}/></label>

  <div className="person-pick"><p className="eyebrow">TARGET PLAYER</p>
   {personLoading?<p className="hint">正在检测首个有人帧…</p>:personFrame?.found?<>
    <div className="person-frame"><img src={`data:image/jpeg;base64,${personFrame.image_jpeg_base64}`} alt="首个人物帧"/>
     {personFrame.persons!.map((p,i)=><button key={i} className={i===selBox?'sel':''} style={{left:`${100*p.bbox[0]/imgW}%`,top:`${100*p.bbox[1]/imgH}%`,width:`${100*(p.bbox[2]-p.bbox[0])/imgW}%`,height:`${100*(p.bbox[3]-p.bbox[1])/imgH}%`}} onClick={()=>setSelBox(i)} title={`conf ${p.confidence}`}><User size={13}/></button>)}
    </div>
    <p className="hint"><Users size={12}/> 点击选择目标人物（当前第 {selBox+1} 个）；默认最大框。丢失时显示“目标丢失”，不会切换到其他人。</p></>
   :<p className="hint">前 4 秒未检测到人物：将以视频内首个检测到的人物为准。</p>}
  </div>

  <button className="primary wide" disabled={running||project.proxy_status!=='ready'} onClick={create}><Play size={17}/>{running?'正在本地分析…':'开始分析'}</button>
  {running&&<button className="ghost wide" onClick={cancel} style={{marginTop:8}}>取消任务</button>}
  {job&&<div className="jobbox"><div><span className={`pill ${job.status}`}>{job.status}</span><span>{job.stage||''} · {job.progress?.frames_done||0}/{job.progress?.frames_total_est||'—'} 帧{job.progress?.device?` · ${job.progress.device}`:''}</span></div><progress value={job.progress?.frames_done||0} max={job.progress?.frames_total_est||1}/></div>}
  <button className="ghost wide" style={{marginTop:10}} onClick={onCompare}>对比已有 run（A/B）</button>
 </div></div></section>
}
