import {SpeedPanel} from './SpeedPanel'
import {useCallback,useEffect,useMemo,useRef,useState} from 'react'
import {Activity,Check,ChevronLeft,CircleDot,Crosshair,Download,Eye,EyeOff,Play,Plus,RefreshCw,Tag,X,ZoomIn} from 'lucide-react'
import {api,fmtTime,poll} from './api'
import type{Evaluation,EventRec,FrameRec,GtData,GtSet,Layers,Project,Run} from './types'
import {OverlayCanvas} from './Overlay'
import {useVideoSync} from './useVideoSync'

const eventText:Record<string,string>={player_hit:'击球',wall_contact_candidate:'撞墙候选',swing_candidate:'挥拍候选',bounce_candidate:'落地候选'}

export function Workbench({project,initialRun,back,setError,onCompare}:{project:Project;initialRun:Run;back:()=>void;setError:(s:string)=>void;onCompare:(runId:string)=>void}){
 const[run,setRun]=useState(initialRun),[events,setEvents]=useState<EventRec[]>([]),[frames,setFrames]=useState<FrameRec[]>([])
 const[view,setView]=useState<'automatic'|'reviewed'>('automatic'),[time,setTime]=useState(initialRun.segment_start_ms)
 const[selected,setSelected]=useState<EventRec|null>(null),[revision,setRevision]=useState(initialRun.correction_revision||0)
 const[exporting,setExporting]=useState(false),[tab,setTab]=useState<'events'|'evidence'|'params'|'annotate'|'results'>('events')
 const[annotate,setAnnotate]=useState(false),[gt,setGt]=useState<GtData>({segment_start_ms:run.segment_start_ms,segment_end_ms:run.segment_end_ms,events:[],ball_points:[]})
 const[gtSets,setGtSets]=useState<GtSet[]>([]),[evaluations,setEvaluations]=useState<Evaluation[]>([])
 const[precise,setPrecise]=useState(false),[preciseSrc,setPreciseSrc]=useState<string|null>(null),[speed,setSpeed]=useState(1)
 const[layers,setLayers]=useState<Layers>({ball:true,predicted:true,candidates:false,skeleton:true,racket:true,person:true,personId:true,hipTrail:false,events:true,gt:false,searchCrop:false})
 const video=useRef<HTMLVideoElement>(null)
 const{mediaTimeMs,playing,rvfcSupported}=useVideoSync(video,!precise)
 const fps=project.nominal_fps||30
 const imgW=project.width||1920,imgH=project.height||1080
 const effectiveTime=precise?time:mediaTimeMs

 const load=useCallback(async(which=view)=>{
  try{const rr=await api.get(`/api/runs/${run.id}`);setRun(rr);setRevision(rr.correction_revision)
   const ev=await api.get(`/api/runs/${run.id}/events?view=${which}`);setEvents(ev.events)
   setSelected(old=>old?ev.events.find((i:EventRec)=>i.id===old.id)||null:old)
   const windows=[]
   for(let s=run.segment_start_ms;s<run.segment_end_ms;s+=12000)windows.push(api.get(`/api/runs/${run.id}/observations?start_ms=${s}&end_ms=${Math.min(s+11999,run.segment_end_ms)}`))
   const chunks=await Promise.all(windows)
   const unique=new Map<number,FrameRec>()
   for(const chunk of chunks)for(const f of chunk.frames)unique.set(f.frame_index,f)
   setFrames([...unique.values()].sort((a,b)=>a.source_pts_ms-b.source_pts_ms))
   setGtSets((await api.get(`/api/projects/${project.id}/ground-truth-sets`)).gt_sets)
   setEvaluations((await api.get(`/api/runs/${run.id}/evaluations`)).evaluations)
  }catch(e:any){setError(e.message)}
 },[run.id,run.segment_start_ms,run.segment_end_ms,view,project.id,setError])
 useEffect(()=>{load()},[load])

 const current=useMemo(()=>frames.reduce<FrameRec|null>((best,f)=>Math.abs(f.source_pts_ms-effectiveTime)<Math.abs((best?.source_pts_ms??Infinity)-effectiveTime)?f:best,null),[frames,effectiveTime])
 const seek=(ms:number)=>{const t=Math.max(run.segment_start_ms,Math.min(run.segment_end_ms,ms));setTime(t);if(video.current&&!precise)video.current.currentTime=t/1000}
 // 键盘：空格播放/暂停，←→ 逐帧，Shift+←→ 1 秒
 useEffect(()=>{
  const onKey=(e:KeyboardEvent)=>{
   const tag=(e.target as HTMLElement)?.tagName
   if(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA')return
   if(e.code==='Space'){e.preventDefault();video.current?.paused?video.current.play():video.current?.pause()}
   else if(e.key==='ArrowLeft'){e.preventDefault();seek((precise?time:mediaTimeMs)-(e.shiftKey?1000:1000/fps))}
   else if(e.key==='ArrowRight'){e.preventDefault();seek((precise?time:mediaTimeMs)+(e.shiftKey?1000:1000/fps))}
  }
  window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey)
 },[fps,precise,time,mediaTimeMs])
 // 精确帧模式：暂停视频，取后端解码帧
 useEffect(()=>{
  if(!precise||!current){setPreciseSrc(null);return}
  let alive=true
  api.get(`/api/projects/${project.id}/frame?index=${current.frame_index}`).then(()=>{}).catch(()=>{})
  fetch(`/api/projects/${project.id}/frame?index=${current.frame_index}`)
   .then(r=>r.ok?r.blob():Promise.reject(new Error('取帧失败')))
   .then(b=>{if(alive)setPreciseSrc(URL.createObjectURL(b))})
   .catch(()=>{if(alive)setPreciseSrc(null)})
  return()=>{alive=false;if(preciseSrc)URL.revokeObjectURL(preciseSrc)}
 },[precise,current?.frame_index,project.id])

 const correct=async(op:any)=>{try{const r=await api.post(`/api/runs/${run.id}/corrections`,{...op,expected_revision:revision});setRevision(r.revision);setView('reviewed');await load('reviewed')}catch(e:any){setError(e.message)}}
 const exportRun=async()=>{setExporting(true)
  try{const r=await api.post(`/api/runs/${run.id}/exports`,{view,layers:{...layers},panel:true,client_request_id:crypto.randomUUID()})
   const j=await poll(r.job_id,()=>{})
   if(j.status==='failed')throw new Error(j.error)
   const ex=await api.get(`/api/exports/${r.export_id}`)
   for(const a of ex.artifacts?.artifacts||[]){const link=document.createElement('a');link.href=`/api/exports/${r.export_id}/files/${encodeURIComponent(a.file)}`;link.download=a.file;link.click()}
  }catch(e:any){setError(e.message)}finally{setExporting(false)}}

 // ---- 标注操作 ----
 const addBallPoint=(x:number,y:number)=>{if(!current)return
  setGt(g=>({...g,ball_points:[...g.ball_points.filter(b=>b.source_time_ms!==current.source_pts_ms),{source_time_ms:current.source_pts_ms,point:[x,y]}]}))}
 const markUnobservable=()=>{if(!current)return
  setGt(g=>({...g,ball_points:[...g.ball_points.filter(b=>b.source_time_ms!==current.source_pts_ms),{source_time_ms:current.source_pts_ms,unobservable:true}]}))}
 const markEvent=(type:string,stroke_type:string)=>{if(!current)return
  setGt(g=>({...g,events:[...g.events.filter(e=>!(e.type===type&&Math.abs(e.source_time_ms-current.source_pts_ms)<200)),{type,source_time_ms:current.source_pts_ms,stroke_type}]}))}
 const saveGt=async(name:string)=>{try{
  await api.post(`/api/projects/${project.id}/ground-truth-sets`,{...gt,name,segment_start_ms:run.segment_start_ms,segment_end_ms:run.segment_end_ms})
  setGtSets((await api.get(`/api/projects/${project.id}/ground-truth-sets`)).gt_sets)
 }catch(e:any){setError(e.message)}}
 const runEvaluation=async(gtSetId:string)=>{try{
  const ev=await api.post(`/api/runs/${run.id}/evaluations`,{gt_set_id:gtSetId,view:'automatic',time_tolerance_ms:100,ball_position_tolerance_px:40})
  setEvaluations([ev,...evaluations])
  setTab('results')
 }catch(e:any){setError(e.message)}}

 const s=run.summary||{},perf=run.performance||{}
 const pt=s.person_tracking||{}
 return <section className="work">
 <div className="work-top"><button className="back" onClick={back}><ChevronLeft/>项目</button>
  <div><p className="eyebrow">RUN · {run.id.slice(0,8)} · {run.mode==='assisted'?'ASSISTED':'AUTOMATIC'}</p><h2>{project.filename}</h2></div>
  <span className={`pill ${run.status}`}>{run.status}</span>
  {run.effective_config?.processing?.detection_mode&&<span className="pill">{run.effective_config.processing.detection_mode}</span>}
  <div className="spacer"/>
  <div className="segmented"><button className={view==='automatic'?'active':''} onClick={()=>setView('automatic')}>自动</button><button className={view==='reviewed'?'active':''} onClick={()=>setView('reviewed')}>已复核 r{revision}</button></div>
  <button className={annotate?'primary small':'ghost'} onClick={()=>setAnnotate(!annotate)}><Crosshair size={14}/>{annotate?'标注中（自动结果已隐藏）':'独立标注'}</button>
  <button className="ghost" onClick={()=>onCompare(run.id)}>对比</button>
  <button className="primary small" onClick={exportRun} disabled={exporting}><Download size={15}/>{exporting?'导出中':'导出'}</button></div>

 <div className="metrics">
  <Metric label="击球候选" value={s.counts?.player_hit??0}/><Metric label="正/反/未知" value={`${s.stroke_counts_auto?.forehand??0}/${s.stroke_counts_auto?.backhand??0}/${s.stroke_counts_auto?.unknown??0}`}/>
  <Metric label="球观测覆盖" value={s.ball_observation_coverage==null?'—':`${(s.ball_observation_coverage*100).toFixed(1)}%`}/>
  <Metric label="人物目标覆盖" value={pt.target_tracked_coverage==null?'—':`${(pt.target_tracked_coverage*100).toFixed(1)}%`}/>
  <Metric label="最长断档" value={`${s.longest_observation_gap_ms??'—'} ms`}/>
  <Metric label="耗时/内存" value={perf.wall_clock_ms?`${(perf.wall_clock_ms/perf.segment_duration_ms).toFixed(1)}× · ${perf.peak_rss_mb||'—'}MB`:''}/></div>

 <SpeedPanel runId={run.id} time={effectiveTime} seek={seek}/>
 <div className="bench"><div className="viewer">
  <OverlayCanvas imgW={imgW} imgH={imgH} current={current} frames={frames} events={annotate?[]:events} gt={gt} layers={annotate?{...layers,events:false,gt:true,predicted:false}:layers} annotate={annotate} onAnnotate={addBallPoint} hideVideo={precise}>
   <video ref={video} src={`/api/projects/${project.id}/media`} playsInline onRateChange={e=>setSpeed(e.currentTarget.playbackRate)}/>
   {precise&&preciseSrc&&<img src={preciseSrc} alt="后端精确帧"/>}
  </OverlayCanvas>
  <span className="frame-tag">{current?`f${current.frame_index} · ${current.source_pts_ms}ms${current.pose?.source?` · pose:${current.pose.source}`:''}${current.ball?.selected?.search_mode?` · ${current.ball.selected.search_mode}`:''}`:'无分析帧'}{!rvfcSupported?' · timeupdate 回退':''}{precise?' · 精确帧模式':''}</span>
  <div className="controls">
   <button onClick={()=>{video.current?.pause();video.current&&(video.current.currentTime=(effectiveTime-1000/fps)/1000)}}>−1s</button>
   <button onClick={()=>{video.current?.pause();seek(effectiveTime-1000/fps)}}>−1帧</button>
   {playing&&!precise?<button onClick={()=>video.current?.pause()}><EyeOff size={13}/>暂停</button>:<button onClick={async()=>{setPrecise(false);setPreciseSrc(null);if(video.current){video.current.currentTime=(time||mediaTimeMs)/1000;await video.current.play()}}}><Play size={13}/>播放</button>}
   <b>{fmtTime(effectiveTime)}</b>
   <button onClick={()=>{video.current?.pause();seek(effectiveTime+1000/fps)}}>+1帧</button>
   <button onClick={()=>{video.current?.pause();seek(effectiveTime+1000)}}>+1s</button>
   <select value={speed} onChange={e=>{setSpeed(+e.target.value);if(video.current)video.current.playbackRate=+e.target.value}}>{[0.25,0.5,1,2].map(v=><option key={v} value={v}>{v}×</option>)}</select>
   <button className={precise?'on':''} onClick={()=>{setPrecise(!precise);video.current?.pause()}}><ZoomIn size={13}/>精确帧</button>
  </div>
  <Timeline run={run} frames={frames} events={annotate?[]:events} gt={gt} time={effectiveTime} seek={seek}/>
  <LayerControls layers={layers} setLayers={setLayers} annotate={annotate}/>
 </div>
 <aside>
  <div className="tabs-row">
   {([['events','事件'],['evidence','证据'],['params','参数'],['annotate','标注'],['results','结果']]as const).map(([k,label])=>
    <button key={k} className={`${tab===k?'active':''}${annotate&&k!=='annotate'&&k!=='results'?'disabled':''}`} onClick={()=>setTab(k)}>{label}</button>)}
  </div>
  {tab==='events'&&!annotate&&<EventsTab events={events} selected={selected} setSelected={e=>{setSelected(e);seek(e.source_time_ms)}} correct={correct}/>}
  {tab==='evidence'&&<EvidenceTab current={current} frames={frames}/>}
  {tab==='params'&&<ParamsTab run={run} project={project} onNewRun={(r)=>{setRun(r);load()}} setError={setError}/>}
  {tab==='annotate'&&<AnnotateTab gt={gt} gtSets={gtSets} current={current} markUnobservable={markUnobservable} markEvent={markEvent} saveGt={saveGt} runEvaluation={runEvaluation} setGt={setGt} seek={seek}/>}
  {tab==='results'&&<ResultsTab run={run} evaluations={evaluations} seek={seek}/>}
 </aside></div></section>
}

function Metric({label,value}:{label:string;value:string|number}){return <div><span>{label}</span><strong>{value}</strong></div>}

function LayerControls({layers,setLayers,annotate}:{layers:Layers;setLayers:(l:Layers)=>void;annotate:boolean}){
 const names:Record<keyof Layers,string>={ball:'球(观测)',predicted:'预测(虚线)',candidates:'全部候选',skeleton:'骨架',racket:'球拍框',person:'人物框',personId:'人物ID',hipTrail:'人体轨迹',events:'事件点',gt:'标注点',searchCrop:'搜索窗'}
 return <div className="layers">{(Object.keys(names)as(keyof Layers)[]).filter(k=>!annotate||k==='gt'||k==='ball'||k==='skeleton'||k==='person').map(k=>
  <button className={layers[k]?'on':''} key={k} onClick={()=>setLayers({...layers,[k]:!layers[k]})}>{layers[k]?<Eye size={13}/>:<EyeOff size={13}/>}{names[k]}</button>)}</div>}

function Timeline({run,frames,events,gt,time,seek}:{run:Run;frames:FrameRec[];events:EventRec[];gt:GtData;time:number;seek:(ms:number)=>void}){
 const ref=useRef<HTMLDivElement>(null)
 const span=run.segment_end_ms-run.segment_start_ms||1
 const pct=(ms:number)=>Math.max(0,Math.min(100,100*(ms-run.segment_start_ms)/span))
 const bands=useMemo(()=>{
  const W=1000,arr=new Uint8Array(W) // 0=无,1=观测,2=预测
  for(const f of frames){const i=Math.min(W-1,Math.floor(pct(f.source_pts_ms)/100*W))
   if(f.ball?.selected?.raw)arr[i]=1;else if(f.ball?.selected?.status==='predicted')arr[i]=2}
  const out:{i:number;v:number}[]=[]
  arr.forEach((v,i)=>{if(v)out.push({i,v})})
  return out
 },[frames])
 return <div className="tl-wrap" ref={ref} onClick={e=>{if(!ref.current)return;const r=ref.current.getBoundingClientRect();seek(run.segment_start_ms+(e.clientX-r.left)/r.width*span)}}>
  <div className="tl-band">{bands.map(({i,v})=>v?<span key={i} className={v===1?'obs':'pred'} style={{left:`${i/10}%`,width:'0.12%'}}/>:null)}</div>
  <div className="tl-track"><div className="tl-cursor" style={{left:`${pct(time)}%`}}/>
   {events.map(e=><button key={e.id} className={`mark ${e.type} ${e.review_status}`} style={{left:`${pct(e.source_time_ms)}%`}} title={`${eventText[e.type]||e.type} ${fmtTime(e.source_time_ms)}`} onClick={ev=>{ev.stopPropagation();seek(e.source_time_ms)}}/>)}
   {gt.events.map((e,i)=><button key={i} className="mark gt" style={{left:`${pct(e.source_time_ms)}%`}} title={`标注 ${e.type}`} onClick={ev=>{ev.stopPropagation();seek(e.source_time_ms)}}/>)}
  </div>
 </div>}

function EventsTab({events,selected,setSelected,correct}:{events:EventRec[];selected:EventRec|null;setSelected:(e:EventRec)=>void;correct:(op:any)=>void}){
 return <>
  <div className="event-list">{events.length===0?<div className="empty compact"><CircleDot/><p>当前参数没有提取到事件。检测覆盖不代表事件正确率。</p></div>:events.map(e=>
   <button key={e.id} className={selected?.id===e.id?'selected':''} onClick={()=>setSelected(e)}><i className={e.review_status}/>
    <div><strong>{eventText[e.type]||e.type}</strong><span>{fmtTime(e.source_time_ms)} · {e.stroke_type||'unknown'}</span></div>
    <small>{e.heuristic_score?.toFixed(2)??'—'}</small></button>)}</div>
  {selected?<div className="inspector"><p className="eyebrow">EVIDENCE</p><h3>{eventText[selected.type]||selected.type}</h3>
   <dl><dt>时间</dt><dd>{fmtTime(selected.source_time_ms)}</dd><dt>来源</dt><dd>{selected.origin}</dd><dt>复核</dt><dd>{selected.review_status}</dd><dt>分数</dt><dd>{selected.heuristic_score?.toFixed(3)??'—'}（非准确率）</dd><dt>速度</dt><dd>{selected.speed_px_s?`${selected.speed_px_s.toFixed(0)} px/s`:'—'}（km/h 不可用）</dd></dl>
   <pre>{JSON.stringify(selected.evidence,null,1)}</pre>
   <div className="review-actions"><button onClick={()=>correct({op:'modify_event',event_id:selected.id,changes:{review_status:'reviewed'}})}><Check/>确认</button>
    <button onClick={()=>correct({op:'modify_event',event_id:selected.id,changes:{stroke_type:selected.stroke_type==='forehand'?'backhand':'forehand'}})}><RefreshCw/>正反手</button>
    <button className="danger" onClick={()=>correct({op:'exclude_event',event_id:selected.id,reason:'前端人工排除'})}><X/>排除</button></div>
  </div>:<div className="inspector muted">点击事件查看证据与复核。</div>}</>}

function EvidenceTab({current,frames}:{current:FrameRec|null;frames:FrameRec[]}){
 if(!current)return <div className="inspector muted">该时间点无分析帧。</div>
 const sel=current.ball?.selected
 const idx=frames.findIndex(f=>f.frame_index===current.frame_index)
 const prev=idx>0?frames[idx-1]:null
 return <div className="inspector evi">
  <p className="eyebrow">FRAME EVIDENCE · f{current.frame_index}</p>
  <dl>
   <dt>PTS</dt><dd>{current.source_pts_ms} ms</dd>
   <dt>耗时</dt><dd>det {current.timings?.detector_ms?.toFixed?.(1)??'—'}ms · pose {current.timings?.pose_ms?.toFixed?.(1)??'—'}ms</dd>
   <dt>轨迹</dt><dd>{sel?`${sel.status}${sel.raw?'（观测）':'（预测）'} · score ${sel.tracking_score?.toFixed(3)??'—'} · conf ${sel.candidate_conf?.toFixed(2)??'—'}`:'lost / 无球'}</dd>
   <dt>搜索</dt><dd>{sel?.search_mode||'—'}{sel?.search_crop?` crop[${sel.search_crop.join(',')}]`:''}</dd>
   <dt>姿态</dt><dd>{current.pose?(current.pose.landmarks?`${current.pose.source||'full_frame'} · 33pts`:'未检出人'):'本帧未采样（按间隔）'}</dd>
   <dt>人物</dt><dd>{current.person?.target?`P${current.person.target.track_id} ${current.person.target.status}${current.person.target.matched?'':'（未匹配）'}`:current.person?.status||'—'}{current.person?.hip_center?` · 髋 ${current.person.hip_center.source}`:''}</dd>
   <dt>断轨</dt><dd>{prev?.ball?.selected?.raw&&!sel?'⚠ 上一帧有观测，本帧丢失':'—'}</dd>
  </dl>
  <p className="eyebrow" style={{marginTop:12}}>候选 {current.ball?.candidates?.length??0} · 拒绝 {current.ball?.rejected?.length??0}</p>
  <div className="cand-list">{(current.ball?.candidates||[]).map((c,i)=>
   <div key={i} className={`cand ${sel&&c.bbox[0]===sel.point[0]?'chosen':''}`}>#{i} {c.cls} conf={c.confidence?.toFixed(3)} bbox=[{(c.bbox as number[]).map((v:number)=>v.toFixed(0)).join(',')}]</div>)}
   {(current.ball?.rejected||[]).map((r,i)=><div key={`r${i}`} className="cand rejected">{r.reason}（conf {(r.confidence as number)?.toFixed?.(2)}{r.score_detail?` · ${JSON.stringify(r.score_detail)}`:''}）</div>)}
  </div></div>}

function ParamsTab({run,project,onNewRun,setError}:{run:Run;project:Project;onNewRun:(r:Run)=>void;setError:(s:string)=>void}){
 const ec=run.effective_config||run.config||{}
 const[mode,setMode]=useState(ec.processing?.detection_mode||'full_frame')
 const[imgsz,setImgsz]=useState(ec.model?.imgsz||960),[conf,setConf]=useState(ec.model?.ball_confidence||.15)
 const[assoc,setAssoc]=useState(ec.tracking?.association_max_dist_px||160),[dedup,setDedup]=useState(ec.events?.deduplicate_window_ms||250)
 const[busy,setBusy]=useState(false)
 const rerun=async()=>{
  setBusy(true)
  try{const body={segment_start_ms:run.segment_start_ms,segment_end_ms:run.segment_end_ms,
   handedness:run.config?.handedness||'unknown',
   model:{imgsz,ball_confidence:conf},
   processing:{...ec.processing,detection_mode:mode},
   tracking:{...(ec.tracking||{}),association_max_dist_px:assoc},
   events:{...(ec.events||{}),deduplicate_window_ms:dedup},
   client_request_id:crypto.randomUUID()}
   const r=await api.post(`/api/projects/${project.id}/runs`,body)
   if(r.reused){onNewRun(await api.get(`/api/runs/${r.run_id}`));return}
   const j=await poll(r.job_id,()=>{})
   if(j.status==='succeeded')onNewRun(await api.get(`/api/runs/${r.run_id}`))
   else setError(j.error||`任务${j.status}`)}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 const rederive=async()=>{
  setBusy(true)
  try{const r=await api.post(`/api/runs/${run.id}/derive`,{events:{deduplicate_window_ms:dedup},client_request_id:crypto.randomUUID()})
   const j=await poll(r.job_id,()=>{})
   if(j.status==='succeeded')onNewRun(await api.get(`/api/runs/${r.run_id}`))
   else setError(j.error||`任务${j.status}`)}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 const modeChanged=mode!==(ec.processing?.detection_mode||'full_frame')||imgsz!==(ec.model?.imgsz||960)||conf!==(ec.model?.ball_confidence||.15)||assoc!==(ec.tracking?.association_max_dist_px||160)
 return <div className="inspector">
  <p className="eyebrow">PARAMS · 实际生效值</p>
  <dl><dt>模式</dt><dd>{ec.processing?.detection_mode||'full_frame'}{ec.processing?.tiles?` ${ec.processing.tiles.cols}×${ec.processing.tiles.rows}`:''}</dd>
  <dt>输入</dt><dd>{ec.model?.imgsz} · conf≥{ec.model?.ball_confidence}</dd>
  <dt>跟踪</dt><dd>关联 {ec.tracking?.association_max_dist_px}px · 缺口 {ec.tracking?.max_prediction_gap_ms}ms</dd>
  <dt>搜索</dt><dd>全图间隔 {ec.processing?.full_search_interval_ms}ms · 局部基础 {ec.tracking?.base_search_radius_px}px</dd></dl>
  <div className="param-form">
   {mode==='tracknet'&&<p className="notice">球使用连续三帧、固定 640×360 输入，热图阈值 0.5；YOLO 球阈值和分块设置不生效。前两帧预热，人物和球拍仍由 YOLO 检测。</p>}<label>检测模式<select value={mode} onChange={e=>setMode(e.target.value)}><option value="tracknet">TrackNet 三帧时序</option><option value="full_frame">整图基线</option><option value="tiled">分块</option><option value="tiled_local">分块+局部</option></select></label>
   <div className="two"><label>输入尺寸<select disabled={mode==='tracknet'} value={imgsz} onChange={e=>setImgsz(+e.target.value)}>{[640,960,1280].map(v=><option key={v} value={v}>{v}</option>)}</select></label>
   <label>球阈值<input type="number" step="0.01" min="0.01" max="0.9" value={conf} onChange={e=>setConf(+e.target.value)}/></label></div>
   <div className="two"><label>关联距离<input type="number" step="10" min="40" value={assoc} onChange={e=>setAssoc(+e.target.value)}/></label>
   <label>事件去重ms<input type="number" step="50" min="0" value={dedup} onChange={e=>setDedup(+e.target.value)}/></label></div>
   <button className="primary wide" disabled={busy} onClick={rerun}>{busy?'运行中…':modeChanged?'重跑（新 run，重新推理）':'以当前参数重跑'}</button>
   <button className="ghost wide" disabled={busy} onClick={rederive} title="复用已保存的检测观测，不重新推理">仅重算事件（复用检测）</button>
   <p className="hint">改动检测模型/输入尺寸/搜索策略会重新推理；仅改轨迹或事件参数时复用已保存检测。调参建议先用短片段。</p>
  </div></div>}

function AnnotateTab({gt,gtSets,current,markUnobservable,markEvent,saveGt,runEvaluation,setGt,seek}:{gt:GtData;gtSets:GtSet[];current:FrameRec|null;markUnobservable:()=>void;markEvent:(t:string,s:string)=>void;saveGt:(n:string)=>void;runEvaluation:(id:string)=>void;setGt:(g:GtData)=>void;seek:(ms:number)=>void}){
 const[name,setName]=useState(`标注 ${new Date().toLocaleDateString()}`)
 const strokeRef=useRef<HTMLSelectElement>(null)
 return <div className="inspector ann">
  <p className="eyebrow">INDEPENDENT ANNOTATION</p>
  <p className="hint">独立标注模式：自动事件与预测轨迹已隐藏。在画面上点击标球；按 100ms 间隔推进并标注，事件附近加密。</p>
  <div className="ann-actions">
   <button onClick={()=>{const t=current?current.source_pts_ms+100:0;seek(t)}}><Activity size={13}/>下一采样点(+100ms)</button>
   <button onClick={markUnobservable}><EyeOff size={13}/>此帧球不可辨认</button>
  </div>
  <div className="ann-actions">
   <select ref={strokeRef}><option value="forehand">正手</option><option value="backhand">反手</option><option value="unknown">unknown</option></select>
   <button onClick={()=>markEvent('player_hit',strokeRef.current?.value||'unknown')}><Tag size={13}/>标记击球时刻</button>
   <button onClick={()=>markEvent('wall_contact','unknown')}><CircleDot size={13}/>标记撞墙</button>
  </div>
  <div className="ann-list">
   <div>球点 {gt.ball_points.length}（含不可辨认 {gt.ball_points.filter(b=>b.unobservable).length}）</div>
   {gt.ball_points.slice(-8).reverse().map((b,i)=><button key={i} onClick={()=>seek(b.source_time_ms)}>{fmtTime(b.source_time_ms)} {b.unobservable?'不可辨认':`(${b.point?.map(v=>v.toFixed(0)).join(',')})`}</button>)}
   <div>事件 {gt.events.length}</div>
   {gt.events.slice(-8).reverse().map((e,i)=><button key={i} onClick={()=>seek(e.source_time_ms)}>{eventText[e.type]||e.type} · {fmtTime(e.source_time_ms)} · {e.stroke_type}</button>)}
  </div>
  {gt.ball_points.length+gt.events.length>0&&<button className="ghost" onClick={()=>setGt({segment_start_ms:gt.segment_start_ms,segment_end_ms:gt.segment_end_ms,events:[],ball_points:[]})}>清空未保存标注</button>}
  <div className="two"><input value={name} onChange={e=>setName(e.target.value)} placeholder="标注集名称"/><button className="primary small" onClick={()=>saveGt(name)}>保存并锁定</button></div>
  {gtSets.length>0&&<div className="gt-existing"><p className="eyebrow">已保存真值集</p>
   {gtSets.map(g=><div key={g.id}><span>{g.name} v{g.version}</span>
    <button onClick={()=>runEvaluation(g.id)}>评估本 run</button>
    <button onClick={()=>{api.get(`/api/projects/0`).catch(()=>{});const data=g.data;alert('已导出 JSON（见评估/导出）');void data}}>查看</button></div>)}</div>}
 </div>}

function ResultsTab({run,evaluations,seek}:{run:Run;evaluations:Evaluation[];seek:(ms:number)=>void}){
 const s=run.summary||{},pt=s.person_tracking||{},perf=run.performance||{}
 return <div className="inspector results">
  <p className="eyebrow">RESULTS · 证据可回跳</p>
  <dl>
   <dt>球观测覆盖</dt><dd>{s.ball_observation_coverage==null?'—':`${(s.ball_observation_coverage*100).toFixed(1)}%（覆盖率≠召回）`}</dd>
   <dt>预测帧</dt><dd>{s.frames_ball_predicted_only??0}</dd>
   <dt>断档</dt><dd>最长 {s.longest_observation_gap_ms??'—'}ms</dd>
   <dt>人物</dt><dd>{pt.target_tracked_coverage==null?'—':`目标覆盖 ${(pt.target_tracked_coverage*100).toFixed(1)}% · 重获 ${pt.reacquire_events??0} 次`}</dd>
   <dt>分项耗时</dt><dd>{Object.entries(perf.stage_ms||{}).map(([k,v])=>`${k} ${((v as number)/1000).toFixed(1)}s`).join(' · ')||'—'}</dd>
   <dt>内存</dt><dd>{perf.peak_rss_mb?`${perf.peak_rss_mb}MB（进程峰值）`:'—'}</dd>
  </dl>
  {evaluations.length===0?<p className="hint">先在“标注”页保存真值集，再点击“评估本 run”生成召回/精确率。评分基于独立人工真值。</p>:
   evaluations.map(ev=><div key={ev.id} className="eval-card">
    <b>{ev.gt_name} v{ev.gt_version} · {ev.view}</b>
    <div className="eval-grid">
     <span>击球 P</span><b>{ev.hit_events.precision==null?'—':ev.hit_events.precision.toFixed(3)}</b>
     <span>击球 R</span><b>{ev.hit_events.recall==null?'—':ev.hit_events.recall.toFixed(3)}</b>
     <span>F1</span><b>{ev.hit_events.f1==null?'—':ev.hit_events.f1.toFixed(3)}</b>
     <span>TP/FP/FN</span><b>{ev.hit_events.tp}/{ev.hit_events.fp}/{ev.hit_events.fn}（分母 预测{ev.hit_events.denominator_pred}/真值{ev.hit_events.denominator_gt}）</b>
     <span>球召回</span><b>{ev.ball_localization?.recall==null?'—':`${(ev.ball_localization.recall*100).toFixed(1)}%（容差${ev.ball_localization.tolerance_px}px，可见 ${ev.ball_localization.gt_frames_observable} 帧）`}</b>
     <span>球误差</span><b>{ev.ball_localization?.mean_error_px==null?'—':`${ev.ball_localization.mean_error_px}px`}</b>
    </div>
    {(ev.false_positives?.length||0)>0&&<div className="fp-list"><p className="eyebrow">误检 FP（点击回跳）</p>
     {ev.false_positives!.map((f,i)=><button key={i} onClick={()=>seek(f.source_time_ms)}>{fmtTime(f.source_time_ms)}</button>)}</div>}
    {(ev.false_negatives?.length||0)>0&&<div className="fp-list fn"><p className="eyebrow">漏检 FN（点击回跳）</p>
     {ev.false_negatives!.map((f,i)=><button key={i} onClick={()=>seek(f.source_time_ms)}>{fmtTime(f.source_time_ms)} {f.stroke_type}</button>)}</div>}
   </div>)}
 </div>}
