import {ExpertPage} from './ExpertPage'
import {ProjectStorage} from './ProjectStorage'
import {useCallback,useEffect,useState} from 'react'
import {Activity,AlertTriangle,Check,CircleDot,FolderOpen,Gauge,RefreshCw,Settings2,Upload,WifiOff,X} from 'lucide-react'
import {api,fmtTime} from './api'
import type {Project,Run} from './types'
import {Setup} from './SetupPage'
import {Workbench} from './Workbench'
import {Compare} from './ComparePage'

type Screen={k:'expert',project:Project}|{k:'projects'}|{k:'setup',project:Project}|{k:'workbench',project:Project,run:Run}|{k:'compare',project:Project,left:string,right:string}|{k:'system'}
const statusText:Record<string,string>={pending:'等待准备',processing:'正在准备',ready:'可分析',queued:'排队中',running:'分析中',succeeded:'已完成',failed:'失败',cancelled:'已取消',interrupted:'已中断'}

function ErrorBanner({message,onClose}:{message:string;onClose:()=>void}){return <div className="error-banner"><AlertTriangle size={18}/><span>{message}</span><button onClick={onClose}><X size={16}/></button></div>}
function Pill({status}:{status:string}){return <span className={`pill ${status}`}>{statusText[status]||status}</span>}

export default function App(){
 const[screen,setScreen]=useState<Screen>({k:'projects'}),[projects,setProjects]=useState<Project[]>([]),[error,setError]=useState('')
 const refresh=useCallback(async()=>{try{setProjects((await api.get('/api/projects')).projects)}catch(e:any){setError(e.message)}},[])
 useEffect(()=>{refresh()},[refresh])
 const open=(p:Project)=>{
  if(p.latest_run?.status==='succeeded'){
   api.get(`/api/runs/${p.latest_run.id}`).then((r:Run)=>setScreen({k:'workbench',project:p,run:r})).catch((e:any)=>{setError(e.message);setScreen({k:'setup',project:p})})
  }else setScreen({k:'setup',project:p})}
 const compare=(runId:string)=>{setScreen(s=>s.k==='workbench'?{k:'compare',project:s.project,left:runId,right:''}:s)}
 return <div className="app">
  <header><button className="brand" onClick={()=>setScreen({k:'projects'})}><span>TL</span><div>Tennis Lab<small>本地视频分析</small></div></button>
  <nav><button className={screen.k==='projects'?'active':''} onClick={()=>setScreen({k:'projects'})}>项目</button><button className={screen.k==='system'?'active':''} onClick={()=>setScreen({k:'system'})}>环境</button></nav>
  <div className="local"><WifiOff size={14}/> 本机离线处理</div></header>
  {error&&<ErrorBanner message={error} onClose={()=>setError('')}/>}
  <main>
   {screen.k==='projects'&&<Projects projects={projects} refresh={refresh} open={open} expert={p=>setScreen({k:'expert',project:p})} setError={setError}/>}
   {screen.k==='setup'&&<Setup project={screen.project} back={()=>setScreen({k:'projects'})} onRun={r=>setScreen({k:'workbench',project:screen.project,run:r})} setError={setError} onCompare={()=>setScreen({k:'compare',project:screen.project,left:'',right:''})}/>}
   {screen.k==='workbench'&&<Workbench project={screen.project} initialRun={screen.run} back={()=>setScreen({k:'projects'})} setError={setError} onCompare={compare}/>}
   {screen.k==='expert'&&<ExpertPage project={screen.project} back={()=>setScreen({k:'projects'})}/>}
   {screen.k==='system'&&<System setError={setError}/>}
   {screen.k==='compare'&&<ComparePicker screen={screen} setScreen={setScreen} setError={setError}/>}
  </main></div>}

function ComparePicker({screen,setScreen,setError}:{screen:Extract<Screen,{k:'compare'}>;setScreen:(s:Screen)=>void;setError:(s:string)=>void}){
 const[runs,setRuns]=useState<Run[]>([]),[left,setLeft]=useState(screen.left),[right,setRight]=useState(screen.right)
 useEffect(()=>{api.get(`/api/projects/${screen.project.id}/runs`).then((r:any)=>{setRuns(r.runs)
  const succ=r.runs.filter((x:Run)=>x.status==='succeeded')
  if(succ.length>=2){setLeft(succ[1].id);setRight(succ[0].id)}
 }).catch((e:any)=>setError(e.message))},[screen.project.id,setError])
 if(left&&right)return <Compare project={screen.project} leftId={left} rightId={right} back={()=>setScreen({k:'projects'})} setError={setError} swap={()=>{setLeft(right);setRight(left)}}/>
 return <section><p className="eyebrow">A/B COMPARE</p><h1>选择两个 run 对比</h1>
  {runs.filter(r=>r.status==='succeeded').length<2?<p className="hint">该项目成功 run 少于 2 个，先在设置页以不同参数分析两次。</p>:
  <div className="compare-picker">
   <div><p className="eyebrow">LEFT (A)</p>{runs.map(r=><button key={r.id} className={left===r.id?'on':''} onClick={()=>setLeft(r.id)}>{r.id.slice(0,8)} · {r.effective_config?.processing?.detection_mode||'full'} · {fmtTime(r.segment_start_ms)}+</button>)}</div>
   <div><p className="eyebrow">RIGHT (B)</p>{runs.map(r=><button key={r.id} className={right===r.id?'on':''} onClick={()=>setRight(r.id)}>{r.id.slice(0,8)} · {r.effective_config?.processing?.detection_mode||'full'} · {fmtTime(r.segment_start_ms)}+</button>)}</div>
  </div>}</section>}

function Projects({projects,refresh,open,expert,setError}:{projects:Project[];refresh:()=>void;expert:(p:Project)=>void;open:(p:Project)=>void;setError:(s:string)=>void}){
 const[trash,setTrash]=useState(false),[trashProjects,setTrashProjects]=useState<Project[]>([]),[managing,setManaging]=useState<Project|null>(null)
 const reload=()=>{refresh();api.get('/api/projects?trash=true').then(d=>setTrashProjects(d.projects)).catch(e=>setError(e.message))}
 useEffect(()=>{reload()},[trash])
 const[input]=useState<{current:HTMLInputElement|null}>({current:null}),[busy,setBusy]=useState(false),[job,setJob]=useState<any>(null)
 const fileInput=(node:HTMLInputElement|null)=>{input.current=node}
 const choose=async(f?:File)=>{
  if(!f)return;setBusy(true)
  try{const r=await api.upload(f)
   const j=await fetch(`/api/jobs/${r.preparation_job_id}`).then(x=>x.json())
   setJob(j)
   const timer=setInterval(async()=>{const jj=await fetch(`/api/jobs/${r.preparation_job_id}`).then(x=>x.json())
    setJob(jj)
    if(['succeeded','failed','cancelled','interrupted'].includes(jj.status)){clearInterval(timer)
     if(jj.status==='succeeded')refresh();else setError(jj.error||`准备${jj.status}`)
     setBusy(false)}},800)
  }catch(e:any){setError(e.message);setBusy(false)}}
 return <section><div className="hero"><div><p className="eyebrow">LOCAL TENNIS VISION</p><h1>把训练视频变成<br/><em>可检查的数据证据</em></h1><p>默认全自动追踪人体与球轨迹，生成动作候选、标注视频和中文报告。人工修正为可选步骤。</p>
  <button className="primary" disabled={busy} onClick={()=>input.current?.click()}><Upload size={18}/>{busy?'正在准备视频…':'导入训练视频'}</button>
  <input ref={fileInput} hidden type="file" accept="video/mp4,video/quicktime,.m4v" onChange={e=>choose(e.target.files?.[0])}/>
  {job&&busy&&<div className="jobline"><span>{statusText[job.status]||job.status} · {job.stage||'等待 worker'}</span><progress value={job.progress?.frames_done||0} max={job.progress?.frames_total_est||1}/></div>}</div>
  <div className="hero-card"><CircleDot/><strong>测量边界</strong><p>自动输出二维轨迹与击球候选；未经可靠三维标定不显示 km/h 或真实落点。</p>
  <div className="mini"><span>模型</span><b>YOLO11m-pose + TrackNet</b></div><div className="mini"><span>设备</span><b>Apple M2 / MPS</b></div></div></div>
 <div className="section-title"><div><p className="eyebrow">PROJECTS</p><h2>{trash?"回收站":"训练项目"}</h2></div><button className="ghost" onClick={()=>setTrash(!trash)}>{trash?"返回项目":"回收站"}</button><button className="ghost" onClick={reload}><RefreshCw size={15}/>刷新</button></div>
 {(trash?trashProjects:projects).length===0?<div className="empty"><FolderOpen/><h3>{trash?"回收站为空":"还没有训练视频"}</h3><p>{trash?"移入回收站的项目会显示在这里，可恢复或彻底删除。":"先导入一段 10–60 秒、固定机位的网球练习。"}</p></div>:
 <div className="project-grid">{(trash?trashProjects:projects).map(p=><div key={p.id}><button className="project-card" style={{width:"100%"}} onClick={()=>trash?setManaging(p):expert(p)}>
  <div className="thumb">{p.proxy_status==='ready'?<img src={`/api/projects/${p.id}/thumb`}/>:<Activity/>}<Pill status={p.proxy_status}/></div>
  <div><h3>{p.filename}</h3><p>{p.width||'—'} × {p.height||'—'} · {fmtTime(p.duration_ms)} · {p.nominal_fps?.toFixed(1)||'—'} fps</p>
  <span>{p.latest_run?`最近分析：${statusText[p.latest_run.status]||p.latest_run.status}`:'尚未分析'}</span></div></button><button className="ghost" disabled={trash} onClick={()=>open(p)}>高级分析 / 旧版结果</button><button className="ghost" onClick={()=>setManaging(p)}>{trash?"恢复 / 彻底删除":"管理 / 删除"}</button></div>)}</div>}
 {managing&&<ProjectStorage project={managing} trashed={trash} close={()=>setManaging(null)} changed={reload}/>}</section>}

function System({setError}:{setError:(s:string)=>void}){
 const[d,setD]=useState<any>(null)
 useEffect(()=>{api.get('/api/system/doctor').then(setD).catch(e=>setError(e.message))},[setError])
 return <section><p className="eyebrow">SYSTEM DOCTOR</p><h1>本机环境</h1>
 {!d?<div className="empty"><Activity className="spin"/>正在检查…</div>:<div className="doctor">
  <div><Gauge/><h3>{d.chip||d.arch}</h3><p>Python {d.python} · macOS {d.macos||'—'} · 磁盘剩余 {d.disk_free_gb}GB</p></div>
  <div className={d.mps_available?'ok':'warn'}>{d.mps_available?<Check/>:<AlertTriangle/>}<h3>MPS {d.mps_available?'可用':'不可用'}</h3><p>分析时会记录实际使用设备</p></div>
  <div className={d.models?.all_present?'ok':'warn'}>{d.models?.all_present?<Check/>:<AlertTriangle/>}<h3>模型 {d.models?.all_present?'已就绪':'缺失或不匹配'}</h3><p>{d.models?.models?.map((m:any)=>`${m.key}: ${m.exists?'存在':'缺失'}`).join(' · ')}</p></div>
  <div className={d.ffmpeg_binary?'ok':'warn'}>{d.ffmpeg_binary?<Check/>:<AlertTriangle/>}<h3>FFmpeg CLI {d.ffmpeg_binary?'可用':'未安装'}</h3><p>当前代理视频由 PyAV 编码，CLI 缺失不阻塞主链路。</p></div>
 </div>}</section>}
