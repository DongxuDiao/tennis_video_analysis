import {useCallback,useEffect,useMemo,useRef,useState} from 'react'
import {ChevronLeft,Eye,Play} from 'lucide-react'
import {api,fmtTime} from './api'
import type{EventRec,FrameRec,Project,Run} from './types'
import {OverlayCanvas} from './Overlay'
import {useVideoSync} from './useVideoSync'

const eventText:Record<string,string>={player_hit:'击球',wall_contact_candidate:'撞墙候选',swing_candidate:'挥拍候选',bounce_candidate:'落地候选'}

export function Compare({project,leftId,rightId,back,setError,swap}:{project:Project;leftId:string;rightId:string;back:()=>void;setError:(s:string)=>void;swap:()=>void}){
 const[left,setLeft]=useState<Run|null>(null),[right,setRight]=useState<Run|null>(null)
 const[leftEv,setLeftEv]=useState<EventRec[]>([]),[rightEv,setRightEv]=useState<EventRec[]>([])
 const[leftFr,setLeftFr]=useState<FrameRec[]>([]),[rightFr,setRightFr]=useState<FrameRec[]>([])
 const videoL=useRef<HTMLVideoElement>(null),videoR=useRef<HTMLVideoElement>(null)
 const{mediaTimeMs}=useVideoSync(videoL,true,!!left&&!!right)
 const[time,setTime]=useState(0)
 const load=useCallback(async()=>{
  try{
   const[l,r]=await Promise.all([api.get(`/api/runs/${leftId}`),api.get(`/api/runs/${rightId}`)])
   setLeft(l);setRight(r)
   const[le,re]=await Promise.all([api.get(`/api/runs/${leftId}/events?view=automatic`),api.get(`/api/runs/${rightId}/events?view=automatic`)])
   setLeftEv(le.events);setRightEv(re.events)
   const win=async(id:string,s:number,e:number)=>{const out:FrameRec[]=[]
    for(let t=s;t<e;t+=12000)out.push(...(await api.get(`/api/runs/${id}/observations?start_ms=${t}&end_ms=${Math.min(t+11999,e)}`)).frames)
    return out}
   const s0=Math.max(l.segment_start_ms,r.segment_start_ms),e0=Math.min(l.segment_end_ms,r.segment_end_ms)
   setTime(s0)
   setLeftFr(await win(leftId,s0,e0));setRightFr(await win(rightId,s0,e0))
  }catch(e:any){setError(e.message)}
 },[leftId,rightId,setError])
 useEffect(()=>{load()},[load])
 useEffect(()=>{
  if(!left||!right)return
  const start=Math.max(left.segment_start_ms,right.segment_start_ms)/1000
  if(videoL.current)videoL.current.currentTime=start
  if(videoR.current)videoR.current.currentTime=start
 },[left,right])
 // 从属同步：右视频跟随左视频 mediaTime（帧步进式，保持同步）
 useEffect(()=>{const v=videoR.current
  if(!v||!left||!right)return
  const s0=Math.max(left.segment_start_ms,right.segment_start_ms)
  const t=Math.max(s0,Math.min(mediaTimeMs,Math.min(left.segment_end_ms,right.segment_end_ms)))
  if(mediaTimeMs>=Math.min(left.segment_end_ms,right.segment_end_ms))videoL.current?.pause()
  if(Math.abs(v.currentTime*1000-t)>40){v.currentTime=t/1000}
 },[mediaTimeMs,left,right])
 const seek=(ms:number)=>{setTime(ms)
  if(videoL.current)videoL.current.currentTime=ms/1000
  if(videoR.current)videoR.current.currentTime=ms/1000}
 const shared=useMemo(()=>{
  if(!left||!right)return null
  const s0=Math.max(left.segment_start_ms,right.segment_start_ms),e0=Math.min(left.segment_end_ms,right.segment_end_ms)
  return{s0,e0,overlap:e0>s0}
 },[left,right])
 if(!left||!right||!shared)return <section><button className="back" onClick={back}><ChevronLeft/>返回</button><p className="hint">加载中…</p></section>
 const curL=nearest(leftFr,mediaTimeMs),curR=nearest(rightFr,mediaTimeMs)
 const imgW=project.width||1920,imgH=project.height||1080
 const lc=left.summary?.ball_observation_coverage,rc=right.summary?.ball_observation_coverage
 return <section className="work">
  <div className="work-top"><button className="back" onClick={back}><ChevronLeft/>返回</button>
   <div><p className="eyebrow">A/B COMPARE</p><h2>{project.filename}</h2></div>
   <div className="spacer"/><button className="ghost" onClick={swap}><ChevronLeft/>⇄ 调换</button>
   <button className="primary small" onClick={()=>{if(videoL.current)videoL.current.playbackRate=.5;if(videoR.current)videoR.current.playbackRate=.5;videoL.current?.play()}}><Play size={14}/>0.5× 同步播放</button>
   <button className="ghost" onClick={()=>{videoL.current?.pause();videoR.current?.pause()}}>暂停</button></div>
  {!shared.overlap&&<div className="notice warn"><Eye/>两个 run 的时间区间不重叠：{fmtTime(left.segment_start_ms)}–{fmtTime(left.segment_end_ms)} vs {fmtTime(right.segment_start_ms)}–{fmtTime(right.segment_end_ms)}</div>}
  <div className="compare-grid">
   <RunPane label={`A · ${left.id.slice(0,8)} · ${left.effective_config?.processing?.detection_mode||left.config?.processing?.detection_mode||'full_frame'}`} run={left} frames={leftFr} events={leftEv} cur={curL} video={videoL} imgW={imgW} imgH={imgH} time={mediaTimeMs} seek={seek}/>
   <RunPane label={`B · ${right.id.slice(0,8)} · ${right.effective_config?.processing?.detection_mode||right.config?.processing?.detection_mode||'full_frame'}`} run={right} frames={rightFr} events={rightEv} cur={curR} video={videoR} imgW={imgW} imgH={imgH} time={mediaTimeMs} seek={seek}/>
  </div>
  <div className="compare-diff">
   <div><p className="eyebrow">参数差异</p><DiffRows a={paramLines(left)} b={paramLines(right)}/></div>
   <div><p className="eyebrow">覆盖/事件差异</p>
    <div className="diff-cell"><span>球覆盖</span><b>{fmtCov(lc)} → {fmtCov(rc)}</b></div>
    <div className="diff-cell"><span>击球数</span><b>{left.summary?.counts?.player_hit??'—'} → {right.summary?.counts?.player_hit??'—'}</b></div>
    <div className="diff-cell"><span>耗时</span><b>{fmtX(left.performance)} → {fmtX(right.performance)}</b></div>
    <div className="diff-cell"><span>内存</span><b>{left.performance?.peak_rss_mb||'—'}MB → {right.performance?.peak_rss_mb||'—'}MB</b></div>
   </div>
   <div><p className="eyebrow">事件变化（±250ms 内视为同一）</p>
    <EventDiff left={leftEv} right={rightEv} seek={seek}/></div>
  </div>
 </section>}
const fmtCov=(c?:number|null)=>c==null?'—':`${(c*100).toFixed(1)}%`
const fmtX=(p?:any)=>p?.wall_clock_ms?`${(p.wall_clock_ms/p.segment_duration_ms).toFixed(1)}×`:'—'
function paramLines(r:Run){const ec=r.effective_config||r.config||{}
 return[`模式 ${ec.processing?.detection_mode||'full_frame'}`,`输入 ${ec.model?.imgsz||'—'} · conf≥${ec.model?.ball_confidence??'—'}`,
 `分块 ${ec.processing?.tiles?`${ec.processing.tiles.cols}×${ec.processing.tiles.rows}@${ec.processing.tiles.overlap}`:'—'}`,
 `关联 ${ec.tracking?.association_max_dist_px||'—'}px · 半径 ${ec.tracking?.base_search_radius_px||'—'}px`,
 `持拍手 ${ec.handedness||'unknown'}`]}
function DiffRows({a,b}:{a:string[];b:string[]}){
 return <>{a.map((l,i)=><div key={i} className="diff-cell"><span>{l}</span><b>{b[i]===l?'相同':b[i]}</b></div>)}</>}
function nearest(frames:FrameRec[],t:number){return frames.reduce<FrameRec|null>((best,f)=>Math.abs(f.source_pts_ms-t)<Math.abs((best?.source_pts_ms??Infinity)-t)?f:best,null)}
function EventDiff({left,right,seek}:{left:EventRec[];right:EventRec[];seek:(ms:number)=>void}){
 const hitsL=left.filter(e=>e.type==='player_hit'),hitsR=right.filter(e=>e.type==='player_hit')
 const usedR=new Set<number>()
 const matchedR=hitsR.filter(r=>{const m=hitsL.find((l,i)=>!usedR.has(i)&&Math.abs(l.source_time_ms-r.source_time_ms)<=250);if(m){usedR.add(hitsL.indexOf(m));return true}return false})
 const onlyL=hitsL.filter((_,i)=>!usedR.has(i)),onlyR=hitsR.filter(r=>!matchedR.includes(r))
 return <div className="ev-diff">
  <div><b>仅 A（{onlyL.length}）</b>{onlyL.map((e,i)=><button key={i} onClick={()=>seek(e.source_time_ms)}>{fmtTime(e.source_time_ms)} {e.stroke_type}</button>)}</div>
  <div><b>仅 B（{onlyR.length}）</b>{onlyR.map((e,i)=><button key={i} onClick={()=>seek(e.source_time_ms)}>{fmtTime(e.source_time_ms)} {e.stroke_type}</button>)}</div>
  <div><b>共同（{matchedR.length}）</b>{matchedR.filter(m=>{const l=hitsL.find(x=>Math.abs(x.source_time_ms-m.source_time_ms)<=250);return l&&l.stroke_type!==m.stroke_type}).map((m,i)=><button key={i} onClick={()=>seek(m.source_time_ms)}>分类变化 {fmtTime(m.source_time_ms)}</button>)}</div>
 </div>}
function RunPane({label,run,frames,events,cur,video,imgW,imgH,time,seek}:{label:string;run:Run;frames:FrameRec[];events:EventRec[];cur:FrameRec|null;video:React.RefObject<HTMLVideoElement|null>;imgW:number;imgH:number;time:number;seek:(ms:number)=>void}){
 const layers={ball:true,predicted:true,candidates:false,skeleton:true,racket:true,person:true,personId:true,hipTrail:false,events:true,gt:false,searchCrop:false}
 const s0=run.segment_start_ms,e0=run.segment_end_ms,span=e0-s0||1
 return <div className="viewer"><div className="pane-label">{label}</div>
  <OverlayCanvas imgW={imgW} imgH={imgH} current={cur} frames={frames} events={events} layers={layers}>
   <video ref={video} src={`/api/projects/${run.project_id}/media`} muted playsInline/>
  </OverlayCanvas>
  <div className="tl-track big"><div className="tl-cursor" style={{left:`${100*Math.max(0,Math.min(1,(time-s0)/span))}%`}}/>
   {events.map(e=><button key={e.id} className={`mark ${e.type}`} style={{left:`${100*Math.max(0,Math.min(1,(e.source_time_ms-s0)/span))}%`}} onClick={()=>seek(e.source_time_ms)}/>)}</div>
  <p className="hint">{cur?`f${cur.frame_index} · 球${cur.ball?.selected?(cur.ball.selected.raw?'观测':'预测'):'丢失'}`:'无帧'}</p>
 </div>}
