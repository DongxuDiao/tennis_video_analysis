import {useEffect,useRef} from 'react'
import type{EventRec,FrameRec,GtData,Layers} from './types'
import {makeMapper} from './useVideoSync'

const POSE_EDGES=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]]

/** 轨迹分段：按 track_id 与时间间隔切分。
 * 两段不同轨迹（track 变化）或时间缺口超过 gapMs 的点不连线——
 * 断轨处的错误连线不能用叉号掩盖，必须从几何上断开。 */
export function segmentTrail(items:Array<{t:number;track?:number;p:number[]}>,gapMs:number):number[][][]{
 const segs:number[][][]=[]
 let cur:number[][]=[]
 let prevT:number|null=null
 let prevTrack:number|undefined
 for(const it of items){
  const trackChanged=it.track!==undefined&&prevTrack!==undefined&&it.track!==prevTrack
  const gapTooLong=prevT!==null&&it.t-prevT>gapMs
  if(prevT!==null&&(trackChanged||gapTooLong)){
   if(cur.length>1)segs.push(cur)
   cur=[]
  }
  cur.push(it.p)
  prevT=it.t
  if(it.track!==undefined)prevTrack=it.track
 }
 if(cur.length>1)segs.push(cur)
 return segs
}

/** 骨架短时复用（仅显示）：后端按 ~66ms 采样姿态，60fps 下多数帧无姿态数据。
 * 在 reuseMs 内取最近含姿态帧显示并标注来源；超窗不显示（不跨长遮挡连线）。 */
export function nearestPoseFrame(frames:FrameRec[],tMs:number,reuseMs=100):{frame:FrameRec;dtMs:number}|null{
 const current=frames.find(f=>f.source_pts_ms===tMs)
 if(current?.person?.status==="lost"||current?.pose?.source==="target_unavailable")return null
 const tid=current?.person?.track_id
 let best:{frame:FrameRec;dtMs:number}|null=null
 for(const f of frames){
  if(!f.pose?.landmarks)continue
  if(tid!=null&&f.person?.track_id!==tid)continue
  if(frames.some(v=>v.source_pts_ms>=Math.min(tMs,f.source_pts_ms)&&v.source_pts_ms<=Math.max(tMs,f.source_pts_ms)&&(v.person?.status==="lost"||v.pose?.source==="target_unavailable")))continue
  const dt=Math.abs(f.source_pts_ms-tMs)
  if(dt<=reuseMs&&(!best||dt<best.dtMs))best={frame:f,dtMs:dt}
 }
 return best
}

/** 叠加画布：观测=实线，预测=虚线，人工=紫色。断轨处画 ×（不用连线掩盖丢失）。 */
export function OverlayCanvas({imgW,imgH,current,frames,events,gt,layers,annotate,onAnnotate,children,hideVideo}:{
 imgW:number;imgH:number
 current:FrameRec|null
 frames:FrameRec[]
 events:EventRec[]
 gt?:GtData|null
 layers:Layers
 annotate?:boolean
 onAnnotate?:(x:number,y:number)=>void
 children?:React.ReactNode
 hideVideo?:boolean
}){
 const wrap=useRef<HTMLDivElement>(null),canvas=useRef<HTMLCanvasElement>(null)
 useEffect(()=>{
  const c=canvas.current,w=wrap.current
  if(!c||!w)return
  const rect=w.getBoundingClientRect(),d=window.devicePixelRatio||1
  c.width=rect.width*d;c.height=rect.height*d
  c.style.width=`${rect.width}px`;c.style.height=`${rect.height}px`
  const x=c.getContext('2d')!
  x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,rect.width,rect.height)
  const m=makeMapper(rect.width,rect.height,imgW,imgH)
  const P=(p:number[])=>{const q=m.toDisplay(p);return[q.x,q.y]as[number,number]}
  const t=current?.source_pts_ms??0

  // ---- 人物框 / ID / 目标丢失 ----
  if(layers.person&&current){
   for(const b of current.person?.boxes||[]){
    const [a,bb]=[P([b[0],b[1]]),P([b[2],b[3]])]
    const isTarget=current.person?.target&&b.every((v,i)=>Math.abs(v-current.person!.target!.box[i])<2)
    x.strokeStyle=isTarget?'#53e6a6':'#5a6b81';x.lineWidth=isTarget?2.5:1.5
    x.strokeRect(a[0],a[1],bb[0]-a[0],bb[1]-a[1])
    if(layers.personId&&isTarget){
     x.fillStyle='#53e6a6';x.font='11px DM Mono,monospace'
     x.fillText(`P${current.person!.target!.track_id} · ${current.person!.target!.status}`,a[0],a[1]-5)
    }
   }
   const st=current.person?.status
   if(st==='lost'||st==='coasting'){
    x.fillStyle='rgba(255,170,60,.92)';x.font='600 13px Manrope,sans-serif'
    x.fillText(st==='lost'?'目标丢失（不切换）':'目标间歇（预测中）',12,rect.height-12)
   }
  }
  // ---- 髋部中心人体轨迹（按人物 track_id 与时间缺口分段）----
  if(layers.hipTrail){
   const items=frames
    .filter(f=>f.source_pts_ms<=t&&f.source_pts_ms>=t-4000&&f.person?.hip_center?.point)
    .map(f=>({t:f.source_pts_ms,track:f.person?.track_id??undefined,p:f.person!.hip_center!.point}))
   for(const seg of segmentTrail(items,500)){
    x.strokeStyle='rgba(83,230,166,.55)';x.lineWidth=1.5;x.setLineDash([])
    x.beginPath();seg.forEach((p,i)=>{const q=P(p);i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])});x.stroke()
   }
  }
  // ---- 骨架（短时复用消除抽帧闪烁；复用时标注，仅显示用途）----
  if(layers.skeleton){
   const poseSrc=current?nearestPoseFrame(frames,current.source_pts_ms,100):null
   if(poseSrc){
    const lm=poseSrc.frame.pose!.landmarks!
    const reused=poseSrc.dtMs>17 // 超过一帧间隔即为复用
    x.strokeStyle='#53e6a6';x.lineWidth=2;x.setLineDash([])
    for(const[a,b]of POSE_EDGES){if(lm[a]?.visibility>.5&&lm[b]?.visibility>.5){const p=P([lm[a].x,lm[a].y]),q=P([lm[b].x,lm[b].y]);x.beginPath();x.moveTo(...p);x.lineTo(...q);x.stroke()}}
    for(const l of lm){if(l.visibility>.5){const p=P([l.x,l.y]);x.fillStyle='#53e6a6';x.fillRect(p[0]-1.5,p[1]-1.5,3,3)}}
    if(reused){
     x.fillStyle='rgba(140,155,175,.85)';x.font='10px DM Mono,monospace'
     x.fillText(`骨架复用 Δ${poseSrc.dtMs}ms（仅显示，非本帧检测）`,12,26)
    }
   }
  }
  // ---- 球候选（含拒绝原因 tooltip 交给证据区）----
  if(layers.candidates&&current){
   for(const b of current.ball?.candidates||[]){
    const [a,bb]=[P([b.bbox[0],b.bbox[1]]),P([b.bbox[2],b.bbox[3]])]
    x.strokeStyle='#94a3b8';x.lineWidth=1;x.setLineDash([3,3]);x.strokeRect(a[0],a[1],bb[0]-a[0],bb[1]-a[1])
    x.fillStyle='#94a3b8';x.font='9px DM Mono';x.fillText((b.confidence??0).toFixed(2),bb[0]+2,a[1])
   }
   x.setLineDash([])
  }
  // ---- 真实轨迹（仅观测点，实线）与预测轨迹（虚线）分别按 track 与时间缺口分段；断轨画 × ----
  const recent=frames.filter(f=>f.source_pts_ms<=t&&f.source_pts_ms>=t-1500)
  if(layers.ball){
   const obsItems=recent
    .filter(f=>f.ball?.selected?.raw&&f.ball.selected.point)
    .map(f=>({t:f.source_pts_ms,track:f.ball!.selected!.track_id??undefined,p:f.ball!.selected!.point}))
   for(const seg of segmentTrail(obsItems,120)){
    x.strokeStyle='#f8e45c';x.lineWidth=2;x.setLineDash([])
    x.beginPath();seg.forEach((p,i)=>{const q=P(p);i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])});x.stroke()
   }
  }
  if(layers.predicted){
   const predItems=recent
    .filter(f=>f.ball?.selected&&!f.ball.selected.raw&&f.ball.selected.point&&f.ball.selected.status==='predicted')
    .map(f=>({t:f.source_pts_ms,track:f.ball!.selected!.track_id??undefined,p:f.ball!.selected!.point}))
   for(const seg of segmentTrail(predItems,120)){
    x.strokeStyle='#ff9f43';x.lineWidth=1.5;x.setLineDash([5,4])
    x.beginPath();seg.forEach((p,i)=>{const q=P(p);i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])});x.stroke()
   }
   x.setLineDash([])
  }
  // 断轨标记：上一帧有球、这一帧 lost → ×
  if(layers.ball){
   const idx=frames.findIndex(f=>f.source_pts_ms===current?.source_pts_ms)
   if(idx>0){const prev=frames[idx-1]
    if(prev?.ball?.selected?.raw&&(!current?.ball?.selected)){
     const p=P(prev.ball.selected.point)
     x.strokeStyle='#ff6478';x.lineWidth=2
     x.beginPath();x.moveTo(p[0]-7,p[1]-7);x.lineTo(p[0]+7,p[1]+7);x.moveTo(p[0]+7,p[1]-7);x.lineTo(p[0]-7,p[1]+7);x.stroke()
    }}
  }
  // 当前球点
  if(layers.ball&&current?.ball?.selected){
   const p=P(current.ball.selected.point)
   x.strokeStyle=current.ball.selected.raw?'#f8e45c':'#ff9f43';x.lineWidth=3
   x.beginPath();x.arc(p[0],p[1],9,0,Math.PI*2);x.stroke()
   if(!current.ball.selected.raw){x.setLineDash([4,3]);x.beginPath();x.arc(p[0],p[1],13,0,Math.PI*2);x.stroke();x.setLineDash([])}
   x.fillStyle='#f8e45c';x.font='10px DM Mono';x.fillText(current.ball.selected.status,p[0]+13,p[1]-8)
  }
  // ---- 搜索窗（局部模式）----
  if(layers.searchCrop&&current?.ball?.selected?.search_crop){
   const[a,b]=[P([current.ball.selected.search_crop[0],current.ball.selected.search_crop[1]]),P([current.ball.selected.search_crop[2],current.ball.selected.search_crop[3]])]
   x.strokeStyle='rgba(114,168,255,.5)';x.lineWidth=1;x.setLineDash([6,5]);x.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);x.setLineDash([])
  }
  // ---- 球拍 ----
  if(layers.racket&&current){for(const b of current.detections?.racket||[]){const[a,bb]=[P([b[0],b[1]]),P([b[2],b[3]])];x.strokeStyle='#fb7185';x.lineWidth=2;x.strokeRect(a[0],a[1],bb[0]-a[0],bb[1]-a[1])}}
  // ---- 事件 ----
  if(layers.events){for(const e of events.filter(e=>Math.abs(e.source_time_ms-t)<200)){
   if(!e.point)continue;const p=P([e.point.x,e.point.y])
   const color=e.review_status==='excluded'?'#666':e.origin!=='automatic'?'#d879ff':e.type==='player_hit'?'#ff566f':'#5aa9ff'
   x.strokeStyle=color;x.lineWidth=3;x.beginPath();x.arc(p[0],p[1],14,0,Math.PI*2);x.stroke()
  }}
  // ---- 人工标注点 ----
  if(layers.gt&&gt){
   for(const bp of gt.ball_points){if(bp.unobservable||!bp.point)continue
    if(Math.abs(bp.source_time_ms-t)>150)continue
    const p=P(bp.point);x.strokeStyle='#d879ff';x.lineWidth=2;x.beginPath();x.arc(p[0],p[1],6,0,Math.PI*2);x.stroke()}
   for(const ev of gt.events){if(Math.abs(ev.source_time_ms-t)>200||!ev.point)continue
    const p=P(ev.point);x.strokeStyle='#d879ff';x.lineWidth=2.5;x.strokeRect(p[0]-9,p[1]-9,18,18)}
  }
 },[current,frames,events,gt,layers,imgW,imgH])
 const handleClick=(e:React.MouseEvent)=>{
  if(!annotate||!onAnnotate||!wrap.current)return
  const rect=wrap.current.getBoundingClientRect()
  const m=makeMapper(rect.width,rect.height,imgW,imgH)
  const img=m.toImage(e.clientX-rect.left,e.clientY-rect.top)
  onAnnotate(img.x,img.y)
 }
 return <div className="overlay-wrap" ref={wrap} onClick={handleClick} style={annotate?{cursor:'crosshair'}:undefined}>
  <div className="media-slot" style={hideVideo?{display:'none'}:undefined}>{children}</div>
  <canvas ref={canvas}/>
 </div>
}
