import {useEffect,useState} from 'react'
import {api} from './api'
const reasons:Record<string,string>={no_observation:'没有球观测',predicted_only:'仅预测点',invalid_position:'位置无效',missing_track:'缺少轨迹编号',track_break:'断轨或轨迹切换',insufficient_points:'连续观测不足3点',unstable_fit:'轨迹拟合不稳定'}
type Sample={source_time_ms:number;speed_px_s:number|null;reason:string|null;track_id:number|null}
export function SpeedPanel({runId,time,seek}:{runId:string;time:number;seek:(t:number)=>void}){
 const[data,setData]=useState<any>(null),[error,setError]=useState('')
 useEffect(()=>{let alive=true;setData(null);setError('');api.get(`/api/runs/${runId}/speed`).then(d=>{if(alive)setData(d)}).catch(e=>{if(alive)setError(e.message)});return()=>{alive=false}},[runId])
 if(error)return <p role="alert">球速加载失败：{error}</p>
 if(!data)return <p>正在计算轨迹速度…</p>
 const rows:Sample[]=data.samples
 const cur=rows.reduce<Sample|null>((a,b)=>Math.abs(b.source_time_ms-time)<Math.abs((a?.source_time_ms??Infinity)-time)?b:a,null)
 const start=rows[0]?.source_time_ms??0,end=rows.at(-1)?.source_time_ms??1,span=Math.max(1,end-start)
 const max=Math.max(1,...rows.map(r=>r.speed_px_s??0))
 const segments:string[][]=[];let last:Sample|null=null
 for(const r of rows){if(r.speed_px_s===null){last=null;continue}if(!last||last.track_id!==r.track_id||r.source_time_ms-last.source_time_ms>80)segments.push([]);segments.at(-1)!.push(`${(r.source_time_ms-start)/span*1000},${110-r.speed_px_s/max*100}`);last=r}
 return <div className="viewer" style={{marginBottom:16}}><b>球的画面速度 · 自动轨迹</b><p>{cur?.speed_px_s!=null?`${cur.speed_px_s.toFixed(0)} px/s`:reasons[cur?.reason??'']??'暂无数据'} · 有效 {data.summary.valid}/{data.summary.total} 帧</p>
 <svg viewBox="0 0 1000 120" style={{width:'100%',height:130,cursor:'crosshair'}} role="img" aria-label="球画面速度曲线，空白表示不可用，点击跳转" onClick={e=>{const b=e.currentTarget.getBoundingClientRect();seek(start+(e.clientX-b.left)/b.width*span)}}>
 {segments.map((p,i)=>p.length===1?<circle key={i} cx={p[0].split(',')[0]} cy={p[0].split(',')[1]} r="2" fill="#ccff66"/>:<polyline key={i} points={p.join(' ')} fill="none" stroke="#ccff66" strokeWidth="2"/>)}
 <line x1={(time-start)/span*1000} x2={(time-start)/span*1000} y1="0" y2="120" stroke="#72a8ff"/><text x="0" y="10" fill="#aab">{max.toFixed(0)} px/s</text></svg>
 <small>{Object.entries(data.summary.reasons).map(([k,v])=>`${reasons[k]??k}：${v}`).join(' · ')}<br/>{data.note}</small></div>
}
