import './project-storage.css'
import {useEffect,useState} from 'react'
import {api} from './api'
import type {Project} from './types'
export function ProjectStorage({project,trashed,close,changed}:{project:Project;trashed:boolean;close:()=>void;changed:()=>void}){
 const[d,setD]=useState<any>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false)
 const load=()=>api.get(`/api/projects/${project.id}/storage`).then(setD).catch(e=>setError(e.message))
 useEffect(()=>{load()},[project.id])
 const act=async(path:string,method:string,prompt:string,done=false)=>{
  if(!window.confirm(prompt))return
  setBusy(true);setError('')
  try{const r=await fetch(path,{method});if(!r.ok){const e=await r.json();throw Error(e.detail||'操作失败，可重试')}const result=await r.json();changed();if(result.cleanup_pending){setError('记录已删除，但部分隔离文件清理失败，磁盘空间尚未全部释放。');await load().catch(()=>{})}else if(done)close();else await load()}catch(e:any){setError(e.message)}finally{setBusy(false)}
 }
 const base=`/api/projects/${project.id}`
 return <div role="dialog" aria-modal="true" aria-label="管理项目存储" style={{position:'fixed',inset:0,background:'#000c',zIndex:50,display:'grid',placeItems:'center'}}><section className="viewer storage-dialog" style={{width:'min(800px,90vw)',maxHeight:'85vh',overflow:'auto',padding:24}}>
 <div className="storage-heading"><div><h2>存储管理</h2><p>{project.filename}</p></div><button className="ghost" disabled={busy} onClick={close}>关闭</button></div>
 {error&&<p role="alert">{error}</p>}{!d?<p>正在统计…</p>:<>
 <p>文件占用 {(d.bytes/1024/1024).toFixed(1)} MB · {d.runs.length} 次分析 · {d.exports.length} 次导出</p>
 <p>回收站保留视频、分析和标注，可恢复，仍占磁盘。彻底删除不可恢复；不删除共享模型或外部原文件。</p>
 {d.jobs.length>0&&<div><p>请先取消下列任务，等待结束后再删除。</p>{d.jobs.map((j:any)=><p key={j.id}>{j.type} · {j.status} <button disabled={busy} onClick={()=>act(`/api/jobs/${j.id}/cancel`,'POST','取消这个任务？')}>取消任务</button></p>)}<button onClick={load}>刷新任务状态</button></div>}
 <fieldset disabled={busy||d.jobs.length>0} style={{border:0,padding:0}}>
 {trashed?<><button className="storage-restore" onClick={()=>act(base+'/restore','POST','恢复此项目？',true)}>恢复项目</button> <button className="storage-danger" onClick={()=>act(base,'DELETE',`彻底删除 ${project.filename} 的全部视频副本、分析、标注、修订和导出文件（${(d.bytes/1024/1024).toFixed(1)} MB）？无法恢复。`,true)}>彻底删除</button></>:<button className="storage-danger" onClick={()=>act(base+'/trash','POST',`将 ${project.filename} 及其分析、标注和导出移入回收站？可以恢复。`,true)}>移入回收站</button>}
 <h3>分析版本</h3><p>删除分析会同时删除其修订、评估和导出；保留项目原视频及人工标注集。</p>
 {d.runs.length===0&&<p className="storage-empty">暂无分析版本</p>}{d.runs.map((r:any)=><p className="storage-row" key={r.id}>{r.id.slice(0,8)} · {r.status} <button className="storage-danger" onClick={()=>act(base+`/runs/${r.id}`,'DELETE','永久删除这次分析及其修订、评估和导出？依赖它的分析必须先删除。')}>删除这次分析</button></p>)}
 <h3>导出文件</h3>{d.exports.length===0&&<p className="storage-empty">暂无导出文件</p>}{d.exports.map((e:any)=><p className="storage-row" key={e.id}>{e.id.slice(0,8)} · {e.status} <button className="storage-danger" onClick={()=>act(base+`/exports/${e.id}`,'DELETE','永久清理此次导出文件？分析结果仍会保留。')}>清理导出</button></p>)}
 </fieldset></>}
 </section></div>
}
