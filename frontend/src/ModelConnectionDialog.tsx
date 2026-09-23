import {useEffect,useState} from 'react'
import {PlugZap,Unplug} from 'lucide-react'
import {SkillChoice} from './SkillChoice'
import {api} from './api'

type Conn={skill_id?:string;id:string;account_kind:string;display_name?:string;has_key:boolean;vision_ok:boolean;text_ok:boolean;eligibility_status:string;enabled:boolean;config_revision:number}

/** 唯一“模型连接”设置入口（克制的 UX）：探测 → 启用 → 断开。
 Key 只存在组件内存与请求瞬间；不写 localStorage。 */
export function ModelConnectionDialog({open,onClose,changed}:{open:boolean;onClose:()=>void;changed:()=>void}){
 const[conns,setConns]=useState<Conn[]>([]),[token,setToken]=useState(''),[doc,setDoc]=useState('')
 const[skillId,setSkillId]=useState('default')
 const[kind,setKind]=useState<'api'|'coding_plan'>('coding_plan'),[key,setKey]=useState('')
 const[probe,setProbe]=useState<any>(null),[busy,setBusy]=useState(false),[err,setErr]=useState('')
 const load=async()=>{try{const d=await api.get('/api/model-connections');setConns(d.connections||[]);setToken(d.session_token||'');setDoc(d.coding_plan_doc||'')}catch(e){setErr(String((e as Error).message))}}
 useEffect(()=>{if(open){setErr('');setProbe(null);load()}},[open])
 useEffect(()=>{const onKey=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose()};if(open)window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey)},[open,onClose])
 if(!open)return null
 const authed=async(p:string,method:string,body?:unknown)=>{
  const r=await fetch(p,{method,headers:{'Content-Type':'application/json','X-Session-Token':token},body:body?JSON.stringify(body):undefined})
  const d=await r.json().catch(()=>({detail:'请求失败'}))
  if(!r.ok)throw Error(d.detail?.message||d.detail||'请求失败')
  return d}
 const doProbe=async()=>{setBusy(true);setErr('');setProbe(null)
  try{const d=await authed('/api/model-connections/probe','POST',{provider:'glm',account_kind:kind,api_key:key})
   setProbe(d);if(!d.ok&&d.error)setErr(d.error);changed()}
  catch(e){setErr(String((e as Error).message))}finally{setBusy(false);setKey('')}}
 const doEnable=async(id:string,on:boolean,rev:number)=>{setBusy(true);setErr('')
  try{await authed(`/api/model-connections/${id}`,'PATCH',{enabled:on,expected_revision:rev});changed();load()}
  catch(e){setErr(String((e as Error).message));load()}finally{setBusy(false)}}
 const doDelete=async(id:string)=>{setBusy(true);setErr('')
  try{await authed(`/api/model-connections/${id}`,'DELETE');setProbe(null);changed();load()}
  catch(e){setErr(String((e as Error).message))}finally{setBusy(false)}}
 const doCreate=async(enable:boolean)=>{if(!probe?.probe_id)return;setBusy(true);setErr('')
  try{await authed('/api/model-connections','POST',{probe_id:probe.probe_id,enable,skill_id:skillId});setProbe(null);setKey('');changed();load()}
  catch(e){setErr(String((e as Error).message))}finally{setBusy(false)}}
 const changeSkill=async(c:Conn,id:string)=>{setBusy(true);setErr('');try{await authed(`/api/model-connections/${c.id}`,'PATCH',{expected_revision:c.config_revision,skill_id:id});await load();changed()}catch(e){setErr((e as Error).message)}finally{setBusy(false)}}
 const active=conns.find(c=>c.enabled)
 return <div className="conn-overlay" role="dialog" aria-label="模型连接" onClick={e=>{if(e.target===e.currentTarget)onClose()}}>
 <div className="conn-panel">
  <div className="conn-head"><h2>模型连接</h2><button className="icon-button" aria-label="关闭" onClick={onClose}>✕</button></div>
  {err&&<p className="conn-err" role="alert">{err}</p>}
  {conns.length>0?<>
   {conns.map(c=><div key={c.id} className="conn-item">
    <div><b>{c.display_name||`GLM ${c.account_kind}`}</b>
     <p>{c.enabled?'已连接':c.vision_ok?'可用（未启用）':'缺少视觉能力'}</p></div>
    {(c.enabled
     ?<button disabled={busy} onClick={()=>doDelete(c.id)}><Unplug size={15}/>断开</button>
     :<button className="primary" disabled={busy} onClick={()=>doEnable(c.id,true,c.config_revision)}><PlugZap size={15}/>启用</button>)
     }
   </div>)}
   {conns.map(c=><SkillChoice key={c.id} token={token} disabled={busy} value={c.skill_id||'default'} onChange={id=>changeSkill(c,id)}/>)}
   <p className="conn-note">{active
    ?'已连接：在分析方式中选择“仅外部模型”后才会发送视频抽帧。本地分析不调用此连接。'
    :'启用连接后，可在分析方式中选择外部模型。'}</p>
  </>:<>
   <label className="conn-label">接入类型
    <select value={kind} onChange={e=>{setKind(e.target.value as 'api'|'coding_plan');setProbe(null)}}>
     <option value="api">GLM 普通 API（独立计费）</option>
     <option value="coding_plan">GLM Coding Plan（套餐）</option>
    </select></label>
   {kind==='coding_plan'&&<p className="conn-note">通过 GLM Anthropic 端点连接，默认模型 glm-5.3-flash。测试通过后即可启用。</p>}
   <label className="conn-label">API Key
    <input type="password" autoComplete="off" value={key} disabled={busy}
     placeholder="粘贴你的 Key" onChange={e=>setKey(e.target.value)}/></label>
   <SkillChoice value={skillId} onChange={setSkillId} token={token} disabled={busy}/>
   <div className="conn-actions">
    <button className="primary" disabled={busy||key.length<8} onClick={doProbe}>{busy?'正在测试…':'测试并保存'}</button>
   </div>
   <p className="conn-note">测试会发送一张合成图片（非你的视频），可能消耗少量额度。</p>
   {probe?.ok&&<div className="conn-probe-ok">
    <b>探测通过{probe.capabilities?.vision_ok?' · 视觉可用':' · 视觉不可用'}</b>
    <p>选择外部分析后：视频抽帧将发送给智谱分析（Key {probe.key_mask}）。原视频不会整段上传。</p>
    <div className="conn-actions">
     <button className="primary" disabled={busy||!probe.capabilities?.vision_ok} onClick={()=>doCreate(true)}>启用连接</button>
     <button disabled={busy} onClick={()=>doCreate(false)}>仅保存（不启用）</button>
    </div></div>}
  </>}
 </div></div>
}
