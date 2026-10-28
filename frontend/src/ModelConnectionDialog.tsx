import {useEffect,useState} from 'react'
import {PlugZap,Unplug,Plus} from 'lucide-react'
import {SkillChoice} from './SkillChoice'
import {api} from './api'

type Conn={skill_id?:string;id:string;provider?:string;account_kind:string;display_name?:string;has_key:boolean;vision_ok:boolean;text_ok:boolean;eligibility_status:string;enabled:boolean;config_revision:number}
type KindInfo={id:string;label:string;docs_url:string;default_model:string;note:string}
type ProviderInfo={id:string;label:string;key_hint:string;docs_url:string;note:string;default_model:string;account_kinds:KindInfo[]}

/** 唯一“模型连接”设置入口（克制的 UX）：探测 → 启用 → 断开。
 服务商/接入类型来自后端 providers 目录（多厂商）；Key 只存在组件内存与请求瞬间，不写 localStorage。 */
export function ModelConnectionDialog({open,onClose,changed}:{open:boolean;onClose:()=>void;changed:()=>void}){
 const[conns,setConns]=useState<Conn[]>([]),[token,setToken]=useState(''),[doc,setDoc]=useState('')
 const[providers,setProviders]=useState<ProviderInfo[]>([]),[providerId,setProviderId]=useState('glm')
 const[skillId,setSkillId]=useState('default')
 const[kind,setKind]=useState('api'),[key,setKey]=useState(''),[modelOv,setModelOv]=useState('')
 const[adding,setAdding]=useState(false)
 const[probe,setProbe]=useState<any>(null),[busy,setBusy]=useState(false),[err,setErr]=useState('')
 const load=async()=>{try{const d=await api.get('/api/model-connections');setConns(d.connections||[]);setToken(d.session_token||'');setDoc(d.coding_plan_doc||'');setProviders(d.providers||[])}catch(e){setErr(String((e as Error).message))}}
 useEffect(()=>{if(open){setErr('');setProbe(null);setAdding(false);load()}},[open])
 useEffect(()=>{const onKey=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose()};if(open)window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey)},[open,onClose])
 if(!open)return null
 const provider=providers.find(p=>p.id===providerId)
 const kindInfo=provider?.account_kinds.find(k=>k.id===kind)
 const authed=async(p:string,method:string,body?:unknown)=>{
  const r=await fetch(p,{method,headers:{'Content-Type':'application/json','X-Session-Token':token},body:body?JSON.stringify(body):undefined})
  const d=await r.json().catch(()=>({detail:'请求失败'}))
  if(!r.ok)throw Error(d.detail?.message||d.detail||'请求失败')
  return d}
 const pickProvider=(id:string)=>{setProviderId(id);const p=providers.find(x=>x.id===id);setKind(p?.account_kinds[0]?.id||'api');setModelOv('');setProbe(null);setErr('')}
 const doProbe=async()=>{setBusy(true);setErr('');setProbe(null)
  try{const d=await authed('/api/model-connections/probe','POST',{provider:providerId,account_kind:kind,api_key:key,model:modelOv||undefined})
   setProbe(d);if(!d.ok&&d.error)setErr(d.error);changed()}
  catch(e){setErr(String((e as Error).message))}finally{setBusy(false);setKey('')}}
 const doEnable=async(id:string,on:boolean,rev:number)=>{setBusy(true);setErr('')
  try{await authed(`/api/model-connections/${id}`,'PATCH',{enabled:on,expected_revision:rev});changed();load()}
  catch(e){setErr(String((e as Error).message));load()}finally{setBusy(false)}}
 const doDelete=async(id:string)=>{setBusy(true);setErr('')
  try{await authed(`/api/model-connections/${id}`,'DELETE');setProbe(null);changed();load()}
  catch(e){setErr(String((e as Error).message))}finally{setBusy(false)}}
 const doCreate=async(enable:boolean)=>{if(!probe?.probe_id)return;setBusy(true);setErr('')
  try{await authed('/api/model-connections','POST',{probe_id:probe.probe_id,enable,skill_id:skillId});setProbe(null);setKey('');setModelOv('');setAdding(false);changed();load()}
  catch(e){setErr(String((e as Error).message))}finally{setBusy(false)}}
 const changeSkill=async(c:Conn,id:string)=>{setBusy(true);setErr('');try{await authed(`/api/model-connections/${c.id}`,'PATCH',{expected_revision:c.config_revision,skill_id:id});await load();changed()}catch(e){setErr((e as Error).message)}finally{setBusy(false)}}
 const active=conns.find(c=>c.enabled)
 const showForm=conns.length===0||adding
 const newForm=showForm&&<><label className="conn-label">服务商
   <select value={providerId} onChange={e=>pickProvider(e.target.value)}>
    {providers.map(p=><option key={p.id} value={p.id}>{p.label}</option>)}
    {!providers.length&&<option value="glm">GLM</option>}
   </select></label>
  {provider&&provider.account_kinds.length>1&&<label className="conn-label">接入类型
   <select value={kind} onChange={e=>{setKind(e.target.value);setProbe(null)}}>
    {provider.account_kinds.map(k=><option key={k.id} value={k.id}>{k.label}（{k.default_model}）</option>)}
   </select></label>}
  {provider&&(kindInfo?.note||provider.note)&&<p className="conn-note">{kindInfo?.note||provider.note}
   {(kindInfo?.docs_url||provider.docs_url)&&<> <a className="conn-doc" href={kindInfo?.docs_url||provider.docs_url} target="_blank" rel="noreferrer">获取 Key ↗</a></>}</p>}
  <label className="conn-label">API Key
   <input type="password" autoComplete="off" value={key} disabled={busy}
    placeholder={provider?.key_hint||'粘贴你的 Key'} onChange={e=>setKey(e.target.value)}/></label>
  <label className="conn-label">模型覆盖（可选）
   <input type="text" autoComplete="off" value={modelOv} disabled={busy}
    placeholder={kindInfo?.default_model||provider?.default_model||'使用默认模型'}
    onChange={e=>{setModelOv(e.target.value);setProbe(null)}}/></label>
  <SkillChoice value={skillId} onChange={setSkillId} token={token} disabled={busy}/>
  <div className="conn-actions">
   <button className="primary" disabled={busy||key.length<8} onClick={doProbe}>{busy?'正在测试…':'测试并保存'}</button>
   {adding&&<button disabled={busy} onClick={()=>{setAdding(false);setProbe(null);setErr('')}}>取消</button>}
  </div>
  <p className="conn-note">测试会发送一张合成图片（非你的视频），可能消耗少量额度。</p>
  {probe?.ok&&<div className="conn-probe-ok">
   <b>探测通过{probe.capabilities?.vision_ok?' · 视觉可用':' · 视觉不可用'}</b>
   <p>选择外部分析后：视频抽帧将发送给 {provider?.label||'该服务'} 分析（Key {probe.key_mask}）。原视频不会整段上传。</p>
   <div className="conn-actions">
    <button className="primary" disabled={busy||!probe.capabilities?.vision_ok} onClick={()=>doCreate(true)}>启用连接</button>
    <button disabled={busy} onClick={()=>doCreate(false)}>仅保存（不启用）</button>
   </div></div>}
  {probe&&!probe.ok&&probe.error_code==='capability_missing'&&
   <p className="conn-note">该服务当前未通过视觉测试，不能用于视频帧分析；请更换服务商或模型后重试。</p>}
  {doc&&providers.some(p=>p.id==='glm')&&<p className="conn-note">GLM Coding Plan 文档：<a className="conn-doc" href={doc} target="_blank" rel="noreferrer">bigmodel.cn ↗</a></p>}</>
 return <div className="conn-overlay" role="dialog" aria-label="模型连接" onClick={e=>{if(e.target===e.currentTarget)onClose()}}>
 <div className="conn-panel">
  <div className="conn-head"><h2>模型连接</h2><button className="icon-button" aria-label="关闭" onClick={onClose}>✕</button></div>
  {err&&<p className="conn-err" role="alert">{err}</p>}
  {conns.length>0?<>
   {conns.map(c=><div key={c.id} className="conn-item">
    <div><b>{c.display_name||`${c.provider||'模型'} ${c.account_kind}`}</b>
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
   {!adding&&<div className="conn-actions"><button disabled={busy} onClick={()=>{setAdding(true);setErr('')}}><Plus size={15}/>添加新连接</button></div>}
   {adding&&newForm}
  </>:newForm}
 </div></div>
}
