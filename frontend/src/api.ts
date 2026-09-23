const parse=async(r:Response)=>{if(!r.ok){let d:any;try{d=await r.json()}catch{d=await r.text()}throw new Error(typeof d==='string'?d:(d.detail?.message||d.detail||d.message||`请求失败 ${r.status}`))}return r.json()}
export const api={
 get:(p:string)=>fetch(p).then(parse),
 post:(p:string,b?:unknown)=>fetch(p,{method:'POST',headers:b?{'Content-Type':'application/json'}:undefined,body:b?JSON.stringify(b):undefined}).then(parse),
 upload:(file:File,mode:string="local",connectionId:string="")=>{const f=new FormData();f.append('file',file);f.append('auto_analyze','true');f.append('analysis_mode',mode);if(connectionId)f.append('connection_id',connectionId);f.append('client_request_id',crypto.randomUUID());return fetch('/api/projects',{method:'POST',body:f}).then(parse)}
}
export const fmtTime=(ms=0)=>{const s=ms/1000;return `${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,'0')}`}
export const poll=async(jobId:string,onUpdate:(j:Job_like)=>void,interval=900):Promise<Job_like>=>{
 for(;;){let j:Job_like;try{j=await api.get(`/api/jobs/${jobId}`)}catch(e){await new Promise(x=>setTimeout(x,interval));continue}
  onUpdate(j)
  if(['succeeded','failed','cancelled','interrupted'].includes(j.status))return j
  await new Promise(x=>setTimeout(x,interval))}}
type Job_like={id:string;type:string;status:string;stage?:string;progress?:any;error?:string;result?:any;cancel_requested?:boolean}
