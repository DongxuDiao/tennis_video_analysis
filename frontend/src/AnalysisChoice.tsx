export type AnalysisMode='local'|'external'
export type ConnectionChoice={id:string;display_name:string;enabled:boolean;vision_ok:boolean}
export function AnalysisChoice({mode,setMode,connections,connectionId,setConnectionId,disabled=false,connect}:{mode:AnalysisMode;setMode:(m:AnalysisMode)=>void;connections:ConnectionChoice[];connectionId:string;setConnectionId:(s:string)=>void;disabled?:boolean;connect:()=>void}){
 return <div className="analysis-choice">
  <label>分析方式 <select aria-label="分析方式" value={mode} disabled={disabled} onChange={e=>setMode(e.target.value as AnalysisMode)}><option value="local">本地分析</option><option value="external">仅外部模型</option></select></label>
  {mode==='external'&&<>{connections.length?<label>模型 <select aria-label="分析模型" value={connectionId} disabled={disabled} onChange={e=>setConnectionId(e.target.value)}><option value="" disabled>选择已连接的模型</option>{connections.map(c=><option key={c.id} value={c.id}>{c.display_name}</option>)}</select></label>:<button disabled={disabled} onClick={connect}>连接外部模型</button>}</>}
  <p>{mode==='local'?'仅在本机识别人物和球，不调用外部模型。':'按所选 Skill 将视频抽帧发送给外部模型，消耗服务额度；跳过本地识别，失败不自动切换。首版约 4 帧/秒，非连续逐帧追踪。'}</p>
 </div>
}
