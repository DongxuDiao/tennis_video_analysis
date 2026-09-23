import {useEffect,useRef,useState} from 'react'

/** 视频时间同步：优先 requestVideoFrameCallback（真实 mediaTime），
 * 回退 timeupdate。仅在视频实际呈现新帧时回调，避免高频 React 重绘。 */
export function useVideoSync(video:React.RefObject<HTMLVideoElement|null>,enabled=true,ready=true){
 const [mediaTimeMs,setMediaTimeMs]=useState(0)
 const [playing,setPlaying]=useState(false)
 const [rvfcSupported,setRvfcSupported]=useState(false)
 const raf=useRef<number>(0)
 useEffect(()=>{
  const v=video.current
  if(!v||!ready)return
  setRvfcSupported('requestVideoFrameCallback' in v)
  const onPlay=()=>setPlaying(true),onPause=()=>setPlaying(false)
  v.addEventListener('play',onPlay);v.addEventListener('pause',onPause)
  let cancelled=false
  let callbackId=0
  const onSeek=()=>setMediaTimeMs(Math.round(v.currentTime*1000))
  v.addEventListener("seeked",onSeek)
  if('requestVideoFrameCallback' in v&&enabled){
   const cb=(_now:number,meta:{mediaTime:number})=>{
    if(cancelled)return
    setMediaTimeMs(Math.round(meta.mediaTime*1000))
    ;callbackId=(v as any).requestVideoFrameCallback(cb)
   }
   ;callbackId=(v as any).requestVideoFrameCallback(cb)
   return()=>{cancelled=true;v.cancelVideoFrameCallback?.(callbackId);v.removeEventListener("seeked",onSeek);v.removeEventListener('play',onPlay);v.removeEventListener('pause',onPause)}
  }
  const onTime=()=>setMediaTimeMs(Math.round(v.currentTime*1000))
  v.addEventListener('timeupdate',onTime)
  return()=>{cancelled=true;v.cancelVideoFrameCallback?.(callbackId);v.removeEventListener("seeked",onSeek);v.removeEventListener('timeupdate',onTime);v.removeEventListener('play',onPlay);v.removeEventListener('pause',onPause)}
 },[video,enabled,ready])
 useEffect(()=>()=>cancelAnimationFrame(raf.current),[])
 return{mediaTimeMs,playing,rvfcSupported}
}

/** 分析图像 → 显示坐标 的等比映射（含 letterbox 偏移）。 */
export function makeMapper(displayW:number,displayH:number,imgW:number,imgH:number){
 const scale=Math.min(displayW/imgW,displayH/imgH)
 const ox=(displayW-imgW*scale)/2,oy=(displayH-imgH*scale)/2
 return{
  scale,ox,oy,
  toDisplay:(p:number[])=>({x:ox+p[0]*scale,y:oy+p[1]*scale}),
  toImage:(x:number,y:number)=>({x:(x-ox)/scale,y:(y-oy)/scale})
 }
}
