"""Render observed/interpolated 2D tracks without inventing a camera model."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
SIDE={'FH':'疑似正手','BH':'疑似反手','UNK':'动作类型未确认'}


def render(work, font):
    def read(n): return json.loads((work/n).read_text())
    meta=read('meta.json');pose=read('pose.json');track=read('auto_tracks.json');shots=read('shots.json');summary=read('auto_summary.json')
    external=summary.get('analysis_mode')=='external'
    external_people=read('external_people.json') if external else {}
    w,h=meta['width'],meta['height'];fps=meta['fps'];n=meta['n_frames']
    scale=min(1.5,1280/w);vw,vh=int(w*scale),int(h*scale)
    panel=350; ow=(vw+panel+15)//16*16;oh=(max(vh+110,720)+15)//16*16
    fonts={s:ImageFont.truetype(font,s) for s in (16,18,22,26)}
    tmp=work/'automatic_silent.mp4';out=work/'annotated.mp4'
    writer=iio.write_frames(str(tmp),(ow,oh),fps=fps,quality=7,macro_block_size=1);writer.send(None)
    try:
        for f in range(n):
            im=cv2.imread(str(work/'frames'/f'f_{f+1:05d}.jpg'))
            if im is None: raise RuntimeError(f'缺失帧 {f}')
            im=cv2.resize(im,(vw,vh)); canvas=np.full((oh,ow,3),(23,19,15),np.uint8);canvas[:vh,:vw]=im
            def xy(p):return tuple(int(round(v*scale)) for v in p)
            for person in external_people.get(str(f),[]):
                cv2.rectangle(canvas,xy(person['box'][:2]),xy(person['box'][2:]),(255,160,80),2)
            p=pose.get(str(f))
            if p:
                k=p['kpts']; box=p['box']
                cv2.rectangle(canvas,xy(box[:2]),xy(box[2:]),(190,190,100),1)
                for a,b in EDGES:
                    if min(k[a][2],k[b][2])>=.45:cv2.line(canvas,xy(k[a][:2]),xy(k[b][:2]),(255,205,70),2,cv2.LINE_AA)
                for x,y,c in k:
                    if c>=.45:cv2.circle(canvas,xy((x,y)),3,(255,230,150),-1)
            # Pelvis image trajectory, rather than unsupported world-space foot position.
            prev=None
            for g in range(max(0,f-round(fps*1.5)),f+1):
                po=pose.get(str(g))
                if not po or min(po['kpts'][11][2],po['kpts'][12][2])<.45:prev=None;continue
                pos=np.mean(np.array(po['kpts'])[11:13,:2],axis=0)
                if prev is not None:cv2.line(canvas,xy(prev),xy(pos),(255,145,70),2)
                prev=pos
            prev=None
            for g in range(max(0,f-round(fps*.6)),f+1):
                b=track.get(str(g))
                if not b:prev=None;continue
                pos=xy((b['x'],b['y']))
                if prev and prev['segment']==b['segment']:
                    col=(0,230,255) if b['status']==prev['status']=='observed' else (140,140,140)
                    cv2.line(canvas,xy((prev['x'],prev['y'])),pos,col,2,cv2.LINE_AA)
                prev=b
            current=track.get(str(f))
            if current:
                cv2.circle(canvas,xy((current['x'],current['y'])),7,(0,230,255) if current['status']=='observed' else (170,170,170),2)
            pil=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
            def text(x,y,t,s=18,c='#e5edf7'):d.text((x,y),t,font=fonts[s],fill=c)
            d.rectangle((8,8,280,42),fill='#101a26');text(16,12,f'{f/fps:05.2f} 秒 · 帧 {f}',18)
            x=vw+18;text(x,20,'网球自动分析',26,'#75c7ff')
            text(x,66,'外部模型 · 抽帧估计' if external else '模型观测 + 时序关联',18)
            text(x,103,f'分析抽帧：{summary["sample_frames"]} 帧' if external else f'人体覆盖：{summary["pose_coverage"]:.1%}')
            text(x,132,f'抽样球检出：{summary["sample_ball_coverage"]:.1%}' if external else f'球观测覆盖：{summary["ball_coverage"]:.1%}')
            text(x,161,f'击球候选：{len(shots)} 次')
            text(x,197,'覆盖率不等于准确率',16,'#f8c474')
            active=next((s for s in shots if abs(f-s['tc'])<=fps*.5),None)
            text(x,242,f'候选 #{active["id"]}' if active else '当前无击球候选',22)
            if active:
                text(x,281,SIDE[active['side']],22,'#ffcf77')
                labels={'audio':'音频冲击','wrist_motion':'挥拍动作','ball_near_wrist':'球接近手腕','external_model':'外部模型候选'}
                for i,e in enumerate(active['evidence']):text(x,321+i*26,labels[e],18)
                text(x,411,'自动判断，未经人工复核',16,'#a4b3c4')
            text(x,464,'真实球速：未测量',18)
            text(x,496,'三维落点：未测量',18)
            text(x,538,'未标定时不推算物理量',16,'#a4b3c4')
            text(16,vh+12,'抽帧上的框与球点为模型估计；未分析帧不补画' if external else '蓝色：人体姿态与移动   黄色：球观测   灰色：短缺口插值',18)
            y=oh-45;d.line((16,y,vw-16,y),fill='#526579',width=3)
            for s in shots:
                xx=16+(vw-32)*s['tc']/max(n-1,1);d.ellipse((xx-4,y-4,xx+4,y+4),fill='#ffce70')
            xx=16+(vw-32)*f/max(n-1,1);d.line((xx,y-9,xx,y+9),fill='#75c7ff',width=3)
            writer.send(np.asarray(pil).copy())
            if f%150==0:print(f'render {f}/{n}',flush=True)
    finally:writer.close()
    if meta.get('audio') and (work/'audio.wav').exists():
        subprocess.run([iio.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(tmp),'-i',str(work/'audio.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-af',f'apad=whole_dur={n/fps:.9f}','-t',f'{n/fps:.9f}','-movflags','+faststart',str(out)],check=True)
        tmp.unlink()
    else:tmp.replace(out)
    print(f'exported {out}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--font',required=True);a=ap.parse_args();render(a.work,a.font)
