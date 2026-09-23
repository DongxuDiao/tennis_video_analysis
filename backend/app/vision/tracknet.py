"""Three-frame TrackNet adapter, compatible with yastrebksv/TrackNet state_dict.
BGR current/previous/previous-previous, 640x360, 256 intensity classes.
Heatmap intensity is a detection score, NOT a calibrated probability.
"""
from collections import deque
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch import nn
from app.vision.detector import Candidate, DetectionResult


class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(b))
    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [9,64,64,128,128,256,256,256,512,512,512,256,256,256,128,128,64,64,256]
        for i, (a,b) in enumerate(zip(channels, channels[1:]), 1):
            setattr(self, f'conv{i}', ConvBlock(a,b))
    def forward(self, x):
        for i in range(1,19):
            x = getattr(self, f'conv{i}')(x)
            if i in (2,4,7):
                x = nn.functional.max_pool2d(x,2)
            if i in (10,13,15):
                x = nn.functional.interpolate(x,scale_factor=2,mode='nearest')
        return x.reshape(x.shape[0],256,-1)


def heatmap_candidates(heat, width, height, threshold=.5):
    mask = (heat >= max(1, threshold*255)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    result = []
    sx, sy = width/heat.shape[1], height/heat.shape[0]
    for k in range(1,count):
        x,y,w,h,area = stats[k]
        if area > 600:  # reject broad activated background
            continue
        cx,cy = centers[k]
        score = float(heat[labels==k].max())/255
        # Heatmap support box, not a measured physical ball diameter.
        bw,bh = max(4,w*sx),max(4,h*sy)
        cx,cy = (cx+.5)*sx,(cy+.5)*sy
        result.append(Candidate('sports ball',score,(cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),(cx,cy),'tracknet'))
    return sorted(result,key=lambda c:c.confidence,reverse=True)[:8]


class TrackNetDetector:
    def __init__(self, path, device='auto_mps_then_cpu', threshold=.5):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f'TrackNet 权重缺失：{path}。请安装兼容的 tennis state_dict；不会回退成 YOLO。')
        self.policy = device
        self.device = 'mps' if device != 'cpu' and torch.backends.mps.is_available() else 'cpu'
        if device == 'mps' and self.device != 'mps':
            raise RuntimeError('指定 MPS 不可用')
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.model.eval().to(self.device)
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.frames = deque(maxlen=3)
        self.last = None
        self.threshold = threshold

    def detect(self, image, pts_ms):
        t0 = time.perf_counter()
        if self.last is not None and (pts_ms <= self.last or pts_ms-self.last > 150):
            self.frames.clear()
        self.last = pts_ms
        self.frames.append(cv2.resize(image,(640,360)))
        result = DetectionResult(mode='tracknet',device=self.device)
        if len(self.frames)<3:
            return result
        arr = np.concatenate(list(self.frames)[::-1],axis=2).transpose(2,0,1).copy()
        inp = torch.from_numpy(arr).float().unsqueeze(0)/255
        devices = [self.device] + (['cpu'] if self.device=='mps' and self.policy=='auto_mps_then_cpu' else [])
        for dev in devices:
            try:
                self.model.to(dev)
                with torch.inference_mode():
                    heat = self.model(inp.to(dev)).argmax(1).reshape(360,640).cpu().numpy().astype(np.uint8)
                self.device = dev
                result.attempts.append({'device':dev,'ok':True})
                break
            except RuntimeError as exc:
                result.attempts.append({'device':dev,'ok':False,'error':str(exc)[:300]})
                if dev==devices[-1]:
                    raise
        result.candidates = heatmap_candidates(heat,image.shape[1],image.shape[0],self.threshold)
        result.elapsed_ms = (time.perf_counter()-t0)*1000
        result.device = self.device
        return result
