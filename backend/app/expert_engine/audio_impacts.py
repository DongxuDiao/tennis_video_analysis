"""Step 3 - locate impact sounds (racket/bat/foot contact) -> impacts.json

The audio track is the cheapest and most accurate contact-time source there is: a racket
impact is a broadband click that is visible in the 2-8 kHz band even when the ball itself
is motion-blurred beyond detection. Sub-frame timing matters, because a trajectory fit
anchored 1 frame (33 ms) off biases the launch speed by several km/h.

  python3 scripts/audio_impacts.py --work work --expect 7
  python3 scripts/audio_impacts.py --work work --band 1500 9000 --min-gap 0.5 --top 12

Cross-check every peak against the video before using it: a bounce, a footstep or a voice
also produce peaks, so the peak list is a candidate list, not a shot list.
"""
import argparse
import json
import os
import wave

import numpy as np


def load_wav(path):
    with wave.open(path, 'rb') as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dt).astype(np.float32)
    if sw == 1:
        x = x - 128
    x = x.reshape(-1, ch).mean(1)
    return x / (np.abs(x).max() + 1e-9), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work')
    ap.add_argument('--audio', default='audio.wav')
    ap.add_argument('--fps', type=float, default=0, help='0 = read from meta.json')
    ap.add_argument('--band', type=float, nargs=2, default=(2000, 8000), help='impact band in Hz')
    ap.add_argument('--min-gap', type=float, default=0.35, help='minimum seconds between impacts')
    ap.add_argument('--top', type=int, default=0, help='keep only the N strongest peaks')
    ap.add_argument('--expect', type=int, default=0, help='how many impacts you expect (report only)')
    ap.add_argument('--prominence', type=float, default=3.0, help='in units of the flux MAD')
    a = ap.parse_args()

    meta = {}
    mp = os.path.join(a.work, 'meta.json')
    if os.path.exists(mp):
        meta = json.load(open(mp))
    fps = a.fps or meta.get('fps') or 30.0

    x, sr = load_wav(os.path.join(a.work, a.audio))
    from scipy import signal
    nper = 512
    f, t, S = signal.spectrogram(x, sr, nperseg=nper, noverlap=nper - 128, mode='magnitude')
    band = (f >= a.band[0]) & (f <= a.band[1])
    e = np.log10(S[band].mean(0) + 1e-8)
    flux = np.maximum(np.diff(e, prepend=e[0]), 0)          # onset strength
    pos = flux[flux > 0]
    scale = float(np.std(pos)) if len(pos) > 10 else 1.0   # robust-enough onset scale
    pk, props = signal.find_peaks(flux, prominence=a.prominence * scale,
                                  distance=max(1, int(a.min_gap / (t[1] - t[0]))))
    peaks = [dict(t=round(float(t[i]), 3), frame=round(float(t[i] * fps), 1),
                  score=round(float(props['prominences'][j] / scale), 2),
                  rms_db=round(float(20 * e[i]), 1))
             for j, i in enumerate(pk)]
    peaks.sort(key=lambda p: -p['score'])
    if a.top:
        peaks = peaks[:a.top]
    peaks.sort(key=lambda p: p['t'])

    out = dict(fps=fps, sr=sr, band=list(a.band), n=len(peaks), peaks=peaks)
    json.dump(out, open(os.path.join(a.work, 'impacts.json'), 'w'), indent=1)
    print('%d candidate impacts (expected %s)' % (len(peaks), a.expect or '?'))
    for p in peaks:
        print('  t=%6.3fs  frame %7.1f  score %5.1f' % (p['t'], p['frame'], p['score']))
    if a.expect and len(peaks) != a.expect:
        print('tune --prominence / --min-gap / --top, or accept extra candidates and '
              'filter them with the video')


if __name__ == '__main__':
    main()
