"""Image-plane speed from contiguous raw observations, independent of events."""
import math
from collections import Counter


def speed_series(frames, window_ms=120, max_gap_ms=80, min_points=3):
    rows, window = [], []
    previous = None
    for f in sorted(frames, key=lambda f: f['source_pts_ms']):
        t = f['source_pts_ms']
        s = (f.get('ball') or {}).get('selected') or {}
        row = dict(source_time_ms=t, frame_index=f['frame_index'], speed_px_s=None,
                   reason=None, points=0, track_id=s.get('track_id'))
        p = s.get('point')
        if not s.get('raw'):
            row['reason'] = 'predicted_only' if s else 'no_observation'
        elif not p or len(p) != 2 or not all(math.isfinite(v) for v in p):
            row['reason'] = 'invalid_position'
        elif s.get('track_id') is None:
            row['reason'] = 'missing_track'
        if row['reason']:
            window = []
            previous = None
        else:
            cut = previous is not None and (s['track_id'] != previous[1] or not 0 < t-previous[0] <= max_gap_ms)
            if cut:
                window = []
            window.append((t, p))
            window = [(tt, pp) for tt, pp in window if t-tt <= window_ms]
            row['points'] = len(window)
            row['reason'] = 'track_break' if cut else 'insufficient_points'
            if len(window) >= min_points:
                ts = [(tt-t)/1000 for tt, _ in window]
                mt = sum(ts)/len(ts)
                den = sum((v-mt)**2 for v in ts)
                means = [sum(pp[k] for _, pp in window)/len(window) for k in (0, 1)]
                if den > 0:
                    slopes = [sum((tt-mt)*(pp[k]-means[k]) for tt, (_, pp) in zip(ts, window))/den for k in (0, 1)]
                    residual = math.sqrt(sum(sum((pp[k]-means[k]-slopes[k]*(tt-mt))**2 for k in (0, 1)) for tt, (_,pp) in zip(ts,window))/len(window))
                    row['fit_error_px'] = round(residual, 2)
                    row['reason'] = 'unstable_fit' if residual > 15 else None
                    if row['reason'] is None:
                        row['speed_px_s'] = round(math.hypot(*slopes), 1)
            previous = (t, s['track_id'])
        rows.append(row)
    valid = [r['speed_px_s'] for r in rows if r['speed_px_s'] is not None]
    return dict(samples=rows, unit='px/s', view='automatic',
                parameters=dict(window_ms=window_ms, max_gap_ms=max_gap_ms, min_points=min_points, max_fit_error_px=15),
                summary=dict(total=len(rows), valid=len(valid), reasons=dict(Counter(r['reason'] for r in rows if r['reason']))),
                note='原始自动观测；预测点不测速。短窗拟合可能平滑击球峰值；无空间标定，不输出 km/h。')
