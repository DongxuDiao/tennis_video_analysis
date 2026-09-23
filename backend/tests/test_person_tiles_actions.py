"""P1/P2/P4 新模块单元测试：人物身份关联、分块几何、挥拍时间窗、挥拍锚定事件。"""
from __future__ import annotations

from app.vision.detector import Detector, plan_tiles
from app.vision.person_tracker import PersonTracker


# ---------------- P1: 人物身份关联 ----------------

def _box(cx, cy, w=120, h=300):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_person_tracker_keeps_identity_and_does_not_switch():
    t = PersonTracker()
    # 0ms 目标在 (600,400)
    r = t.step(0, [_box(600, 400), _box(1200, 400)])
    assert r["track_id"] == 1 and r["matched"]
    tid = r["track_id"]
    # 500ms 目标移动 60px，旁边的人也还在
    r = t.step(500, [_box(660, 400), _box(1200, 400)])
    assert r["matched"] and r["track_id"] == tid
    assert abs(r["box"][0] - 600) < 40  # 跟到移动后的框
    # 1000ms 目标消失（只有另一个人）→ 不串人
    r = t.step(1000, [_box(1200, 400)])
    assert not r["matched"] and r["track_id"] == tid
    # 1200ms 目标回来 → 再关联（短暂缺席未超 lost 窗时状态仍为 tracked，重关联即可）
    r = t.step(1200, [_box(700, 400), _box(1200, 400)])
    assert r["matched"] and r["track_id"] == tid and r["status"] in ("tracked", "reacquired")


def test_person_tracker_hint_selects_clicked_box():
    t = PersonTracker(target_hint={"box": _box(1200, 400)})
    r = t.step(0, [_box(600, 400), _box(1200, 400)])
    # _box 中心为 (1200,400)，选中框左边缘应为 1200-60
    assert abs((r["box"][0] + r["box"][2]) / 2 - 1200) < 5  # 选中点击的（较小的）框


def test_person_tracker_lost_after_window():
    t = PersonTracker(lost_keep_ms=800)
    t.step(0, [_box(600, 400)])
    r = t.step(2000, [_box(600, 400)])
    # 长时间丢失后（超出保留窗）回到画面：仍尝试关联；这里距离近应可重关联
    assert r["track_id"] == 1
    r2 = t.step(2500, [])
    assert not r2["matched"]


def test_crop_with_margin_covers_swing():
    t = PersonTracker(target_hint={"box": (500, 100, 640, 500)})
    t.step(0, [(500, 100, 640, 500)])
    x1, y1, x2, y2 = t.crop_with_margin(1920, 1080, margin_ratio=0.35)
    assert x1 <= 500 and y1 <= 100 and x2 >= 640 and y2 >= 500


# ---------------- P2: 分块几何 ----------------

def test_plan_tiles_covers_whole_image_with_overlap():
    tiles = plan_tiles(1920, 1080, cols=3, rows=2, overlap=0.25)
    assert len(tiles) == 6
    xs = sorted({(r.x, r.x2) for r, _ in tiles})
    # 横向覆盖无缺口
    covered = []
    for r, _ in tiles:
        covered.append((r.x, r.x2))
    covered.sort()
    cur = 0
    for a, b in covered:
        assert a <= cur, f"分块存在横向缺口：{a} > {cur}"
        cur = max(cur, b)
    assert cur == 1920
    # 纵向
    cur = 0
    for r, _ in sorted(tiles, key=lambda t: t[0].y):
        assert r.y <= cur
        cur = max(cur, r.y2)
    assert cur == 1080


def test_tiled_detection_maps_back_coordinates(monkeypatch):
    d = Detector.__new__(Detector)
    d.imgsz = 640
    d.device_policy = "cpu"
    d._model = object()
    d._resolved_device = "cpu"

    def fake_infer(crop):
        from app.vision.detector import Candidate, bbox_center_xyxy
        h, w = crop.shape[:2]
        # 在每块中心返回一个球
        c = (w / 2, h / 2, w / 2 + 20, h / 2 + 20)
        return ([Candidate("sports ball", 0.9, c, bbox_center_xyxy(c), tile=None)],
                [], 1.0, "cpu")

    monkeypatch.setattr(Detector, "_infer_crop", lambda self, crop: fake_infer(crop))
    import numpy as np
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = d.detect_tiled(img, cols=3, rows=2, overlap=0.0)
    # 无重叠时 6 块各产生一个候选，互不重复合并
    assert len(res.candidates) == 6
    for c in res.candidates:
        x1, y1, x2, y2 = c.bbox
        assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080


def test_tiled_merge_dedupes_overlapping():
    d = Detector.__new__(Detector)
    d.imgsz = 640
    d.device_policy = "cpu"
    d._resolved_device = "cpu"

    import numpy as np
    from app.vision.detector import Candidate, bbox_center_xyxy, Detector as D

    def fake_infer(self, crop):
        # 从块内容找绿色标记（写在全局 (960,540)），返回块局部坐标
        mask = crop[:, :, 1] > 200
        if not mask.any():
            return ([], [], 1.0, "cpu")
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        c = (cx - 10, cy - 10, cx + 10, cy + 10)
        return ([Candidate("sports ball", 0.9, c, bbox_center_xyxy(c))], [], 1.0, "cpu")

    orig = D._infer_crop
    D._infer_crop = lambda self, crop: fake_infer(self, crop)
    try:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[535:546, 955:966, 1] = 255  # 全局标记 (960,540)
        res = d.detect_tiled(img, cols=3, rows=2, overlap=0.4)
        # 所有覆盖该点的块都回映到同一全局位置 → 合并为 1
        assert len(res.candidates) == 1, f"应合并为 1，得到 {len(res.candidates)}"
        c = res.candidates[0]
        assert abs(c.center[0] - 960) < 2 and abs(c.center[1] - 540) < 2
        assert c.tile_count >= 2
    finally:
        D._infer_crop = orig


# ---------------- P4: 挥拍时间窗 ----------------

def _pose_frame(t_ms, wrist_xy, vis=0.9):
    lms = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)]
    for idx, v in ((11, (640, 260)), (12, (700, 260)), (23, (650, 500)), (24, (690, 500))):
        lms[idx] = {"x": v[0], "y": v[1], "z": 0, "visibility": vis}
    # MediaPipe：15=left_wrist，16=right_wrist，14=right_elbow
    lms[16] = {"x": wrist_xy[0], "y": wrist_xy[1], "z": 0, "visibility": vis}  # right_wrist
    lms[15] = {"x": 500, "y": 400, "z": 0, "visibility": vis}                 # left_wrist（静止）
    lms[14] = {"x": 700, "y": 350, "z": 0, "visibility": vis}
    return {"frame_index": t_ms // 33, "source_pts_ms": t_ms,
            "pose": {"landmarks": lms, "person_found": True}}


def test_detect_swings_finds_time_window():
    from app.vision.actions import detect_swings
    frames = []
    # 0-1s 静止 → 1-1.4s 挥拍（腕 600→900，约 776px/s）→ 1.4-1.9s 减速回位 → 静止
    for i in range(90):
        t = i * 33
        if 1000 <= t < 1400:
            f = (t - 1000) / 400
            wrist = (600 + 300 * f, 400 + 80 * f)
        elif 1400 <= t < 1900:
            f = (t - 1400) / 500
            wrist = (900 - 300 * f, 480 - 80 * f)
        else:
            wrist = (600, 400)
        frames.append(_pose_frame(t, wrist))
    segs = detect_swings(frames, handedness="right", min_speed_px_s=600)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.side == "right"
    assert seg.peak_speed_px_s > 600
    assert seg.start_ms <= seg.peak_ms <= seg.end_ms
    assert 900 <= seg.peak_ms <= 1500


def test_swing_without_ball_is_swing_candidate_not_hit():
    from app.vision.events import extract_events
    frames = [_pose_frame(0, (600, 400)), _pose_frame(100, (750, 420)), _pose_frame(200, (900, 440))]
    events = extract_events(frames, run_id="t", handedness="right", swings=None)
    hits = [e for e in events if e["type"] == "player_hit"]
    swings = [e for e in events if e["type"] == "swing_candidate"]
    assert hits == [] and len(swings) >= 1
    assert "不计入确认击球" in swings[0]["evidence"]["note"]


def test_hit_requires_ball_proximity_near_swing():
    from app.vision.events import extract_events

    def frame(t, wrist, ball=None, racket=None):
        f = _pose_frame(t, wrist)
        f["ball"] = {"candidates": [], "selected": ({"point": list(ball), "raw": True,
                                                     "status": "tracked", "track_id": 1} if ball else None),
                     "rejected": []}
        f["detections"] = {"racket": [racket] if racket else [], "person": []}
        return f

    frames = []
    # 挥拍 + 球从远处飞向腕附近再离开（腕 640,350）
    for i, (wrist, ball) in enumerate([
            ((600, 400), None), ((640, 380), (950, 520)), ((680, 360), (760, 430)),
            ((720, 350), (660, 360)), ((700, 380), (460, 220)), ((660, 400), (300, 120))]):
        frames.append(frame(i * 66, wrist, ball))
    events = extract_events(frames, run_id="t", handedness="right")
    hits = [e for e in events if e["type"] == "player_hit"]
    assert len(hits) >= 1
    ev = hits[0]["evidence"]
    assert ev["swing_evidence"] is True
    assert ev.get("near_wrist") or ev.get("near_racket") or ev.get("direction_change_deg") is not None


def test_unsampled_person_frames_preserve_prediction_without_faking_observation():
    t = PersonTracker()
    t.step(0, [_box(600, 400)])
    for ms in (66, 132, 198, 264):
        r = t.step(ms, [], sampled=False)
        assert r['status'] == 'predicted' and r['box'] is not None
        assert not r['matched']
        assert t.state.last_seen_ms == 0
    assert t.step(300, [_box(620, 400)])['matched']
    assert t.step(2000, [], sampled=False)['status'] == 'lost'


def test_long_loss_does_not_adopt_distant_similar_person():
    t = PersonTracker()
    t.step(0, [_box(600, 400)])
    r = t.step(3000, [_box(1400, 400)])
    assert not r['matched'] and r['status'] == 'lost'
    assert t.state.last_seen_ms == 0


def test_invalid_user_hint_never_falls_back_to_another_person():
    t = PersonTracker(target_hint={'box': _box(600, 400)})
    r = t.step(0, [_box(1400, 400)])
    assert not r['matched'] and r['box'] is None
