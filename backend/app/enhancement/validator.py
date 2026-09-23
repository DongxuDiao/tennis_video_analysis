"""模型提议验证（方案 §7.3）：本地证据裁决 accepted / rejected / unresolved。

每类判定必须通过对应本地证据规则；未通过的提议只进审计，不进入测量统计。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.skill_runtime.contracts import Decision, EvidenceManifest


@dataclass
class Verdict:
    status: str                    # accepted | rejected | unresolved
    reason_code: str
    detail: str = ""
    applied: dict = field(default_factory=dict)


def _local_ball_points(work, frame: int) -> list[dict]:
    import json
    p = work / "ball_cand.json"
    if not p.is_file():
        return []
    data = json.loads(p.read_text())
    return data.get(str(frame), [])


def _local_poses(work, frame: int) -> dict | None:
    import json
    p = work / "pose.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    return data.get(str(frame))


def validate_decision(decision: Decision, manifest: EvidenceManifest, work) -> Verdict:
    kind = decision.kind
    if kind == "contact_candidate":
        return _validate_contact(decision, manifest, work)
    if kind == "ball_pick":
        return _validate_ball(decision, manifest, work)
    if kind == "person_id":
        return _validate_person(decision, manifest, work)
    if kind in ("hit_window", "scene_note", "unknown_note"):
        # 仅作提示，不直接修改数值类结果 → 接受为"提示性"但不 apply
        if not decision.evidence_frames:
            return Verdict("rejected", "no_evidence_frames")
        return Verdict("accepted", "advisory_only", applied={})
    return Verdict("rejected", "unknown_kind")


def _validate_contact(decision: Decision, manifest: EvidenceManifest, work) -> Verdict:
    """触球归属：需要本地挥拍/音频/球接近任一支持（§7.3）。"""
    import json
    import math
    if decision.person_id is None:
        return Verdict("unresolved", "no_person_id")
    if not decision.frame_range:
        return Verdict("unresolved", "no_frame_range")
    a, b = decision.frame_range
    meta = json.loads((work / "meta.json").read_text())
    fps = float(meta.get("fps") or 30)
    # 本地证据：音频峰或 tracks 在窗口内接近该人物
    impacts = json.loads((work / "impacts.json").read_text()) if (work / "impacts.json").is_file() else {"peaks": []}
    audio_hit = any(a / fps - 0.15 <= float(p.get("t", -9)) <= b / fps + 0.15
                    for p in impacts.get("peaks", []))
    tracks = json.loads((work / "auto_tracks.json").read_text()) if (work / "auto_tracks.json").is_file() else {}
    ball_near = False
    for f in range(a, b + 1):
        t = tracks.get(str(f))
        if t and t.get("status") == "observed":
            pose = _local_poses(work, f)
            if pose:
                # 触球近似：球点距人物框中心 < 框高
                bx, by = float(t["x"]), float(t["y"])
                px = (pose["box"][0] + pose["box"][2]) / 2
                py = (pose["box"][1] + pose["box"][3]) / 2
                if math.hypot(bx - px, by - py) < max(80, (pose["box"][3] - pose["box"][1])):
                    ball_near = True
                    break
    if audio_hit or ball_near:
        return Verdict("accepted", "local_evidence_supported",
                       applied={"person_id": decision.person_id,
                                "frame_range": [a, b]})
    return Verdict("unresolved", "insufficient_local_evidence",
                   "无音频峰或球接近支持")


def _validate_ball(decision: Decision, manifest: EvidenceManifest, work) -> Verdict:
    """球点复核：候选必须在本地候选集中且画面内；坐标只产生待重检区域。"""
    if not decision.ball_candidate_ids:
        if decision.visibility == "not_visible":
            return Verdict("accepted", "marked_not_visible",
                           applied={"not_visible_frames": decision.evidence_frames})
        return Verdict("unresolved", "no_candidates")
    ok_ids, bad = [], []
    for f in decision.evidence_frames:
        local = _local_ball_points(work, f)
        for cid in decision.ball_candidate_ids:
            # b<frame>_<i> 形式必须与本地帧号一致
            try:
                cf = int(cid.split("_")[0][1:])
                idx = int(cid.split("_")[1])
                if cf == f and 0 <= idx < len(local):
                    ok_ids.append(cid)
                else:
                    bad.append(cid)
            except (ValueError, IndexError):
                bad.append(cid)
    if bad:
        return Verdict("rejected", "candidate_not_in_local_set", f"{bad[:3]}")
    if not ok_ids:
        return Verdict("unresolved", "no_valid_candidates")
    # V1：接受为“候选优先级提示”，不直接作为观测点
    return Verdict("accepted", "candidate_priority_hint",
                   applied={"preferred_candidates": sorted(set(ok_ids))})


def _validate_person(decision: Decision, manifest: EvidenceManifest, work) -> Verdict:
    """人物身份：需跨帧位置/尺度连续；单帧断言不足以覆盖轨迹。"""
    if decision.person_id is None:
        return Verdict("unresolved", "no_person_id")
    if not decision.evidence_frames or len(decision.evidence_frames) < 2:
        return Verdict("rejected", "single_frame_insufficient",
                       "人物身份判定需至少两帧证据")
    import json
    people = json.loads((work / "people.json").read_text()) if (work / "people.json").is_file() else {}
    boxes = []
    for f in decision.evidence_frames:
        frame_people = people.get(str(f), [])
        if not frame_people:
            continue
        box = frame_people[0]["box"]
        boxes.append((f, box))
    if len(boxes) < 2:
        return Verdict("unresolved", "local_person_boxes_missing")
    # 尺度/位移连续性：相邻帧位移 < 1.2×框高、尺度变化 < 40%
    import math
    for (fa, ba), (fb, bb) in zip(boxes, boxes[1:]):
        ha = ba[3] - ba[1]
        dx = (ba[0] + ba[2]) / 2 - (bb[0] + bb[2]) / 2
        dy = (ba[1] + ba[3]) / 2 - (bb[1] + bb[3]) / 2
        if math.hypot(dx, dy) > 1.2 * max(ha, 1):
            return Verdict("rejected", "identity_not_continuous",
                           f"帧 {fa}→{fb} 位移超界")
        hb = bb[3] - bb[1]
        if ha and abs(hb - ha) / ha > 0.4:
            return Verdict("rejected", "scale_jump")
    return Verdict("accepted", "cross_frame_continuous",
                   applied={"person_id": decision.person_id})
