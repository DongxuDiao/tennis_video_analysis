"""证据构建（方案 §7.1）：裁剪关键帧、写 manifest、生成受限提示词。

- 发送原始帧裁剪（非渲染答案图）；帧号/PTS/裁剪/哈希全部入 manifest。
- 允许的人物/球候选 ID 由本地面 data 提供（people.json / ball_cand.json）。
- 提示词固定模板 + 白名单编号；不渲染任何“预期答案”。
"""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from app.skill_runtime.contracts import EvidenceFrame, EvidenceManifest
from app.skill_runtime.policy import TOOL_POLICY_NOTE

MAX_CROP_SIDE = 1280
JPEG_QUALITY = 82


def build_evidence(work: Path, *, project_id: str, job_id: str, question: str,
                   frames: list[dict], allowed_person_ids: list[str],
                   allowed_ball_ids: list[str], audio_candidates: list[dict] | None = None,
                   rotation_deg: int = 0, stabilization_transform_id: str | None = None,
                   max_frames: int = 8) -> tuple[EvidenceManifest, list[str], list[str]]:
    """写 evidence/<evidence_id>/ 并返回 (manifest, data_urls, 文件绝对路径列表)。"""
    import cv2
    meta = json.loads((work / "meta.json").read_text())
    width, height = int(meta["width"]), int(meta["height"])
    evidence_id = f"ev_{secrets.token_hex(4)}"
    out_dir = work / "evidence" / evidence_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames: list[EvidenceFrame] = []
    data_urls: list[str] = []
    file_paths: list[str] = []
    for fr in frames[:max_frames]:
        f = int(fr["frame"])
        src = work / "frames" / f"f_{f+1:05d}.jpg"
        if not src.is_file():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        x1, y1, x2, y2 = _crop_box(img.shape, fr.get("roi"))
        crop = img[y1:y2, x1:x2]
        # 限边：超长缩到 1280
        scale = min(1.0, MAX_CROP_SIDE / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        asset = f"f{f}_crop.jpg"
        (out_dir / asset).write_bytes(buf.tobytes())
        digest = hashlib.sha256(buf.tobytes()).hexdigest()
        manifest_frames.append(EvidenceFrame(
            frame=f, pts_ms=float(fr.get("pts_ms", f / float(meta["fps"]) * 1000)),
            asset=asset, original_size=[width, height],
            crop_xyxy=[x1, y1, x2, y2],
            encoded_size=[crop.shape[1], crop.shape[0]],
            coordinate_space="upright_original",
            transform_id=stabilization_transform_id, sha256=digest))
        data_urls.append("data:image/jpeg;base64," +
                         __import__("base64").b64encode(buf.tobytes()).decode())
        file_paths.append(str((out_dir / asset).resolve()))
    if not manifest_frames:
        raise ValueError("无可用证据帧")

    manifest = EvidenceManifest(
        schema_version="1.0", evidence_id=evidence_id, project_id=project_id,
        job_id=job_id, question=question, frames=manifest_frames,
        allowed_person_ids=allowed_person_ids[:8],
        allowed_ball_ids=allowed_ball_ids[:16],
        audio_candidates=(audio_candidates or [])[:8],
        rotation_deg=rotation_deg,
        stabilization_transform_id=stabilization_transform_id)
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=1))
    return manifest, data_urls, file_paths


def _crop_box(shape, roi=None) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    if roi and len(roi) == 4:
        x1, y1 = max(0, int(roi[0])), max(0, int(roi[1]))
        x2, y2 = min(w, int(roi[2])), min(h, int(roi[3]))
        if x2 - x1 > 32 and y2 - y1 > 32:
            return x1, y1, x2, y2
    return 0, 0, w, h


PROMPT_TEMPLATES = {
    "attribute_contact": (
        "这些是同一段网球训练视频的连续帧（按发送顺序，帧号见 JSON）。\n"
        "人物编号：{person_ids}。任务：判断帧 {focus} 附近哪个人物编号最可能刚完成触球，"
        "或回答无法确定。\n只依据画面证据；不要猜测画面外的信息。"),
    "person_identity": (
        "这些是连续帧。人物编号：{person_ids}。任务：判断画面中是否发生了人物身份切换"
        "（同一编号前后明显不是同一人），或回答无法确定。"),
    "pick_ball_candidate": (
        "这些是连续帧。候选球编号见 JSON allowed_ball_ids（对应画面标注由本地复核使用）。"
        "任务：指出你认为哪些帧中球可见、更可能在哪个位置区域；全部不可见就说明。"
        "不要虚构不存在的候选。"),
    "hit_window": (
        "这些是连续帧。任务：估计其中是否包含一次清晰的正手或反手挥拍触球，"
        "并给出你依据的帧号区间；证据不足回答无法确定。"),
    "scene_check": (
        "这些是视频中的帧。任务：描述画面中是否存在球场线、比分条或明显机位移动迹象；"
        "仅描述可见内容。"),
}


def build_prompt(manifest: EvidenceManifest, extra: dict | None = None) -> str:
    base = PROMPT_TEMPLATES.get(manifest.question, PROMPT_TEMPLATES["scene_check"])
    prompt = base.format(
        person_ids="、".join(manifest.allowed_person_ids) or "（无编号人物）",
        focus=extra.get("focus_frame", "") if extra else "")
    # 帧号/编号清单必须显式给出，模型只能引用这些（响应契约据此校验）
    frame_list = ",".join(str(f.frame) for f in manifest.frames)
    prompt += (f"\n\n本次发送帧号（顺序与图片一致，只允许引用这些帧）：{frame_list}")
    prompt += ("\n\n输出要求：仅一个 JSON 对象，schema_version=\"1.0\"，"
               f"evidence_id=\"{manifest.evidence_id}\"，decisions 数组"
               "（每条含 kind/person_id/frame_range/ball_candidate_ids/visibility/"
               "confidence/evidence_frames/reason），needs_more_evidence 布尔。"
               "只允许引用给出的帧号与编号；不确定就写 unknown，允许空 decisions。")
    prompt += ("\n\n" + TOOL_POLICY_NOTE)
    return prompt


def candidate_ids_for(work: Path, frames: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """从本地产物取允许引用的人物/球候选 ID 与音频候选（给定窗口内）。"""
    people = json.loads((work / "people.json").read_text()) if (work / "people.json").is_file() else {}
    ball = json.loads((work / "ball_cand.json").read_text()) if (work / "ball_cand.json").is_file() else {}
    impacts = json.loads((work / "impacts.json").read_text()) if (work / "impacts.json").is_file() else {"peaks": []}
    meta = json.loads((work / "meta.json").read_text())
    fps = float(meta.get("fps") or 30)

    person_ids: set[str] = set()
    f_lo = min(f["frame"] for f in frames)
    f_hi = max(f["frame"] for f in frames)
    for f in range(max(0, f_lo - 3), f_hi + 4):
        for p in people.get(str(f), []):
            pid = p.get("track_id") or p.get("id")
            if pid is not None:
                person_ids.add(f"person_{pid}")
    person_ids.add("main")
    ball_ids: list[str] = []
    for f in range(f_lo, f_hi + 1):
        for i in range(min(3, len(ball.get(str(f), [])))):
            ball_ids.append(f"b{f}_{i}")
    audio = [{"pts_ms": round(float(p.get("t", 0)) * 1000, 1), "source": "spectral_peak"}
             for p in impacts.get("peaks", [])
             if f_lo / fps - 0.3 <= float(p.get("t", 0)) <= f_hi / fps + 0.3][:8]
    return sorted(person_ids), ball_ids[:16], audio
