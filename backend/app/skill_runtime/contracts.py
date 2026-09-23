"""证据与模型响应的严格契约（方案 §7.1/§7.2）。

Pydantic 严格模式：禁额外字段、NaN/Infinity；限制尺寸/长度。
EvidenceManifest 由本地构建（模型不可伪造）；ModelReviewResponse 解析自模型文本，
任何越界（未发送帧号、未提供人物/球 ID、越界坐标、超长）整条拒绝。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QUESTION_KINDS = ("attribute_contact", "pick_ball_candidate", "hit_window",
                  "scene_check", "person_identity")


class EvidenceFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    frame: int = Field(ge=0)
    pts_ms: float = Field(ge=0)
    asset: str = Field(min_length=1, max_length=120)
    original_size: list[int] = Field(min_length=2, max_length=2)
    crop_xyxy: list[int] = Field(min_length=4, max_length=4)
    encoded_size: list[int] = Field(min_length=2, max_length=2)
    coordinate_space: Literal["upright_original"]
    transform_id: str | None = None
    sha256: str = Field(min_length=64, max_length=64)


class EvidenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    evidence_id: str = Field(min_length=3, max_length=40)
    project_id: str
    job_id: str
    question: Literal["attribute_contact", "pick_ball_candidate", "hit_window",
                      "scene_check", "person_identity"]
    frames: list[EvidenceFrame] = Field(min_length=1, max_length=8)
    allowed_person_ids: list[str] = Field(default_factory=list, max_length=8)
    allowed_ball_ids: list[str] = Field(default_factory=list, max_length=16)
    audio_candidates: list[dict] = Field(default_factory=list, max_length=8)
    rotation_deg: int = 0
    stabilization_transform_id: str | None = None

    def frame_ids(self) -> set[int]:
        return {f.frame for f in self.frames}


class Decision(BaseModel):
    """单条模型判定。confidence 是模型自评，不是准确率。"""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["contact_candidate", "ball_pick", "person_id", "hit_window",
                  "scene_note", "unknown_note"]
    person_id: str | None = None
    frame_range: list[int] | None = Field(default=None, min_length=2, max_length=2)
    ball_candidate_ids: list[str] = Field(default_factory=list, max_length=8)
    visibility: Literal["visible", "uncertain", "not_visible", "unknown"] = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_frames: list[int] = Field(default_factory=list, max_length=8)
    reason: str = Field(default="", max_length=400)
    crop_hint_xyxy: list[int] | None = Field(default=None, min_length=4, max_length=4)

    @field_validator("frame_range")
    @classmethod
    def _ordered(cls, v: list[int]):
        if v and v[0] > v[1]:
            raise ValueError("frame_range 需升序")
        return v


class ModelReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    evidence_id: str = Field(min_length=3, max_length=40)
    decisions: list[Decision] = Field(default_factory=list, max_length=24)
    needs_more_evidence: bool = False


MAX_RESPONSE_BYTES = 64 * 1024


def parse_model_response(text: str, manifest: EvidenceManifest) -> ModelReviewResponse:
    """解析 + 边界校验。任何违反 → ValueError（调用方丢弃并记录，不重试）。"""
    if not text or len(text.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("响应为空或超过 64KB")
    import json as _json
    try:
        data = _json.loads(text, parse_constant=_reject_constant)
    except _json.JSONDecodeError as e:
        raise ValueError(f"非法 JSON：{e}") from None
    try:
        parsed = ModelReviewResponse.model_validate(data)
    except Exception as e:
        raise ValueError(f"响应不符合契约：{str(e)[:200]}") from None
    if parsed.evidence_id != manifest.evidence_id:
        raise ValueError("evidence_id 不匹配")
    allowed_frames = manifest.frame_ids()
    allowed_persons = set(manifest.allowed_person_ids)
    allowed_balls = set(manifest.allowed_ball_ids)
    for d in parsed.decisions:
        for f in d.evidence_frames:
            if f not in allowed_frames:
                raise ValueError(f"引用未发送的帧 {f}")
        if d.frame_range:
            fr = d.frame_range
            if fr[0] not in allowed_frames and fr[1] not in allowed_frames:
                raise ValueError(f"frame_range 引用未发送帧 {fr}")
        if d.person_id is not None and allowed_persons and d.person_id not in allowed_persons:
            raise ValueError(f"引用未提供的人物 ID {d.person_id}")
        for b in d.ball_candidate_ids:
            if allowed_balls and b not in allowed_balls:
                raise ValueError(f"引用未提供的球候选 {b}")
    return parsed


def _reject_constant(name: str):
    raise ValueError(f"JSON 常量 {name} 不允许（NaN/Infinity）")
