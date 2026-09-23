"""Provider 抽象：能力探测与受限视觉复核。

约定（方案 §5.1/§5.2）：
- probe 用合成图（不用用户视频），分别探测文本与视觉能力。
- review_images 只发送证据清单内的图片与编号，返回严格受限的 JSON。
- 能力来自探测结果，不按模型名称猜测。
- Key 只从 SecretStore 读取，不进日志/异常文本。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.model_providers.errors import ProviderError


@dataclass
class Capabilities:
    protocol: str                    # openai_compatible | anthropic | mcp
    text_ok: bool = False
    vision_ok: bool = False
    multi_image: bool = False
    structured_output: bool = False  # 是否支持 response_format/json
    cancel_supported: bool = False
    model_id: str = ""
    usage_source: str = "unknown"    # response.usage | mcp | none
    notes: dict = field(default_factory=dict)


@dataclass
class ProbeResult:
    ok: bool
    capabilities: Capabilities
    error_code: str | None = None
    detail: str = ""                 # 脱敏后的简短说明


@dataclass
class ReviewRequest:
    question: str                    # attribute_contact | pick_ball_candidate | hit_window ...
    prompt: str                      # 由 EvidenceBuilder 生成的受限提示词
    images: list[str]                # data URL 列表（HTTP 适配器），与 manifest 帧一一对应
    frame_ids: list[str]             # 帧标识（编号），供响应引用校验
    image_paths: list[str] | None = None   # 本地证据文件路径（MCP 适配器）
    max_tokens: int = 2048
    timeout_s: float = 90.0


@dataclass
class ReviewResponse:
    ok: bool
    content: str = ""                # 原始文本（仅审计/解析用）
    provider_request_id: str = ""
    usage: dict = field(default_factory=dict)
    error_code: str | None = None
    error_message: str = ""
    diagnostics: dict = field(default_factory=dict)


class ProviderAdapter(Protocol):
    """所有接入类型（glm_api/glm_coding/fake）实现同一协议。"""

    def probe(self, api_key: str) -> ProbeResult: ...

    def review_images(self, api_key: str, request: ReviewRequest) -> ReviewResponse: ...

    def optional_review_video(self, api_key: str, path, meta: dict) -> ReviewResponse:
        # V1 不上传整段视频；默认显式禁用
        raise ProviderError("V1 不上传整段视频")

    def cancel(self) -> None:
        return None
