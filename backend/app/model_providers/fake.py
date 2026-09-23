"""FakeProvider：离线端到端与契约测试用，不联网。

脚本化响应：按 question 返回固定/可注入的 JSON 字符串；
也可注入错误码测试失败路径。真实 GLM 测试由显式环境开关启用，CI 默认不消费额度。
"""
from __future__ import annotations

import json

from app.model_providers.base import Capabilities, ProbeResult, ReviewRequest, ReviewResponse


class FakeProvider:
    account_kind = "api"
    protocol = "openai_compatible"

    def __init__(self, responses: dict[str, str] | None = None,
                 error_code: str | None = None):
        self.responses = responses or {}
        self.error_code = error_code
        self.calls: list[ReviewRequest] = []

    def probe(self, api_key: str) -> ProbeResult:
        if self.error_code:
            return ProbeResult(False, Capabilities(protocol=self.protocol),
                               self.error_code, "fake error")
        caps = Capabilities(protocol=self.protocol, text_ok=True, vision_ok=True,
                            multi_image=True, structured_output=True,
                            model_id="fake-vision", usage_source="response.usage")
        return ProbeResult(True, caps)

    def review_images(self, api_key: str, request: ReviewRequest) -> ReviewResponse:
        self.calls.append(request)
        if self.error_code:
            return ReviewResponse(ok=False, error_code=self.error_code,
                                  error_message="fake error")
        content = self.responses.get(request.question,
                                     json.dumps({"schema_version": "1.0",
                                                 "evidence_id": "unknown",
                                                 "decisions": [],
                                                 "needs_more_evidence": False}))
        return ReviewResponse(ok=True, content=content,
                              provider_request_id="fake-req", usage={"total_tokens": 1})

