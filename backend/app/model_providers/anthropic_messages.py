"""Anthropic Messages 协议通用适配器（多厂商共享基类）。

从 glm_coding.py 提炼：端点/模型/鉴权方式/业务错误码由构造参数注入，
厂商差异（GLM Plan 业务码、thinking disabled）通过子类/参数表达。

- 消息转换：OpenAI 风格 image_url data URL → base64 source blocks；
  非法 data URL 拒绝为 invalid_response（不透传原文）。
- json_mode：system 提示约束 JSON（无 response_format，无可去参的"修复重发"，
  故 review_images 单次调用、无隐藏重发）。
- thinking 参数默认不发送（官方 API 无 disabled 类型；GLM 子类按其端点要求发送）。
- 探测继承基类多候选 + 严格视觉语义。
"""
from __future__ import annotations

import base64
import re

import httpx

from app.model_providers.base import ReviewRequest, ReviewResponse
from app.model_providers.openai_compat import OpenAICompatAdapter
from app.model_providers import errors as E

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicMessagesAdapter(OpenAICompatAdapter):
    protocol = "anthropic"
    # 官方 API 无 thinking disabled 类型；GLM Coding Plan 端点要求显式关闭时由子类设置
    json_mode_thinking: dict | None = None
    auth_mode = "bearer"          # bearer（Authorization）| x_api_key（x-api-key）

    def __init__(self, transport=None, base_url: str = "", model: str | None = None,
                 auth_mode: str | None = None, json_mode_thinking: dict | None = None,
                 **kwargs):
        super().__init__(transport=transport, base_url=base_url, model=model, **kwargs)
        if auth_mode:
            self.auth_mode = auth_mode
        if json_mode_thinking is not None:
            self.json_mode_thinking = json_mode_thinking

    def _headers(self, api_key: str) -> dict:
        h = {"anthropic-version": ANTHROPIC_VERSION}
        if self.auth_mode == "x_api_key":
            h["x-api-key"] = api_key
        else:
            h["Authorization"] = "Bearer " + api_key
        return h

    def _business_error(self, err: dict) -> ReviewResponse | None:
        """429/错误体业务码 → 稳定错误。厂商专属码在子类扩展。"""
        code = str(err.get("code", ""))
        stable = self._business_codes.get(code) if code else None
        if stable:
            return ReviewResponse(ok=False, error_code=stable,
                                  error_message=E.USER_MESSAGES.get(stable, "服务请求失败"))
        return None

    # override 基类 _chat_once（而非 _chat）：单次请求的协议转换在这里，
    # 退避重试等通用策略由基类 _chat 包装统一提供
    def _chat_once(self, api_key: str, model: str, messages: list[dict],
                   max_tokens: int, timeout_s: float, json_mode: bool = False) -> ReviewResponse:
        converted = []
        for message in messages:
            blocks = message["content"]
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
            content = []
            for block in blocks:
                if block["type"] == "text":
                    content.append(block)
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    try:
                        header, data = url.split(",", 1)
                        media = header[5:].split(";")[0]
                        if (not header.startswith("data:") or ";base64" not in header
                                or media not in ("image/png", "image/jpeg", "image/webp")):
                            raise ValueError()
                        base64.b64decode(data, validate=True)
                    except (ValueError, TypeError):
                        return ReviewResponse(ok=False, error_code="invalid_response",
                                              error_message="图片格式无效")
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": media, "data": data}})
            converted.append({"role": message["role"], "content": content})
        body = {"model": model, "max_tokens": max(1024, max_tokens), "messages": converted}
        if json_mode:
            body["system"] = "Return only valid JSON matching the requested schema."
            if self.json_mode_thinking is not None:
                body["thinking"] = dict(self.json_mode_thinking)
        try:
            # 连接池由基类 _client() 持有并复用；这里不能 with-close 它
            client = self._client()
            response = client.post(self._base + "/v1/messages", json=body,
                                   headers=self._headers(api_key),
                                   timeout=httpx.Timeout(timeout_s, connect=10))
        except httpx.TimeoutException:
            return ReviewResponse(ok=False, error_code="timeout_unknown",
                                  error_message="请求超时，结果未知")
        except httpx.HTTPError:
            return ReviewResponse(ok=False, error_code="network_error",
                                  error_message="网络连接失败")
        code = response.status_code
        if code != 200:
            err = ("auth_failed" if code in (401, 403)
                   else "rate_limited" if code == 429
                   else "server_error" if code >= 500 else "provider_error")
            detail_message = f"请求失败（HTTP {code}）"
            try:
                provider_error = response.json().get("error") or {}
                business = self._business_error(provider_error)
                if business is not None:
                    return business
            except (ValueError, AttributeError):
                pass
            return ReviewResponse(ok=False, error_code=err, error_message=detail_message)
        try:
            data = response.json()
            blocks = data.get("content")
            if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
                raise ValueError()
            text = "".join(b["text"] for b in blocks
                           if b.get("type") == "text" and isinstance(b.get("text"), str))
            stop = data.get("stop_reason")
            diagnostics = {"http_status": code,
                           "stop_reason": stop if stop in ("end_turn", "max_tokens", "tool_use",
                                                           "stop_sequence", "pause_turn",
                                                           "refusal") else "unknown",
                           "block_types": [b.get("type") for b in blocks
                                           if b.get("type") in ("text", "thinking", "tool_use",
                                                                "redacted_thinking")],
                           "text_chars": len(text)}
            usage = {k: v for k, v in (data.get("usage") or {}).items()
                     if isinstance(v, (int, float))}
            if stop == "max_tokens":
                return ReviewResponse(ok=False, error_code="output_truncated",
                                      error_message="模型输出达到长度上限，结果不完整",
                                      usage=usage, diagnostics=diagnostics)
            if stop == "tool_use" or any(b.get("type") == "tool_use" for b in blocks):
                return ReviewResponse(ok=False, error_code="tool_required",
                                      error_message="模型要求调用视觉工具，当前直连接口未执行该工具",
                                      usage=usage, diagnostics=diagnostics)
            if not text.strip():
                return ReviewResponse(ok=False, error_code="empty_response",
                                      error_message="模型未返回可用文本",
                                      usage=usage, diagnostics=diagnostics)
            return ReviewResponse(ok=True, content=text, provider_request_id=data.get("id", ""),
                                  usage=usage, diagnostics=diagnostics)
        except (ValueError, KeyError, TypeError, AttributeError):
            return ReviewResponse(ok=False, error_code="invalid_response",
                                  error_message="模型响应结构异常",
                                  diagnostics={"http_status": code})

    def review_images(self, api_key: str, request: ReviewRequest) -> ReviewResponse:
        # JSON 经 system 提示约束（无 response_format），单次调用、无隐藏重发
        content = ([{"type": "text", "text": request.prompt}]
                   + [{"type": "image_url", "image_url": {"url": u}} for u in request.images])
        return self._chat(api_key, self.model, [{"role": "user", "content": content}],
                          request.max_tokens, request.timeout_s, True)
