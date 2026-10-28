"""Anthropic Messages 通用基类契约测试：合成厂商 spec，mock transport，不联网。

覆盖：URL 与版本头、双鉴权、图片 base64 转换与非法拒绝、stop_reason 三映射、
业务码钩子、401 不重试、thinking 默认不发、空图片列表可达模型（glm 契约兼容）。
"""
import json

import httpx
import pytest

from app.model_providers.anthropic_messages import AnthropicMessagesAdapter
from app.model_providers.base import ReviewRequest

_TEXT_OK = {"id": "r", "content": [{"type": "text", "text": "蓝色"}],
            "stop_reason": "end_turn", "usage": {"output_tokens": 5}}


def _adapter(handler, **kw):
    defaults = dict(base_url="https://vendor.example", model_candidates=("cm-1",))
    defaults.update(kw)
    return AnthropicMessagesAdapter(transport=httpx.MockTransport(handler), **defaults)


def _req(images=("data:image/png;base64,QUJD",)):
    return ReviewRequest("hit_window", "p", list(images), ["1"])


def test_messages_url_and_version_header_with_bearer():
    captured = {}
    def handler(request):
        captured["url"] = str(request.url)
        captured["version"] = request.headers.get("anthropic-version")
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_TEXT_OK)
    a = _adapter(handler)
    assert a.probe("k").ok
    assert captured["url"] == "https://vendor.example/v1/messages"
    assert captured["version"] == "2023-06-01" and captured["auth"] == "Bearer k"


def test_x_api_key_auth_mode():
    captured = {}
    def handler(request):
        captured["x"] = request.headers.get("x-api-key")
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_TEXT_OK)
    a = _adapter(handler, auth_mode="x_api_key")
    assert a.probe("k-vis").ok
    assert captured["x"] == "k-vis" and captured["auth"] is None


def test_image_data_url_converted_to_base64_source():
    captured = {}
    def handler(request):
        body = json.loads(request.content)
        captured["blocks"] = body["messages"][0]["content"]
        return httpx.Response(200, json=_TEXT_OK)
    a = _adapter(handler)
    assert a.review_images("k", _req(["data:image/jpeg;base64,QUJD"])).ok
    img = [b for b in captured["blocks"] if b["type"] == "image"][0]
    assert img["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}


def test_invalid_image_data_url_rejected_without_provider_text():
    r = _adapter(lambda req: httpx.Response(200, json=_TEXT_OK)).review_images(
        "k", _req(["data:text/html;base64,PHNjcmlwdD4="]))
    assert not r.ok and r.error_code == "invalid_response"


@pytest.mark.parametrize("blocks,stop,expected", [
    ([{"type": "text", "text": "{\"p\":"}], "max_tokens", "output_truncated"),
    ([{"type": "tool_use", "name": "v", "input": {}}], "tool_use", "tool_required"),
    ([], "end_turn", "empty_response"),
])
def test_stop_reason_mappings(blocks, stop, expected):
    def handler(request):
        return httpx.Response(200, json={"id": "r", "content": blocks,
                                         "stop_reason": stop,
                                         "usage": {"output_tokens": 9}})
    r = _adapter(handler).review_images("k", _req())
    assert not r.ok and r.error_code == expected
    assert r.diagnostics["stop_reason"] == stop and r.usage["output_tokens"] == 9


def test_business_code_hook_maps_quota():
    def handler(request):
        return httpx.Response(429, json={"error": {"code": "insufficient_quota",
                                                   "message": "private-value"}})
    r = _adapter(handler, business_codes={"insufficient_quota": "quota_exhausted"}).probe("k")
    assert r.error_code == "quota_exhausted"
    assert "private-value" not in (r.detail or "")


def test_auth_failure_never_retries():
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(401)
    r = _adapter(handler).probe("k")
    assert not r.ok and r.error_code == "auth_failed" and len(calls) == 1


def test_json_mode_does_not_send_thinking_by_default():
    captured = {}
    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_TEXT_OK)
    _adapter(handler).review_images("k", _req())
    assert "thinking" not in captured["body"]
    assert captured["body"]["system"].startswith("Return only valid JSON")
    assert captured["body"]["max_tokens"] >= 1024


def test_empty_images_still_reach_model():
    """glm 契约兼容：anthropic 路径不做空证据拦截（帧坐标由解析层校验）。"""
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=_TEXT_OK)
    assert _adapter(handler).review_images("k", _req([])).ok and len(calls) == 1
