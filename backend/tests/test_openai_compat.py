"""OpenAI 兼容通用基类契约测试：合成 spec，mock transport，不联网。

覆盖：探测成功/严格视觉/多候选与短路、401/429(业务码)/5xx/超时、
json_mode 修复重发、Bearer 与图片顺序、model 覆盖、max_tokens_field 切换、重试诊断。
"""
import json

import httpx
import pytest

from app.model_providers.base import ReviewRequest
from app.model_providers.openai_compat import OpenAICompatAdapter

_OK = lambda r: httpx.Response(200, json={"id": "r", "choices": [
    {"message": {"content": "蓝色"}}], "usage": {}})


def _adapter(handler, **kw):
    defaults = dict(base_url="https://vendor.example/v1",
                    model_candidates=("m1", "m2"))
    defaults.update(kw)
    return OpenAICompatAdapter(transport=httpx.MockTransport(handler), **defaults)


def _req(images=("data:image/jpeg;base64,AAA",)):
    return ReviewRequest("hit_window", "p", list(images), ["1"])


def test_probe_text_and_vision_success():
    r = _adapter(_OK).probe("sk")
    assert r.ok and r.capabilities.text_ok and r.capabilities.vision_ok
    assert r.capabilities.model_id == "m1" and r.capabilities.multi_image


def test_wrong_color_answer_is_strict_capability_missing():
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        content = body["messages"][0]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "红色" if "颜色" in text else "正常"}}]})
    r = _adapter(handler).probe("sk")
    assert not r.ok and r.error_code == "capability_missing"
    assert not r.capabilities.vision_ok and r.detail


def test_vision_request_failure_tries_next_candidate():
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body["model"])
        is_vision = "颜色" in str(body)
        if body["model"] == "m1" and is_vision:
            return httpx.Response(400, json={"error": "unsupported"})   # m1 视觉请求失败
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "蓝色" if is_vision else "正常"}}]})
    r = _adapter(handler).probe("sk")
    assert r.ok and r.capabilities.model_id == "m2" and seen == ["m1", "m1", "m2", "m2"]


def test_auth_failure_short_circuits_candidates():
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={})
    r = _adapter(handler).probe("sk")
    assert not r.ok and r.error_code == "auth_failed" and len(calls) == 1


def test_quota_error_short_circuits_candidates():
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
    r = _adapter(handler, business_codes={"insufficient_quota": "quota_exhausted"}).probe("sk")
    assert r.error_code == "quota_exhausted" and len(calls) == 1


def test_review_sends_bearer_and_image_order():
    captured = {}
    def handler(request):
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "{}"}}]})
    resp = _adapter(handler).review_images("sk-x", _req(
        ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"]))
    assert resp.ok and captured["auth"] == "Bearer sk-x"
    parts = captured["body"]["messages"][0]["content"]
    assert [p["image_url"]["url"] for p in parts if p["type"] == "image_url"] == \
        ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"]


def test_rate_limited_retries_once_with_diagnostics():
    calls = []
    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "{}"}}]})
    resp = _adapter(handler).review_images("sk", _req())
    assert resp.ok and resp.diagnostics.get("retried") is True and len(calls) == 2


def test_server_error_and_timeout_not_retried_beyond_contract():
    r = _adapter(lambda req: httpx.Response(503, json={})).review_images("sk", _req())
    assert r.error_code == "server_error"

    def hang(request):
        raise httpx.ReadTimeout("t")
    r2 = _adapter(hang).review_images("sk", _req())
    assert r2.error_code == "timeout_unknown"


def test_invalid_json_repair_drops_response_format():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content).get("response_format"))
        if len(calls) == 1:
            return httpx.Response(200, text="<not json>")
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "{\"ok\":1}"}}]})
    resp = _adapter(handler).review_images("sk", _req())
    assert resp.ok and calls[0] == {"type": "json_object"} and calls[1] is None


def test_model_override_lands_in_request_body():
    captured = {}
    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "蓝色"}}]})
    r = _adapter(handler, model="m-override").probe("sk")
    assert r.ok and r.capabilities.model_id == "m-override"
    assert {json.loads(json.dumps(c))["model"] for c in [captured["body"]]} == {"m-override"}


def test_max_tokens_field_switch():
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "正常"}}]})
    a = _adapter(handler, model_candidates=("gpt-5x",), max_tokens_field="max_completion_tokens")
    a.probe("sk")
    # 首个文本探测请求即使用切换后的参数名
    assert bodies[0]["max_completion_tokens"] == 16 and "max_tokens" not in bodies[0]
