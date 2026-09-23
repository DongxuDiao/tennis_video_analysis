"""GLM API 适配器契约测试：mock transport，不联网。

覆盖：成功（文本+视觉）、401/429/5xx、超时（timeout_unknown）、
非法 JSON、格式修复重试、多图顺序。
"""
import json

import httpx
import pytest

from app.model_providers.glm_api import GlmApiAdapter
from app.model_providers.base import ReviewRequest


def _adapter(handler):
    return GlmApiAdapter(transport=httpx.MockTransport(handler))


def _ok(content="回答"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "req-1", "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 10}})
    return handler


def test_probe_text_and_vision_success():
    a = _adapter(_ok("蓝色"))
    result = a.probe("sk-test")
    assert result.ok and result.capabilities.text_ok and result.capabilities.vision_ok
    assert result.capabilities.model_id


def test_probe_auth_failure():
    def handler(request):
        return httpx.Response(401, json={"error": "bad key"})
    a = _adapter(handler)
    r = a.probe("sk-bad")
    assert not r.ok and r.error_code == "auth_failed"


def test_review_images_sends_bearer_and_data_urls():
    captured = {}
    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "req-2", "choices": [
            {"message": {"content": "{}"}}], "usage": {}})
    a = _adapter(handler)
    req = ReviewRequest(question="attribute_contact", prompt="p",
                        images=["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"],
                        frame_ids=["10", "12"])
    resp = a.review_images("sk-xyz", req)
    assert resp.ok
    assert captured["auth"] == "Bearer sk-xyz"
    parts = captured["body"]["messages"][0]["content"]
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert urls == req.images          # 顺序与证据一一对应


def test_rate_limited_maps_to_retryable():
    def handler(request):
        return httpx.Response(429, json={}, headers={"Retry-After": "2"})
    a = _adapter(handler)
    r = a.review_images("sk", ReviewRequest("hit_window", "p", ["data:image/jpeg;base64,A"], ["1"]))
    assert r.error_code == "rate_limited"


def test_glm_429_balance_code_maps_to_quota_exhausted():
    """GLM 以 HTTP 429 + error.code=1113 表示余额不足：不可重试的 quota_exhausted。"""
    def handler(request):
        return httpx.Response(429, json={"error": {"code": "1113",
                                                   "message": "余额不足或无可用资源包"}})
    a = _adapter(handler)
    r = a.review_images("sk", ReviewRequest("hit_window", "p", ["data:image/jpeg;base64,A"], ["1"]))
    assert r.error_code == "quota_exhausted"    # 不是 rate_limited


def test_server_error_and_timeout():
    a = _adapter(lambda r: httpx.Response(503, json={}))
    r = a.review_images("sk", ReviewRequest("hit_window", "p", ["data:image/jpeg;base64,A"], ["1"]))
    assert r.error_code == "server_error"

    def hang(request):
        raise httpx.ReadTimeout("read timed out")
    a2 = _adapter(hang)
    r2 = a2.review_images("sk", ReviewRequest("hit_window", "p", ["data:image/jpeg;base64,A"], ["1"]))
    assert r2.error_code == "timeout_unknown"   # 结果未知，不自动重发


def test_invalid_json_then_repair_without_json_mode():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content).get("response_format"))
        if len(calls) == 1:
            return httpx.Response(200, text="<not json>")
        return httpx.Response(200, json={"id": "r", "choices": [
            {"message": {"content": "{\"schema_version\":\"1.0\"}"}}]})
    a = _adapter(handler)
    resp = a.review_images("sk", ReviewRequest("hit_window", "p", ["data:image/jpeg;base64,A"], ["1"]))
    # 首次 json_mode 请求失败（响应非 JSON）→ 第二次去掉 json_mode 重发一次
    assert resp.ok and calls[0] == {"type": "json_object"} and calls[1] is None


def test_coding_plan_anthropic_request_and_vision_probe():
    from app.model_providers.glm_coding import GlmCodingAdapter
    calls=[]
    def handler(request):
        assert str(request.url)=='https://open.bigmodel.cn/api/anthropic/v1/messages'
        assert request.headers['Authorization']=='Bearer fake-key'
        body=json.loads(request.content);calls.append(body)
        assert body['model']=='glm-5.3-flash'
        return httpx.Response(200,json={'id':'test','content':[{'type':'text','text':'蓝色'}],'stop_reason':'end_turn'})
    result=GlmCodingAdapter(transport=httpx.MockTransport(handler)).probe('fake-key')
    assert result.ok and result.capabilities.vision_ok
    assert calls[1]['messages'][0]['content'][1]['source']['type']=='base64'
    assert result.capabilities.protocol=='anthropic'


def test_coding_plan_does_not_claim_vision_for_wrong_answer():
    from app.model_providers.glm_coding import GlmCodingAdapter
    a=GlmCodingAdapter(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'content':[{'type':'text','text':'看不到图片'}]})))
    result=a.probe('fake-key')
    assert not result.ok and not result.capabilities.vision_ok


def test_coding_auth_failure_never_falls_back():
    from app.model_providers.glm_coding import GlmCodingAdapter
    calls=[]
    def handler(r):
        calls.append(str(r.url));return httpx.Response(401)
    result=GlmCodingAdapter(transport=httpx.MockTransport(handler)).probe('fake-key')
    assert not result.ok and result.error_code=='auth_failed' and len(calls)==1


@pytest.mark.parametrize('code,expected', [('1308','plan_limit_reached'), ('1310','plan_limit_reached'), ('1309','plan_expired'), ('1113','quota_exhausted'), ('1302','rate_limited')])
def test_coding_business_errors_are_not_all_rate_limits(code, expected):
    from app.model_providers.glm_coding import GlmCodingAdapter
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={'error':{'code':code,'message':'private-value 限额将在 2026-09-18 10:00:47 重置'}})
    result=GlmCodingAdapter(transport=httpx.MockTransport(handler)).probe('fake-key')
    assert result.error_code == expected
    assert len(calls)==1
    assert 'private-value' not in result.detail
    if expected=='plan_limit_reached':
        assert '2026-09-18 10:00:47' in result.detail


@pytest.mark.parametrize('blocks,stop,expected', [
    ([{'type':'thinking','thinking':'private reasoning'}], 'max_tokens', 'output_truncated'),
    ([{'type':'text','text':'{"partial":'}], 'max_tokens', 'output_truncated'),
    ([{'type':'tool_use','name':'vision','input':{}}], 'tool_use', 'tool_required'),
    ([], 'end_turn', 'empty_response'),
])
def test_coding_incomplete_response_diagnostics(blocks,stop,expected):
    from app.model_providers.glm_coding import GlmCodingAdapter
    a=GlmCodingAdapter(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'content':blocks,'stop_reason':stop,'usage':{'output_tokens':4096}})))
    result=a.review_images('fake-key',ReviewRequest('test','p',[],[]))
    assert not result.ok and result.error_code==expected
    assert result.usage['output_tokens']==4096
    assert result.diagnostics['stop_reason']==stop
    assert 'private reasoning' not in json.dumps(result.diagnostics)


def test_coding_structured_review_disables_thinking():
    from app.model_providers.glm_coding import GlmCodingAdapter
    calls=[]
    def handler(request):
        body=json.loads(request.content);calls.append(body)
        assert body['thinking']=={'type':'disabled'}
        return httpx.Response(200,json={'content':[{'type':'text','text':'{"frames":[]}'}],'stop_reason':'end_turn','usage':{'output_tokens':8}})
    result=GlmCodingAdapter(transport=httpx.MockTransport(handler)).review_images('fake-key',ReviewRequest('external_analysis','p',[],[]))
    assert result.ok and len(calls)==1 and result.diagnostics['text_chars']>0
