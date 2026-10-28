"""OpenAI 兼容 chat/completions 通用适配器（多厂商共享基类）。

从 glm_api.py 提炼：端点/模型/业务错误码/参数名由构造参数注入，
厂商差异（GLM 业务码等）通过 business_codes 映射表达。

- 探测：文本 + 合成图答颜色，多候选按序尝试，成功即冻结 model_id；
  视觉答案不符预期 → capability_missing（严格语义：不允许"能收图即视觉可用"）。
  鉴权/额度/套餐类失败短路，不换模型重试（计费隔离、不浪费额度）。
- 重试：rate_limited/network_error 退避重发一次；timeout_unknown 不重发（可能已计费）。
- 稳定错误映射（errors.py），厂商错误体只进脱敏 diagnostics。
- 可注入 httpx.MockTransport 做离线契约测试。
"""
from __future__ import annotations

import base64
import random
import time

import httpx

from app.model_providers.base import Capabilities, ProbeResult, ReviewRequest, ReviewResponse
from app.model_providers import errors as E

CONNECT_TIMEOUT_S = 10.0

# 合成探测图：程序生成（红底、中心蓝块 PNG），不使用用户视频。
_SYNTH_PNG: bytes = b""


def _synth_png() -> bytes:
    global _SYNTH_PNG
    if not _SYNTH_PNG:
        import cv2
        import numpy as np
        img = np.zeros((32, 32, 3), np.uint8)
        img[:, :] = (60, 60, 200)          # BGR 红
        cv2.rectangle(img, (14, 14), (18, 18), (255, 128, 0), -1)  # 蓝块
        ok, buf = cv2.imencode(".png", img)
        _SYNTH_PNG = buf.tobytes()
    return _SYNTH_PNG


def _synth_data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(_synth_png()).decode()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def image_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + _b64(jpeg_bytes)


# 这些错误换模型无意义：直接短路（不试下一候选），保持计费隔离
_PROBE_SHORT_CIRCUIT = frozenset({
    "auth_failed", "quota_exhausted", "plan_limit_reached", "plan_expired",
    "not_activated"})
# 视觉测试未通过时的稳定表述（与既有 glm_coding 契约一致）
_VISION_MISS_DETAIL = "文本连接可用，但视觉测试未通过"


class OpenAICompatAdapter:
    account_kind = "api"
    protocol = "openai_compatible"
    # 可退避重试一次（errors.py 契约：429 非业务码 / 连接层失败）。
    # 无隐藏重试契约的子类（如 Coding Plan）清空此元组。
    retryable_codes = ("rate_limited", "network_error")

    def __init__(self, transport=None, base_url: str = "",
                 model_candidates: tuple[str, ...] = (), model: str | None = None,
                 business_codes: dict | None = None,
                 probe_max_tokens_vision: int = 32,
                 max_tokens_field: str = "max_tokens",
                 caps_defaults: dict | None = None):
        self._transport = transport
        self._base = base_url.rstrip("/")
        self._candidates = tuple(model_candidates)
        # model 覆盖：探测/复核只用该模型（覆盖即冻结，不再尝试候选列表）
        self._pinned = model is not None
        self.model = model or (self._candidates[0] if self._candidates else "")
        self._business_codes = dict(business_codes or {})
        self._probe_vision_tokens = probe_max_tokens_vision
        self._max_tokens_field = max_tokens_field
        self._caps_defaults = dict(caps_defaults or {})
        self._http: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        # 连接池复用：每次 _chat 新建 Client 会让每个请求都重付一次
        # TCP+TLS 握手（批量复核任务可达几十次请求）。
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(transport=self._transport, timeout=httpx.Timeout(
                CONNECT_TIMEOUT_S, read=95.0, write=30.0, pool=CONNECT_TIMEOUT_S))
        return self._http

    def close(self) -> None:
        if self._http is not None and not self._http.is_closed:
            self._http.close()

    # ---- 基础请求 ----
    def _chat(self, api_key: str, model: str, messages: list[dict],
              max_tokens: int, timeout_s: float, json_mode: bool = False) -> ReviewResponse:
        resp = self._chat_once(api_key, model, messages, max_tokens, timeout_s, json_mode)
        # timeout_unknown 不重发——请求可能已到达并计费。
        if (not resp.ok and self.retryable_codes
                and resp.error_code in self.retryable_codes):
            time.sleep(1.5 + random.random() * 1.5)
            resp = self._chat_once(api_key, model, messages, max_tokens, timeout_s, json_mode)
            resp.diagnostics = {**(resp.diagnostics or {}), "retried": True}
        return resp

    def _chat_once(self, api_key: str, model: str, messages: list[dict],
                   max_tokens: int, timeout_s: float, json_mode: bool = False) -> ReviewResponse:
        url = f"{self._base}/chat/completions"
        body: dict = {"model": model, "messages": messages,
                      self._max_tokens_field: max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            c = self._client()
            r = c.post(url, json=body,
                       headers={"Authorization": f"Bearer {api_key}"},
                       # 总超时 timeout_s，但连接建立仍受 10s 上限约束：
                       # 直接传 timeout_s 会把 connect 也放宽到总超时，
                       # 坏网络路径下阻塞整个 worker 线程
                       timeout=httpx.Timeout(timeout_s, connect=CONNECT_TIMEOUT_S))
        except Exception as exc:  # 连接层错误
            name = type(exc).__name__
            if "ReadTimeout" in name or "timeout" in str(exc).lower():
                return ReviewResponse(ok=False, error_code="timeout_unknown",
                                      error_message="请求超时，结果未知")
            return ReviewResponse(ok=False, error_code="network_error",
                                  error_message="网络连接失败")
        if r.status_code in (401, 403):
            return ReviewResponse(ok=False, error_code="auth_failed",
                                  error_message="Key 无效或无权限")
        if r.status_code == 429:
            # 部分厂商用 429 携带业务码（如 GLM 1113=余额不足）：
            # 按 business_codes 映射为稳定码，未命中回落 rate_limited
            code = ""
            try:
                code = str((r.json().get("error") or {}).get("code", ""))
            except Exception:
                pass
            if code and code in self._business_codes:
                stable = self._business_codes[code]
                return ReviewResponse(ok=False, error_code=stable,
                                      error_message=E.USER_MESSAGES.get(stable, "服务请求失败"))
            return ReviewResponse(ok=False, error_code="rate_limited",
                                  error_message="请求过于频繁")
        if r.status_code >= 500:
            return ReviewResponse(ok=False, error_code="server_error",
                                  error_message="远端服务错误")
        if r.status_code != 200:
            return ReviewResponse(ok=False, error_code="provider_error",
                                  error_message=f"HTTP {r.status_code}")
        try:
            data = r.json()
        except Exception:
            return ReviewResponse(ok=False, error_code="invalid_response",
                                  error_message="响应不是 JSON")
        text = ""
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except Exception:
            return ReviewResponse(ok=False, error_code="invalid_response",
                                  error_message="响应缺少 content")
        return ReviewResponse(ok=True, content=text,
                              provider_request_id=data.get("id", ""),
                              usage=data.get("usage") or {})

    # ---- 探测 ----
    def probe(self, api_key: str) -> ProbeResult:
        candidates = (self.model,) if self._pinned else (
            self._candidates or ((self.model,) if self.model else ()))
        caps = Capabilities(protocol=self.protocol, usage_source="response.usage")
        caps.notes.update(self._caps_defaults.get("notes", {}))
        last_text_fail: ReviewResponse | None = None
        vision_missed = False
        for cand in candidates:
            caps.model_id = cand   # 记录实际尝试的模型
            # 1) 文本
            t = self._chat(api_key, cand,
                           [{"role": "user", "content": "回复两个字：正常"}],
                           max_tokens=16, timeout_s=30.0)
            if not t.ok:
                if t.error_code in _PROBE_SHORT_CIRCUIT:
                    return ProbeResult(False, caps, t.error_code, t.error_message)
                last_text_fail = t
                continue
            # 2) 视觉：合成图，要求回答方块颜色
            v = self._chat(api_key, cand,
                           [{"role": "user", "content": [
                               {"type": "text", "text": "图中心小方块是什么颜色？只答颜色词。"},
                               {"type": "image_url", "image_url": {"url": _synth_data_url()}}]}],
                           max_tokens=self._probe_vision_tokens, timeout_s=45.0)
            if v.ok and ("蓝" in v.content or "blue" in v.content.lower()):
                caps.text_ok = True
                caps.vision_ok = True
                caps.multi_image = self._caps_defaults.get("multi_image", True)
                caps.structured_output = self._caps_defaults.get("structured_output", True)
                caps.model_id = cand
                return ProbeResult(True, caps, None, "")
            # 视觉请求失败或答案不符预期：记下并尝试下一候选
            vision_missed = vision_missed or v.ok
        if vision_missed:
            return ProbeResult(False, caps, "capability_missing", _VISION_MISS_DETAIL)
        if last_text_fail is not None:
            return ProbeResult(False, caps, last_text_fail.error_code,
                               last_text_fail.error_message)
        return ProbeResult(False, caps, "capability_missing", "无可用视觉模型")

    # ---- 视觉复核 ----
    def review_images(self, api_key: str, request: ReviewRequest) -> ReviewResponse:
        if not request.images:
            return ReviewResponse(ok=False, error_code="invalid_response",
                                  error_message="证据为空")
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        for url in request.images:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [{"role": "user", "content": content}]
        resp = self._chat(api_key, self.model, messages,
                          max_tokens=request.max_tokens, timeout_s=request.timeout_s,
                          json_mode=True)
        if not resp.ok and resp.error_code == "invalid_response":
            # 一次格式修复：去掉 json_mode 重发（计入调用预算由调用方管理）
            resp = self._chat(api_key, self.model, messages,
                              max_tokens=request.max_tokens, timeout_s=request.timeout_s,
                              json_mode=False)
        return resp
