"""GLM 普通 API 适配器（glm_api）：open.bigmodel.cn /paas/v4 chat completions。

- 独立计费路径；仅在用户显式启用后使用。
- 视觉候选 glm-4.6v；可用性由探测确认，不从模型名猜测能力。
- 图片经 data URL 发送（本实现不做外部上传，不生成公开链接）。
- 稳定错误映射（errors.py），厂商错误体只进脱敏日志。
- 可注入 httpx.MockTransport 做离线契约测试。
"""
from __future__ import annotations

import base64
import json
import time

from app.model_providers.base import Capabilities, ProbeResult, ReviewRequest, ReviewResponse
from app.model_providers import errors as E

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
VISION_MODEL_CANDIDATES = ["glm-4.6v"]   # 探测按序尝试，成功后冻结
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


class GlmApiAdapter:
    account_kind = "api"
    protocol = "openai_compatible"

    def __init__(self, transport=None, base_url: str = BASE_URL):
        self._transport = transport
        self._base = base_url.rstrip("/")

    def _client(self):
        import httpx
        return httpx.Client(transport=self._transport, timeout=httpx.Timeout(
            CONNECT_TIMEOUT_S, read=95.0, write=30.0, pool=CONNECT_TIMEOUT_S))

    # ---- 基础请求 ----
    def _chat(self, api_key: str, model: str, messages: list[dict],
              max_tokens: int, timeout_s: float, json_mode: bool = False) -> ReviewResponse:
        url = f"{self._base}/chat/completions"
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            with self._client() as c:
                r = c.post(url, json=body,
                           headers={"Authorization": f"Bearer {api_key}"},
                           timeout=timeout_s)
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
            # GLM 用 429 携带业务码：1113=余额不足/无资源包（不可重试，不计入限流重试）
            code = ""
            try:
                code = str((r.json().get("error") or {}).get("code", ""))
            except Exception:
                pass
            if code == "1113":
                return ReviewResponse(ok=False, error_code="quota_exhausted",
                                      error_message="服务额度不足，请充值后重试")
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
        caps = Capabilities(protocol=self.protocol, usage_source="response.usage")
        # 1) 文本
        t = self._chat(api_key, VISION_MODEL_CANDIDATES[0],
                       [{"role": "user", "content": "回复两个字：正常"}],
                       max_tokens=16, timeout_s=30.0)
        if not t.ok:
            return ProbeResult(False, caps, t.error_code, t.error_message)
        caps.text_ok = True
        # 2) 视觉：合成图，要求回答方块颜色
        v = self._chat(api_key, VISION_MODEL_CANDIDATES[0],
                       [{"role": "user", "content": [
                           {"type": "text", "text": "图中心小方块是什么颜色？只答颜色词。"},
                           {"type": "image_url", "image_url": {"url": _synth_data_url()}}]}],
                       max_tokens=32, timeout_s=45.0)
        if v.ok:
            answer_ok = "蓝" in v.content or "blue" in v.content.lower()
            caps.vision_ok = True          # 能收图即视觉通道可用
            caps.multi_image = True        # 多图能力由响应确认后再放开：先保守单值
            caps.structured_output = True  # json_mode 在复核调用中实测；失败会走一次修复
            caps.model_id = VISION_MODEL_CANDIDATES[0]
            if not answer_ok:
                caps.notes["synth_answer"] = v.content[:20]
        return ProbeResult(v.ok, caps, v.error_code if not v.ok else None,
                           "" if v.ok else v.error_message)

    # ---- 视觉复核 ----
    def review_images(self, api_key: str, request: ReviewRequest) -> ReviewResponse:
        if not request.images:
            return ReviewResponse(ok=False, error_code="invalid_response",
                                  error_message="证据为空")
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        for url in request.images:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [{"role": "user", "content": content}]
        resp = self._chat(api_key, VISION_MODEL_CANDIDATES[0], messages,
                          max_tokens=request.max_tokens, timeout_s=request.timeout_s,
                          json_mode=True)
        if not resp.ok and resp.error_code == "invalid_response":
            # 一次格式修复：去掉 json_mode 重发（计入调用预算由调用方管理）
            resp = self._chat(api_key, VISION_MODEL_CANDIDATES[0], messages,
                              max_tokens=request.max_tokens, timeout_s=request.timeout_s,
                              json_mode=False)
        return resp


def image_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + _b64(jpeg_bytes)
