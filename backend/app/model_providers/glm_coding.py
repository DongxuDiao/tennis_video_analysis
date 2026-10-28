"""GLM Coding Plan through the user-selected Anthropic Messages endpoint.
Uses application-scoped credentials; never reads or edits global Claude settings.

多厂商改造后为 Anthropic Messages 基类的 GLM 薄壳：协议实现移至
anthropic_messages.AnthropicMessagesAdapter；GLM 专属的 Plan 业务码
（1308/1310 额度上限+重置时间、1309 到期、1113 余额）保留在本模块。
"""
from __future__ import annotations

import re

from app.model_providers.anthropic_messages import AnthropicMessagesAdapter
from app.model_providers.base import ReviewResponse
from app.model_providers import errors as E

BASE_URL = 'https://open.bigmodel.cn/api/anthropic'
MODEL = 'glm-5.3-flash'


def eligibility_status():
    # Legacy DB field only; successful capability testing governs activation.
    return 'confirmed'


class GlmCodingAdapter(AnthropicMessagesAdapter):
    account_kind = 'coding_plan'
    retryable_codes = ()  # 契约：无隐藏重试（额度重置类错误重试也无意义）
    # Coding Plan 端点能力口径：单图、无 response_format（json 经 system 提示约束）
    _CAPS_DEFAULTS = {'multi_image': False, 'structured_output': False}

    def __init__(self, transport=None, base_url=BASE_URL, model: str | None = None):
        super().__init__(transport=transport, base_url=base_url,
                         model_candidates=(MODEL,), model=model,
                         json_mode_thinking={'type': 'disabled'},
                         caps_defaults=self._CAPS_DEFAULTS)

    def _business_error(self, err: dict) -> ReviewResponse | None:
        """GLM Plan 业务码 → 稳定错误。只提取时间戳等安全片段，不透传原文。"""
        code = str(err.get('code', ''))
        if code == '1113':
            return ReviewResponse(ok=False, error_code='quota_exhausted',
                                  error_message='服务额度不足，请充值后重试')
        if code in ('1308', '1310'):
            message = 'Coding Plan 使用额度已达上限，请等待额度重置后重试'
            # Extract only a timestamp; never expose arbitrary provider text or secrets.
            reset = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                              str(err.get('message', '')))
            if reset:
                message += f'；平台返回的重置时间：{reset.group(0)}'
            return ReviewResponse(ok=False, error_code='plan_limit_reached',
                                  error_message=message)
        if code == '1309':
            return ReviewResponse(ok=False, error_code='plan_expired',
                                  error_message=E.USER_MESSAGES['plan_expired'])
        return None
