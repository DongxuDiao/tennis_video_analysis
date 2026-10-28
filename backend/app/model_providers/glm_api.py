"""GLM 普通 API 适配器（glm_api）：open.bigmodel.cn /paas/v4 chat completions。

多厂商改造后为 OpenAI 兼容基类的 GLM 薄壳：协议实现移至
openai_compat.OpenAICompatAdapter（端点/模型/业务码参数化）。
本模块保留原类名、常量与构造签名，既有直接构造方（测试/service）不受影响。

- 独立计费路径；仅在用户显式启用后使用。
- 视觉候选 glm-4.6v；可用性由探测确认，不从模型名猜测能力。
- 稳定错误映射（errors.py），厂商错误体只进脱敏日志。
"""
from __future__ import annotations

from app.model_providers.openai_compat import (
    CONNECT_TIMEOUT_S,
    OpenAICompatAdapter,
    image_bytes_to_data_url,
)

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
VISION_MODEL_CANDIDATES = ["glm-4.6v"]   # 探测按序尝试，成功后冻结
# GLM 以 429 携带业务码：1113=余额不足/无资源包（quota_exhausted，不可重试）
_BUSINESS_CODES = {"1113": "quota_exhausted"}


class GlmApiAdapter(OpenAICompatAdapter):
    account_kind = "api"
    protocol = "openai_compatible"

    def __init__(self, transport=None, base_url: str = BASE_URL, model: str | None = None):
        super().__init__(transport=transport, base_url=base_url,
                         model_candidates=tuple(VISION_MODEL_CANDIDATES), model=model,
                         business_codes=_BUSINESS_CODES)
