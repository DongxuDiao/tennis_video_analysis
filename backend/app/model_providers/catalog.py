"""多厂商 Provider 目录：端点/模型/协议/错误码的唯一数据源。

设计约束（自带模型与GLM接入_技术落地方案.md §4.1/§6.2、GLM_Anthropic接入更新.md）：
- V1 无自定义 URL：端点与模型走本目录预设；探测成功后 model 冻结进 DB。
- 模型名时效性：目录单点可编辑 + 探测把关 + 可选 model 覆盖三重兜底。
- 计费隔离：每个 (provider, account_kind) 独立适配器实例，失败不自动切换。
- glm 条目是存量行为的逐字节搬迁（display_name、BASE_URL、模型名与既有常量一致），
  存量 provider='glm' 连接不迁移、继续可用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.model_providers.base import ProviderAdapter


@dataclass(frozen=True)
class AccountKindSpec:
    id: str                 # "api" | "coding_plan"
    label: str              # display_name 片段："普通API" / "Coding Plan"
    docs_url: str = ""
    # —— 覆盖 provider 级默认（glm 的 coding_plan 与 api 走不同协议/端点）——
    protocol: str = ""      # 覆盖协议
    base_url: str = ""      # 覆盖端点
    models: tuple[str, ...] = ()       # 覆盖候选模型
    auth_mode: str = ""                # 覆盖鉴权（bearer | x_api_key）
    json_mode_thinking: dict | None = None
    max_tokens_field: str = ""         # 覆盖 max_tokens 参数名
    probe_max_tokens_vision: int = 0   # 覆盖探测视觉 max_tokens（0=用 provider 级）
    caps_defaults: dict = field(default_factory=dict)
    note: str = ""          # 覆盖前端接入说明


@dataclass(frozen=True)
class ProviderSpec:
    id: str                          # DB provider 列取值；"glm" 为存量保留值
    label: str                       # 中文名（display_name 前缀 / source_line）
    protocol: str                    # openai_compatible | anthropic
    base_url: str
    vision_models: tuple[str, ...]   # 探测候选，按序尝试，成功即冻结
    text_model: str = ""             # 预留：纯文本模型（当前探测全用视觉模型）
    account_kinds: tuple[AccountKindSpec, ...] = ()
    key_hint: str = ""               # 前端 Key 输入 placeholder
    docs_url: str = ""               # 获取 Key 的文档
    note: str = ""                   # 前端接入说明
    probe_max_tokens_vision: int = 32
    max_tokens_field: str = "max_tokens"   # gpt-5 系需 "max_completion_tokens"
    auth_mode: str = "bearer"        # anthropic 协议：bearer | x_api_key
    json_mode_thinking: dict | None = None
    business_codes: dict = field(default_factory=dict)   # 429 body error.code → 稳定码
    caps_defaults: dict = field(default_factory=dict)    # multi_image/structured_output 覆盖


def _K(kind_id: str, label: str, **kw) -> AccountKindSpec:
    return AccountKindSpec(id=kind_id, label=label, **kw)


_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="glm", label="GLM",
        protocol="openai_compatible",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        vision_models=("glm-4.6v",),
        account_kinds=(
            _K("api", "普通API",
               docs_url="https://docs.bigmodel.cn/cn/develop/apis/quickstart"),
            _K("coding_plan", "Coding Plan",
               docs_url="https://docs.bigmodel.cn/cn/coding-plan/quick-start",
               protocol="anthropic",
               base_url="https://open.bigmodel.cn/api/anthropic",
               models=("glm-5.3-flash",),
               json_mode_thinking={"type": "disabled"},
               caps_defaults={"multi_image": False, "structured_output": False}),
        ),
        docs_url="https://open.bigmodel.cn",
        note="智谱 GLM；普通 API 与 Coding Plan 套餐为独立计费路径，互不降级。",
        business_codes={"1113": "quota_exhausted"},
    ),
    ProviderSpec(
        id="openai", label="OpenAI",
        protocol="openai_compatible",
        base_url="https://api.openai.com/v1",
        vision_models=("gpt-4o", "gpt-4o-mini"),
        account_kinds=(_K("api", "API"),),
        key_hint="sk-…",
        docs_url="https://platform.openai.com/api-keys",
        note="GPT-4o 系支持图像输入。gpt-5 系需改用 max_completion_tokens 参数，"
             "如需使用请在模型覆盖中指定并确认端点兼容。",
        business_codes={"insufficient_quota": "quota_exhausted"},
    ),
    ProviderSpec(
        id="anthropic", label="Anthropic Claude",
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        vision_models=("claude-sonnet-5", "claude-sonnet-4-5"),
        account_kinds=(_K("api", "API"),),
        key_hint="sk-ant-…",
        docs_url="https://console.anthropic.com/settings/keys",
        note="官方 API 使用 x-api-key 鉴权；JSON 输出经系统提示约束（无 response_format）。",
        auth_mode="x_api_key",
        caps_defaults={"structured_output": False},
    ),
    ProviderSpec(
        id="gemini", label="Google Gemini",
        protocol="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        vision_models=("gemini-2.5-flash", "gemini-2.5-pro"),
        account_kinds=(_K("api", "API"),),
        key_hint="AIza…",
        docs_url="https://aistudio.google.com/apikey",
        note="走 OpenAI 兼容端点（AI Studio Key）。thinking 模型可能消耗输出预算，"
             "探测已放宽输出上限；若视觉探测仍失败请直接反馈。",
        probe_max_tokens_vision=256,
    ),
    ProviderSpec(
        id="deepseek", label="DeepSeek",
        protocol="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        vision_models=("deepseek-chat",),
        account_kinds=(_K("api", "API"),),
        key_hint="sk-…",
        docs_url="https://platform.deepseek.com/api_keys",
        note="DeepSeek 当前无公开视觉模型：文本探测可通过，但视觉测试不会通过，"
             "无法用于视频帧分析。目录保留条目以便其视觉模型上线后启用。",
    ),
    ProviderSpec(
        id="moonshot", label="Kimi（月之暗面）",
        protocol="openai_compatible",
        base_url="https://api.moonshot.cn/v1",
        vision_models=("kimi-latest",),
        account_kinds=(_K("api", "API"),),
        key_hint="sk-…",
        docs_url="https://platform.moonshot.cn/console/api-keys",
        note="kimi-latest 支持图像输入；旧的 vision-preview 模型已下线。",
    ),
    ProviderSpec(
        id="qwen", label="通义千问（百炼）",
        protocol="openai_compatible",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        vision_models=("qwen-vl-max-latest", "qwen-vl-plus-latest"),
        account_kinds=(_K("api", "API"),),
        key_hint="sk-…",
        docs_url="https://bailian.console.aliyun.com/",
        note="百炼 OpenAI 兼容模式；视觉模型名随版本迭代较快，以探测结果为准。",
    ),
    ProviderSpec(
        id="doubao", label="豆包（火山方舟）",
        protocol="openai_compatible",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        vision_models=("doubao-seed-1.6-flash-250815", "doubao-1.5-vision-pro-32k"),
        account_kinds=(_K("api", "API"),),
        key_hint="火山方舟 API Key",
        docs_url="https://console.volcengine.com/ark",
        note="方舟亦接受 ep- 开头的推理接入点 ID：请在“模型覆盖”中填入你的接入点。",
    ),
)

_SPEC_INDEX: dict[str, ProviderSpec] = {s.id: s for s in _SPECS}


def _eff(spec: ProviderSpec, kind: AccountKindSpec, attr: str):
    """kind 级覆盖优先，回落 provider 级。"""
    override = getattr(kind, attr, "")
    if override:
        return override
    return getattr(spec, attr)


def require(provider: str, account_kind: str) -> tuple[ProviderSpec, AccountKindSpec]:
    spec = _SPEC_INDEX.get(provider)
    if spec is None:
        raise ValueError(f"不支持的模型服务：{provider}")
    for k in spec.account_kinds:
        if k.id == account_kind:
            return spec, k
    raise ValueError(f"{spec.label} 不支持接入类型：{account_kind}")


def kinds(provider: str) -> tuple[AccountKindSpec, ...]:
    spec = _SPEC_INDEX.get(provider)
    if spec is None:
        raise ValueError(f"不支持的模型服务：{provider}")
    return spec.account_kinds


def label_of(provider: str) -> str:
    spec = _SPEC_INDEX.get(provider)
    return spec.label if spec else provider


def display_name(provider: str, account_kind: str, masked_key: str) -> str:
    spec, kind = require(provider, account_kind)
    return f"{spec.label} {kind.label} · {masked_key}"


def adapter_for(provider: str, account_kind: str, *, model_id: str | None = None,
                transport=None) -> ProviderAdapter:
    """构造适配器。model_id（DB 冻结值或用户覆盖）作为唯一候选；缺省用目录候选。"""
    spec, kind = require(provider, account_kind)
    model = model_id or None
    protocol = _eff(spec, kind, "protocol") or spec.protocol
    # glm 保留原类：存量行为（重试契约/业务码文案/能力口径）逐字节一致
    if provider == "glm" and account_kind == "api":
        from app.model_providers.glm_api import GlmApiAdapter
        return GlmApiAdapter(transport=transport, model=model)
    if provider == "glm" and account_kind == "coding_plan":
        from app.model_providers.glm_coding import GlmCodingAdapter
        return GlmCodingAdapter(transport=transport, model=model)
    base_url = _eff(spec, kind, "base_url") or spec.base_url
    candidates = tuple(kind.models or ()) or spec.vision_models
    if protocol == "anthropic":
        from app.model_providers.anthropic_messages import AnthropicMessagesAdapter
        return AnthropicMessagesAdapter(
            transport=transport, base_url=base_url, model=model or (candidates[0] if candidates else None),
            auth_mode=_eff(spec, kind, "auth_mode") or spec.auth_mode,
            json_mode_thinking=kind.json_mode_thinking or spec.json_mode_thinking,
            model_candidates=candidates,
            caps_defaults=kind.caps_defaults or spec.caps_defaults)
    from app.model_providers.openai_compat import OpenAICompatAdapter
    return OpenAICompatAdapter(
        transport=transport, base_url=base_url,
        model_candidates=candidates, model=model,
        business_codes=dict(spec.business_codes),
        probe_max_tokens_vision=(kind.probe_max_tokens_vision
                                 or spec.probe_max_tokens_vision),
        max_tokens_field=_eff(spec, kind, "max_tokens_field"),
        caps_defaults=kind.caps_defaults or spec.caps_defaults)


def catalog_payload() -> list[dict]:
    """GET /api/model-connections 的 providers 目录（JSON 可序列化，不含工厂细节）。"""
    out = []
    for spec in _SPECS:
        out.append({
            "id": spec.id,
            "label": spec.label,
            "key_hint": spec.key_hint,
            "docs_url": spec.docs_url,
            "note": spec.note,
            "default_model": spec.vision_models[0] if spec.vision_models else "",
            "account_kinds": [{
                "id": k.id, "label": k.label, "docs_url": k.docs_url or spec.docs_url,
                "default_model": (k.models[0] if k.models else
                                  (spec.vision_models[0] if spec.vision_models else "")),
                "note": k.note or "",
            } for k in spec.account_kinds],
        })
    return out
