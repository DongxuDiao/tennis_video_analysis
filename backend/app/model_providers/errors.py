"""Provider 稳定错误枚举：不透传厂商错误体给前端。"""
from __future__ import annotations


class ProviderError(Exception):
    code = "provider_error"

    def __init__(self, message: str = "", *, retryable: bool = False, raw: str | None = None):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.retryable = retryable
        # raw 仅进服务端日志（脱敏后），不返回给前端
        self.raw = (raw or "")[:500]


class AuthError(ProviderError):
    code = "auth_failed"          # 401/403：Key 无效或无权限，不重试


class QuotaError(ProviderError):
    code = "quota_exhausted"      # 余额不足/限流耗尽，不重试


class RateLimited(ProviderError):
    code = "rate_limited"         # 429：可退避重试一次


class ServerError(ProviderError):
    code = "server_error"         # 5xx：仅在明确未执行时可重试


class NetworkError(ProviderError):
    code = "network_error"        # 连接失败/断网，可重试一次


class RequestTimeout(ProviderError):
    code = "timeout_unknown"      # 发送后超时：结果未知，默认不重发（可能已计费）


class ResponseInvalid(ProviderError):
    code = "invalid_response"     # 非法 JSON/超限/缺字段；一次修复重试后放弃


class CapabilityMissing(ProviderError):
    code = "capability_missing"   # 视觉不可用等


class NotActivated(ProviderError):
    code = "not_activated"        # 资格未确认（Coding Plan），拒绝真实调用


USER_MESSAGES = {
    "auth_failed": "API Key 无效或无权限，请检查后重试",
    "quota_exhausted": "服务额度不足，已保留本地结果",
    "rate_limited": "服务暂时限流，请稍后重试",
    "plan_limit_reached": "Coding Plan 使用额度已达上限，请等待额度重置后重试",
    "plan_expired": "Coding Plan 套餐已到期，请检查套餐状态",
    "server_error": "远端服务暂时不可用，已保留本地结果",
    "network_error": "网络连接失败，已保留本地结果",
    "timeout_unknown": "请求超时（结果未知，不自动重发），已保留本地结果",
    "output_truncated": "模型输出达到长度上限，结果不完整",
    "tool_required": "模型要求调用工具，当前直连模式不支持该工具",
    "empty_response": "模型未返回可用文本",
    "invalid_response": "服务返回格式异常，已丢弃并保留本地结果",
    "capability_missing": "当前连接缺少视觉能力",
    "not_activated": "该接入类型资格待确认，未启用",
    "provider_error": "服务调用失败，已保留本地结果",
}
