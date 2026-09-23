"""GLM Coding Plan 适配器（glm_coding）：官方 Claude 式 MCP stdio 接入。

参照用户提供的 Claude mcpServers 配置（command=npx -y @z_ai/mcp-server，
env Z_AI_API_KEY / Z_AI_MODE=ZHIPU），但按方案约束落地为：
- 我们自行 spawn 固定版本实例（默认显式安装在 backend/node_modules，或
  TENNIS_ZAI_MCP_CMD 指定），不读取/不修改用户 Claude 配置，不运行时自动下载。
- 子进程仅获得 PATH/HOME/Z_AI_API_KEY/Z_AI_MODE 四个环境变量。
- probe = 启动 + initialize + tools/list（不调用模型，不消耗额度）；
  schema 哈希与冻结值比对，变更 → 待重测。
- review_images 用实测工具 analyze_image（本地证据文件路径 + 提示词），
  参数严格按 tools/list 返回的 inputSchema 校验后组装，禁止编造。
- 资格（eligibility）未确认前：所有真实调用抛 NotActivated。
"""
from __future__ import annotations

import json
import os

from app.model_providers import errors as E
from app.model_providers.base import (Capabilities, ProbeResult, ReviewRequest,
                                      ReviewResponse)
from app.model_providers.mcp_client import (EXPECTED_TOOLS_SHA256, PINNED_VERSION,
                                            McpError, McpStdioClient, tools_hash)

_ELIGIBILITY_ENV = "TENNIS_GLM_CODING_ELIGIBILITY_CONFIRMED"
VISION_TOOL = "analyze_image"          # 2026-09-17 实测实名（非 image_analysis）
VIDEO_TOOL = "analyze_video"


def eligibility_status():
    return 'confirmed'


class GlmCodingAdapter:
    account_kind = "coding_plan"
    protocol = "mcp"

    def _gate(self):
        pass

    # ---- probe：无模型调用，不消耗额度 ----
    def probe(self, api_key: str) -> ProbeResult:
        caps = Capabilities(protocol=self.protocol, usage_source="mcp")
        try:
            client = McpStdioClient(api_key)
        except McpError as e:
            return ProbeResult(False, caps, "server_not_installed", str(e)[:160])
        try:
            client.start()
            tools = client.list_tools()
        except McpError as e:
            code = {"server_exited": "auth_failed",     # 启动即校验 Key 格式
                    "timeout_unknown": "timeout_unknown"}.get(e.code, e.code)
            return ProbeResult(False, caps, code, str(e)[:160])
        finally:
            client.close()
        names = {t.get("name") for t in tools}
        caps.text_ok = True                       # 协议层已通（initialize 应答）
        caps.vision_ok = VISION_TOOL in names or VIDEO_TOOL in names
        caps.model_id = f"zai-mcp-server@{PINNED_VERSION}"
        caps.structured_output = False             # MCP 无 response_format；靠本地解析
        caps.multi_image = False                   # analyze_image 单图/次
        current_hash = tools_hash(tools)
        caps.notes = {"tools": sorted(names), "tools_sha256": current_hash,
                      "schema_matches_frozen": current_hash == EXPECTED_TOOLS_SHA256}
        if not caps.vision_ok:
            return ProbeResult(False, caps, "capability_missing",
                               f"服务器未提供 {VISION_TOOL}/{VIDEO_TOOL}")
        if current_hash != EXPECTED_TOOLS_SHA256:
            # schema 变更：能力待重测，不按旧 schema 编造参数
            return ProbeResult(False, caps, "capability_missing",
                               "工具 schema 与冻结版本不一致，需重新核验后更新适配")
        return ProbeResult(True, caps)

    # ---- 视觉复核：analyze_image（本地证据文件路径） ----
    def review_images(self, api_key: str, request: ReviewRequest) -> ReviewResponse:
        self._gate()
        paths = getattr(request, "image_paths", None) or []
        if not paths:
            return ReviewResponse(ok=False, error_code="invalid_response",
                                  error_message="MCP 路径需要本地证据文件路径")
        try:
            client = McpStdioClient(api_key)
        except McpError as e:
            return ReviewResponse(ok=False, error_code="capability_missing",
                                  error_message=str(e)[:200])
        try:
            client.start()
            tools = client.list_tools()
            schema = next((t.get("inputSchema") for t in tools
                           if t.get("name") == VISION_TOOL), None)
            if schema is None:
                return ReviewResponse(ok=False, error_code="capability_missing",
                                      error_message=f"服务器缺少 {VISION_TOOL}")
            required = set(schema.get("required", []))
            props = set(schema.get("properties", {}).keys())
            # 组装参数必须完全落在 schema 内且覆盖全部 required
            if not required <= {"image_source", "prompt"} or \
               not required <= props:
                return ReviewResponse(ok=False, error_code="capability_missing",
                                      error_message="工具 schema 与预期不符，需重测适配")
            texts = []
            for path in paths:
                args = {"image_source": str(path), "prompt": request.prompt}
                text = client.call_tool(VISION_TOOL, args,
                                        timeout_s=request.timeout_s)
                texts.append(text)
            return ReviewResponse(ok=True, content="\n".join(texts),
                                  provider_request_id="", usage={})
        except McpError as e:
            code = {"timeout_unknown": "timeout_unknown",
                    "server_not_installed": "capability_missing",
                    "server_spawn_failed": "capability_missing",
                    "server_exited": "server_error"}.get(e.code, "provider_error")
            return ReviewResponse(ok=False, error_code=code,
                                  error_message=str(e)[:200])
        finally:
            client.close()
