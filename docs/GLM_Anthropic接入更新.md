# GLM Coding Plan Anthropic 接入更新

2026-09-17，按用户要求替代原资格确认门槛。

- 前端默认 Coding Plan，API Key 与测试按钮可用；删除资格提示和禁用逻辑。
- 后端不读取 TENNIS_GLM_CODING_ELIGIBILITY_CONFIRMED；遗留 eligibility_status 仅为数据库兼容字段，不代表任何供应商确认。
- 应用独立调用 https://open.bigmodel.cn/api/anthropic/v1/messages，Authorization Bearer，anthropic-version=2023-06-01，默认 glm-5.3-flash。
- 对应用户 ANTHROPIC_BASE_URL、ANTHROPIC_AUTH_TOKEN、SONNET/OPUS 模型映射。无需运行 Claude Code，所以不写 hasCompletedOnboarding、自动压缩窗口或全局 settings.json；无 Haiku 路由，因此不触发 flash 计费调用。请求时间保留本项目有界预算，不把 Claude 的3000000ms直接用于网页任务。
- 探测真实文本与合成图片，图片答案不符合预期时不允许误报视觉可用；支持图片的真实服务能力仍需用户 Key 联调。服务器只返回 tool_use 而没有文字时会提示能力问题，不伪装成功。
- 不自动切换普通 API；旧 MCP 适配源码保存在 glm_coding_mcp.py 用于兼容测试，默认连接已切换至 Anthropic。
- 本次验证使用 MockTransport 和本地测试，不包含真实Key、套餐额度或付费调用验收。

这份更新覆盖早期方案中的 eligibility_pending 发布门槛；供应商对账户的实际鉴权/限流/能力响应仍原样决定连接能否成功。
