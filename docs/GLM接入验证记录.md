# GLM 接入验证记录

状态：**资格未确认（eligibility_pending）** · 记录日期：2026-09-16 · 本文档不含任何 API Key。

## 1. 官方信息核验（页面级，非账号实测）

以下信息来自方案编写时对官方文档页面的核验，接入实现时以实际探测为准，不宣称长期不变：

| 项 | 官方页面信息 | 结论 |
|---|---|---|
| Coding Plan 端点 | Anthropic 协议 `https://open.bigmodel.cn/api/anthropic`；OpenAI 协议 `https://open.bigmodel.cn/api/coding/paas/v4` | 协议地址存在，但**不构成 Tennis Lab 后台用途获准使用套餐的证明**（[快速开始](https://docs.bigmodel.cn/cn/coding-plan/quick-start)） |
| 视觉 MCP | `@z_ai/mcp-server`，`image_analysis`/`video_analysis`，`Z_AI_API_KEY`/`Z_AI_MODE=ZHIPU`，Node 18+，本地视频工具标注最大 8M | 云端视觉能力；Local MCP ≠ 离线推理（[视觉 MCP](https://docs.bigmodel.cn/cn/coding-plan/mcp/vision-mcp-server)） |
| 普通 API | GLM-4.6V 走 `https://open.bigmodel.cn/api/paas/v4/chat/completions`，支持图片内容 | 可作为独立通用视觉服务的适配目标（[GLM-4.6V](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v)） |
| 计费边界 | Coding Plan 与普通 API 分开；端点用错可能按其他资源扣费；HTTP 200 不证明消耗套餐额度 | 套餐资格确认前不得向套餐端点发送网球分析请求（[套餐 FAQ](https://docs.bigmodel.cn/cn/coding-plan/faq)） |

## 2. 接入路径状态

| 路径 | 状态 | 门槛 |
|---|---|---|
| `glm_coding_plan` | **eligibility_pending（适配待激活）** | 智谱明确确认本产品用途/调用形态可使用套餐 + 真实账单核验。代码已实现接口与模拟测试，不向套餐端点发真实请求 |
| `glm_api`（普通 API） | 适配完成，**仅用户单独启用后联调** | 用户明确选择普通 API、了解独立计费并主动启用；不得从套餐失败静默降级到此路径 |
| 其他供应商/本地端点 | 接口预留，V1 不在 UI 展开 | 后续独立评估 |

- 不伪造 User-Agent 或工具身份；不修改用户 `~/.claude` 等全局配置；不以文本模型冒充视觉模型。
- 套餐资格未确认期间，任何真实调用验收都必须标注“未完成”。

## 3. 能力探测结果（真实 Key 联调后回填）

| 探测项 | glm_api | glm_coding_plan |
|---|---|---|
| 文本能力（合成图探测） | **2026-09-17 实测：认证有效，未到模型调用** | 未执行（资格未确认） |
| 视觉能力（合成图探测） | 未到（余额不足，见下） | 未执行 |
| 模型名（探测后冻结） | 待探测成功后回填 | 待填 |
| MCP schema hash | 不适用 | 待填 |
| 账单资源类型 | 未产生消费 | 待填 |

### MCP（Coding Plan 路径）实测 · 2026-09-17

参考的用户 Claude 配置（`mcpServers: zai-mcp-server, command=npx -y @z_ai/mcp-server, env Z_AI_API_KEY/Z_AI_MODE=ZHIPU`）。本产品不读取/不修改用户 Claude 配置，自行 spawn 固定版本实例（`scripts/install_mcp.sh` 显式安装 0.1.5 到 backend/node_modules，或 `TENNIS_ZAI_MCP_CMD` 覆盖）；子进程仅获得 PATH/HOME/Z_AI_API_KEY/Z_AI_MODE（测试断言环境隔离）。

- 协议实测：newline JSON-RPC 2.0，initialize→tools/list→tools/call；服务器启动即校验 Key 格式。
- **工具实名 `analyze_image`/`analyze_video`**（设计文档假设的 image_analysis/video_analysis 不存在）；输入为本地文件路径或 URL（非 data URL）。
- probe = 握手 + tools/list，**不消耗额度**；工具 schema 哈希冻结（`179334c1…`，逐字快照内嵌 mcp_client.FROZEN_TOOLS），变更即拒绝调用。
- 真实 `analyze_image` 调用（1 张测试图）：**成功返回中文图像描述** —— 同一 Key 走普通 API 报 1113 余额不足、走 MCP 却成功，表明该 Key 绑定 Coding Plan 套餐且 MCP 视觉调用由套餐资源覆盖。
- **资格边界不变**：技术上链路已通，但" Tennis Lab 后台自动复核用途是否符合 Coding Plan 使用范围"仍待智谱确认；确认前默认 `eligibility_pending`，联调开关仅环境变量显式打开。

**2026-09-17 真实 Key 联调记录（用户提供，未入库）**：文本探测返回 HTTP 429 + `error.code=1113`（"余额不足或无可用资源包"）——Key 认证有效（非 401），但账户无可用额度。由此修复适配器缺陷：GLM 以 429 携带业务码，1113 现映射为 `quota_exhausted`（不可重试），其余 429 仍为 `rate_limited`；已补契约测试。**充值后需重跑**：合成图探测 → 启用 → 一次完整 automatic 分析+增强复核 → 账单核对。

## 4. 凭据与安全基线（已实现，见代码）

- Key 存 macOS Keychain（服务名 `TennisLab-ModelConnection-<id>`）；数据库仅存 `secret_ref`。
- Keychain 不可用时明确报错，**不落明文**。
- GET 接口只返回 `has_key` 与掩码别名；Key 不进前端 localStorage、URL、任务 payload、日志或进程参数。
- 变更接口校验 loopback Host/Origin + 本机会话 token（CSRF）。

## 4.1 已实施的离线验证（2026-09-16，无真实 Key）

- 适配器契约：mock transport 覆盖成功/401/429/5xx/超时/非法 JSON/格式修复重试/多图顺序（tests/test_glm_adapters.py，8 项）。
- 连接底座：探测→启用→轮换/启停→删除、Keychain 隔离、乐观锁、Coding Plan 拒绝远端探测（tests/test_model_connections.py，8 项）。
- Skill 快照：路径穿越（含 `root//绝对路径` 变体，测试发现并已修复）、符号链接、可执行位（官方包数据文件带执行位——改为安装时去除并记录）、炸弹、SHA 钉住、真实 ZIP 安装（tests/test_skill_runner.py，7 项）。
- 证据契约：严格 schema/未发送帧/未提供编号/NaN/超长拒绝、提示注入不改预算、FakeProvider 离线端到端（tests/test_enhancement_contracts.py，9 项）。
- 浏览器实测：连接抽屉（未配置/套餐待确认/假 Key 401 明确报错）；跨站 Origin 403、无会话 token 403（中间件先于 body 解析）。
- 未做（需用户 Key）：真实视觉探测、真实增强复核、账单核对——见下表与 §5。

## 5. 真实调用验收清单（待用户启用后执行）

- [ ] 用户显式启用 glm_api 并确认计费理解
- [ ] 合成图探测通过（文本+视觉分别记录）
- [ ] 一次完整 automatic 分析 + 增强复核，记录请求数/耗时/用量
- [ ] 账单页核对资源类型与次数
- [ ] 失败路径：断网/额度不足/401 各一次，确认本地结果保留
