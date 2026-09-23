# Tennis Lab 自带模型与 GLM 接入技术落地方案

状态：开发设计，尚未实现或使用真实 Key 联调。适用：现有 React + FastAPI + SQLite + 单重任务 worker。目标读者：负责实现、测试和验收的开发 Agent。

## 1. 目标与必须先澄清的条件

用户配置自己的模型服务后，仍按“上传 → 自动分析 → 回看/下载”使用产品。后台按最新版 sports-video-motion-analysis skill 做视角体检、人体/球候选提取、证据复核、确定性计算和报告。模型负责辅助识别与解释，本地程序负责坐标变换、轨迹关联、统计、物理计算和最终验证。

“自己的模型”第一版定义为自带账号/Key 接入模型服务，不包含上传任意 .pt 文件训练、运行任意 Python 插件或部署模型服务器。接口预留后续本地服务适配。

### 1.1 已核验的官方信息

官方快速开始将 Coding Plan 限定于支持的指定工具与产品环境，列出 Anthropic 协议端点 `https://open.bigmodel.cn/api/anthropic` 和 OpenAI 协议端点 `https://open.bigmodel.cn/api/coding/paas/v4`。这些是协议地址，不构成 Tennis Lab 获准使用套餐的证明。[官方快速开始](https://docs.bigmodel.cn/cn/coding-plan/quick-start)

官方视觉 MCP 提供 `image_analysis`、`video_analysis`，通过 `@z_ai/mcp-server` 使用；环境变量为 `Z_AI_API_KEY`、`Z_AI_MODE=ZHIPU`。文档要求 Node.js 18+，本地视频工具标注最大 8M。它提供云端视觉能力，Local MCP 不代表离线推理。[视觉 MCP](https://docs.bigmodel.cn/cn/coding-plan/mcp/vision-mcp-server)

普通模型 API 与 Coding Plan 分开：官方 GLM-4.6V 示例使用 `https://open.bigmodel.cn/api/paas/v4/chat/completions`，支持图片内容；可作为独立的通用视觉服务适配目标。[GLM-4.6V](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v)

套餐端点错误可能导致余额不足或按其他资源扣费；官方提供费用明细核查方法。因此连接成功、HTTP 200 均不能证明消耗套餐额度。[套餐 FAQ](https://docs.bigmodel.cn/cn/coding-plan/faq)

以上为本次官方页面核验，不是账号实测。具体模型列表、可用额度与 MCP schema 在实际接入时重新核验，不在代码中宣称长期不变。

### 1.2 V1 决策：GLM 优先，但分清产品接入资格

| 路径 | V1 处理 | 发布门槛 |
|---|---|---|
| `glm_coding_plan` | 优先开发适配和离线契约测试，生产默认 `eligibility_pending` | 智谱明确确认 Tennis Lab 的实际用途/调用形态可使用套餐，并完成真实账单核验 |
| 官方支持的 Coding 工具 | 可独立用于开发/人工实验；不是后台自动代跑的当然授权 | 运行网球分析这一用途仍需确认适用范围，不以包一层 Claude Code 绕过限制 |
| `glm_api` | 独立的普通视觉 API 适配，可作为条件不满足时的可发布路径 | 用户明确选择普通 API、了解独立计费并启用；不得从套餐失败静默降级到此路径 |
| 其他供应商/本地端点 | 定义接口，不在 V1 默认 UI 展开 | 后续独立能力及安全测试 |

不伪造 User-Agent、工具身份，不修改用户 ~/.claude 或其他全局配置，不将一般文本模型当作视觉模型。若套餐资格未确认，交付状态必须写“适配待激活”，不能宣称“已支持套餐额度”。这不阻塞本地流程、凭据管理、证据包与验证器开发。

## 2. 当前代码基线与改造位置

- `frontend/src/App.tsx`：简化列表、结果页和上传流程；新增唯一“模型连接”设置入口。
- `backend/app/api/main.py`：上传、任务、expert、文件输出接口；增加 connection API，不复用 system doctor 暴露敏感信息。
- `backend/app/worker/main.py`：导入成功接续 automatic、任务取消、失败状态；保留单个重分析槽。
- `backend/app/services/expert.py`：当前自动流水线脚本调度、素材缓存、心跳、子进程清理；接入连接配置快照与增强阶段。
- `backend/app/expert_engine/scene_quality.py`：小幅机位运动检测/保守稳定化。
- `pose_extract.py`、`person_quality.py`：主球员连续性、多人候选、关键点质量。
- `automatic.py`、`quality_review.py`：轨迹/事件、本地短缺口复核。
- `render_automatic.py`：自动结果渲染；需要支持模型证据来源与机位提示。
- `backend/app/config.py`：当前 CODE_VERSION 未覆盖所有 expert_engine 文件；本次应增加管线版本及文件哈希，不能只沿用原版本。

现有主流程尚未完成自由主点标定、双人长期身份跟踪和 BVP 全自动求解。新功能必须逐项迁移，不能仅把 SKILL.md 放入提示词就称为“执行完整 skill”。

## 3. 最新 skill 固定与执行约定

用户包：`/Users/bytedance/Downloads/user_skills.zip`。

- ZIP SHA256：`b4c904fda19a1e2abf9b79600038936b2608ea7e3d6d3bc22bc950a2809d0c4a`
- SKILL.md SHA256：`9700b2919289a0d03c75bba0bb470c1d252bd632af021c7321135e2ef9b46131`
- 已读参考位于 `work/skill-update-review/sports-video-motion-analysis/`，此目录只保存与旧版不同的文件，**不是完整安装包**。

实现时从 ZIP 安全解压完整副本到 `backend/app/skill_runtime/vendor/sports-video-motion-analysis/<sha>/`；禁止越界路径、符号链接和解压炸弹，记录逐文件哈希。保留项目已有旋转、音频有限补齐、取消子进程和缓存修复，不能整体覆盖项目脚本。

将 skill 编译成有边界的工作流，不允许模型自由执行 shell、安装包、改代码、调用飞书发布或读取其他文件。skill 的报告发布要求不等于用户授权云端发布；V1 报告仍在本地。

### 3.1 阶段分工与退出条件

| 阶段 | 本地执行 | 模型允许提供 | 不满足时 |
|---|---|---|---|
| 视角体检 | 视频方向、PTS、机位残差、裁剪/黑边、尺度检查 | 图像中的球场/人物/遮挡类型候选 | 不可靠机位仅做低置信二维分析 |
| 稳定化 | 当前保守匹配、变换记录、坐标转换 | 提示疑似比分条区域 | 匹配不足保留原画面，禁止静默套用变换 |
| 人物提取 | YOLO、候选身份关联、跳变/腕部质量检测 | 在给定人物 ID 中选择目标或返回 unknown；候选持拍手/朝向 | 局部重跑，仍冲突则弃用受影响动作指标 |
| 触球时刻/归属 | 音频峰、挥拍峰、多人物时间窗交集 | 在有编号的连续帧中定位可能触球范围、归属候选 | 歧义保留，不强制选人 |
| 球点复核 | TrackNet、运动过滤、短缺口裁剪重跑 | 选候选球 ID、指出不可见、建议更精确裁剪 | 无本地像素/时序证据则不补观测 |
| 场地标定 A/B | 标定/verify、独立尺度约束、残差检查 | 给已检测线分配身份；不生成虚构场地尺寸 | 不输出三维速度/真实落点 |
| 轨迹与速度 | 观测拟合或条件具备的 BVP、旋转敏感性、区间 | 只提出待验证的接触/弹地证据 | 不可辨识标不可测，不计零 |
| 动作指标 | 持拍手与局部朝向规则、代理指标、冲突检查 | 提议复核，不直接覆盖正反手/三维角度 | 标签 unknown，指标带来源 |
| 渲染与报告 | 确定性统计、画面抽检、音视频完整性 | 根据经过验证的 JSON 组织中文解释 | 文本失败使用本地模板 |

### 3.2 分层交付范围

V1 必须完成二维流程的模型复核闭环；A/B 几何分支必须有明确状态和门槛。BVP 不默认生成，用于后续实验开关：需要已验证相机、逐次触球身份/时间/三维接触点、稳定坐标和充分边界约束。无法满足时显示“未测量”，这属于正确执行 skill 的边界要求。

不能把“回合继续所以落点在界内”作为独立证明再报告“识别出界内”；模型推断出的旋转、接触高度和落点必须标为假设。最后一拍滚动球位置不等于第一次弹地点，不得直接代替，除非时间/位置关系有独立验证。

## 4. 克制的 UX

### 4.1 唯一设置入口

主页右上角菜单增加“模型连接”。不增加模型选择页、聊天框、Agent 思考面板或每次上传的参数表。

连接抽屉默认只显示：服务“GLM”、接入类型、API Key 密码框、“测试并保存”。模型名和端点按服务预设；V1 无自定义 URL。接入类型分清 Coding Plan 与普通 API；资格待确认的套餐显示不可启用的说明和官方文档链接，而非让用户反复输入 Key 猜测。

成功连接后显示“已连接 GLM”、掩码标识、增强分析开关、断开按钮。文本与视觉两种能力分别探测，但对用户只显示“可增强分析”或具体缺失能力。文本可用而视觉不可用不得显示视觉就绪。

首次启用明确说明：“将向智谱发送选中的关键帧/短片段与分析数据，用于复核；原视频不会默认整段上传。可能消耗你的服务额度。”确认范围默认为后续新分析任务；历史结果不自动重跑。关闭后新任务只用本地；已运行任务在下一次云端请求前检查撤回状态。

### 4.2 日常流程

上传 → 本地分析 → 必要时增强复核 → 视频与报告。进度只有“分析中 / 复核关键片段 / 生成结果”。用户可离开页面。无疑难片段时跳过云端；页面说明“本次无需增强复核”，不能仅因配置了 Key 就显示“已使用 GLM”。

结果最多一行来源提示：“本地分析 · GLM 复核 4 段”；失败则“增强服务暂不可用，已保留本地结果”。展开详情可看接受/拒绝提议数量和限制，无原始提示词/堆栈/密钥。

本地隐私徽标应动态变为“本地处理 + 已启用智谱复核”，不能联网传视频同时写“全程离线”。

## 5. 总体架构

```text
React 模型连接抽屉
  → FastAPI connections + 本机凭据存储
  → SQLite 连接非敏感元信息

现有 automatic job
  → 固定版本 SkillRunner
  → 本地确定性初分析 + 异常计划
  → EvidenceBuilder（关键帧/裁剪/时间与坐标清单）
  → ProviderAdapter（GLM 普通API / 经确认可用的 Coding Plan）
  → SchemaValidator + EvidenceValidator
  → 受限局部重跑 / 接受或拒绝提议
  → 重新计算轨迹、事件、统计
  → 渲染 + 报告 + 来源审计
```

编排由状态机控制。V1 不需要引入通用多 Agent 框架，也不让一个文本模型自由决定运行多少工具。每个任务最多两轮增强，第二轮仅处理首轮未解决且仍有新证据的窗口。

### 5.1 推荐模块

```text
backend/app/model_connections/{schemas,service,secret_store,capabilities}.py
backend/app/model_providers/{base,glm_api,glm_coding,mcp_client,errors}.py
backend/app/skill_runtime/{manifest,runner,contracts,policy}.py
backend/app/enhancement/{planner,evidence,validator,apply,audit}.py
backend/app/api/model_connections.py
frontend/src/ModelConnectionDialog.tsx
backend/tests/test_model_connections.py
backend/tests/test_enhancement_contracts.py
backend/tests/test_glm_adapters.py
backend/tests/test_skill_runner.py
```

`base.py` 定义 probe、review_images、optional_review_video、optional_write_report、cancel。能力包括协议、输入 MIME/大小、视觉、多图、结构化输出、取消、用量来源，不按模型名称猜测能力。

### 5.2 GLM 两类适配

`glm_api`：HTTP 客户端调用普通 chat completions，Bearer Key，视觉模型候选 `glm-4.6v`；具体可用性由账户探测确认。优先图片，不默认上传整段视频。多图顺序与 evidence manifest 一一对应。若当前服务支持 data URL 经实测验证后直接发送；否则明确设计私有短期上传及删除生命周期，在验证前禁用该传输方式，不生成公开链接。

`glm_coding`：未满足资格条件前只允许模拟测试，不向套餐端点发送真实网球分析请求。获准后按官方确认的接入协议实现；视觉使用明确获准的工具/接口，不假设文本端点接收图片。若需 MCP，使用固定经过测试的 `@z_ai/mcp-server` 版本、stdio 连接，启动时 initialize、tools/list，按返回 schema 调用 image_analysis/video_analysis。禁止编造参数；schema 哈希变更时 capability 变为待重测。

MCP 子进程仅获得任务证据目录及必要环境；关闭 stdin 后等待有界时间，再清理整个进程组。启动不读用户其他 MCP 配置，不带入无关环境变量。安装是显式设置步骤，运行任务期间不执行 npx latest 自动下载安装。

Coding Plan 与普通 API 使用不同配置 ID、计费标识、探测记录，不能因一个失败自动调用另一个。模型名从探测成功配置冻结到任务快照，禁止后台无通知换成其他计费产品。

## 6. 存储、API 与凭据

### 6.1 数据表（增量迁移）

`model_connections`：id、provider、account_kind、protocol、display_name、model_id、secret_ref、capabilities_json、eligibility_status、eligibility_evidence_ref、enabled、consent_version、consented_at、config_revision、created_at、updated_at。

`enhancement_requests`：id、job_id、connection_id、config_revision、evidence_hash、skill_hash、prompt_version、round、status、attempt_count、provider_request_id、started_at、finished_at、usage_json、error_code、response_artifact。

唯一键 `(job_id,evidence_hash,connection_id,config_revision,prompt_version,round)` 保证本地重复调用去重。服务端未提供幂等能力时，不能承诺超时重试不会重复扣费。

不要复用项目视频 hash 代替证据 hash；证据内容、帧时刻、裁剪与稳定化变换、模型版本、提示词任一变化都应失效。任务启动冻结配置，但每次发送前仍检查连接启用/撤回状态与凭据版本。

### 6.2 凭据策略

Mac 使用系统 Keychain，数据库仅保存 secret_ref。容器/无 Keychain 环境用运维注入的 secret 文件（只读权限）或系统凭据服务；不得用硬编码加密 key 加密后放同一个数据库冒充安全存储。

Key 经本机 POST 送入后端后立即清空前端状态；不写 localStorage、URL、任务 payload、trace、异常或 MCP 启动参数。GET 只回 has_key 与随机连接别名。轮换时新版本原子生效；删除连接清理凭据，历史审计仅保留非敏感来源。Keychain 不可用要明确报错，不静默落明文。

本机服务保持 loopback；校验 Host/Origin，增加随机本机会话 token 和写入 CSRF 防护，拒绝任意网页跨站保存 Key 或触发云端发送。日志默认不记录 HTTP body/Authorization。

### 6.3 API 契约

| 方法/路径 | 作用 | 重要约束 |
|---|---|---|
| GET /api/model-connections | 列出掩码状态、可用能力 | 无 secret、无完整响应日志 |
| POST /api/model-connections/probe | Key + provider + account_kind；返回 probe_id、能力、状态 | 短期探测凭据，合成图、不使用用户视频；60秒内过期 |
| POST /api/model-connections | probe_id + enable + consent_version | 只有能力通过且资格允许才能启用；保存凭据前二次校验 |
| PATCH /api/model-connections/{id} | 启停、轮换、重新验证 | 乐观锁 config_revision；不能靠客户端设置 eligibility |
| DELETE /api/model-connections/{id} | 撤回连接与凭据 | 停止后续发送，不删除本地分析结果 |
| GET /api/projects/{pid}/expert/{sid}/enhancement | 用户可读复核摘要 | 已有项目访问/路径校验复用 |

probe 不是免费调用保证：按钮提示“测试可能消耗少量服务额度”。套餐资格待确认时不执行远端探测。返回错误采用稳定枚举，不直接展示厂商错误体。

自动分析 API 保持原参数兼容，后端解析全局已启用连接，无需前端发送 Key、模型参数或 endpoint。每个 job 记录 effective_connection_id 和 skill_hash。

## 7. 证据包与模型结果合同

### 7.1 选片策略（产品初值，非供应商限制）

按影响排序：疑似换人 > 击球归属冲突 > 触球附近球轨迹断裂 > 正反手冲突 > 场地线身份待确认。先本地局部重跑；仍不确定才调用云端。

每视频默认最多6个窗口，每窗口约1–2秒，最多8张关键帧，最多2轮，总计12次视觉请求；调用数为硬预算，tokens/费用为软估计。超过预算直接保持未知。来自同一时段的多个问题合并一次请求。优先只发原始图和候选编号，避免渲染的“答案”反过来诱导模型。

Evidence manifest 示例：

```json
{
  "schema_version":"1.0",
  "evidence_id":"ev_001",
  "project_id":"...","job_id":"...",
  "question":"attribute_contact",
  "frames":[{"frame":64,"pts_ms":2133.33,"asset":"f64_crop.jpg",
    "original_size":[960,540],"crop_xyxy":[300,120,700,450],
    "encoded_size":[800,660],"coordinate_space":"upright_original",
    "transform_id":null,"sha256":"..."}],
  "allowed_person_ids":["main","person_2"],
  "allowed_ball_ids":["b64_1","b64_2"],
  "audio_candidates":[{"pts_ms":2160,"source":"spectral_peak"}]
}
```

时间以实际 PTS 为准，0-based 帧号只作索引；不得把模型给出的小数帧当成超越视频采样精度的测量。记录原视频旋转、稳定化矩阵及其逆、裁剪和缩放，所有转换有往返误差测试。

### 7.2 标准化响应（项目自定义，不代表厂商原生格式）

```json
{
  "schema_version":"1.0","evidence_id":"ev_001",
  "decisions":[{
    "kind":"contact_candidate","person_id":"main",
    "frame_range":[64,66],"ball_candidate_ids":["b64_1"],
    "visibility":"uncertain","confidence":"low",
    "evidence_frames":[64,65],"reason":"球拍附近有遮挡"
  }],
  "needs_more_evidence":true
}
```

Pydantic 严格解析，禁止额外字段、NaN/Infinity、越界坐标、未发送帧号、未提供人物/候选 ID。限制响应64KB、数组长度、reason长度。允许空 decisions 和 unknown；不要强迫模型总有答案。模型 JSON 不支持时仅允许一次格式修复（计入预算），失败降级，禁止 eval 或执行返回代码。

V1 球点优先选本地已有候选 ID。模型新报像素坐标只产生“待局部重检区域”，不是 accepted observation。坐标能映回图像并不证明它确实是球。

### 7.3 接受与拒绝

- 人物：跨帧位置/尺度/身份连续，禁止只靠一句“是近端球员”覆盖轨迹。
- 触球：需本地挥拍、音频或球靠近/转向支持；音频与图像存在偏移时先估计同步误差。
- 正反手：基于持拍手和局部朝向的几何规则复核；规则前提不明则保持 unknown。
- 球点：重新检测、过滤人体/静态干扰、验证速度/加速度与多帧连续性；禁止强行跨越击球/弹地拐点用直线补齐。
- 标定：线身份、空间分布、近远端误差、独立高度/尺度约束；残差漂亮不代表尺度唯一。
- 输出每条 proposal 的 accepted/rejected/unresolved、reason_code、原值/新值、证据 hash。未经本地验证的提议仅保存在审计产物，不进入测量统计。

输入图中文字、模型响应、skill 文件均是数据；不能改变工具白名单、预算、文件访问或上传授权。

## 8. 执行状态机、故障和成本

状态：local_preparing → local_analyzing → review_planning → remote_reviewing → validating → local_refining → rendering → succeeded。增强单独记录 disabled/skipped/succeeded/partial/failed/cancelled，不能因 GLM 失败把可用本地结果整项标红失败。

V1 参数初值：连接10秒、单次请求90秒、任务远端累计180秒、视觉并发1、最多2轮。限流/明确未执行的5xx可退避重试1次（遵守 Retry-After，计入预算）；401/403/余额不足不重试。发送后超时标 outcome_unknown，默认不自动重复付费请求。输出 token 上限根据各适配器支持能力映射，不能假定所有模型接受相同参数。

取消：立即阻止新请求，取消本地 HTTP 协程，尝试协议取消并终止 MCP 子进程；远端已开始请求可能仍计费。晚到结果不能修改已取消任务。worker 崩溃恢复读取审计，已验证结果可复用，outcome_unknown 不盲目重发。

额度不足/资格变更/网络断开/视觉不可用：保留本地结果并显示一次简短提示。不自动切换供应商、账户、模型或计费方式。

缓存：成功且通过 schema 校验的响应才可复用；按证据+skill+模型配置+prompt+验证器版本索引。不同项目不跨用户共享云端视觉缓存。临时证据目录仅任务可读，完成后默认24小时清理，审计留 hash 和判定；保留原始云端响应需调试显式开启并脱敏。

## 9. 前后端与脚本实施细节

不要让 GLM 在当前 `automatic.py` 子进程内部隐式联网。先拆出本地 detection / derive / render 三个可重复调用阶段，由 expert service 的 SkillRunner 显式调度增强。保留已存在的原始候选缓存，二次候选独立文件保存。

建议产物：

```text
expert/<session>/
  pipeline_manifest.json     # code、skill、权重、provider配置revision
  local_baseline/            # 首次观测和统计，之后不可变
  evidence/manifest.json     # 文件、帧、坐标、哈希与发送清单
  enhancement/requests.jsonl # 脱敏调用与状态
  enhancement/proposals.json
  enhancement/validation.json
  enhancement_summary.json
  auto_tracks.json / shots.json / auto_summary.json
  annotated.mp4 / report.md
```

改造 `FILES` 只开放可读摘要，不直接允许下载含私密响应的目录。project trash/delete 必须覆盖新产物；本地任务取消/删除并发互斥沿用已有 idle 检查。渲染必须区分 observed / interpolated / reconstructed；若将来启用BVP，用不同线型与“重建”标签，不能混为真实检测轨迹。

`httpx` 当前是 dev 依赖，如采用其异步客户端需移至生产依赖并更新 uv.lock。MCP SDK/Keychain库分别锁版本；本机 M2 默认单视觉请求，不能同时开多个大模型推理任务抢内存。

## 10. 开发任务拆分与验收

### 阶段0：资格和能力验证

交付 `docs/GLM接入验证记录.md`，记录官方范围确认、账户类型、协议、模型/视觉工具、schema hash、账单资源类型，不存 Key。若资格未确认，套餐状态保持pending；普通API仅在用户单独启用后联调。不把“模拟通过”当真实套餐接入验收。

### 阶段1：连接底座

连接API、Keychain、启停/轮换、合成图探测、GLM provider interfaces、mock transport。验收：断网也能保存本地分析；Key 不出现在前端存储、DB、日志或进程参数；错误 Key 清晰提示；关闭后没有新发送。

### 阶段2：证据与复核

从 people/person_quality/scene_quality/quality_review 生成异常计划；构建带帧号证据、调用视觉能力、严格解析；先只支持人物归属/击球窗口/候选球筛选。验收：拒绝虚构帧/人物/坐标；一次调用不能直接覆盖baseline；所有修改有来源和验证记录。

### 阶段3：技能编排和结果

固定skill快照、A/B能力门槛、两轮预算、重算统计、模板报告兜底、来源提示。连接抽屉只增加最少设置，默认分析页不增加方法菜单。验收：额度耗尽、MCP崩溃、取消、进程恢复均能保留本地结果。

### 阶段4：真实效果评测与发布

至少准备固定近景、双人对拉、左手/面向镜头、移动机位、低清/遮挡、无音频六类片段；建立人工真值小集合，按整段视频划分验证集，不能只调一个样例。对比 baseline / local-review / GLM-review 三组。

指标：人物ID切换、腕关键点定位误差、触球事件 precision/recall（预声明时间容差）、归属正确率与unknown率、球检测precision/recall（预声明像素/球径容差）、错误新增球点数、正反手准确率及未确认比例、耗时、请求数、实际额度/费用。覆盖率单列，不能替代准确率。

建议产品发布门槛（待基线测量后冻结）：已接受的新增球点人工精确率≥95%；事件精确率相对基线下降不超过2个百分点；关键错误不能增加；每个指标报告分母和样本不足说明。未达到则默认关闭影响数值的修改，仅保留复核提示。不得根据模型自评conf或几段看起来顺的视频宣布成功。

## 11. 必须覆盖的测试清单

- 单元：严格schema、坐标正反变换、PTS映射、提示词注入不改变工具权限、证据去重、预算和缓存版本。
- Provider契约：文本成功视觉失败、工具名/schema改变、空结果、拒答、截断、非法JSON、401/403/429/5xx、发送后超时。
- 生命周期：连接关闭/轮换/删除、取消晚到响应、worker重启、重复点击分析、不重复合并提议。
- 凭据：读API脱敏，日志/异常/进程参数无Key，Origin/Host/会话检查，临时证据不暴露任意文件路径。
- 前端：未配置/待资格确认/已连接/视觉缺失/额度不足；375/768/1440px、键盘焦点、密码框不回显、关闭后隐私徽标更新。
- 媒体验收：旋转正确、导出帧数/时长/音频一致、稳定化坐标正确、观测与重建视觉区分。
- 集成：提供FakeProvider固定响应即可离线跑完整管线；真实GLM测试由显式环境开关启用，CI默认不消费额度。

## 12. 给开发 Agent 的执行约束与完成定义

先读取当前代码及本方案，不覆盖既有修复。先完成连接底座与FakeProvider端到端，再接真实服务；真实调用前确认启用范围和具体账号计费类型。不要安装或配置用户全局 Claude Code，也不要把 API Key 写进文档、命令或版本库。

每个阶段交付代码、迁移、针对性测试、真实运行记录和限制。完成定义：用户只配置一次合法可用的连接，上传后后台可完成本地分析、可选视觉复核和校验，失败也有可用本地结果；关键帧发送透明，结果来源可追踪；Coding Plan 若未获资格确认必须仍标未激活，不能用普通API成功来冒充套餐成功。

本方案仅为技术文档。没有保存任何 Key、调用付费模型、上传视频或修改现有运行流程。
