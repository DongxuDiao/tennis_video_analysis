# 本地 Skill 引擎 · P0 验证记录

- 日期：2026-09-20（完成）
- 机器：MacBook Air Mac14,2（Apple M2，16 GB），macOS Darwin 25.6.0
- 依据：《Tennis Lab：Qwen3-VL + 可执行 Skill 本地分析引擎改造方案》v1.1 §14 P0
- 状态：**P0 六项全部在本机实测完成**（推理 / 取消卸载重载 / 沙箱真实脚本 / ARM64 依赖 / LangGraph 崩溃恢复 / 版本冻结）。本记录只写实测事实；未测项见 §4，不写"达标"。

## 0. 环境事实

| 项 | 值 |
|---|---|
| Docker | 29.6.2，server linux/aarch64，可用 |
| 磁盘可用 | 28 Gi（卷 460 Gi） |
| Python（系统） | 3.13.3（Homebrew）；包管理用 `uv`（venv 不预装 pip） |
| Xcode license | 未接受：`git` 直用报错，需 `DEVELOPER_DIR=/Library/Developer/CommandLineTools` 前缀（本记录中 git 命令均带此前缀） |
| 仓库 git 状态 | **无任何提交**（全部文件 untracked）——追溯性风险，建议尽快初始提交 |
| 默认 Skill 包 | `/Users/bytedance/Downloads/user_skills (1).zip`，SHA-256 `1d55f6f304b59bb728f78b23d704775a6822dce6760eb1f2d0aa1a93f8fcfee0`，与方案 §2.2 一致 ✅（31 个条目，解包检查无路径穿越） |

## 1. 版本锁定（P0 冻结值）

### 1.1 模型服务独立环境 `local_services/mlx_model/.venv`

完整冻结清单已导出：`local_services/mlx_model/requirements-frozen.txt`（uv pip freeze）。关键版本：

| 包 | 版本 |
|---|---|
| python | 3.13.3 |
| mlx | 0.32.2 |
| mlx-vlm | 0.7.1 |
| transformers | 5.17.0 |
| huggingface-hub | 1.32.0 |
| numpy | 2.5.3 |
| pillow | 12.3.0 |
| psutil | 7.2.2（内存采样用） |

### 1.2 权重

- 仓库：`mlx-community/Qwen3-VL-4B-Instruct-4bit`，许可 **apache-2.0**（README 声明）
- **revision（冻结）：`2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`**（main @ 2026-09-20）
- 本地路径：`data/local_models/Qwen3-VL-4B-Instruct-4bit/`（14 个文件）
- 仓库已知怪癖：`model.safetensors.index.json` 是陈旧的两分片映射，实际附带单个合并 `model.safetensors`（3,093,767,283 字节）——加载无碍，哈希以下表为准。

### 1.3 编排环境与沙箱镜像

LangGraph 验证环境 `work/p0/lg_venv`（完整冻结：`work/p0/lg_venv-freeze.txt`）：
langgraph 1.2.11 / langgraph-checkpoint 4.2.0 / langgraph-checkpoint-sqlite 3.1.1 / langchain-core 1.6.3 / aiosqlite 0.22.1。

沙箱镜像：`tennis-sandbox:sports-python-cpu-v1-p0`
**ID=sha256:8c467492fc980a0d875f94f405001ed3b0dfcbc086abde76aab88db9c855ecc4**
（base `python:3.12-slim`；apt 只装 `fonts-noto-cjk`；运行用户 uid 10001 非 root；Dockerfile：`runtime_images/sports-python-cpu-v1/Dockerfile`）
镜像内冻结依赖（pip freeze 实录）：

```
imageio-ffmpeg==0.6.0        # 静态 ffmpeg, ffmpeg-linux-aarch64-v7.0.2
numpy==2.5.3
opencv-python-headless==4.14.0.94
pillow==11.3.0
```

### 1.4 权重文件 sha256（revision 2fd8dacb…）

```
d0761303276b99006238f4a2c88f153109025c796b5e929145985b05700d1042  README.md
c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680  added_tokens.json
3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4  chat_template.jinja
6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4  chat_template.json
07406d087dfb8a8849427a4da81bc9edd1dd942e518493629b5a983169b47820  config.json
8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44  generation_config.json
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt
90eeb02604181dbcccd0a30a1f550a4a8928ca7dcbee4aee1449239306cfdfca  model.safetensors
58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b  model.safetensors.index.json
93585062a80db5e8ca038efc7726a3e6411d9db948472d81d63c6303993be8c5  preprocessor_config.json
76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd  special_tokens_map.json
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json
81ec7bb9530159b326c0bef1d0b6c33d392090524014ea3f0123a3c1eb9c2af5  tokenizer_config.json
59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1  video_preprocessor_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
```

## 2. 验证结果（全部 2026-09-20 本机实测）

### 2.1 LangGraph 检查点 + 模拟崩溃恢复 ✅

- 脚本：`work/p0/test_langgraph_recovery.py`；checkpointer：`SqliteSaver`（独立 SQLite 文件）
- 图：A→B→C；`FAIL_AT=B` 使 B 抛异常模拟崩溃；journal.jsonl 作独立执行账本
- 结果（journal 时序）：`A, B(崩溃)` → 新进程恢复 → `B(重执行), C` ✅
- **协议踩坑（重要）**：同 thread 恢复必须 `invoke(None, config)`；
  错误示范 `invoke({"steps": []}, config)` 会从 START 重头执行——实测 A 重复执行、
  状态因 reducer 累加出现 `['A','A','B','C']`。runtime coordinator 封装恢复入口时必须只走 `None` 路径
  （判断条件：`graph.get_state(cfg).next` 非空 → 传 `None`）。
- 语义确认：B 崩溃后无 checkpoint → 恢复时 B 按至少一次调度重执行；A 已有 checkpoint 不重跑。

### 2.2 默认包脚本沙箱执行 ✅（ARM64 CPU 容器，无网络）

- 输入：真实项目视频副本（7 MB，18.8 s，764×960 竖屏，test.mov）
- 运行 flags：`--network none --cap-drop ALL --security-opt no-new-privileges`，容器内 uid 10001，工作盘目录宿主侧 chmod 777 挂载到 `/work`
- 实测结果：

| 测试 | 结果 | 耗时/产物 |
|---|---|---|
| `video_prep.py`（cv2+numpy+imageio_ffmpeg 懒加载） | ✅ | 6.5 s；565 帧 `f_00001…f_00565.jpg`（1-based=帧0）+ audio.wav + bg_median.png + meta.json |
| `ball_detect.py`（纯经典 cv2/numpy） | ✅ | 5.8 s；224 帧处理出 360 个球候选（ball_cand.json） |
| 相对导入（camera.py / court_calib.py 经 scripts 目录 sys.path） | ✅ | 按包内既有机制工作 |
| 中文字体渲染（Noto CJK） | ✅ | cn_font_test.png 生成正常 |
| H.264 编码（imageio-ffmpeg 静态 ffmpeg，aarch64） | ✅ | encode_test.mp4 生成正常 |

- 结论：方案 §6 的 ARM64 CPU 沙箱路线在依赖预装 + 无网络 + 降权条件下可跑默认包的真实计算脚本。

### 2.3 Qwen3-VL-4B（MLX 4-bit）推理 ✅（含一次测试脚本 bug 的诚实更正）

脚本：`work/p0/test_mlx_inference.py`（+ 事后复测）。证据图长边 672。

| 测试 | 结果 | 实测值 |
|---|---|---|
| 加载 | ✅ | 首次 6.7 s（冷缓存）；复测 3.5 s（热缓存） |
| A 单图中文场景描述 | ✅ 准确 | 10.8 s。描述"两名球员、发球准备、KIA/墨尔本广告、转播视角"——已用独立视觉工具核对原帧：该视频确为**手机竖屏录制的电视转播画面**（MELBOURNE 字样、KIA 广告、FEDERER/NADAL 记分牌均在画面中），描述属实，非幻觉 |
| B 多图顺序 | ✅（复测） | 见下方更正框 |
| C 受约束动作 JSON | ✅ | 输出可解析，`action=call_tool`，工具名落在给定工具列表内（skill.ball_track.*） |
| D 流式取消 | ✅ | 收 4 个 chunk 后主动中断，0.8 s |
| E 卸载/重载 | ✅ | del + gc + `mx.clear_cache()` → RSS 回落；重载 6.5 s，重载后请求正常应答 |

> **B 项更正记录（诚实报告要求）**：首轮 B 判"模型答错"**无效**——测试脚本引用 `cn_font_test.png` 路径错误，回退成了网球帧，三张图里根本没有含中文的图，问题不可答。复测（正确路径）：
> - B1 中文图放第 3 张 → 答 3 ✅
> - B2 中文图放第 1 张 → 答 1 ✅
> - B3 两图各述内容 → 位置与内容均对，且能读出测试卡上的文字 ✅
>
> 结论：≤3 图时顺序映射实测正确（双向）。但首轮乌龙也说明：**对模型的每个判定必须用可客观核验的证据构造**，不能依赖"答案看起来合理"。

### 2.4 内存（系统级口径）与连续负载

- **RSS 口径误导**：推理期进程 RSS 仅 ~0.5 GiB，Metal 统一内存常驻不体现在 RSS——内存结论必须用系统可用内存（`psutil.virtual_memory().available`）口径。
- 本机空闲基线就紧：**未加载模型时系统可用仅 4.22 GiB**（整机 16 GiB，其他常驻约 11.8 GiB）。加载后可用 2.30 GiB → 模型实际占用 ≈1.9 GiB。
- 轻量连续负载：4 次串行单图请求，延迟 3.7 / 4.1 / 4.1 / 5.0 s，系统可用稳定在 2.3–2.7 GiB，回答内容一致且正确（费德勒发球准备/纳达尔接发，与画面相符）。

### 2.5 发现的模型能力边界（P1 设计必须吸收）

1. **结构性感知可靠、细粒度事实会编**：B3 复测中模型正确读出画面全部结构元素，但断言"2018 年澳网男单半决赛"——费纳澳网交手实际为 2012 决赛/2014 半决赛/2017 决赛，无 2018。印证方案约束：模型只输出受约束动作 JSON，其自由文本一律不得进入结果字段。
2. 系统内存余量小（加载后 ~2.3 GiB 可用）→ runtime 必须**单模型实例、串行决策请求**；并发多任务编排（沙箱脚本可并行、模型节点不可）需在 P5/P7 设计时作为硬约束。
3. HF 下载坑：xet 分块协议在固定字节偏移卡死（换镜像无效）；`HF_HUB_DISABLE_XET=1` + `https://hf-mirror.com` 后约 7 MB/s 完成。下载器实现须默认带此开关并做断点重试。

## 3. P0 结论

方案 §14 六项 P0 全部实测通过：本地 4-bit 权重可用受约束推理、可取消/卸载/重载、默认包计算脚本可在无网络降权 ARM64 容器真实执行、中文/编码/依赖全过、LangGraph 崩溃恢复可用（含 resume 协议）、全部版本与哈希已冻结。**具备进入批次 2（Registry/导入器/Reader）的条件。**

## 4. 未测项（保持"待验证"，不得宣称完成）

- [ ] 多任务**并发**下的内存/延迟行为（本记录只测了串行 4 请求；需 runtime 就位后按真实调度形态测）
- [ ] >3 图、同类型图交错时的顺序稳定性
- [ ] 长时间连续推理的热/降频行为（M2 无风扇）
- [ ] 前端跑通默认 Skill 全链路（§16 要求，属批次 4–8 交付后的验收项）
