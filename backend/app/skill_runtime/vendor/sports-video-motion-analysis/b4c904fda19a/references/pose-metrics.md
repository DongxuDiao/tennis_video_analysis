# 动作指标：定义、阈值与建议映射

单机位拿不到真三维关节角，但教学关心的大多是**相对量与时序**，这些是可测的。原则：报差值和趋势，不报绝对角度。

## 旋转代理量怎么来的

肩线（左肩 ↔ 右肩）在图像上的水平投影宽度随身体转离相机而缩短。取整段视频里该宽度的 96 分位数作为"正对相机"基准 `SW`，则

```
rot(f) = arccos( width(f) / SW )
```

得到 0°（正对）到 90°（完全侧对）的旋转代理量。髋部同理，基准为 `HW`。

这个量的**绝对值不可信**（受机位角度、关键点抖动、96 分位取值影响），但同一片段内的**差值稳定可比**，因为所有偏差都作用在同一个基准上。

## 指标清单（`technique.json` 每行）

| 字段 | 含义 | 单位 |
| --- | --- | --- |
| `f_coil` | 触球前 `--prep-window` 帧内旋转峰值所在帧（引拍最大位） | 帧 |
| `shoulder_rot_prep` | 引拍峰值时的肩旋转 | ° |
| `hip_rot_prep` | 同一帧的髋旋转 | ° |
| `x_factor` | `shoulder_rot_prep - hip_rot_prep`，躯干分离度 | ° |
| `shoulder_rot_contact` | 触球帧肩旋转 | ° |
| `shoulder_rot_finish` | 触球后 6–12 帧的平均肩旋转 | ° |
| **`unwind_to_contact`** | `prep - contact`，**躯干真正打进球里的释放量** | ° |
| `unwind_to_finish` | `contact - finish`，随挥中的继续转动 | ° |
| `coil_to_contact_frames` / `_s` | 引拍峰值到触球的时间 | 帧 / 秒 |
| `elbow_angle_contact` | 触球时肩–肘–腕夹角 | ° |
| `wrist_speed_px` | 触球窗口内腕点最大帧间速度 | px/帧 |
| `contact_h` | 击球点高度（来自 `shots.json` 的三维拟合） | m |
| `contact_lateral_sw` | 击球点相对肩线中点的横向偏移，**以肩宽为单位** | 肩宽 |
| `contact_vs_shoulder_px` | 击球点相对肩线的垂直位置（负=高于肩） | px |
| `contact_ahead_m` | 击球点在双脚连线之前多少米 | m |
| `hip_y_range_px` | 触球前 18 帧到后 10 帧髋部垂直行程（起伏/蹬伸） | px |
| `head_move_px` | 引拍峰值到随挥中头部位移（稳定性） | px |
| `side_suggest` / `side_conflict` | 由 `contact_lateral_sw` 反推的正反手及冲突标记 | — |

身份与规则字段（每行都有，报告引用时用得到）：`who`（球员键，来自 `contacts.json`）、`hand`、`facing`、`side_rule`（人可读的当拍判定规则，如 `hand=left,facing=away -> FH on image left`）、`pose_file`（这一拍用的哪个 pose 文件）。

有 `calib.json` 时还会补一批米制量：`wrist_speed_ms`（腕速，只含平行于像平面的分量）、`lat_from_body_m`、`contact_h_pose` 与 `contact_h_source`（`shot_field` / `contact_point` / `pose_wrist`，说明击球高度是标注给的还是姿态推的）、`backswing_to_contact_f`、`ground` / `behind_baseline`（球员站位）、`prep_h` / `followthrough_h` / `hip_h`。没有标定时腕速退回 px/帧并标 `wrist_speed_unit`。

标定基准的三个百分位不要混：`shoulder_rot_*` 用肩线**水平投影宽度**的 96 分位（`SW`），髋部同理（`HW`），`contact_lateral_sw` 的分母是肩线**段长**的 96 分位（`SL`），而躯干代理量另有自己的 `--torso-pct`（默认 92），见下一节。

## 正反手判定：接触点在躯干中线的画面哪一侧

这条规则以前写错过，是错误率最高的单点。**不要**写死"右手持拍者正手在身体右侧"。判据是接触点相对**躯干中线**（两肩中点）的**画面左右**，而画面左右与正反手的对应关系由持拍手和朝向共同决定：

```
正手出现在画面右侧  ⟺  (hand == "right") == (facing == "away")
```

推论：右手持拍且**面向镜头** → 正手在画面**左**（镜像）；左手持拍且**背对镜头** → 正手也在画面**左**。实测那段双球员片段里两名球员恰好都是"正手在画面左"，一个因为镜像、一个因为背身，13/13 非发球拍与规则一致——凭直觉判会全部反过来。

`pose_metrics.py` 的参数：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--hand` | `right` / `left`，或按球员 `N=left,F=right` | 决定持拍侧腕/肘关键点与规则的一半 |
| `--facing` | `away`（镜头在球员背后，历史默认）/ `toward`，或 `N=away,F=toward` | 规则的另一半 |
| `--pose-map` | `N=pose_near.json,F=pose_far.json` | 按 `contacts.json` 每拍的 `who` 自动选对应球员的 pose 文件与 hand/facing |

`contacts.json` 里 `players[*].hand` / `players[*].facing` **优先于**命令行同名参数，所以双球员片段把这两项写进 `contacts.json` 最省事。默认 `--hand right --facing away` 就是历史硬编码规则，单人片段行为不变。

```bash
python3 $SKILL/scripts/pose_metrics.py --work $WORKS --shots contacts.json \
    --pose-map N=pose_near.json,F=pose_far.json
```

两条同样重要的经验：

- **多模态视觉模型给出的动作名称不可采信。**实测它对若干远端击球给出的正反手标签是错的。它的正确用途是定位帧、确认球在画面里的可见位置、判断落点；动作标签一律交给上面的几何规则。
- 反过来，几何规则报出的 `side_conflict` **值得当真**。实测它抓出一个被误标成正手的反手（近端球员左手持拍、背对镜头，接触点落在躯干中线的画面右侧 = 反手侧），修正后该球员统计从"正手 5 / 反手 1"变成"正手 4 / 反手 2"，正反手均速的结论随之反转。`|contact_lateral_sw|` 小于 `--side-tol`（默认 0.2 肩宽）时 `side_suggest` 给 `?`，不报冲突。

## 躯干代理指标（3D 角退化时自动接管）

**触发条件**（`pose_metrics.py` 逐 pose 文件判定一次，`--torso-source` 可强制）：3D 肩髋角可用帧比例 < `--rot3d-min-frac`（0.5）、角度离散度 < `--rot3d-min-std`（1.0°，即"全是 0"的退化）、球员 bbox 高度中位 < `--rot3d-min-box-h`（240 px）、逐帧抖动中位 > `--rot3d-max-jitter`（3.0°/帧）、肩髋关键点置信度中位 < `--rot3d-min-conf`（0.5）——命中任一条就降级。脚本会把命中的原因逐条打印出来。

**定义**：图像躯干宽高比 = 肩宽 / 躯干长度（两肩中点到两髋中点），随身体转离镜头轴而变小；用整段视频该比值的 `--torso-pct`（默认 92）分位作为"正对镜头"基准 `ratio_max`，则关闭角 = `arccos(ratio / ratio_max)`。取百分位而不是最大值，一坏帧就定不了标。扫描窗口是 `--torso-window`（默认触球前 12 帧到触球后 6 帧）。

**字段**：

| 字段 | 含义 |
| --- | --- |
| `torso_closed_deg` | 触球时刻的躯干关闭角 |
| `torso_closed_max_deg` | 窗口内的引拍峰值关闭角 |
| `torso_unwind_deg` | `torso_closed_max_deg − torso_closed_deg`，躯干解旋幅度，**要引用的就是这个数** |
| `torso_source` | `shoulder_hip_3d` 或 `torso_ratio_proxy`，标明这一行的来源 |
| `hip_closed_deg` / `torso_separation_deg` | 仅 3D 路线才有（髋线关闭角、肩髋分离） |

注意字段名是 `torso_source`（不是 `source`），而且它是**行级**的：同一行的三个躯干角必然同源。

**实测事实**：那段双球员片段 14 拍**全部**降级为 `torso_ratio_proxy`，原因是 3D 角逐帧抖动 7.2°、球员在画面里尺度过小。降级后的度数只能做同片段内相对比较，绝对值不能当真三维关节角引用，报告里必须带 `torso_source`。

代理量的**可用性依赖球员在画面里的像素高度**：近端球员 6 拍底线球上 `torso_unwind_deg` 与出球速度的相关系数达 0.94（n=6；剔除单个极值后 0.68），远端球员看不到同样的关系——尺度不够，分辨力不足。所以远端球员的躯干结论要显式降权，不要和近端球员直接比大小。

## 解读阈值（业余到进阶，网球正反手抽击）

这些是经验带，用于生成建议，不是判决：

| 指标 | 偏低 | 合理 | 说明 |
| --- | --- | --- | --- |
| `unwind_to_contact` | <20° | 30–50° | 低 = 手臂主导，躯干没参与，力量与稳定性都吃亏 |
| `x_factor` | <15° | 20–40° | 低 = 上下身一起转，失去弹性势能 |
| `coil_to_contact_frames` | <4 帧（30 fps） | 6–12 帧 | 太短 = 引拍不足/仓促，被动挡球的典型特征 |
| `contact_h` | <1.0 m | 1.1–1.5 m（腰到肩之间） | 太低 = 让球下落后才打，主动性差 |
| `contact_lateral_sw` | \|值\| <1.0 | 1.5–3.0 | 太小 = 击球点贴身，无法伸展。**符号只表示画面左右**，正手在哪一侧由 `hand`/`facing` 决定，别按"右手正手为正"读 |
| `contact_ahead_m` | <0.2 m | 0.3–0.8 m | 太小 = 击球点偏后，力量传不出去 |
| `elbow_angle_contact` | — | 正手 110–160°，反手（单反）150–175° | 过小 = 肘部锁死/夹臂 |
| `head_move_px` | — | <画面身高的 8% | 大 = 重心不稳或过早抬头看球 |
| `hip_y_range_px` | 过小 = 无蹬伸 | — | 与 `unwind_to_contact` 一起看 |

## 建议怎么写

**逐拍指标 → 分组均值 → 建议**，每条建议必须挂在具体数字上。实测样例（一段 35 秒、7 拍的网球训练视频）：

| 指标 | 正手（4 拍） | 反手（3 拍） | 读出的问题 |
| --- | --- | --- | --- |
| `unwind_to_contact` | 13.8° | 46.1° | 正手躯干几乎不参与，仅为反手的 30% |
| `coil_to_contact_frames` | 3.0 | 9.3 | 正手引拍仓促，0.1 s 就要触球 |
| `contact_h` | 1.52 m | 1.37 m | 正手偏高，反手偏低 |
| `contact_lateral_sw` | +2.67 | −1.42 | 反手击球点偏贴身，伸展不足 |

对应建议：

- 正手："躯干释放量仅为反手的 30%（13.8° vs 46.1°），引拍到触球只有 3 帧（0.1 s）——属于手臂主导的仓促挥拍。练习：分离式起动后先完成肩部转位再启动挥拍，用节拍器把引拍到触球拉长到 0.2 s 以上；靠墙做'先转肩、后送拍'的分解慢动作。"
- 反手："击球点距身体中心仅 1.4 个肩宽、高度 1.37 m，属于贴身偏低——转体虽充分（46.1°）但没转化成伸展空间。练习：多退半步给出手臂空间，在腰高、身前 0.5 m 处设置目标点做定点击球。"

**不要这么写**："正手转体只有 13.8 度，远低于职业选手的 90 度。"——绝对角度不可测，跨片段更不可比。

B 模式（双球员）引用时多两条：躯干数字用 `torso_unwind_deg` 并注明 `torso_source`；分球员分组，远端球员的躯干结论只作参考，不与近端球员比大小。

## 常见故障

- **指标为 `null`**：该拍附近腕/髋/踝关键点置信度不足。回到 `pose_extract.py` 换 `yolo11m-pose.pt` 或提高 `--imgsz`。
- **`side_conflict: true`**：画面判断的正反手与击球点横向位置矛盾。**先怀疑标签，再怀疑脚本**：回看那几帧，改 `shots_input.json`（A 模式）或 `contacts.json`（B 模式）的 `side` 后重跑第 6、7 步；如果整名球员都在冲突，那是 `hand`/`facing` 写反了，改 `players` 里的这两项。
- **`shoulder_rot_prep` 全都很大**：基准 `SW` 被低估了，通常因为整段视频里球员从未正对相机。此时旋转代理量整体偏移，差值仍可用，但绝对值更不能提。
- **左手持拍**：传 `--hand left`，脚本会切换持拍侧腕/肘关键点，并按上面的规则重算正反手对应的画面侧。朝向不是"背对镜头"时必须同时传 `--facing toward`，只改 `--hand` 会把结论整体反过来。
- **躯干角全部为 0 或方差极小**：3D 路线退化了，正常情况下脚本会自动降级到代理量并打印原因；如果 `torso_source` 仍是 `shoulder_hip_3d`，检查是不是用 `--torso-source` 强制了。
- **双球员时所有拍用了同一个 pose**：`--pose-map` 没写或 `who` 与 map 的键不一致，落回 `--pose` 的默认文件。看每行的 `pose_file` 就能确认。
