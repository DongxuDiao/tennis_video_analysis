# 流水线细则：调参、迭代与失败处置

每一步都写清楚：跑什么、怎么判断结果对不对、不对时改什么。顺序不能跳——标定不过关就不要去拟弹道，速度全部会错。

下面第 1–9 步是 **A 模式**（固定机位、单人训练视频）的主线。B 模式（转播裁剪 / 跟拍 / 双球员）的完整步骤序列在文末「B 模式流水线」，两条路的差异只集中在标定、球速和主体选择三处。

## 0. 视角体检：先定 A 还是 B

**这一步不能跳，也不能凭印象跳。**抽两帧相距 ≥100 帧的画面（`video_prep.py grab`，或直接看 `frames/` 里两张 JPEG），把同一个球场角点的像素坐标写下来比一比：

| 现象 | 走哪条 |
| --- | --- |
| 角点位移 < 2 px，画幅是相机原生比例，只分析 1 个人 | A 模式 |
| 角点位移 2–10 px（轻微抖动） | A 模式可用，但建议仍做稳定化 |
| 角点位移 > 10 px，或球场在两帧间明显缩放 | B 模式 |
| 竖屏裁剪、加黑边、数码变焦（画幅不是原生比例） | B 模式（主点不在画面中心） |
| 需要分析 2 人及以上 | B 模式（击球归属） |
| 球在画面直径只有 2–4 px | B 模式（放弃逐帧球检测） |
| 角点在画面中心稳、边缘发散（推车 / 走动 / 无人机 = 相机平移） | 两条路都不支持，别硬做 |

命中任意一条 B 就走 B。按 A 模式硬做**不会报错**：标定残差照样 1–2 px，球速照样输出，只是系统性偏 10–20%。完整判据与失败症状见 `broadcast-view.md`。片段里有镜头切换或回放，先按切点切成独立片段，每段自己选参考帧、自己标定。

## 0.5 A / B 的分叉点与汇合点

- **两边共用**：`video_prep.py`、`pose_extract.py`、`audio_impacts.py`、`pose_metrics.py`、`render_annotated.py`、报告。
- **B 模式专属替代品**：`calib_broadcast.py` 替代 `court_calib.py`（9 参数、自由主点）；`flight_bvp.py` 替代 `flight_fit.py`（接触点边界值求解，不需要球点）。
- **B 模式专属新增**：`stabilize.py`（1.5 步）、`swing_peaks.py`（3.5 步，多人必做）。
- **B 模式通常整条跳过**：`ball_detect.py` → `flight_fit.py`。
- 分叉点在第 1 步之后（都从 `video_prep.py` 的产物出发），汇合点在 `pose_metrics.py` / `render_annotated.py` / 报告——这三步两边都跑，只是参数不同（B 要 `--pose-map`、`--stabilize` / `--src-frames`）。

B 模式要维护**两个工作目录**：`$WORK`（原始帧）和 `$WORKS`（稳定化画布）。第 1.5 步之后所有脚本都跑在 `$WORKS` 上，只有渲染要回头读 `$WORK/frames`。

## 1. 视频探测与抽帧

```bash
python3 $SKILL/scripts/video_prep.py prep --video clip.mov --work work [--width 960] [--bg-samples 40] [--no-audio]
```

产出 `meta.json`（宽、高、fps、帧数、时长、有无音频、`rotation_deg`）、`frames/f_%05d.jpg`、`audio.wav`、`bg_median.png`。

- **`rotation_deg` 必须看**。手机拍的 `.mov` 常带 ±90/180° display matrix。ffmpeg 抽帧会自动应用它，`cv2.VideoCapture` 视版本可能不应用。本 skill 全程只用 `frames/` 里的 JPEG，不要中途改用 `cv2` 直接读视频，否则姿态、球点、标定会分属两个坐标系。
- **分辨率**：960 px 宽足够跑通全流程（网球在这个尺度下直径 3–8 px，仍能测视差与表观大小）。更高分辨率会让球检测更稳但姿态推理线性变慢；`--width` 只在原片超过 1920 时才需要。
- **背景中位图**：`--bg-samples 40` 抽 40 帧取中位数，得到没有运动物体的干净场地，供检线和球检测的静态掩膜使用。如果视频里相机有位移或缩放，背景图会糊，此时整条几何流水线都不成立——需要改成分段处理。
- **`grab` 子命令**用来把任意视频（包括渲染结果）的指定帧存成 PNG 看细节：

```bash
python3 $SKILL/scripts/video_prep.py grab --video annotated.mp4 --frames 64 65 66 --out chk
```

## 2. 人体姿态

```bash
python3 $SKILL/scripts/pose_extract.py --work work --model yolo11m-pose.pt --imgsz 960 --conf 0.35 --min-area 6000
```

每帧只保留一个主体（面积最大且过 `--min-area` 的人），输出 COCO-17 关键点。

- 模型选择：`yolo11n-pose.pt` 最快，适合先验证流程能跑通；`yolo11m-pose.pt` 是分析用的默认档，手腕/脚踝明显更稳。首次运行会自动下载权重。
- 跑完看脚本打印的覆盖率。经验阈值：整体检出 >95%，肩 >95%，髋 >95%，腕 >80%，踝 >60%。腕覆盖率低会直接毁掉挥拍速度和击球点估计，此时换更大模型或提高 `--imgsz`。
- 画面里有对面球员、教练、旁观者时，用 `--roi X0 Y0 X1 Y1` 把注意力锁定在近端半场，`--min-area` 天然会滤掉远处的人（远端球员像素面积通常只有近端的 1/4）。
- `--device` 默认 cpu；有 GPU 时传 `cuda:0`，`--batch` 可提到 32。
- 多人片段每名球员各跑一次，用 `--label` 分文件：`--label far` 写 `pose_far.json`，默认 `--label main` 仍写 `pose.json`（单人流程不受影响）。`--subject near|far` 在 ROI 之后再按"大而低 / 小而高"选主体，`auto` 是历史行为（取最大框）。
- 跑完必看脚本打印并写进 json 的 `qc` 块，**`n_jumps` 不为 0 就不要往下走**：那意味着"主体"在不同人之间跳。判据是 bbox 中心位移超过 `--jump-disp` 个框宽或面积变化超过 `--jump-area` 倍。压制手段按效力排序：`--roi` → `--subject` → 提高 `--conf` / `--min-area`。
- 已经生成的 pose 文件可以单独复检：`pose_extract.py --qc-only pose_far.json --work $WORKS`（不加载模型，不需要帧）。

## 3. 击球时刻

```bash
python3 $SKILL/scripts/audio_impacts.py --work work --band 2000 8000 --min-gap 0.35 --top 12 --expect 7
```

拍击是宽带瞬态，在 2–8 kHz 的谱通量上是尖峰。脚本输出 `impacts.json`，每条含小数帧号 `frame`、秒 `t`、`score`。

- **音频给的是最准的时间锚**（±0.5 帧），比看视频找触球帧可靠得多，因为 30 fps 下球拍与球接触只有约 1 ms，几乎不会正好落在某一帧上。
- 峰不只有击球：球落地弹跳、脚步摩擦、球撞围网、说话都会进榜。用 `--top` 取比预期多 50%–100% 的候选，然后**逐条用视频复核**：真击球峰前后几帧一定能看到球从来向变出向，且人处于挥拍中段。
- `--min-gap` 是同一次击球的多峰合并窗口，网球对拉取 0.35 s；乒乓球快攻要降到 0.15 s，羽毛球杀球连击取 0.25 s。
- `--prominence` 以正谱通量的标准差为单位。峰太少就降到 2.0，噪声太多提到 4.0。
- 完全没有音频（`meta.json` 里 `has_audio=false`）时退化方案：用手腕关键点速度的极小值/方向反转帧作为 `tc` 的初值，精度会掉到 ±2 帧，报告里要相应放宽速度区间。

## 4. 球场标定（A：`court_calib.py`）

见 `calibration.md`，这是整条流水线唯一必须人来做身份判断的几何环节。三步不能省：

```bash
python3 $SKILL/scripts/court_calib.py detect --work work --skip-top 0.5 --max-lines 14
python3 $SKILL/scripts/court_calib.py fit    --work work --ident work/ident.json --band 6
python3 $SKILL/scripts/court_calib.py verify --work work --row-min 100 --tol 8
```

B 模式换成 `calib_broadcast.py fit/verify`（9 参数、自由主点，`verify` 用 `--tol-near` / `--tol-far` 两档容差）。什么时候必须换、为什么固定主点会让残差好看而尺度漂，见 `calibration.md`。

## 5. 球检测（A 模式专属）

```bash
python3 $SKILL/scripts/ball_detect.py --work work --colour yellow_green --min-area 4 --max-area 400 --max-dim 26 --row-min 180 --bg-thresh 22 --diff-thresh 18
```

逻辑是三重与：颜色在预设范围内 ∧ 与背景中位图有差异 ∧ 与前后帧有差异，再减掉"静止球色物体"的膨胀掩膜（地上的球、绿色围网、场地贴纸），最后按面积/长宽/长宽比筛连通域。输出 `ball_cand.json`，键为帧号，值为候选列表，字段 `x y a w h gb sat val n`（`gb` 是绿蓝差，`n` 是该帧候选总数）。

- **正常产出量**：每帧 2–5 个候选。>10 说明掩膜没生效或阈值太松；<1 说明太紧。
- **`n == 1` 的帧最可信**，优先用它们锚定轨迹，再向两侧扩展。一段好的出球轨迹在图像上应该单调、逐帧位移接近（30 fps、80 km/h 下约 10–25 px/帧）。
- 调参顺序：先用 `--row-min` 切掉观众席/横梁（取近端底线上方一点的行号）→ 再收 `--max-dim` 挤掉球拍框和人的肢体 → 最后用 `--colour` 预设匹配球色（`yellow_green` / `white` / `orange` / `any`）。
- 拍面遮挡、球与亮色背景重合、球速过快导致长条残影，都会造成**整段丢球**。这不是 bug，是这类视频的固有限制：接受"该拍出球不可测"，不要靠放松阈值把噪声凑成轨迹。
- 球在画面只有 2–4 px（转播裁剪）时这一步整条跳过：实测全局候选只有 40/440 帧能被物理轨迹穿过，引导式检测的点与拟合弧线中位偏差 28 px。走 `flight_bvp.py`。

## 6. 弹道与球速（A：`flight_fit.py`）

```bash
python3 $SKILL/scripts/flight_fit.py --work work --input shots_input.json \
    --spins 0 1000 1500 2000 3000 --nominal-spin 1500 --bootstrap 40
```

输入是人工确认过的球点（结构见 `data-schema.md`），输出 `shots.json`。内部做四件事：

1. **来球段拟合**，反演击球点 `contact`（观测 ≥4 点）；也可在输入里直接给 `contact`。
2. **出球段拟合**，用重力 + 空气阻力（`--cd 0.55`，`--mass 0.057`）解初速度，图像位置残差 + 表观直径深度项联合最小化。
3. **bootstrap 区间**：对像素与直径加噪 40 次重采样，给出 `kmh_lo`/`kmh_hi`（16/84 分位）。
4. **旋转带**：对每个 `--spins` 值算落点、过网余量、结论；全带一致才 `result_firm=true`。

置信度规则（脚本自动打）：`high` = 出球观测 ≥4 且残差 <2.5 px 且区间相对宽度 <0.2；否则 `medium`；观测 ≤2 或残差大 → `low`；无法求解 → `none`。**`low` 的速度值只能当假设，`none` 记"不可测"。**

结论字段 `result` ∈ `IN` / `NET` / `OUT_LONG` / `OUT_WIDE` / `LONG?` / `UNK`，`result_cn` 是中文版，`result_note` 给出外推提示。

## 7. 动作指标

```bash
python3 $SKILL/scripts/pose_metrics.py --work work --shots shots.json --hand right --prep-window 20
```

输出 `technique.json`，一行一拍。指标含义与阈值见 `pose-metrics.md`。跑完先看两件事：

- `side_conflict` 是否为空。冲突表示画面判断的正反手与击球点横向位置不符，回到那几帧重看，改 `shots_input.json` 的 `side` 后重跑 6、7 两步。
- 关键指标是否有 `null`。缺失通常源于该拍附近腕/髋关键点没检出，回到第 2 步换模型。

`--hand` / `--facing` 决定"画面左右 ↔ 正反手"的映射（`fh_on_image_right = (hand == 'right') == (facing == 'away')`），默认 `--hand right --facing away` 就是历史硬编码规则。两个参数都支持按球员写：`--hand N=left,F=right`、`--facing N=away,F=toward`；`contacts.json` 的 `players[*].hand/facing` 优先级高于命令行。

躯干指标有两条路，脚本自动选并把结果写进每行的 `torso_source`：`shoulder_hip_3d`（三维肩髋方位角）或 `torso_ratio_proxy`（图像肩宽/躯干长比值代理）。降级条件由 `--rot3d-min-frac` / `--rot3d-min-std` / `--rot3d-min-box-h` / `--rot3d-max-jitter` / `--rot3d-min-conf` 控制，脚本会把降级原因逐条打印。要强制走某一条用 `--torso-source`。降级后的度数只能做同片段内相对比较。

## 8. 渲染

```bash
python3 $SKILL/scripts/render_annotated.py --work work --frames 60 80 --out check.mp4   # 抽检
python3 $SKILL/scripts/render_annotated.py --work work --scale 1.5 --panel 300 --out annotated.mp4
```

**永远先抽检再全量**。全量 1000 帧约 3–5 分钟，版面问题（俯视图压住文字、面板串行、图例超出画布）只有在真实帧上才看得见：用 `grab` 把 check.mp4 的几帧存成 PNG 逐个看。配置项见 `render-config.md`。

`--panel` 现在接 `auto` / `right` / `bottom`（也仍接一个数字，等价于历史的 `--panel 300`），面板尺寸走 `--panel-size`；竖屏（H > W）下 `auto` 自动改成底部面板。稳定化片段必须加 `--stabilize` + `--src-frames`，多球员用 `--pose` 多文件 + `--pose-who` / `--pose-label`，BVP 球路用 `--flights`，见下方 B 模式流水线。

## 9. 报告

按 `report-template.md` 的骨架组织飞书文档，图表走 lark-htmlbox skill，文档与视频附件走 lark-doc skill。发布前对齐三处口径：四类落点次数之和 = 总拍数；平均速度只统计可测拍；每个"出界/下网"结论都标了是实测还是外推。

## B 模式流水线（转播裁剪 / 跟拍 / 双球员）

约定 `SKILL=user_skills/sports-video-motion-analysis`、`WORK=work`（原始帧）、`WORKS=works`（稳定化画布）。每步下面那行「门禁」不达标就**停在这一步**，不要往下走：往下走的成本是全盘重算。深层原理与实测基准见 `broadcast-view.md`。

**B1 探测与抽帧**（与 A 完全相同）

```bash
python3 $SKILL/scripts/video_prep.py prep --video clip.mov --work $WORK
```

**B2 稳定化**——先干跑只算 homography，通过了再落盘 warp 帧（565 帧干跑 <1 min，落盘要几分钟、几百 MB）。

```bash
python3 $SKILL/scripts/stabilize.py --work $WORK --out $WORKS --ref auto \
        --mask-rows 0 160 --emit-frames false
python3 $SKILL/scripts/stabilize.py --work $WORK --out $WORKS --ref 280 --mask-rows 0 160
```

`--mask-rows TOP BOTTOM` 排除烧入比分条/台标（它们不随场景动，会主导 ORB 匹配）；`--ref auto` 按"画布外扩面积最小"选参考帧，`--ref 280` 手工指定；参考帧一旦定了不要中途换（画布尺寸和主点会变，下游全部要重跑）。`--max-pad` / `--max-pad-v` 限制画布外扩，被 clamp 的帧脚本会列出来。

> 门禁：`weak_frac` > 0.05（脚本直接 WARNING）或残差 p95 > 3 px 就停——前者说明 overlay 掩膜没做对或有镜头切换，后者说明帧间映射不是纯 homography（相机在平移）。

稳定化**不会**复制音频，脚本只打印提示。下一步之前手动补上：`cp $WORK/audio.wav $WORKS/`，否则 `audio_impacts.py --work $WORKS` 找不到 wav。

**B3 姿态：两名球员各提一次**

```bash
python3 $SKILL/scripts/pose_extract.py --work $WORKS --label near --subject near
python3 $SKILL/scripts/pose_extract.py --work $WORKS --label far  --subject far \
        --roi 60 200 1060 400 --min-area 2500
```

ROI 坐标是**稳定化画布**里的坐标。远端球员像素小，`--min-area` 要降到 2000–3000，否则整段检不出来。

> 门禁：两个文件的 `qc.n_jumps` 都必须是 **0**。实测远端不加 ROI 是 34 次跳变（最差一帧位移 8.88 个框宽），加了 ROI 是 0 次；34 次跳变足以让下游全盘失真。

**B4 击球时刻**

```bash
python3 $SKILL/scripts/audio_impacts.py --work $WORKS --top 40
```

双人对拉的峰比单人多得多（弹地、脚步、喊声都在里面），`--top` 取比预期拍数多两三倍，宁滥不缺，后面用姿态去筛。

**B5 谁在击球**

```bash
python3 $SKILL/scripts/swing_peaks.py --work $WORKS \
        --pose $WORKS/pose_near.json $WORKS/pose_far.json \
        --hand left right --thr 0.45 --window 6 --match-impacts $WORKS/impacts.json
```

产物是每个 pose 文件一份 `swing_<label>.json`（label 默认从文件名推，`pose_near.json` → `near`）加一份 `swing_match.json`。默认 `--thr 0.55 --window 4` 偏保守（实测召回 8/14），`--thr 0.45 --window 6` 提到 12/14。`--hand` 支持每个 pose 文件给一个值；**不要用 `auto`**，实测它在远端球员上判错，且脚本自己会打印 margin 警告。已知接触帧可用 `--truth 34 55 85` 直接算召回。

> 门禁：`swing_match.json` 里 `ambiguous=true` 的峰必须人工看过那几帧才能定归属，不要让脚本替你选。

**B6 标定：9 参数自由主点**

```bash
python3 $SKILL/scripts/calib_broadcast.py fit    --work $WORKS --ident $WORKS/ident.json
python3 $SKILL/scripts/calib_broadcast.py verify --work $WORKS --zoom
```

`ident.json` 除 `manual` 线提示外**必须**有 `net_tape` 窗口（`fit` 无条件读它），并强烈建议给 `player_height` 和 `principal_prior`。参数、失败模式和实测基准值见 `calibration.md`。

> 门禁：`verify` 打印 `PASS`（近端线 ≤ `--tol-near` 2 px、网带与远端线 ≤ `--tol-far` 8 px）才算过。出现 `FAIL` 就回去改线的身份 / `net_tape` 窗口 / 身高约束再重拟，别看叠加图自我安慰。`no-line`（那一列附近根本没有白线）既不算过也不算不过，要换 `--columns` 再验一次。

**B7 人工确认接触帧与接触点 → `contacts.json`**

这是 B 模式唯一集中的人工环节，写好它后面都是确定性计算。接触帧来自 B4×B5 的交集；接触点 `point` 是 3D 米坐标：脚踝落点给深度、持拍手腕射线与合理击球高度相交给横向与高度。每拍还要给 `who`、`side`，回合最后一拍要给 `final_landing`，弹地音频候选给 `bounce_audio_candidates`。结构见 `data-schema.md`。

> 门禁：`flight_bvp.py` 会检查接触帧递增、`point` 长度为 3、相邻接触点分居球网两侧（同侧会 WARNING）。这三条里任何一条报出来，都是标注错了，不是脚本挑食。

**B8 球速：接触点边界值求解**

```bash
python3 $SKILL/scripts/flight_bvp.py --selftest
python3 $SKILL/scripts/flight_bvp.py --work $WORKS --contacts $WORKS/contacts.json
```

先跑 `--selftest`（校验积分器、Magnus 符号、BVP 反演自洽，不需要数据）。默认在 `--spins 0,800,1600,2400,3200` × `--angles -8,0,8,16,24` 上搜，用合法性 + 音频弹地峰打分挑解。

> 门禁：`conf == "low"` 的拍**不进任何平均**，并且要读它的 `flip_reason`——`err_m` 超 `--err-max` 通常意味着接触点或接触帧标错了，回 B7 改，不要靠调 `--err-max` 把它蒙过去。

**B9 动作指标**

```bash
python3 $SKILL/scripts/pose_metrics.py --work $WORKS --shots contacts.json \
        --pose-map N=pose_near.json,F=pose_far.json
```

`--shots` 直接吃 `contacts.json`（脚本按有没有 `contacts` 键自动识别格式），`who` 通过 `--pose-map` 选对应球员的 pose 文件，持拍手与朝向从 `players` 块读，所以命令行不必再写 `--hand` / `--facing`。

> 门禁：`side_conflict` 必须逐条回看。实测它抓出过一个被误标成正手的反手，改完之后该球员正反手统计和均速结论都反转了。

**B10 渲染**

```bash
python3 $SKILL/scripts/render_annotated.py --work $WORKS --src-frames $WORK/frames \
        --stabilize stabilize.json --shots contacts.json --flights flights.json \
        --pose $WORKS/pose_near.json $WORKS/pose_far.json --pose-who N F \
        --pose-label "近端=255,190,90" "远端=170,255,170" \
        --panel bottom --frames 330 340 --out check.mp4
```

`--stabilize` + `--src-frames` 是 B 模式的必需组合：叠加层画在画布上，再用每帧 homography 的逆 warp 回原始帧做 alpha 合成（`--overlay-alpha` 调不透明度）。缺了任一个就满屏错位。`--flights` 让球路按每拍**自己的** rpm 物理重建，面板会自动带上"球路为重建非检测"的说明。

> 门禁：抽检帧里不能有贯穿画面的脏线（视锥/边界裁剪没生效），俯视小图近端必须在下方，面板不能压住画面主体。确认无误再全量。

**B11 报告**：口径与 A 相同（见第 9 步），另外双球员片段要分球员各一行再给合计，出界/下网按回合级判定统计，见 `broadcast-view.md` 第九节。

## 收尾

`work/frames/`（1000 帧 JPEG，约 100 MB）、无声中间视频 `*_noaudio.mp4`、`__pycache__` 都可以删。要留的是 `meta.json`、`calib.json`、`shots_input.json`、`shots.json`、`technique.json`——有这五个文件就能在不重抽帧的前提下复算全部结论（渲染除外）。

B 模式要留的是 `$WORKS` 里的 `meta.json`、`stabilize.json`、`calib.json`、`calib_verify.json`、`contacts.json`、`flights.json`、`technique.json`。`stabilize.json` 千万别删——它是画布与原始帧之间唯一的桥，没有它就重渲染不了，也没法把画布坐标换回原始帧坐标；而 `$WORKS/frames/`（warp 帧）删了可以按同一个 `--ref` 重新生成。
