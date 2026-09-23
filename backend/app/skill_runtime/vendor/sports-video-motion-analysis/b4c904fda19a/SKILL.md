---
name: sports-video-motion-analysis
description: 单机位运动视频（网球/羽毛球/乒乓球等挥拍类及类似球类）的逐帧打标与技术分析流水线。能力：抽帧与音频冲击检测、人体姿态提取（支持双球员对拉、ROI 约束与跨人跳变自检）、球场标定与相机重建（含转播裁剪视角的自由主点标定与跟拍/变焦稳定化）、小球检测、三维弹道与球速反演（观测点拟合与接触点边界值求解两条路径，含旋转不确定性带与置信度分级）、动作技术指标计算（躯干旋转、引拍时序、击球点高度与横向位置、稳定性，持拍手与朝向可配置）、渲染带骨架/球轨迹/俯视图/统计面板的打标视频（横屏右侧面板与竖屏底部面板）、并输出带置信度边界的中文飞书分析报告。适用场景：把一段训练视频或转播比赛片段做逐帧打标并输出打标视频、统计正反手次数与出界/下网次数、估算平均球速与最快球速（km/h）、给出正反手动作优化建议、生成运动技术分析报告或数据可视化，也适用于任何"给视频里的人体动作与球/物体轨迹逐帧标注并量化分析"的需求。
author: 刁栋旭
---

# 运动视频逐帧打标与技术分析

一条已经跑通的流水线：单机位视频 → 逐帧标注视频 + 可量化的技术结论 + 带置信度边界的中文报告。

核心难点不是画框，而是**从一个机位把像素还原成米和 km/h，并诚实地说明哪一部分是测量、哪一部分是外推**。所有脚本都围绕这两件事组织：先建立可验证的几何，再在几何之上做物理反演，最后把不确定性写进产物而不是藏起来。

约定：`SKILL=user_skills/sports-video-motion-analysis`，`WORK=work`（工作目录放在当前工作区，不要放进 skill 目录）。所有脚本都接受 `--work $WORK`，帧号一律 **0-based**，磁盘上的 `frames/f_00001.jpg` 就是第 0 帧。

## 第 0 步：先做视角体检，再决定走哪条流水线

**这一步不能跳。**流水线有两套几何假设，选错了后面每个数字都是错的，而且错得很安静——标定残差照样能收到 1–2 px，球速照样输出，只是系统性偏 10–20%。

抽两帧相距 100 帧以上的画面，逐项对照：

| 体检项 | 固定机位（A 模式） | 转播 / 跟拍（B 模式） |
| --- | --- | --- |
| 球场角点在两帧间的位移 | < 2 px | 明显移动，或球场缩放 |
| 画面是否被裁剪 / 竖屏重构图 | 否，是原始相机帧 | 是（9:16 竖屏、数码变焦、加黑边） |
| 需要分析的人 | 1 名 | 2 名及以上 |
| 有无烧入 overlay（比分条、台标） | 一般没有 | 通常有，且不随场景移动 |
| 球在画面的直径 | ≥ 5 px，可逐帧检测 | 2–4 px，逐帧检测不可用 |

只要 B 模式命中任意一项，就走 B 模式。命中"球场角点移动"却按 A 模式做，是这条流水线最贵的一类错误。完整判据、失败症状与处置见 [broadcast-view.md](references/broadcast-view.md)。

一个额外的排除项：如果相机是**平移**的（推车、走动拍摄、无人机），帧间映射不是 homography，稳定化后拼图看着没问题而几何已经错了。症状是画面中心残差很小、边缘炸开，且 `verify` 在近端线上永远到不了 2 px。这种片段本流水线不支持，别硬做。

## 流水线

| 步骤 | 脚本 | 产物 | 需要判断的地方 |
| --- | --- | --- | --- |
| 1 探测与抽帧 | [`video_prep.py`](scripts/video_prep.py) prep | `meta.json` `frames/` `audio.wav` `bg_median.png` | 是否降分辨率 |
| 1.5 **稳定化（仅 B）** | [`stabilize.py`](scripts/stabilize.py) | `$WORKS/frames/` `stabilize.json` | 参考帧、overlay 掩膜、weak 帧比例 |
| 2 人体姿态 | [`pose_extract.py`](scripts/pose_extract.py) | `pose_<label>.json` | ROI 与主体选择、**跨人跳变必须为 0** |
| 3 击球时刻 | [`audio_impacts.py`](scripts/audio_impacts.py) | `impacts.json` | 哪些峰是击球、哪些是弹地/脚步 |
| 3.5 **谁在击球（多人时必做）** | [`swing_peaks.py`](scripts/swing_peaks.py) | `swing_<label>.json` `swing_match.json` | `ambiguous=true` 的峰要人看 |
| 4 球场标定 | [`court_calib.py`](scripts/court_calib.py)（A）／[`calib_broadcast.py`](scripts/calib_broadcast.py)（B） | `calib.json` `calib_overlay.png` | **线的身份识别（必须人看）** |
| 5 球检测 | [`ball_detect.py`](scripts/ball_detect.py) | `ball_cand.json` | 颜色/尺寸/行范围阈值；B 模式常直接放弃 |
| 6 轨迹与球速 | [`flight_fit.py`](scripts/flight_fit.py)（A）／[`flight_bvp.py`](scripts/flight_bvp.py)（B） | `shots.json` / `flights.json` | A：逐拍挑可信球点；B：接触点与飞行时长 |
| 7 动作指标 | [`pose_metrics.py`](scripts/pose_metrics.py) | `technique.json` | 持拍手/朝向、正反手标签复核 |
| 8 渲染打标视频 | [`render_annotated.py`](scripts/render_annotated.py) | `annotated.mp4`（含原声） | 版面是否碰撞 |
| 9 报告 | 见 [report-template.md](references/report-template.md) | 飞书文档 + 图表 + 视频附件 | 结论口径 |

A 模式（固定机位单人训练视频）：

```bash
SKILL=user_skills/sports-video-motion-analysis; WORK=work
python3 $SKILL/scripts/video_prep.py prep --video clip.mov --work $WORK
python3 $SKILL/scripts/pose_extract.py --work $WORK --model yolo11m-pose.pt
python3 $SKILL/scripts/audio_impacts.py --work $WORK --top 12
python3 $SKILL/scripts/court_calib.py detect --work $WORK --skip-top 0.5
#   看 court_lines.png → 写 ident.json → 拟合 → 量化验证
python3 $SKILL/scripts/court_calib.py fit    --work $WORK --ident $WORK/ident.json
python3 $SKILL/scripts/court_calib.py verify --work $WORK
python3 $SKILL/scripts/ball_detect.py --work $WORK --colour yellow_green --row-min 180
#   逐拍整理 shots_input.json（结构见下）
python3 $SKILL/scripts/flight_fit.py --work $WORK --input shots_input.json
python3 $SKILL/scripts/pose_metrics.py --work $WORK --shots shots.json
python3 $SKILL/scripts/render_annotated.py --work $WORK --frames 60 80 --out check.mp4
python3 $SKILL/scripts/render_annotated.py --work $WORK --out annotated.mp4
```

B 模式（转播裁剪 / 跟拍 / 双球员对拉）：

```bash
SKILL=user_skills/sports-video-motion-analysis; WORK=work; WORKS=works
python3 $SKILL/scripts/video_prep.py prep --video clip.mov --work $WORK
#   先只算 homography 看质量，通过了再落盘 warp 帧
python3 $SKILL/scripts/stabilize.py --work $WORK --out $WORKS --ref auto \
        --mask-rows 0 160 --emit-frames false
python3 $SKILL/scripts/stabilize.py --work $WORK --out $WORKS --ref auto --mask-rows 0 160
#   两名球员各提一次姿态，用 ROI 压住跨人跳变
python3 $SKILL/scripts/pose_extract.py --work $WORKS --label near --subject near
python3 $SKILL/scripts/pose_extract.py --work $WORKS --label far  --subject far \
        --roi 60 200 1060 400
python3 $SKILL/scripts/audio_impacts.py --work $WORKS --top 40
python3 $SKILL/scripts/swing_peaks.py --work $WORKS --pose pose_near.json pose_far.json \
        --hand left right --thr 0.45 --window 6 --match-impacts impacts.json
python3 $SKILL/scripts/calib_broadcast.py fit    --work $WORKS --ident $WORKS/ident.json
python3 $SKILL/scripts/calib_broadcast.py verify --work $WORKS
#   逐拍整理 contacts.json（结构见下）
python3 $SKILL/scripts/flight_bvp.py --work $WORKS --contacts contacts.json
python3 $SKILL/scripts/pose_metrics.py --work $WORKS --shots contacts.json \
        --pose-map N=pose_near.json,F=pose_far.json
python3 $SKILL/scripts/render_annotated.py --work $WORKS --src-frames $WORK/frames \
        --stabilize stabilize.json --flights flights.json --panel bottom \
        --pose pose_near.json pose_far.json --frames 330 340 --out check.mp4
```

所有脚本共用 [`camera.py`](scripts/camera.py) 里的相机与场地模型（世界坐标定义、场地预设、投影/反投影、球的表观直径，主点可通过 `pp` 指定）。单脚本的完整参数看 `--help`；中间产物的 JSON 结构看 [data-schema.md](references/data-schema.md)。

## 四个必须守住的环节

### 标定要用数字验证，不能靠眼睛

叠加图"看起来差不多贴上了"完全不可靠：室内场馆的横梁、窗框、栏杆会产生大量长白线，自动检线经常把它们排在真实底线前面。所以 `fit` 之后必须跑 `verify`，它把模型预测的关键线行号与图像实测白线行号逐条列出。近端线（底线、发球线）偏差应在 2 px 内，网带与远端线在 8 px 内；超了就说明线的身份认错了，或标定落进了错解。

单机位只用场地线拟合存在**焦距与机位距离的耦合解**：同一组线可以同时被 f≈385 px / 机位靠前 和 f≈554 px / 机位后退 拟合到 1–2 px，两者近端几乎一致、远端相差 20 px 量级，而深度尺度会线性传导到球速上。要打破耦合必须补一个独立尺度约束：网带高度、远端底线，或 `ident.json` 里的 `player_height`。

B 模式还多一层：画面被裁剪或稳定化后**主点不在画布中心**，`court_calib.py fit` 的中心主点假设会把误差吸收进偏航/俯仰/焦距，残差照样漂亮而深度尺度系统性漂移。这时改用 `calib_broadcast.py`（9 参数：位姿 6 + 焦距 1 + 主点 2），并且**必须**同时给网带高度或球员身高约束——地面线只有 8 个自由度，撑不起 9 个未知量。细节见 [calibration.md](references/calibration.md) 与 [broadcast-view.md](references/broadcast-view.md)。

### 谁在击球、哪只手持拍，是所有标签的前提

单人视频里这不是问题，双人对拉里这是**最大的错误来源**：音频冲击峰只知道"有东西被击中了"，不知道是谁击的。把 458 帧的峰记到近端球员名下，那一拍的接触点、球速、正反手标签全错，还会顺着 BVP 求解污染前后两拍。

做法是让音频与姿态求交：`swing_peaks.py` 给出每名球员持拍手腕的速度峰，与 `impacts.json` 做时间窗匹配，只有窗内恰好一名球员有挥拍峰才算确定；两人都有峰时输出 `ambiguous=true`，必须回去看那几帧。默认阈值偏保守，`--thr 0.45 --window 6` 在实测上把召回从 8/14 提到 12/14。

正反手判定**不要**写死"右手持拍者正手在身体右侧"。判据是接触点相对躯干中线的**画面左右**，而画面左右与正反手的对应关系由持拍手和朝向共同决定：

```
正手在画面右侧  ⟺  (持拍手 == right) == (朝向 == away)
```

所以右手持拍且面向镜头的球员，正手出现在**画面左侧**（镜像）；左手持拍且背对镜头的球员，正手也在画面左侧。`pose_metrics.py` 用 `--hand` / `--facing` 承接这两个参数。同样重要的是：**多模态视觉模型给出的动作名称不可采信**，它只用来定位帧；动作标签一律由上面的几何规则判定，实测中模型对若干远端击球给出的正反手是错的。

### 球点要人工确认；确认不了就换求解路线

`ball_detect.py` 输出的是候选而不是球：球拍框、拍线残影、鞋面反光、地上静止的球都会混进来。做法是以音频击球帧为锚，在 `tc` 前后各 20 帧内逐帧核对候选（`n=1` 的帧最可信），把确认的点写进 `shots_input.json`。出球段观测点少于 3 个时最小二乘会给出"完美拟合"的过拟合值——本流水线曾在只有 1 个出球点的一拍上得到 140 km/h，而同一动作的合理值在 78 km/h 附近。

但转播裁剪片段里球只有 2–4 px，逐帧检测**根本不成立**：实测全局候选只有 40/440 帧能与物理轨迹对上，引导式检测拿到 32 点、中位偏差 28 px，也就是说那些"检测"根本不是球。这时不要再挣扎着拟合，改走 `flight_bvp.py`：把飞行当成两点边界值问题——已知第 i 次触球的 3D 接触点 A、第 i+1 次触球的接触点 B 和飞行时长 dt，在"重力 + 二次阻力 + Magnus + 一次硬地反弹"下求初速度 V0。三个未知量三个方程，球速、落点、过网余量、出界与否都是这条轨迹的推论。

回合的最后一拍没有下一次触球提供边界条件（回合正是以出界/下网结束的）。用视觉定位的滚动球落点补上，并在输出里标 `boundary_source=visual_landing`，与正常拍区分开。

### 落点是外推，必须给不确定性

出球弧段通常只被观测 0.15–0.35 s，这个时长既测不出旋转，也不足以约束落点，而上旋恰恰决定球在哪落地：同一组初速在 0 rpm 下过底线、在 2000 rpm 下落在场内。

A 模式（`flight_fit.py`）的做法是**扫旋转带**：每拍在 0/1000/1500/2000/3000 rpm 各算一次落点，只有整条带结论一致才算稳健（`result_firm=true`）。B 模式（`flight_bvp.py`）的做法不同：端点被钉死了，旋转变成**由合法性约束挑选的自由参数**——必须过网、若回合继续则落点必须在对方有效区内、发球须落在发球区、预测弹地帧应对上某个音频峰。换个旋转就把 IN 翻成 OUT 的拍标 `low` 并附 `flip_reason`，不进平均。两种模式的置信度语义不同，报告里不要混着写，判定与措辞见 [measurement-limits.md](references/measurement-limits.md)。

零旋转口径系统性高估飞行深度，也就系统性高报出界，不要当默认答案。

## 逐拍数据怎么整理

人工判断全部集中在一个输入文件里，写好它，后面都是确定性计算。

A 模式用 `shots_input.json`：

```json
{"fps": 30,
 "shots": [{"id": 1, "side": "FH", "tc": 64.8,
            "incoming": [[44, 442, 369], [45, 437, 361]],
            "outgoing": [[66, 460, 276], [67, 468, 259], [70, 474, 235]]}]}
```

- `tc` 用音频峰的**小数**帧号，1 帧偏差就是几 km/h 的偏差。
- `incoming` ≥4 点才能反演击球点；不足时用 `"contact": [x, y, z]` 直接给。

B 模式用 `contacts.json`，把"谁、什么时候、在哪、正手还是反手"全部显式写出来：

```json
{"fps": 30,
 "players": {"N": {"label": "近端", "hand": "left",  "facing": "away"},
             "F": {"label": "远端", "hand": "right", "facing": "toward"}},
 "contacts": [{"id": 1, "frame": 34, "who": "N", "side": "SERVE", "point": [-0.8, 10.41, 2.62]},
              {"id": 2, "frame": 55, "who": "F", "side": "BH",    "point": [5.35, -13.49, 1.06]}],
 "bounce_audio_candidates": {"1": [43.3], "3": [109.8]},
 "final_landing": {"point": [4.35, 8.5], "frame": 550, "source": "rolling ball at f549-557"}}
```

- `point` 是 3D 接触点：脚踝落点给深度，持拍手腕射线与合理击球高度相交给位置。
- `bounce_audio_candidates` 每拍可以给多个候选峰（弹地一般落在飞行的 50–80% 区间），由物理去挑，不要自己先定。
- `side` 用几何规则判定后再让 `pose_metrics.py` 复核，冲突时脚本报 `side_conflict`——**这个冲突值得当真**，实测中它抓出了一个被误标成正手的反手。

## 动作结论用相对量，不用绝对角度

单机位拿不到真三维关节角。可用的是**同一片段内的相对变化**：肩线在图像上的投影宽度随转体缩短，`arccos(width / width_when_square)` 给出旋转代理量——绝对值不可信，但"引拍峰值 − 触球时刻"这个差值（`torso_unwind_deg`）稳定可比，也正是教学上关心的躯干参与度。同理，击球点横向位置用肩宽为单位而不是像素。

球员正对或背对镜头、或远端球员在画面里只有几十像素高时，三维肩髋角会**退化**（实测算出来全是 0，因为左右肩/髋被投到了同一深度平面）。`pose_metrics.py` 会自动降级到图像躯干宽高比代理量，并在每个躯干指标上标 `source`（`shoulder_hip_3d` 或 `torso_ratio_proxy`）。报告里引用时必须带上这个来源，降级后的度数只能做同片段内比较。

报告里写"正手躯干释放 13.8°、反手 46.1°，正手仅为反手的 30%"是成立的；写"正手转体只有 13.8 度"就越界了。指标定义与解读阈值见 [pose-metrics.md](references/pose-metrics.md)。

## 渲染与检查

渲染是最贵的一步（1000 帧量级 3–5 分钟），所以先 `--frames a b` 渲十几帧，再用 `video_prep.py grab` 抽成 PNG 看版面：俯视小图与文字、面板与说明经常重叠，只有真实帧能暴露。

B 模式下几何是在**稳定化画布**里成立的，必须把球场线、球路、骨架画在画布上，再用 homography 的逆 warp 回原始帧合成（`--stabilize` + `--src-frames`）。直接画在原始帧上就会满屏错位。同时注意两件事：投影点落到画布外或深度为负会产生贯穿全屏的脏线，绘制前必须做视锥与边界裁剪；每拍要用**它自己的** rpm 重建球路，全片共用一个 rpm 会让轨迹和落点标记对不上。BVP 模式下球路是重建而非检测，面板上必须有一行明确说明。

竖屏视频右侧没有放面板的空间，`--panel auto` 会在 `H > W` 时自动改成底部面板。中文必须走 PIL + NotoSansCJK（`cv2.putText` 画不了中文）；输出尺寸自动对齐到 16 的倍数以避免编码器静默缩放；原声通过内置 ffmpeg（`imageio_ffmpeg.get_ffmpeg_exe()`，系统 PATH 里通常没有 ffmpeg）重新混流。面板字段、阶段划分、图例、语言都在 `--config` 里改，见 [render-config.md](references/render-config.md)。

## 报告

默认产物是**飞书文档**：结论摘要 → 统计总览表 → 逐拍明细 → 数据图表（HTMLBox）→ 失误成因 → 正反手优化建议 → 训练优先级 → 测量方法与置信度边界 → 打标视频附件。图表用 lark-htmlbox skill，文档创建与视频上传用 lark-doc skill。章节骨架、图表选型与措辞模板见 [report-template.md](references/report-template.md)。

统计口径固定为：总击球数；正手/反手/发球次数；出界/下网/界内/未判定次数；可测球速的平均值与最大值（另单独给高置信平均）；正反手分组均值。不可测的拍不进平均，也不记 0。

双球员片段要多守一条：**出界与下网是回合级判定，正反手次数是球员级统计**，两者不能共用一个分母。分球员各给一行，再给一行合计，并在开头点明谁是主视角球员。

## 容易踩的坑

- 字段名和路径先看真实样本再写批量代码：`ball_cand.json` 里是 `a` 不是 `area`，`pose.json` 里是 `kpts` 不是 `kp`，帧文件名是 1-based。先 print 一条记录能省掉一轮 KeyError。
- `camera.py` 的 `Camera` 要用 `Camera(calib_file=...)` 构造；`ray()` **只返回方向向量**，相机中心是 `cam.C`；落到平面上用 `img2plane()` / `img2ground()`。
- 用脚本量化代替反复看图：行号、像素差、残差这些数字才是证据，读图既有配额也容易误判。必须看图时先把关键区域裁剪放大，或走 `analyze_media` 的 `analyze_image`（JSON 走 stdin，不要用 `--images/--prompt`）。
- `.mov` 常带 180°/90° display matrix，ffmpeg 抽帧会自动转正而 `cv2.VideoCapture` 不一定，同一次分析不要混用两个来源（`meta.json` 里有 `rotation_deg`）。
- 烧入的比分条和台标不随场景移动，会主导 ORB 匹配，稳定化时必须用 `--mask-rows` 排除。
- 姿态提取一定要看跨人跳变数：实测同一段远端球员不加 ROI 是 34 次跳变、加了 ROI 是 0 次，而 34 次跳变足以让下游全盘失真。
- 非线性最小二乘的残差向量在"坏解"分支必须与正常分支**长度一致**，否则 `least_squares` 直接抛错或返回 RMS≈1000。
- Magnus 力方向写反会让"加上旋反而飞得更远"，物理上必须是 `a -= (0.5ρA·Cl·v²/m) · (v̂ × axis)`。
- 静止的球和绿色栅栏要用背景中位图生成的静态掩膜剔除，否则每帧几十个"球"。
- 俯视小图的近远端方向容易画反：世界坐标 `y > 0` 是近端球员半场，小图上近端应在下方。
- 渲染前后清理中间帧和无声临时视频，1000 帧 JPEG 很快就上百 MB。

## 参考文档

- [broadcast-view.md](references/broadcast-view.md)：视角体检判据、稳定化、自由主点标定、双球员与 BVP 求解的完整 B 模式流程
- [pipeline.md](references/pipeline.md)：每一步的调参、迭代方法和失败处置
- [calibration.md](references/calibration.md)：场地尺寸、检线/拟合/验证协议、尺度歧义
- [measurement-limits.md](references/measurement-limits.md)：置信度分级、旋转带、可测边界与措辞
- [pose-metrics.md](references/pose-metrics.md)：动作指标定义、解读阈值、建议映射
- [render-config.md](references/render-config.md)：渲染配置项与版面规范
- [data-schema.md](references/data-schema.md)：全部中间产物的 JSON 结构
- [report-template.md](references/report-template.md)：飞书报告骨架与图表规格
