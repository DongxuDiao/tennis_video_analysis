# 标定：从像素到米

标定是整条流水线的地基。球速、落点、击球点高度全都按标定给的深度尺度线性缩放——标定错 10%，球速就错 10%，出界判定可能整体反转。

## 世界坐标系

`camera.py` 定义（右手系，米）：

- **X** 横向，从相机看向右为正
- **Y** 沿场地纵深：近端底线 `y=+half`，网 `y=0`，远端底线 `y=-half`
- **Z** 向上，地面 `z=0`
- 相机通常在近端底线之后，即 `Cy > half`

相机用 **7 个物理参数** `[X0, Y0, Z0, yaw, pitch, roll, f_px]` 存储，而不是不透明的 3×4 矩阵。这样每个量都能做常识检查：机位高度 15 m、焦距 3000 px 这种解一眼就知道是错的，而同样的错误藏在投影矩阵里根本看不出来。

`court_calib.py`（下面第一到第四步）解的就是这 7 个参数，**主点固定在图像中心**。裁剪过、稳定化过、数码变焦过的画面这个假设不成立，要换成 9 参数的 `calib_broadcast.py`，见后面「9 参数自由主点标定」。

## 场地预设

`COURTS`（`camera.py`）内置，单位米：

| 运动 | half（底线到网） | dbl（外边线半宽） | sgl（单打半宽） | svc（发球线） | 网中/网柱高 | 球直径 |
| --- | --- | --- | --- | --- | --- | --- |
| tennis | 11.885 | 5.485 | 4.115 | 6.40 | 0.914 / 1.07 | 0.067 |
| badminton | 6.70 | 3.05 | 2.59 | 1.98 | 1.524 / 1.55 | 0.058 |
| table_tennis | 1.37 | 0.7625 | 0.7625 | — | 0.1525 | 0.040 |
| pickleball | 6.70 | 3.05 | 3.05 | 2.13 | 0.864 / 0.914 | 0.074 |
| volleyball | 9.00 | 4.50 | 4.50 | 3.00 | 2.43 | 0.21 |

非标准场地在 `ident.json` 里传 dict 覆盖任意字段即可（例如短式网球、青少年场地）。

## 第一步：detect

```bash
python3 $SKILL/scripts/court_calib.py detect --work work --image bg_median.png \
    --sat 70 --val 150 --tophat 16 --band 6 --skip-top 0.5 --max-lines 14
```

在背景中位图上做低饱和 + 高亮度掩膜 + tophat，再用 Hough 取长线，输出 `court_lines.json` 与可视化 `court_lines.png`（每条线带编号）。

**必须人眼看 `court_lines.png`。** 自动检测在室内场馆几乎一定会把非场地结构排进前列：屋顶横梁、窗框、栏杆、地面接缝、墙面踢脚线，它们比真实底线更长更直更亮。`--skip-top 0.5` 先把上半幅涂掉是最有效的一招（场地线通常只出现在画面下半部）。

## 第二步：ident.json —— 人来指定线的身份

可复制 [assets/ident_example.json](../assets/ident_example.json) 改坐标。

```json
{
  "sport": "tennis",
  "manual": {
    "near_baseline":  [[215, 431], [955, 456]],
    "near_service":   [[400, 349], [690, 346]],
    "left_sideline":  [[210, 430], [440, 302]],
    "right_sideline": [[955, 452], [665, 302]]
  },
  "player_height": [
    {"frame": 235, "ankle": [308, 513], "nose": [347, 281], "height": 1.60}
  ]
}
```

- `manual` 里每条线只要**两个粗略端点**（±5 px 足够），脚本会在 `--band` 像素带内用白线掩膜精修到子像素。
- 可用的线名：`near_baseline` `far_baseline` `near_service` `far_service` `left_sideline` `right_sideline` `net_line` `centre_service`。给得越多越稳，**至少要有近端底线 + 一条边线**。
- `player_height` 是打破尺度歧义的关键项：给一帧里球员的脚踝像素、头顶（或鼻尖）像素和真实身高，脚本用它约束纵深尺度。可从 `pose.json` 里读关键点坐标。`--player-weight` 调这一项的权重。
- 没有 `court_lines.json` 也能只用 `manual` 拟合（脚本会从 `bg_median.png` 取画幅尺寸）。

## 第三步：fit

```bash
python3 $SKILL/scripts/court_calib.py fit --work work --ident work/ident.json \
    --trials 300 --band 6 --cam-back-max 18 --rms-warn 3.0
```

多起点 Levenberg-Marquardt 拟合 7 参数，输出 `calib.json` 和 `calib_overlay.png`。

物理合理性检查（脚本会告警，也要自己看）：

- 机位高度 `Z0`：手持/三脚架 1.2–3.0 m；看台机位 3–8 m。>10 m 基本是错解。
- 机位纵深 `Y0 - half`：底线后 2–18 m。`--cam-back-max` 是硬上限。
- 焦距：手机横拍在 960 px 宽画幅下典型 350–900 px。
- RMS 残差：<2 px 好，2–3 px 可接受，>3 px 说明线认错了或场地预设不对。

## 第四步：verify —— 这一步不能省

```bash
python3 $SKILL/scripts/court_calib.py verify --work work --frame 0 --row-min 100 --tol 8
```

`verify` 把模型预测的关键线行号和图像实测白线行号逐条列出来，并输出放大对比图 `calib_zoom.png`。

**判定标准**（近端优先，因为分析主体在近端）：

| 线 | 容许偏差 |
| --- | --- |
| 近端底线、近端发球线 | ≤ 2 px |
| 网带 | ≤ 8 px |
| 远端发球线、远端底线 | ≤ 8 px |

超标就回到 `ident.json` 改线的身份或补线，不要接受"叠加图看着还行"。**叠加图会骗人**：透视投影下焦距与纵深有耦合，错解在近端能贴得很好。

## 尺度歧义：必须知道的坑

只用场地线拟合单机位视频时，**焦距和机位纵深近似共线**。实测例子（960×540 画幅、同一段网球视频、同一组线）：

| 解 | 焦距 | 机位 Y0 | 机位 Z0 | 近端底线 | 远端底线 |
| --- | --- | --- | --- | --- | --- |
| A | 385.3 px | 16.22 m | 1.90 m | 吻合 | 吻合 |
| B | 553.6 px | 17.94 m | 2.28 m | 吻合 | 偏差 ~20 px |

两个解在近端几乎无差别，但纵深尺度差约 10%，直接传导到球速上（80 km/h 会变成 88 km/h）。破解办法，按可靠性排序：

1. **远端底线或网带**入镜时把它写进 `manual`——纵深跨度越大，耦合越弱。
2. **`player_height`**：竖直方向的已知长度对焦距敏感、对纵深不敏感，正好互补。
3. **球的表观直径**：球在不同深度的像素直径变化本身就是尺度探针，`flight_fit.py` 已把它作为残差项，若某个标定下拟合出的球直径系统性偏离 `ball_d`，说明这个标定的尺度不对。
4. 都拿不到时：在报告里明确写"深度尺度存在约 ±10% 系统不确定性，球速数值应按相对比较使用"。

## 9 参数自由主点标定（`calib_broadcast.py`）

### 什么时候必须用它

只要画面**不是相机原生帧**，7 参数解就有一个错误的前提：光心（主点）已经不在画面中心了。三种典型情况：

- 16:9 转播帧的**竖屏裁剪**——光心被裁掉了；
- `stabilize.py` 的**稳定化画布**——参考帧的光心落在 padding 决定的位置，不是画布中心；
- 任何**数码变焦、加黑边、重构图**的片段。

### 为什么这是最危险的失败模式

把主点强行钉在中心**不会**产生大残差。求解器会把误差吸收进偏航/俯仰/焦距，照样收到 1–2 px，`fit` 打印的 rms 甚至更漂亮。代价是**深度尺度系统性漂移**——而深度尺度正是把像素变成 km/h 的那个量。典型症状：近端线贴得很紧，远端底线偏 10–20 px，球速整体偏 10–20%，而所有数字看起来都"正常"。上面那张尺度歧义表格里的两个解至少还能靠远端底线区分开；固定主点的错解连这个信号都没有。

所以这里解 9 个未知量：位姿 6 + 焦距 1 + **主点 2**。

### 必须补的平面外尺度约束

地面线全在一个平面上，只提供 8 个自由度，撑不起 9 个未知量。**至少要给一个平面外的独立尺度约束**，否则拟合会心安理得地在错误的尺度上返回 rms < 0.5 px：

| 约束 | 写在哪 | 权重参数 |
| --- | --- | --- |
| **网带**（首选）：已知高度、横跨全画幅、位于已知深度的曲线，从背景中位图亚像素提取 | `ident.json` 的 `net_tape`: `{"y_range": [v0, v1], "x_range": [u0, u1]}` | `--net-weight`（默认 1.5，0 关闭）、`--tape-min-val`、`--net-shape`（`quad` / `linear`） |
| **球员身高**：脚踝像素给落点、鼻尖像素给已知站立高度 | `ident.json` 的 `player_height`（可加 `ankle_height`，默认 0.09 m） | `--player-weight`（默认 1.0） |
| **主点软先验**：稳定化画布上就是 `source_centre + canvas_offset`，通常准到几十像素 | `ident.json` 的 `principal_prior`: `[cu, cv]` | `--pp-weight`（默认 0.05）、`--pp-range DU DV`（默认 260 220，硬边界） |

注意 `fit` **无条件读 `net_tape`**：这个块缺了脚本直接退出，即使把 `--net-weight` 设成 0 也一样。`player_height` 和 `--net-weight` 同时缺席时脚本会警告"几何看着合理、尺度是错的"。

### fit

```bash
python3 $SKILL/scripts/calib_broadcast.py fit --work $WORKS --ident $WORKS/ident.json \
    --trials 200 --band 6 --max-pts 120 --net-weight 1.5 --player-weight 1.0 \
    --pp-weight 0.05 --pp-range 260 220 --cam-h 3 20 --cam-back 1 35 --rms-warn 1.0
```

`--ident` 里必须有 `manual` 线提示（自动检线的编号撑不起 9 参数拟合），`--image` 默认 `bg_median.png`。输出写进 `$WORKS/calib.json`（`--out` 可改）**外加一份固定的 `calib_freepp.json` 副本**，方便和中心主点的解逐字段 diff；同时写 `net_tape.png`（网带提取可视化）和 `calib_overlay.png`。落盘的 `calib.json` 带 `pp` 字段，`camera.py` 认这个字段，忽略它就等于回到错的尺度。

脚本还会打印每条线的 mean/max 残差、网带残差、每条身高约束的鼻尖误差，以及主点相对画布中心与相对先验的偏移。搜索范围里 `--cam-h`（机位高度）和 `--cam-back`（底线后距离）默认放得比 A 模式宽，因为转播机位常在 9–10 m 高、38 m 远；俯仰初值范围也覆盖大俯角。

实测基准（那段澳网竖屏切片，画布 1272×1127），可以当量级参照，不要当阈值照抄：

| 量 | 值 |
| --- | --- |
| RMS | **0.356 px** |
| 主点 `pp` | **(562, 745)**，相对画布中心偏 −74 / +181 px |
| 相机位置 | x≈0.05、**y≈38.28**、**z≈9.26** m |
| 焦距 | **f≈3201 px**（水平视场约 22.5°） |
| `verify` | 12 条关键线**全部**在阈值内（近端线 2 px、网带与远端线 8 px） |

### verify

```bash
python3 $SKILL/scripts/calib_broadcast.py verify --work $WORKS --tol-near 2 --tol-far 8 \
    --row-min 100 --max-search 20 --zoom
```

和 `court_calib.py verify` 同一个思路，但容差分两档（`--tol-near` 用于 `near_baseline` / `near_service`，`--tol-far` 用于 `far_service` / `net_tape` / `far_baseline`），并且结果落盘到 `calib_verify.json`。逐行 `verdict` 有四种：`ok`、`FAIL`、`no-line`（该列带 `--max-search` 内根本没有白线，线被挡住或没画——既不算过也不算不过，换 `--columns U0 U1` 再验）、`info`（`net_base` 这类地面上没有涂线的参考项）。`n_fail == 0` 才打印 PASS，`--zoom` 额外出 `calib_zoom.png`。

**FAIL 时不要动容差**，回去改线的身份、`net_tape` 窗口或身高约束。`verify` 在近端线上死活到不了 2 px，还有一种可能是这段素材的相机在**平移**（推车/走动/无人机），那属于本流水线不支持的情形。

### 解落在 `--pp-range` 边界上

脚本会警告"the fit is constrained, not converged"。正确处置顺序是：**先修 `principal_prior`**（转播帧的裁剪中心不等于光心，先验本身很可能写错了），再考虑放宽 `--pp-range`。直接放宽范围等于允许主点去吸收标定里其他环节的误差，rms 会更好看，深度尺度会更错。放宽之后一定要对比 rms 和相机距离有没有跟着跑——跑了就说明这个自由度在替别人背锅。

## 相机 API

`camera.Camera(work=...)` 载入 `calib.json` 后可用：

| 方法 | 用途 |
| --- | --- |
| `project(Pw)` | 世界点 → 像素 |
| `ray(uv)` | 像素 → 视线方向（单位向量） |
| `img2plane(uv, z)` | 像素 → 指定高度平面上的世界点 |
| `img2ground(uv)` | 像素 → 地面点（`z=0`） |
| `ground2img(xy)` | 地面点 → 像素 |
| `apparent_diam_px(uv, height)` | 该位置球的理论像素直径（深度线索） |
| `net_height(x)` | 横向位置 x 处的网高（中间低、网柱高） |
| `net_top_image_y(u)` | 网带上沿在图像该列的行号 |
| `court_polylines()` | 渲染用的场地线框（世界坐标） |
| `describe()` | 打印人可读的相机描述，做常识检查 |

`ray()` 返回单位方向向量，**不是**世界点——要用 `img2plane`/`img2ground` 才能得到位置。这是本流水线出过的一个 bug。

主点：`Camera` 读 `calib.json` 里的 `pp`，没有这个字段才回落到 `(W/2, H/2)`，取到的值在 `cam.CU` / `cam.CV`。所以 `calib_broadcast.py` 的产物可以被任何脚本直接使用，不需要额外传参；反过来，自己写的分析代码如果手算 `(u - W/2) / f`，在 B 模式下就错了。要载入非默认名的标定用 `Camera(work, calib_file=...)`（`describe()` 会打印相机位置、焦距、hfov 和 rms，先看一眼再往下算）。
