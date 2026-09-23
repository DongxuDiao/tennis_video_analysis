# 渲染配置与版面规范

`render_annotated.py` 把前面所有产物画到帧上并混回原声。所有文案、阶段划分、面板字段都在 `--config` 的 JSON 里，换运动或换语言不用改代码。

## 画面里有什么

| 元素 | 数据来源 | 说明 |
| --- | --- | --- |
| 场地 + 球网线框 | `calib.json` | 也是标定正确性的持续自检——线框跑偏立刻能看到 |
| 人体骨架 | `pose.json` | 左臂黄、右臂蓝、躯干绿、下肢粉 |
| 挥拍阶段标签 | `technique.json` 的 `f_coil` + `tc` | 准备 / 引拍转体 / 击球瞬间 / 随挥 / 移动还原 |
| 球检测点 + 拖尾 | `shots_input.json` | 青色=来球，黄色=出球，拖尾渐隐 |
| 重建球路 + 弹地标记 | `flights.json`（`--flights`） | 逐拍按自己的 `rpm` 积分重放，球随深度改变大小；弹地处画扩散圈，界内绿、界外红 |
| 模型弹道 | `shots.json` 的 `out_path` | 实线=已发生，虚线=预测 |
| 击球闪烁 | `tc` | 触球帧前后闪一下，便于逐帧核对 |
| 右侧信息卡 | `shots.json` + `technique.json` | 当前拍的速度、区间、仰角、过网余量等 |
| 俯视小地图 | `world_path` + `spin_band` | 世界坐标俯视，×标不同上旋的落点 |
| 累计统计 | 全部拍 | 正手/反手/出界/下网/界内计数；多球员时分球员各一行（`player_tally`） |
| 时间轴 | 全部拍 | 底部标出每拍位置与当前进度，多球员时刻度按球员配色 |

## 命令行参数

```bash
python3 $SKILL/scripts/render_annotated.py --work work \
    --shots shots.json --input shots_input.json --technique technique.json \
    --config render.json --scale 1.5 --panel right --panel-size 300 \
    --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
    --quality 7 --frames 60 80 --out check.mp4
```

- `--frames a b`：只渲 `[a, b)`，**迭代版面时必用**。
- `--scale`：画面放大倍数（960 宽 × 1.5 = 1440），决定字号相对大小。
- `--panel`：面板位置，`auto`（默认，竖屏走底部、横屏走右侧）/ `right` / `bottom`。传一个纯数字仍然兼容老写法，等价于把它当作 `--panel-size` 再走 `auto`。
- `--panel-size`：右侧面板的宽度或底部面板的高度（px），默认 300。字段多或中文长时右侧加到 340、底部加到 304 以上。
- `--quality`：imageio 的 1–10，7 已足够，10 会让文件翻倍。
- `--no-audio`：跳过混流（调试时更快）。
- `--fps 0`：从 `meta.json` 读。

B 模式追加的参数：

- `--stabilize stabilize.json` + `--src-frames $WORK/frames`：几何画在稳定化画布上，再逆 warp 回原始帧合成。
- `--overlay-alpha`：warp 回来的叠加层不透明度，默认 0.75。
- `--flights flights.json`：球路改为 `flight_bvp.py` 的物理重建（逐拍用自己的 `rpm`），并在面板打出 `ball_source_note`。给了 `--flights` 时不再读 `--input` 的检测点。
- `--pose a.json b.json`：一名球员一个 pose 文件。
- `--pose-who N F`：每个 pose 文件对应 `contacts.json` 里的 `who`，统计与信息卡靠它归属。
- `--pose-label "近端 · 纳达尔=255,190,90" "远端 · 费德勒=170,255,170"`：显示名，`=` 后面是 **BGR** 三元组。不指定颜色时用内置 `PALETTE` 依次分配。

多 pose（`len(--pose) > 1`）时骨架按球员整体着色、每人一个名字标签、俯视图上一个点、统计里一行；单 pose 时保留历史的按肢体配色（左臂黄、右臂蓝、躯干绿、下肢粉）。

## 稳定化画布渲染

跟拍片段的标定是在稳定化画布上做的，**几何只在画布坐标里成立**。所以渲染流程是：把球场线、球网、球路、骨架、击球闪烁全部画进一张画布尺寸的叠加层，用该帧 homography 的逆把叠加层 warp 回原始帧，再做 alpha 合成——视频底图始终是没被 warp 过的原始帧，画质不损失。

```bash
python3 $SKILL/scripts/render_annotated.py --work $WORKS \
    --stabilize stabilize.json --src-frames $WORK/frames \
    --shots contacts.json --flights flights.json --technique technique.json \
    --pose pose_near.json pose_far.json --pose-who N F \
    --panel auto --frames 330 340 --out check.mp4
```

直接把几何画在原始帧上会满屏错位（球场线离球场几米远），而且错得很整齐，看一帧未必发现，一定要抽帧确认线框贴住球场。

三个必查项，都是实测踩过的坑：

- **脏线**：投影点落到画布外或深度为负时，`cv2.line` 会画出一条贯穿整个画面的直线。所有连线都要走边界与视锥裁剪（脚本里是 `safe_poly()`，按**线段**逐段丢弃而不是整条丢，这样出画的球场线还能保留可见部分）。
- **每拍用自己的 rpm**：`--flights` 下每条飞行按它自己的 `rpm` 积分重放。全片共用一个 rpm 时，球路和落点标记会明显对不上——对一拍看着对，其余全偏。
- **俯视小图方向**：世界坐标 `y > 0` 是近端球员半场，小图上近端必须在**下方**。这个方向实测画反过一次，而且反了之后画面仍然"看着合理"，只有对着真实站位才看得出来。多球员时小图上下会打出 `half_labels` 的"远端半场 / 近端半场"，正好用来自检。

## 竖屏底部面板

竖屏视频右侧没有放面板的空间。`--panel auto` 在 `H > W` 时自动切底部面板：实测 764×960 的画面 + 304 px 面板 = 输出 768×1264（宽高都向上对齐到 16 的倍数）。底部面板的版式是左侧俯视小图、中间当前拍信息卡（球速与区间用大号字）、右侧累计统计与分球员统计，最下面一行时间轴加 `timeline_note`。

BVP 模式下面板**必须**有一行明确写出球路是物理重建而非逐帧检测，这行由配置键 `ball_source_note` 提供（默认"球路：接触点物理重建（非逐帧检测）"），给了 `--flights` 就会自动打出来。删掉这行读者会把重建轨迹当成真实检测。

## 配置 JSON 的键

`--config` 只需写要覆盖的键，其余用默认值。可直接复制 [assets/render_tennis_cn.json](../assets/render_tennis_cn.json)（中文网球版，与下例一致）。

```json
{
  "title": "动作 / 球路逐帧分析",
  "phases": [
    ["准备 / 迎球",  "coil",    -12, "coil",    0,  [150, 150, 150]],
    ["引拍 / 转体",  "coil",      0, "contact", -3, [0, 200, 255]],
    ["击球瞬间",     "contact",  -3, "contact",  2, [60, 60, 255]],
    ["随挥 / 收拍",  "contact",   2, "contact", 16, [90, 230, 120]]
  ],
  "idle_phase": "移动 / 还原",
  "panel_fields": [
    ["出球速度",     "kmh",       "%s km/h"],
    ["速度区间",     "_range",    "%s"],
    ["出球仰角",     "elev",      "%.1f°"],
    ["过网余量",     "net_clear", "%+.2f m"],
    ["击球点高度",   "contact_h", "%.2f m"],
    ["来球速度",     "feed_kmh",  "%.0f km/h"]
  ],
  "tally": [["正手", "side", "FH"], ["反手", "side", "BH"],
            ["出界", "result", "OUT"], ["下网", "result", "NET"], ["界内", "result", "IN"]],
  "legend": ["青色=来球  黄色=出球  虚线=模型预测",
             "×=落点推算（不同上旋）  红=出界 绿=界内"],
  "side_names": {"FH": "正手", "BH": "反手"},
  "unmeasurable": "无法测量"
}
```

- `phases`：`[名称, 起点参考, 起点偏移, 终点参考, 终点偏移, BGR]`，参考点只能是 `coil` 或 `contact`，偏移单位是帧。**颜色是 BGR 不是 RGB**（OpenCV 约定）。换运动时按该项目的动作时长改偏移（乒乓球全部收窄到 ±6 帧内）。
- `panel_fields`：`[标签, shots.json 字段名, 格式串]`。`_range` 是特殊键，渲染成 `kmh_lo–kmh_hi`。字段缺失时该行自动跳过；`kmh` 缺失时显示 `unmeasurable` 的文案。
- `tally`：`[标签, 字段, 匹配值]`。`result` 的匹配用前缀，所以 `OUT` 同时覆盖 `OUT_LONG` 和 `OUT_WIDE`。
- `legend`：图例行，每行一句，别超过面板宽度。

B 模式 / 多球员新增的键（[assets/render_tennis_cn.json](../assets/render_tennis_cn.json) 已经带上，直接复用即可）：

| 键 | 用途 |
| --- | --- |
| `ball_source_note` | 球路来源说明，`--flights` 时打在面板上。**BVP 模式不可省** |
| `player_names` | `who` → 显示名的映射（如 `{"N": "近端球员", "F": "远端球员"}`），`--pose-label` 优先级更高 |
| `player_tally` | 分球员统计的项目，格式同 `tally`（通常只留正手/反手） |
| `half_labels` | 俯视小图上下的半场标签 `{"near": ..., "far": ...}`，兼作方向自检 |
| `minimap_title` / `stat_title` / `landing_title` | 底部面板的三个小标题 |
| `landing_fmt` | 落点坐标格式串，两个 `%+.1f`（x、y，米） |
| `verdict_fmt` | 判定行格式串，一个 `%s` |
| `speed_stat` | 两个格式串（平均球速、最快球速），各一个 `%.0f` |
| `shot_fmt` | 当前拍标题，三个 `%s`：拍号、球员前缀、正反手 |
| `waiting` | 两拍之间的占位文案 |
| `timeline_note` | 时间轴下方的说明 |
| `result_names` | `result` → 中文（`IN`/`OUT`/`NET`/`UNK`），`flights.json` 自带 `result_cn` 时优先用它 |
| `side_names` | `FH`/`BH`/`SERVE` → 中文 |

`--flights` 模式下 `panel_fields` 里的字段名**不用改**：渲染时会把 `flights.json` 的字段别名过去（`kmh`←`v0_kmh`、`kmh_lo`/`kmh_hi`←`v0_lo`/`v0_hi`、`elev`←`angle_deg`、`apex`←`apex_m`、`f_land`←`bounce_frame`、`land`←`bounce`，`net_clear` 同名），所以一份给 A 模式写的 config 在 B 模式照样能用。`conf == 'low'` 的拍，速度区间那一行会自动补"（低置信）"。

## 版面必须实机检查

先渲十几帧，抽出来逐张看：

```bash
python3 $SKILL/scripts/render_annotated.py --work work --frames 60 76 --out check.mp4
python3 $SKILL/scripts/video_prep.py grab --video check.mp4 --frames 0 5 10 --out chk
```

历史上真实踩过的版面问题，全部只有在真实帧上才暴露：

- 俯视小地图压住图例文字（面板高度不够）
- 信息卡与统计块行距重叠
- 长中文字段名（"击球点横向位置"）超出面板右边界
- 弹道虚线跑到画布外导致边缘出现杂线
- 时间轴刻度和当前进度标记在拍数多时挤成一团

稳定化 / 竖屏 / 双球员追加的检查项：

- 球场线框是否贴住原始帧里的球场（叠加层 warp 回来的正确性，一帧就能看出来）
- 球路与弹地标记是否落在同一位置（对不上先查是不是逐拍 rpm）
- 俯视小图里近端球员在下方、远端在上方
- 底部面板三列（小图 / 信息卡 / 统计）有没有互相挤压，中文是否被截断
- 球员名字标签是否跟骨架重叠或跑出画面（标签只在框中心落在画面内时才画）

## 技术约束

- **中文必须用 PIL + NotoSansCJK**。`cv2.putText` 只支持 ASCII，中文会变成 `????`。字体路径默认 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`，不存在时装 `fonts-noto-cjk` 或用 `--font` 指到别处。
- **输出尺寸自动向上对齐到 16 的倍数**（例：1740×810 → 1744×816）。不对齐时 imageio/ffmpeg 会静默缩放，导致输出比预期模糊；同时传 `macro_block_size=1` 阻止隐式 resize。
- **系统 PATH 里通常没有 ffmpeg**，一律用 `imageio_ffmpeg.get_ffmpeg_exe()`。
- 混流必须显式映射流，否则可能丢音频：

```bash
$FFMPEG -y -i out_noaudio.mp4 -i audio.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest out.mp4
```

  混流后用 `ffprobe`（或 `ffmpeg -i`）确认输出里同时有 `Video: h264` 和 `Audio: aac` 两条流。

- 渲染成本：960×540、1050 帧、`--scale 1.5` 约 3–5 分钟，输出约 35–40 MB。稳定化模式每帧多一次 `warpPerspective` 加 alpha 合成，会更慢一些。
- `--frames a b` 渲片段时混流会按 `-ss a/fps` 对齐音频，所以检查片段的声音和画面仍然同步。

## 换运动时要改的

1. `calib.json` 里的 `sport` 换预设（线框自动适配）。
2. `phases` 的帧偏移按动作时长缩放（乒乓球 ÷2，羽毛球杀球 ×0.7）。
3. `panel_fields` 增删项目特有字段。
4. `tally` 的类别（排球可换成"扣球/拦网/失误"）。
5. `--scale` / `--panel-size` 按原片分辨率调；竖屏片段让 `--panel auto` 自己选底部面板。
