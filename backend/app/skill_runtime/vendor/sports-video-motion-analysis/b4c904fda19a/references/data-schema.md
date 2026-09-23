# 中间产物 JSON 结构

全部文件都在 `--work` 目录下。**帧号一律 0-based**；磁盘上的 `frames/f_00001.jpg` 是第 0 帧（`f_%05d.jpg % (f+1)`）。

写批量代码前先 `print` 一条真实记录再对字段名下手——本流水线在 `a`/`area`、`kpts`/`kp` 上都栽过。

每节标题后的标记表示适用范围：**[通用]** 两种模式都产出；**[A]** 只有固定机位单人流程有；**[B]** 只有转播裁剪 / 跟拍 / 双球员流程有。B 模式的工作目录是稳定化画布目录（下文记作 `$WORKS`），里面的所有像素坐标都是**画布坐标**，不是原始帧坐标。

## meta.json — `video_prep.py prep` [通用]

```json
{"video": "/abs/path/clip.mov", "width": 960, "height": 540,
 "fps": 30.0, "n_frames": 1050, "duration": 35.0,
 "has_audio": true, "rotation_deg": -180.0,
 "frames_dir": "frames", "audio": "audio.wav", "background": "bg_median.png"}
```

`rotation_deg` 来自容器的 display matrix。ffmpeg 抽帧时已应用，`cv2.VideoCapture` 不一定——所以全程只读 `frames/` 里的 JPEG。

`stabilize.py` 在 `$WORKS/meta.json` 里复制这份 meta 并改写/追加下面几项（只在 `--emit-frames true` 时写）：

| 字段 | 说明 |
| --- | --- |
| `width` / `height` | **改成画布尺寸**，不再是原片尺寸 |
| `stabilized` | `true`，下游据此知道自己在画布上 |
| `stabilize_ref` | 参考帧号（0-based） |
| `canvas_offset` | `[dx, dy]`，原始帧坐标系原点在画布里的偏移 |
| `source_size` | `[W, H]` 原片尺寸，渲染回原始帧时要用 |

`audio` 字段被原样继承，但 `stabilize.py` **不会**复制 `audio.wav`，要自己 `cp` 过去。

## stabilize.json — `stabilize.py` [B]

写在 `--out` 目录下（`--emit-frames false` 时也写，这是"先看质量再决定落盘"的依据）。

```json
{"ref": 280, "ref_mode": "given", "offset": [254, 84], "canvas": [1272, 1127],
 "source_size": [764, 960], "mask_rows": [0, 160], "emitted_frames": true,
 "n_frames": 565, "n_weak": 3, "weak_frac": 0.0053,
 "inliers_median": 295.0, "residual_median_px": 0.9, "residual_p95_px": 1.06,
 "quality": [{"frame": 0, "inliers": 214, "residual_px": 0.83, "kind": "direct"}],
 "weak": [{"frame": 417, "inliers": 0, "residual_px": null, "kind": "copied"}],
 "H": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]}
```

| 字段 | 说明 |
| --- | --- |
| `ref` / `ref_mode` | 参考帧号、`auto` 还是 `given`。换参考帧等于换一套画布与主点，下游全部要重跑 |
| `offset` | 画布偏移 `[dx, dy]`。**这个键叫 `offset`**，`meta.json` 里同一个量叫 `canvas_offset`，读的时候两个都要认（`render_annotated.py` 就是先 `offset` 再回落 `canvas_offset`） |
| `canvas` / `source_size` | 画布尺寸 / 原片尺寸 |
| `H` | 长度 = `n_frames` 的列表，每项是 3×3 嵌套数组，表示"该帧 → 参考帧"的 homography（**不含** `offset` 平移，落盘 warp 时脚本额外左乘了平移矩阵） |
| `quality[i]` | 每帧 `frame`、`inliers`、`residual_px`（RANSAC 内点重投影中位误差，可能为 `null`）、`kind` ∈ `ref` / `direct` / `chained` / `copied` / `missing` |
| `weak` | `kind != "direct"` 的帧子集，就是不可信的帧 |
| `n_weak` / `weak_frac` | 弱帧数与占比。`weak_frac > 0.05`（`--weak-warn-frac`）就不要在这个画布上做 3D 结论 |
| `inliers_median` / `residual_median_px` / `residual_p95_px` | 整体质量。实测健康值：内点中位 295、残差中位 0.90 px、p95 1.06 px；p95 > 3 px 说明帧间映射不是纯 homography |
| `mask_rows` / `emitted_frames` | 当次用的 overlay 掩膜行数、是否落盘了 warp 帧 |

## pose.json / `pose_<label>.json` — `pose_extract.py` [通用]

键是帧号字符串，每帧只保留一个主体。COCO-17 顺序。多球员时每人一个文件：`--label near` → `pose_near.json`，默认 `--label main` → `pose.json`。

```json
{"137": {"box": [x1, y1, x2, y2], "conf": 0.93,
         "kpts": [[x, y, conf], ...17 项...]},
 "qc": {"coverage": 0.982, "n_jumps": 0, "jump_frames": [],
        "kpt_confident_frac": {"shoulders": 0.99, "wrists": 0.87, "hips": 0.98, "ankles": 0.71},
        "box_h_median_px": 118.4, "missing_runs": [[401, 403]], "n_missing": 3},
 "meta": {"label": "far", "subject": "far", "roi": [60, 200, 1060, 400],
          "min_area": 2500, "conf": 0.35, "model": "yolo11m-pose.pt",
          "imgsz": 960, "frames": [0, 565], "n_frames_dir": 565}}
```

字段名是 **`kpts`**。关键点索引：0 鼻，5/6 左/右肩，7/8 左/右肘，9/10 左/右腕，11/12 左/右髋，13/14 膝，15/16 踝。未检出的帧直接缺键，取值前先判 `None`。

`qc` 和 `meta` 是唯一两个**非数字**的顶层键，所有读取方都必须跳过非数字键（`pose_metrics.py`、`render_annotated.py`、`swing_peaks.py` 已经这么做了）。`qc` 里最该看的是 `n_jumps`（跨人跳变次数，判据是 bbox 中心位移 > `jump_disp` 个框宽或面积变化 > `jump_area` 倍）和 `coverage`。`meta` 记下了这次是怎么选主体的，回头复现 ROI 靠它。

## impacts.json — `audio_impacts.py` [通用]

```json
{"fps": 30.0, "sr": 48000, "band": [2000, 8000], "n": 10,
 "peaks": [{"t": 2.16, "frame": 64.8, "score": 20.21, "rms_db": -75.1}]}
```

`frame` 是**小数**帧号，直接拿去当 `shots_input.json` 的 `tc`，不要取整。`score` 以正谱通量标准差为单位。

## `swing_<label>.json` — `swing_peaks.py` [B]

每个 `--pose` 文件一份，label 默认从文件名推（`pose_near.json` → `near`），文件名是 `swing_<label>.json`。

```json
{"pose": "works/pose_near.json", "label": "near", "hand": "left", "fps": 30.0,
 "n": 565, "thr": 0.45, "minsep": 8, "n_used": 540,
 "peaks": [{"frame": 34, "peak": 0.83, "peak_px": 21.4, "peak_ms": 14.6,
            "backswing_start": 26, "lat": -0.72, "hgt": 1.14}],
 "spd": [null, 0.12, 0.31], "spd_px": [null, 3.1, 8.0],
 "lat": [null, -0.2], "hgt": [null, 0.4], "spd_ms": [null, 2.1]}
```

| 字段 | 说明 |
| --- | --- |
| `hand` | 实际使用的持拍手（`--hand auto` 时是脚本推断的结果，margin 小的时候不可信） |
| `peaks[].peak` | 腕速峰值，单位是**肩宽/帧**（归一化后跨深度可比），`peak_px` 是原始像素/帧 |
| `peaks[].peak_ms` | 近似 m/s，只有给了 `--calib` 才有，否则为 `null`；只含平行于像平面的分量 |
| `peaks[].backswing_start` | 峰前速度首次低于峰值 35% 的帧 = 引拍结束 |
| `peaks[].lat` / `hgt` | 腕相对肩线中点的横向偏移（肩宽为单位，+ = 画面右）／相对髋线的高度（躯干长为单位） |
| `spd` / `spd_px` / `lat` / `hgt` / `spd_ms` | 逐帧全序列，长度 = `n`，无效帧是 `null`。`spd_ms` 只在给了 `--calib` 时才有 |

## swing_match.json — `swing_peaks.py --match-impacts` [B]

音频冲击峰与各球员腕峰求交的结果，默认文件名 `swing_match.json`（`--match-out` 可改）。

```json
{"summary": {"window": 6.0, "n_audio": 34, "n_matched": 24, "n_ambiguous": 6,
             "per_player": {"near": 13, "far": 11},
             "truth": {"n": 14, "recalled": 12, "recall": 0.857, "missed": [230.0, 402.0], "tol": 4.0}},
 "matches": [{"audio_frame": 55.2, "score": 18.4, "best_player": "far",
              "delta_frames": -0.2, "swing_frame": 55, "ambiguous": false,
              "candidates": [{"player": "far", "swing_frame": 55, "delta_frames": -0.2, "peak": 0.71}]}]}
```

| 字段 | 说明 |
| --- | --- |
| `best_player` | 归属的球员 label；`null` 表示窗内没人挥拍（大概率是弹地/脚步峰，正常） |
| `delta_frames` / `swing_frame` | 腕峰相对音频峰的帧差、腕峰所在帧 |
| `ambiguous` | `true` = 窗内两名球员都有腕峰，**必须人工看那几帧**，不要采信 `best_player` |
| `candidates` | 窗内全部候选，按帧差绝对值排序 |
| `summary.truth` | 只在给了 `--truth` 时出现，是对已知接触帧的召回自检 |

## calib.json — `court_calib.py fit` [A]

```json
{"width": 960, "height": 540, "sport": "tennis", "court": {"half": 11.885, "...": "..."},
 "cam_params": [0.078, 16.223, 1.896, -0.1427, -0.0088, -0.0266, 385.31],
 "Hground": [9 个 float（行优先展开的 3×3）], "rms": 1.09,
 "lines_used": {"near_baseline": 400, "left_sideline": 400},
 "n_player_constraints": 1}
```

`cam_params` = `[X0, Y0, Z0, yaw, pitch, roll, f_px]`，长度单位米、角度弧度、焦距像素。**没有 `pp` 字段时读取方按图像中心当主点**（`camera.py` 的默认行为）。`Hground` 是地面平面到图像的单应，磁盘上是**展平的 9 个数**，`camera.py` 会 `reshape(3, 3)`。`sport` 也可以是覆盖了预设字段的 dict，落盘时展开后的场地参数存在 `court` 里。用 `camera.Camera(work).describe()` 打印人可读版本做常识检查。

## calib.json / calib_freepp.json — `calib_broadcast.py fit` [B]

同名 `calib.json`（`--out` 可改）加一份固定的 `calib_freepp.json` 副本，便于和中心主点的解 diff。比 A 模式多出主点等字段：

```json
{"width": 1272, "height": 1127, "sport": "tennis", "court": {"...": "..."},
 "cam_params": [0.05, 38.28, 9.26, 0.0031, 0.6612, -0.0042, 3201.4],
 "pp": [562.3, 745.1], "rms": 0.356,
 "per_line": {"near_baseline": [0.21, 0.55], "net_tape": [0.42, 1.10]},
 "player_residual_px": [[-1.2, 0.8]],
 "net_shape": "quad", "principal_prior": [636.0, 564.0],
 "lines_used": {"near_baseline": 120}, "Hground": [9 个 float],
 "solver": "calib_broadcast.py (9 params, free principal point)",
 "note": "principal point is NOT the image/canvas centre - readers must honour \"pp\""}
```

| 字段 | 说明 |
| --- | --- |
| `pp` | 主点 `[cu, cv]`，**B 模式的关键字段**。`camera.py` 有 `pp` 就用它，没有才回落到画布中心。忽略它 = 深度尺度系统性偏 |
| `rms` | 全残差 RMS（px），含线、网带、身高、主点先验各项 |
| `per_line` | 每条线的 `[mean, max]` 点到线距离（px），`net_tape` 也在里面 |
| `player_residual_px` | 每条 `player_height` 约束的鼻尖预测误差 `[du, dv]` |
| `principal_prior` | 拟合时用的主点先验，和 `pp` 差太多就说明先验或线的身份有问题 |
| `net_shape` | 网带高度模型，`quad`（默认）或 `linear` |
| `solver` / `note` | 提醒读取方这份标定的主点不在画布中心 |

## calib_verify.json — `calib_broadcast.py verify` [B]

```json
{"calib": "works/calib.json", "n_fail": 0, "n_nocheck": 2,
 "tol_near": 2.0, "tol_far": 8.0,
 "checks": [{"line": "near_baseline", "cols": [572, 699], "predicted": 801.4,
             "white_row": 802, "delta": 0.6, "tol": 2.0, "verdict": "ok"}]}
```

`verdict` ∈ `ok` / `FAIL` / `no-line` / `info`。`no-line`（该列带附近 `--max-search` 内没有白线，线被挡住或没画）**既不算过也不算不过**，换 `--columns` 再验；`info` 是 `net_base` 这类没有涂线的参考项，只作参考。`n_fail == 0` 才是 PASS。A 模式的 `court_calib.py verify` 只打印不落盘。

## court_lines.json — `court_calib.py detect` [A]

```json
{"width": 960, "height": 540,
 "lines": [{"id": 0, "p1": [x, y], "p2": [x, y], "len": 741.2, "angle": 1.9, "score": 88.4}]}
```

仅是候选，**身份要人来定**（写进 `ident.json`）。配套可视化 `court_lines.png` 带编号。

## ident.json — 人工编写 [通用]

```json
{"sport": "tennis",
 "manual": {"near_baseline":  [[215, 431], [955, 456]],
            "near_service":   [[400, 349], [690, 346]],
            "left_sideline":  [[210, 430], [440, 302]],
            "right_sideline": [[955, 452], [665, 302]]},
 "player_height": [{"frame": 235, "ankle": [308, 513], "nose": [347, 281],
                    "height": 1.60, "ankle_height": 0.09}],
 "net_tape": {"y_range": [382, 416], "x_range": [124, 872]},
 "principal_prior": [822, 525]}
```

线名可用：`near_baseline` `far_baseline` `near_service` `far_service` `left_sideline` `right_sideline` `net_line` `centre_service`。两个粗略端点即可（±5 px），脚本会在白线掩膜内精修。`player_height` 用于打破焦距/纵深尺度歧义，见 `calibration.md`；`ankle_height` 可选（默认 0.09 m，脚踝关键点离地高度）。`sport` 也可以写成 `court` 并给 dict 覆盖预设字段。

后两个块是 **[B] 专属**，`court_calib.py` 不读：

| 字段 | 说明 |
| --- | --- |
| `net_tape` | 网带亚像素提取的搜索窗 `{"y_range": [v0, v1], "x_range": [u0, u1]}`。`calib_broadcast.py fit` **无条件读它**，缺了直接退出（即使把 `--net-weight` 设成 0 也一样） |
| `principal_prior` | 主点软先验 `[cu, cv]`。稳定化画布上就写 `source_centre + canvas_offset`；缺省时脚本回落到画布中心，等于把最危险的假设又请回来了 |

## ball_cand.json — `ball_detect.py` [A]

```json
{"66": [{"x": 460.2, "y": 276.4, "a": 21, "w": 6, "h": 5,
         "gb": 42, "sat": 118, "val": 205, "n": 2}]}
```

面积字段是 **`a`**（不是 `area`）。`gb` 绿蓝差，`sat`/`val` HSV 分量，`n` 该帧候选总数。`n == 1` 的帧最可信。这些是**候选**，不是球。

## shots_input.json — 人工编写（A 模式核心输入） [A]

```json
{"fps": 30,
 "shots": [{"id": 1, "side": "FH", "tc": 64.8,
            "incoming": [[44, 442, 369], [45, 437, 361]],
            "outgoing": [[66, 460, 276], [67, 468, 259], [70, 474, 235]],
            "contact": [-0.33, 14.54, 1.61],
            "diam": {"66": 6.2}}]}
```

| 字段 | 说明 |
| --- | --- |
| `id` | 拍序号，1 起 |
| `side` | `FH` / `BH`（也可自定义如 `SERVE`） |
| `tc` | 触球时刻，小数帧号，取自 `impacts.json` |
| `incoming` | 来球观测 `[frame, u, v]`，≥4 点才能反演击球点 |
| `outgoing` | 出球观测 `[frame, u, v]`，≥4 点才有 `high` 置信 |
| `contact` | 可选，直接给三维击球点 `[x, y, z]`（米）；给了就不做来球反演 |
| `diam` | 可选，`{帧号: 像素直径}`，人工量的球径，强化深度约束 |

## shots.json — `flight_fit.py` [A]

```json
{"id": 1, "side": "FH", "tc": 64.8, "t_s": 2.16,
 "contact_src": "incoming_fit", "in_rms": 0.72, "n_in": 11, "feed_kmh": 18.3,
 "contact": [-0.33, 14.54, 1.61], "contact_h": 1.61, "behind_baseline": 2.65,
 "n_out": 6, "out_rms": 0.9, "V0": [-4.06, -31.64, 6.8],
 "kmh": 117.4, "elev": 12.0, "azim": -7.3,
 "kmh_lo": 111.3, "kmh_hi": 123.6, "conf": "high",
 "spin_band": [{"rpm": 0, "result": "OUT", "net_clear": 2.47,
                "land": [-4.51, -18.0], "depth": 6.12, "wide": 0.39}],
 "result": "IN", "result_firm": false, "net_clear": 1.7, "land": [-3.62, -11.07],
 "result_cn": "界内",
 "result_note": "按 1000 rpm 上旋推算为界内；旋转带内结果会翻转（界内/出界），落点属外推",
 "t_land": 1.07, "out_path": [[u, v, t]], "world_path": [[x, y, z]]}
```

| 字段 | 说明 |
| --- | --- |
| `contact_src` | `given`（输入给定）/ `incoming_fit`（来球反演） |
| `in_rms` / `out_rms` | 来球/出球段重投影残差（px），>2.5 需警惕 |
| `n_in` / `n_out` | 观测点数，决定置信度 |
| `feed_kmh` | 来球速度（喂球或对手球） |
| `behind_baseline` | 击球点在近端底线之后多少米（负=站在线内） |
| `V0` | 出球初速度向量（m/s，世界坐标） |
| `kmh` / `kmh_lo` / `kmh_hi` | 出球速率与 bootstrap 16/84 分位区间 |
| `elev` / `azim` | 出球仰角 / 方位角（度） |
| `conf` | `high` / `medium` / `low` / `none` |
| `spin_band` | 每个上旋档位的落点与结论 |
| `result` | `IN` `NET` `OUT_LONG` `OUT_WIDE` `LONG?` `UNK` |
| `result_firm` | 旋转带内结论是否一致；`false` 则必须写"外推" |
| `net_clear` | 过网余量（米，负=下网） |
| `land` | 落点 `[x, y]`（米） |
| `depth` / `wide`（band 内） | 超出底线/边线的米数，负=在界内 |
| `out_path` / `world_path` | 渲染用轨迹（图像 `[u,v,t]` / 世界 `[x,y,z]`） |

`kmh` 存在但 `conf == "low"` 时**不能当结论**；`conf == "none"` 时无 `kmh`，报告写"不可测"。

A 模式 `conf` 的语义是**观测充分性**：`high` = 出球观测 ≥4 点、残差 <2.5 px、bootstrap 区间相对宽度 <0.2；`medium` = 能解但上述任一项不达标；`low` = 观测 ≤2 或残差大；`none` = 解不出来。B 模式的 `conf` 是另一套语义，见下面 flights.json。

## contacts.json — 人工编写（B 模式核心输入） [B]

B 模式所有人工判断集中在这一个文件里，`flight_bvp.py` 和 `pose_metrics.py` 都读它。

```json
{"fps": 30,
 "players": {"N": {"label": "近端", "hand": "left",  "facing": "away"},
             "F": {"label": "远端", "hand": "right", "facing": "toward"}},
 "contacts": [{"id": 1, "frame": 34, "who": "N", "side": "SERVE", "point": [-0.8, 10.41, 2.62]},
              {"id": 2, "frame": 55, "who": "F", "side": "BH",    "point": [5.35, -13.49, 1.06]}],
 "bounce_audio_candidates": {"1": [43.3], "3": [109.8]},
 "final_landing": {"point": [4.35, 8.5], "frame": 550,
                   "source": "rolling ball at f549-557"}}
```

| 字段 | 说明 |
| --- | --- |
| `fps` | 必填（也可由 `--fps` 或 `meta.json` 提供） |
| `players` | 键是球员代号（与 `who` 对应），每人 `label`（报告显示名）、`hand`（`left`/`right`）、`facing`（`away`/`toward`）。`pose_metrics.py` 读这里的 `hand`/`facing`，优先级高于命令行 |
| `contacts[].id` | 拍序号，缺省按顺序补 `i+1`，不能重复 |
| `contacts[].frame` | 接触帧，0-based，可以是小数；**必须严格递增**，否则脚本退出 |
| `contacts[].who` | 击球者代号，缺省 `"?"`。写错这个字段那一拍的接触点、球速、正反手全错，还会顺着 BVP 污染前后两拍 |
| `contacts[].side` | `FH` / `BH` / `SERVE`（以 `SERVE` 开头会启用发球区合法性约束），缺省 `"?"` |
| `contacts[].point` | 3D 接触点 `[x, y, z]`，米，必须是 3 个数。相邻两拍的 `y` 必须一正一负（分居球网两侧），同侧会 WARNING |
| `contacts[].bounce_audio_cand` | 可选，写在单拍里的弹地候选帧，等价于顶层 `bounce_audio_candidates` 里对应的项 |
| `bounce_audio_candidates` | 键是**接触 id 的字符串**，值是候选弹地帧列表，可以给多个（弹地一般落在飞行的 50–80% 区间），由物理去挑 |
| `final_landing` | 回合最后一拍的边界条件：`point` 是 `[x, y]`（也可给第三个数当高度，否则用 `--landing-z`）、`frame` 是落点帧、`source` 是文字说明。不给的话最后一拍直接跳过 |
| `sport` / `court` | 可选，覆盖场地预设 |

## flights.json — `flight_bvp.py` [B]

```json
{"fps": 30.0, "sport": "tennis", "source": "contacts.json",
 "method": "contact-to-contact boundary value problem (flight_bvp.py)",
 "summary": {"n_flights": 14, "n_in_average": 14, "excluded_low_conf": [],
             "overall": {"n": 14, "mean_kmh": 117.4, "max_kmh": 177.7},
             "by_player": {"N": {"n": 8, "mean_kmh": 123.8, "max_kmh": 177.7, "label": "近端"}},
             "by_side": {"FH": {"n": 6, "mean_kmh": 112.0, "max_kmh": 131.4}},
             "results": {"IN": 13, "OUT": 1, "NET": 0, "UNK": 0},
             "conf": {"high": 8, "medium": 6, "low": 0}},
 "flights": [{"id": 1, "fa": 34.0, "fb": 55.0, "dt": 0.7, "who": "N", "side": "SERVE",
              "player": "近端", "contact": [-0.8, 10.41, 2.62], "target": [5.35, -13.49, 1.06],
              "boundary_source": "next_contact",
              "rpm": 1600.0, "V0": [8.62, -48.55, -3.0], "v0_kmh": 177.7,
              "v0_lo": 168.2, "v0_hi": 186.1, "angle_deg": -3.5, "apex_m": 2.62,
              "net_clear": 0.31, "net_x": 1.42, "err_m": 0.019,
              "bounce": [2.31, -5.12], "bounce_frame": 43.1, "bounce_in": true,
              "bounce_audio_cand": [43.3], "dt_bounce": 0.2,
              "n_branches": 25, "conf": "high", "flip_reason": null,
              "spin_band": [{"rpm": 0, "v0_kmh": 172.0, "angle_deg": 2.1,
                             "bounce": [2.5, -6.4], "bounce_in": true,
                             "err_m": 0.031, "dt_bounce": 1.4, "legal": true}],
              "result": "IN", "result_cn": "界内"}]}
```

| 字段 | 说明 |
| --- | --- |
| `fa` / `fb` / `dt` | 本拍接触帧、下一个边界的帧、飞行时长（秒） |
| `who` / `side` / `player` | 击球者代号 / 正反手标签 / `players[who].label` |
| `contact` / `target` | 边界值问题的两个端点 A、B（米） |
| `boundary_source` | `next_contact`（下一次触球提供边界）或 `visual_landing`（最后一拍，边界来自 `final_landing`） |
| `rpm` | **被选中的**上旋值，渲染重建球路必须按每拍自己的 rpm |
| `V0` / `v0_kmh` | 初速度向量（m/s）与速率（km/h） |
| `v0_lo` / `v0_hi` | 速度区间：旋转带各档的速度 ∪ 接触帧 ±`--dt-sens` 帧的重解，取最小/最大值。这是诚实的不确定性，不是噪声 |
| `angle_deg` / `apex_m` | 发射仰角（度）、轨迹最高点（米） |
| `net_clear` / `net_x` | 过网余量（米，负 = 下网）与过网时的横向位置 |
| `err_m` | 端点残差：按解出的 V0 飞 `dt` 秒后与 B 的距离（米）。超 `--err-max`（默认 0.25）就不是这个边界值问题的解 |
| `bounce` / `bounce_frame` / `bounce_in` | 首次弹地点 `[x, y]`、弹地帧、是否落在单打有效区内；`boundary_source == "visual_landing"` 时 `bounce` 就是实测落点，另有 `bounce_sim`（模拟弹地点，两者不一致时能看见）和 `note` |
| `bounce_audio_cand` / `dt_bounce` | 输入的候选弹地帧、预测弹地帧与最近候选的帧差（证据强度） |
| `n_branches` | 收敛的（旋转 × 仰角）分支数 |
| `spin_band[]` | 每个 rpm 档位的最优分支：`v0_kmh`、`angle_deg`、`bounce`、`bounce_in`、`err_m`、`dt_bounce`、`legal` |
| `flip_reason` | 只有 `conf == "low"` 时非空，写明为什么低置信（端点残差过大 / 全带都落不进场 / 换旋转就 IN↔OUT 翻转且无音频仲裁 / 速度带过宽） |
| `result` / `result_cn` | `IN` / `OUT` / `NET` / `UNK` 与中文 |

**B 模式 `conf` 的语义与 A 模式不同**，报告里不要混着写：`high` = 端点残差 < `--err-good` 且落点合法且对上了音频弹地峰；`medium` = 能解但缺其中一项证据；`low` = 端点残差超限、或旋转带里 IN/OUT 会翻转而没有音频仲裁、或速度带超过 `--band-max-frac`——**`low` 不进任何平均**（`summary.excluded_low_conf` 列出被剔的 id）。也就是说 A 模式的 `conf` 说的是"观测够不够"，B 模式的 `conf` 说的是"这条轨迹被多少独立证据钉住"。详见 `broadcast-view.md` 第六节。

## technique.json — `pose_metrics.py` [通用]

数组，一行一拍。字段含义与阈值见 `pose-metrics.md`：

```json
[{"id": 1, "side": "FH", "tc": 64.8, "frame": 64, "who": "N", "f_coil": 58,
  "hand": "left", "facing": "away", "side_rule": "hand=left,facing=away -> FH on image left",
  "pose_file": "pose_near.json",
  "shoulder_rot_prep": 41.2, "hip_rot_prep": 22.0, "x_factor": 19.2,
  "shoulder_rot_contact": 27.4, "shoulder_rot_finish": 12.1,
  "unwind_to_contact": 13.8, "unwind_to_finish": 15.3,
  "coil_to_contact_frames": 3, "coil_to_contact_s": 0.1,
  "elbow_angle_contact": 142.6, "wrist_speed_px": 31.4, "wrist_speed_ms": 12.4,
  "torso_source": "torso_ratio_proxy",
  "torso_closed_deg": 21.4, "torso_closed_max_deg": 35.2, "torso_unwind_deg": 13.8,
  "contact_h": 1.61, "contact_h_source": "contact_point", "contact_h_pose": 1.55,
  "lat_from_body_m": 0.62, "hip_h": 0.95, "ground": [-0.3, 10.9],
  "behind_baseline": -0.98, "backswing_to_contact_f": 7,
  "followthrough_h": 2.1, "prep_h": 0.8,
  "contact_lateral_sw": 2.67, "contact_vs_shoulder_px": -18.2,
  "contact_ahead_m": 0.42, "hip_y_range_px": 26.0, "head_move_px": 14.7,
  "side_suggest": "FH"}]
```

指标缺失时为 `null`（关键点置信度不足）。出现 `"side_conflict": true` 说明正反手标签与击球点横向位置矛盾，需回看。

躯干与正反手规则相关的字段：

| 字段 | 说明 |
| --- | --- |
| `torso_source` | 躯干指标的来源，取值 `shoulder_hip_3d`（三维肩髋方位角）或 `torso_ratio_proxy`（图像肩宽/躯干长比值代理）。**这是整行躯干指标共用的一个字段**，不是每个指标各带一个 |
| `torso_closed_deg` | 触球时刻的躯干关闭角（度），0 = 正对球网 |
| `torso_closed_max_deg` | `--torso-window`（默认 −12…+6 帧）内的关闭角峰值 = 引拍最深处 |
| `torso_unwind_deg` | `torso_closed_max_deg − torso_closed_deg`，躯干解旋幅度，是**最该引用的那个数**；由已四舍五入的两个值相减，保证读者自己算得一样 |
| `hip_closed_deg` / `torso_separation_deg` | 只有走 `shoulder_hip_3d` 时才有（髋线关闭角、肩髋分离角） |
| `hand` / `facing` / `side_rule` | 该拍实际生效的持拍手、朝向和推导出的规则文字，出问题时先看这行 |
| `pose_file` | 这一拍用的是哪个 pose 文件（多球员时按 `who` + `--pose-map` 选） |
| `contact_h` / `contact_h_source` | 击球高度（米）与来源：`shot_field`（输入自带）/ `contact_point`（`contacts.json` 的 `point[2]`）/ `pose_wrist`（腕关键点反投影） |
| `contact_h_pose` / `lat_from_body_m` / `hip_h` / `ground` / `behind_baseline` | 由腕/肩/髋/踝反投影得到的米制量，需要标定；`ground` 是球员在场地上的位置 `[x, y]` |
| `wrist_speed_ms` / `wrist_speed_px` | 腕速峰值。有标定才有 m/s（只含平行像平面分量），没有标定时另有 `wrist_speed_unit: "px/frame"` |
| `backswing_to_contact_f` / `prep_h` / `followthrough_h` | 引拍最慢帧到触球的帧数、引拍最低腕高、随挥最高腕高（米） |

降级为 `torso_ratio_proxy` 后度数只能做**同片段内相对比较**，报告里必须带来源说明。远端球员（画面里只有几十像素高）的躯干结论要降权，理由见 `broadcast-view.md` 第七节。

## 最小复算集

A 模式留下这五个文件就能在不重新抽帧的前提下复算全部统计与结论（渲染除外）：

`meta.json` · `calib.json` · `shots_input.json` · `shots.json` · `technique.json`

B 模式对应的七个：`$WORKS/meta.json` · `stabilize.json` · `calib.json` · `calib_verify.json` · `contacts.json` · `flights.json` · `technique.json`。少了 `stabilize.json` 就再也回不到原始帧坐标（渲染、复核都做不了）。

`frames/`（1000 帧约 100 MB）、`audio.wav`、`bg_median.png`、`ball_cand.json`、`swing_*.json`、`*_noaudio.mp4` 都可以删。
