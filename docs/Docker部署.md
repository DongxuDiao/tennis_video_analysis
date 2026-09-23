# Tennis Lab Docker 部署

## 启动与管理

在项目根目录执行：

```sh
./scripts/docker.sh up -d --build
./scripts/docker.sh ps
./scripts/docker.sh logs --tail=100 app
./scripts/docker.sh down
```

访问 http://127.0.0.1:8765 。仅绑定本机 loopback，不开放局域网访问。

`Dockerfile` 构建前端静态页面和 Python 后端；后端自身管理唯一 worker。不要增加 uvicorn workers 或扩容 app 副本。

## 数据保留

Compose 将当前 checkout 的 `data/` 挂载到容器内相同绝对路径，并设置 `TENNIS_DATA_DIR`。这是为了兼容数据库和历史分析 JSON 中的绝对路径。数据库、视频、模型、分析与导出均保留，删除容器不会删除数据。

`./scripts/docker.sh` 自动提供宿主 UID/GID 和目录。自定义数据目录可设置 `TENNIS_HOST_DATA_DIR`；已有数据库不能仅复制到新路径就认为所有历史绝对路径自动迁移。

**原生服务和容器服务不能同时使用同一 data 目录。** 启动前应停止原生 API/worker，并确认没有进行中的分析。此次部署前已创建 SQLite 一致性备份，位于 `.docker-state/backups/`。

## 运行平台与能力

- 镜像原生运行于 Linux ARM64，避免 Apple Silicon 上 amd64/Rosetta 执行 PyTorch 时的 worker `SIGBUS`。
- 模型使用 CPU，不使用 macOS MPS。0.5 秒、15 帧的合成验收视频完整分析实测 352.9 秒，因此长视频更适合使用原生 Mac/MPS 运行。
- MediaPipe 0.10.35 没有 Linux ARM64 wheel，容器健康信息中 `mediapipe` 为 `null`。当前默认的 YOLO11m-pose + TrackNet 自动分析不依赖 MediaPipe；旧版 MediaPipe 分析路径需在原生 macOS 环境使用。
- Docker 下人物与 TrackNet 重模型阶段采用阻塞等待，避免模型运行期间频繁读 bind mount 日志和写 SQLite 导致 worker 崩溃。当前阶段结束前，取消操作不会即时生效。
- 前端构建使用宿主架构，减小构建开销。
- Docker Python 依赖锁定在 `docker/requirements-final.lock`，包含文件哈希；PyTorch/torchvision 指向官方 ARM64 CPU wheel，避免安装 CUDA 包。OpenCV core 与 contrib 固定为相同版本。
- 基础镜像固定 digest；Python 镜像通过 DaoCloud 公共镜像源获取。依赖更新后需要重新生成并验证 Docker 锁文件。
- Qwen/MLX P0 验证脚本尚未接入产品。此次镜像部署的是现有业务应用，不包含 macOS Metal 推理服务，也不代表可执行 Skill 改造完成。
- 此应用容器不是用户上传 Python 的通用安全沙箱；没有挂 Docker socket。

重新生成依赖锁：

```sh
uv pip compile docker/requirements.in \
  --python-version 3.12 --python-platform aarch64-manylinux_2_28 \
  --generate-hashes --output-file docker/requirements-final.lock
```

更新依赖后必须执行 `pip check` 并验证 `/app/requirements.lock` 与实际安装版本。PyTorch 与 torchvision 继续使用官方 CPU wheel 的明确 URL 和 SHA256，不能改为默认 PyPI Linux 包，否则会带入 CUDA 运行库。

## 2026-09-21 部署验收

- 容器 `tennis-lab-app-1` 为 `linux/arm64`，状态 `healthy`，仅监听 `127.0.0.1:8765`。
- API、worker、前端静态页面和历史视频 Range 请求均通过；部署前的两个项目保留，验收完成后仍为两个项目，无 queued/running 任务。
- 0.5 秒 H.264 合成视频完整执行导入、人物检测、球场候选、球候选、TrackNet、渲染；任务成功并生成 0.5 秒、30 fps、832×720 H.264 标注视频。合成图没有真人，覆盖率为 0 属于测试输入预期，不代表真实视频精度。
- 视频旋转元数据回归测试通过；60 fps 视频截取后音频保留、时长和起始时间回归测试通过。
- 最终测试产物保存在 `.docker-state/validation/`；合成测试项目已通过产品删除接口清理，未改动原有项目。

## 外部模型凭据

macOS Keychain 无法从 Linux 容器直接读取。容器启用已有的 FileSecretStore：

- 宿主目录 `.docker-state/secrets/`，权限 0700，包含允许 UI 保存的 `.writable` 标记。
- 凭据文件按程序设置为 0600。该存储是受文件权限保护的明文，不等同于 Keychain 加密。
- 目录不进入 Git 或镜像。不要将其包含在公开备份中。
- 历史连接元数据仍保留，但 Keychain 中的 Key 不会自动复制。页面可能仍显示原连接，应断开后重新输入 Key 测试并保存。
- 重新连接是用户操作，部署过程不调用 GLM 付费接口、不提取 Keychain 凭据。

## 故障与更新

镜像更新：`./scripts/docker.sh up -d --build`。先等待或取消正在执行的分析，再重建重启。

健康检查验证 API、依赖导入和 worker 存活。真实模型结果仍需通过分析任务验证，容器 healthy 不代表模型精度达标。

查看 `./scripts/docker.sh logs --tail=100 app`。容器资源初始限制为 2 CPU、4 GB 内存；如果长视频 OOM，先检查宿主 Docker VM 总内存和其他容器占用，按实际负载调整 Compose。

停止容器后可恢复原生运行；容器中新保存的文件凭据不会自动导入 macOS Keychain。
