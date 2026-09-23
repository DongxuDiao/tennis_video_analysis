# sports-python-cpu-v1 — 可执行 Skill 包的 ARM64 CPU 沙箱（P0 spike）
#
# 设计约束（改造方案 §6）：
# - 非 root、无网络（运行时由 --network none 保证）、依赖全部预装，分析期不 pip install
# - 中文字体内置（render_annotated 报告用）
# - ffmpeg 来自 imageio-ffmpeg 静态二进制（video_prep.py 依赖）
# - 具体版本以镜像内 pip freeze 为准，记录进 docs/本地Skill引擎_P0验证记录.md
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "opencv-python-headless>=4.10,<5" \
    "pillow>=10,<12" \
    "imageio-ffmpeg>=0.6,<1"

RUN useradd -m -u 10001 sandbox
WORKDIR /work
USER sandbox
