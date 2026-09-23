"""全局配置与路径解析。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

# 仓库根 = backend/ 的上一级
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

PREPROCESS_VERSION = "v1"
# 代码版本：对关键算法文件内容取哈希，保证可追溯且随代码变化
_CODE_FILES = [
    "model_providers/glm_coding.py", "model_providers/base.py",
    "skill_runtime/default_external_instructions.md",
    "services/external_analysis.py", "services/analysis_mode.py", "model_connections/skills.py",
    "vision/tracknet.py", "vision/detector.py", "vision/pose.py", "vision/tracker.py",
    "vision/events.py", "vision/pipeline.py", "vision/geometry.py",
    # 管线版本需覆盖 expert 引擎与模型增强编排（GLM 接入方案 §2）
    "expert_engine/automatic.py", "expert_engine/quality_review.py",
    "expert_engine/person_quality.py", "expert_engine/scene_quality.py",
    "enhancement/planner.py", "enhancement/evidence.py",
    "enhancement/validator.py", "enhancement/audit.py",
]


def _code_version() -> str:
    h = hashlib.sha256()
    for rel in _CODE_FILES:
        p = Path(__file__).parent / rel
        if p.exists():
            h.update(rel.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


CODE_VERSION = _code_version()


def load_config() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_CONFIG = load_config()


def cfg() -> dict[str, Any]:
    return _CONFIG


def data_dir() -> Path:
    env = os.environ.get("TENNIS_DATA_DIR")
    d = Path(env) if env else REPO_ROOT / _CONFIG["paths"]["data_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "app.sqlite"


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def projects_dir() -> Path:
    d = data_dir() / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exports_root() -> Path:
    d = data_dir() / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path() -> Path:
    return REPO_ROOT / "models" / "manifest.json"


def model_path_for(key: str) -> Path | None:
    import json
    m = json.loads(manifest_path().read_text(encoding="utf-8"))
    for item in m["models"]:
        if item["key"] == key:
            return models_dir() / item["filename"]
    return None
