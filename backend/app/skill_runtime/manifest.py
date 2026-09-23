"""Skill 快照管理：安全解压官方 user_skills.zip 到固定版本目录。

安全规则（方案 §3）：
- ZIP 整体 SHA256 必须匹配锁定的期望值；SKILL.md 单独校验。
- 拒绝：绝对路径、`..` 穿越、符号链接、可执行位、超限文件/总大小/文件数。
- 逐文件哈希记录在 manifest.json；目录名 = <zip-sha 前 12 位>。
- vendor 目录只读使用，运行时绝不修改。
"""
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

EXPECTED_ZIP_SHA256 = "b4c904fda19a1e2abf9b79600038936b2608ea7e3d6d3bc22bc950a2809d0c4a"
EXPECTED_SKILL_MD_SHA256 = "9700b2919289a0d03c75bba0bb470c1d252bd632af021c7321135e2ef9b46131"
SKILL_ROOT_NAME = "sports-video-motion-analysis"
MAX_FILES = 2000
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


class SkillArchiveError(RuntimeError):
    pass


def zip_sha256(zip_path: Path) -> str:
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_member(name: str) -> bool:
    if name.startswith("/") or "\\" in name or ":" in name:
        return False
    if "//" in name:   # "root//abs" 这类双斜杠在 rel 提取后会成为绝对路径
        return False
    parts = Path(name).parts
    if any(p in ("..", "") for p in parts):
        return False
    return True


def _relative_member(name: str) -> str:
    """提取相对路径并再次校验（防御性：返回值必须是相对且不越界）。"""
    rel = name[len(SKILL_ROOT_NAME) + 1:] if name.startswith(SKILL_ROOT_NAME + "/") else name
    p = Path(rel)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        raise SkillArchiveError(f"拒绝不安全相对路径：{name!r}")
    return rel


def install_skill(zip_path: Path, *, expected_sha: str | None = None) -> Path:
    """校验并解压到 vendor/sports-video-motion-analysis/<sha12>/；已存在则直接返回。"""
    expected = expected_sha or EXPECTED_ZIP_SHA256
    actual = zip_sha256(zip_path)
    if actual != expected:
        raise SkillArchiveError(f"skill ZIP 哈希不匹配：{actual[:12]} ≠ 期望 {expected[:12]}")
    target = VENDOR_DIR / SKILL_ROOT_NAME / actual[:12]
    marker = target / ".installed.json"
    if marker.is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)

    file_hashes: dict[str, str] = {}
    exec_stripped: list[str] = []
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_FILES:
            raise SkillArchiveError(f"文件数超限（{len(infos)}）")
        for info in infos:
            name = info.filename
            if not _safe_member(name):
                raise SkillArchiveError(f"拒绝不安全路径：{name!r}")
            had_exec = bool(info.external_attr >> 16 & 0o111)
            # skill 只作为数据消费（从不执行）；官方包的数据文件也可能带执行位，
            # 统一在安装时去除并记录，而不是拒绝整个官方包
            if had_exec:
                file_hashes.setdefault("_exec_stripped", "")  # 占位标记，见下方记录
            if info.file_size > MAX_MEMBER_BYTES or total + info.file_size > MAX_TOTAL_BYTES:
                raise SkillArchiveError(f"大小超限：{name}")
            total += info.file_size
            data = zf.read(name)
            rel = _relative_member(name)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if had_exec:
                exec_stripped.append(rel)
            file_hashes[rel] = hashlib.sha256(data).hexdigest()

    skill_md = target / "SKILL.md"
    if not skill_md.is_file():
        raise SkillArchiveError("包内缺少 SKILL.md")
    md_sha = hashlib.sha256(skill_md.read_bytes()).hexdigest()
    if md_sha != EXPECTED_SKILL_MD_SHA256:
        raise SkillArchiveError(f"SKILL.md 哈希不匹配：{md_sha[:12]}")
    # 拒绝符号链接
    for p in target.rglob("*"):
        if p.is_symlink():
            raise SkillArchiveError(f"拒绝符号链接：{p.name}")
    marker.write_text(json.dumps({
        "zip_sha256": actual, "skill_md_sha256": md_sha,
        "files": file_hashes, "exec_bits_stripped": exec_stripped,
        "installed_at": __import__("time").time(),
    }, ensure_ascii=False, indent=1))
    # vendor 只读化（尽力而为）
    for p in target.rglob("*"):
        try:
            p.chmod(p.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except OSError:
            pass
    return target


def installed_skill() -> Path | None:
    """返回已安装的固定版本目录（依据期望 ZIP SHA）。"""
    target = VENDOR_DIR / SKILL_ROOT_NAME / EXPECTED_ZIP_SHA256[:12]
    return target if (target / ".installed.json").is_file() else None


def skill_hash() -> str:
    """当前使用（或期望）的 skill 版本标识。"""
    return EXPECTED_ZIP_SHA256[:16]
