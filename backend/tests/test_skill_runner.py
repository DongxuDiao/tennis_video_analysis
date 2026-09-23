"""skill 快照安全测试：穿越/符号链接/可执行位/哈希钉住/炸弹拒绝；真实 ZIP 可安装。"""
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from app.skill_runtime import manifest as M


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _write_zip(tmp_path, members, sha=None):
    p = tmp_path / "skill.zip"
    p.write_bytes(_zip_bytes(members))
    return p, (sha or hashlib.sha256(p.read_bytes()).hexdigest())


def test_install_rejects_path_traversal(tmp_path):
    p, sha = _write_zip(tmp_path, {"sports-video-motion-analysis/../../evil.txt": b"x"})
    with pytest.raises(M.SkillArchiveError):
        M.install_skill(p, expected_sha=sha)


def test_install_rejects_absolute_and_backslash(tmp_path):
    for name in ("/etc/passwd", "a\\b.txt"):
        p, sha = _write_zip(tmp_path, {"sports-video-motion-analysis/" + name: b"x"})
        with pytest.raises(M.SkillArchiveError):
            M.install_skill(p, expected_sha=sha)


def test_install_rejects_executable_bit(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zi = zipfile.ZipInfo("sports-video-motion-analysis/run.sh")
        zi.external_attr = (0o755 << 16)
        zf.writestr(zi, b"#!/bin/sh\n")
    p = tmp_path / "skill.zip"
    p.write_bytes(buf.getvalue())
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(M.SkillArchiveError):
        M.install_skill(p, expected_sha=sha)


def test_install_rejects_zip_bomb(tmp_path):
    big = b"\0" * (M.MAX_TOTAL_BYTES + 1)
    p, sha = _write_zip(tmp_path, {"sports-video-motion-analysis/big.bin": big})
    with pytest.raises(M.SkillArchiveError):
        M.install_skill(p, expected_sha=sha)


def test_install_rejects_wrong_sha(tmp_path):
    p, _ = _write_zip(tmp_path, {"sports-video-motion-analysis/SKILL.md": b"content"})
    with pytest.raises(M.SkillArchiveError):
        M.install_skill(p, expected_sha="0" * 64)


def test_skill_md_sha_pinned(tmp_path):
    # SKILL.md 哈希不匹配（即使整体 zip sha 允许自定义）→ 拒绝
    import unittest.mock
    members = {"sports-video-motion-analysis/SKILL.md": b"different content",
               "sports-video-motion-analysis/x.txt": b"1"}
    p, sha = _write_zip(tmp_path, members)
    with pytest.raises(M.SkillArchiveError):
        M.install_skill(p, expected_sha=sha)


def test_real_zip_installs_and_freezes_version():
    """官方 user_skills.zip 存在时：可安装、逐文件哈希记录、只读。"""
    zip_path = Path("/Users/bytedance/Downloads/user_skills.zip")
    if not zip_path.is_file():
        pytest.skip("用户 skill ZIP 不存在（非本机环境）")
    target = M.install_skill(zip_path)
    assert target.name == M.EXPECTED_ZIP_SHA256[:12]
    marker = (target / ".installed.json")
    data = json.loads(marker.read_text())
    assert data["zip_sha256"] == M.EXPECTED_ZIP_SHA256
    assert data["skill_md_sha256"] == M.EXPECTED_SKILL_MD_SHA256
    assert data["files"]["SKILL.md"]
    assert M.installed_skill() == target
    assert M.skill_hash() == M.EXPECTED_ZIP_SHA256[:16]


