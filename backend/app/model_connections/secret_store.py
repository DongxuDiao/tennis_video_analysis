"""凭据存储：macOS Keychain 优先；不可用时明确报错，绝不静默落明文。

DB 只保存 secret_ref（Keychain 服务名）。测试环境通过 SecretStore 的
内存实现注入，不触碰真实 Keychain。
"""
from __future__ import annotations

import secrets as py_secrets
import subprocess
import sys
from typing import Protocol

SERVICE_PREFIX = "TennisLab-ModelConnection"


class SecretStoreError(RuntimeError):
    """Keychain 不可用或操作失败；消息不包含 Key 本身。"""


class SecretStore(Protocol):
    def save(self, secret_ref: str, api_key: str) -> None: ...
    def load(self, secret_ref: str) -> str: ...
    def delete(self, secret_ref: str) -> None: ...
    def exists(self, secret_ref: str) -> bool: ...


def new_secret_ref() -> str:
    return f"{SERVICE_PREFIX}-{py_secrets.token_hex(8)}"


def _run(*args: str) -> subprocess.CompletedProcess:
    # -w 0 避免交互式询问；超时防挂起。任何输出都不回显 Key 到日志。
    return subprocess.run(args, capture_output=True, text=True, timeout=15)


class KeychainStore:
    """macOS 安全项存储：generic password，服务名 = secret_ref，账户固定。"""

    ACCOUNT = "tennis-lab"

    def _available(self) -> bool:
        if sys.platform != "darwin":
            raise SecretStoreError("Keychain 仅支持 macOS；当前平台请配置外部凭据文件环境变量 TENNIS_SECRET_DIR")
        r = _run("/usr/bin/security", "list-keychains")
        if r.returncode != 0:
            raise SecretStoreError(f"Keychain 不可用：{r.stderr.strip()[:120]}")

    def save(self, secret_ref: str, api_key: str) -> None:
        self._available()
        r = _run("/usr/bin/security", "add-generic-password",
                 "-a", self.ACCOUNT, "-s", secret_ref,
                 "-w", api_key, "-U")
        if r.returncode != 0:
            raise SecretStoreError(f"写入 Keychain 失败（code {r.returncode}）")

    def load(self, secret_ref: str) -> str:
        self._available()
        r = _run("/usr/bin/security", "find-generic-password",
                 "-a", self.ACCOUNT, "-s", secret_ref, "-w")
        if r.returncode != 0:
            raise SecretStoreError("凭据不存在或无法读取（需要登录 Keychain）")
        key = r.stdout.strip()
        if not key:
            raise SecretStoreError("凭据为空")
        return key

    def delete(self, secret_ref: str) -> None:
        self._available()
        r = _run("/usr/bin/security", "delete-generic-password",
                 "-a", self.ACCOUNT, "-s", secret_ref)
        if r.returncode != 0 and self.exists(secret_ref):
            raise SecretStoreError("删除 Keychain 凭据失败")

    def exists(self, secret_ref: str) -> bool:
        if sys.platform != "darwin":
            return False
        r = _run("/usr/bin/security", "find-generic-password",
                 "-a", self.ACCOUNT, "-s", secret_ref)
        return r.returncode == 0


class FileSecretStore:
    """容器/无 Keychain 环境的降级：运维注入的只读目录（TENNIS_SECRET_DIR）。

    目录必须已存在且权限不宽于 0700，否则拒绝使用——不创建、不写明文到任意位置。
    仅供测试或受控部署；本机默认仍是 Keychain。
    """

    def __init__(self, root: str):
        from pathlib import Path
        import stat
        self.root = Path(root)
        if not self.root.is_dir():
            raise SecretStoreError(f"凭据目录不存在：{self.root}")
        mode = stat.S_IMODE(self.root.stat().st_mode)
        if mode & 0o077:
            raise SecretStoreError("凭据目录权限过宽（需 0700）")
        self._write: Path | None = self.root / ".writable" if (self.root / ".writable").exists() else None

    def _path(self, secret_ref: str) -> "Path":
        # secret_ref 由本模块生成（TennisLab-ModelConnection-hex），仍做白名单校验
        import re
        if not re.fullmatch(r"[A-Za-z0-9\-]{1,80}", secret_ref):
            raise SecretStoreError("凭据引用非法")
        return self.root / secret_ref

    def save(self, secret_ref: str, api_key: str) -> None:
        if self._write is None:
            raise SecretStoreError("文件凭据目录为只读（无 .writable 标记），不能写入")
        self._path(secret_ref).write_text(api_key)
        self._path(secret_ref).chmod(0o600)

    def load(self, secret_ref: str) -> str:
        p = self._path(secret_ref)
        if not p.is_file():
            raise SecretStoreError("凭据不存在")
        key = p.read_text().strip()
        if not key:
            raise SecretStoreError("凭据为空")
        return key

    def delete(self, secret_ref: str) -> None:
        self._path(secret_ref).unlink(missing_ok=True)

    def exists(self, secret_ref: str) -> bool:
        return self._path(secret_ref).is_file()


class MemoryStore:
    """测试用内存实现。"""

    def __init__(self):
        self._data: dict[str, str] = {}

    def save(self, secret_ref: str, api_key: str) -> None:
        self._data[secret_ref] = api_key

    def load(self, secret_ref: str) -> str:
        if secret_ref not in self._data:
            raise SecretStoreError("凭据不存在")
        return self._data[secret_ref]

    def delete(self, secret_ref: str) -> None:
        self._data.pop(secret_ref, None)

    def exists(self, secret_ref: str) -> bool:
        return secret_ref in self._data


def default_store() -> SecretStore:
    import os
    env_dir = os.environ.get("TENNIS_SECRET_DIR")
    if env_dir:
        return FileSecretStore(env_dir)
    if sys.platform == "darwin":
        return KeychainStore()
    raise SecretStoreError("当前平台无默认凭据存储；请设置 TENNIS_SECRET_DIR")


def mask_key(api_key: str) -> str:
    """掩码别名：前3后4，中间以长度表示；不足8位全掩码。"""
    if len(api_key) < 8:
        return "****"
    return f"{api_key[:3]}…{api_key[-4:]}"


def alias_for(api_key: str) -> str:
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]
