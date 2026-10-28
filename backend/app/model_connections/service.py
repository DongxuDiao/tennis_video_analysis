"""模型连接服务：探测 → 启用 → 轮换/启停 → 删除。

- probe 产生的 probe_id 60 秒有效，启用时二次校验，避免“探测成功但保存错 Key”。
- 厂商/接入类型由 model_providers/catalog.py 目录驱动（多厂商）。
- 以实际能力探测为准，不使用资格环境变量拦截。
- config_revision 乐观锁；删除连接清理凭据。
- 凭据只经 SecretStore；本模块任何返回值不含 Key。
"""
from __future__ import annotations

import secrets
import time

from app.model_providers import catalog
from app.model_providers.base import ProbeResult
from app.model_providers.errors import USER_MESSAGES
from app.model_connections import secret_store as ss

CONSENT_VERSION = "v1-providers-2026-10-28"
PROBE_TTL_S = 60.0

# 进程内短期探测凭据（不落盘）；probe_id → {key_mask, capabilities, expires_at, kind}
_pending_probes: dict[str, dict] = {}


def _adapter_for(provider: str, account_kind: str):
    """统一 provider 工厂（目录驱动）。

    保持两位置参数签名：本函数是测试注入 FakeProvider 的猴补接缝；
    需要 DB 冻结模型时由调用方做属性注入（adapter.model = row["model_id"]）。
    """
    return catalog.adapter_for(provider, account_kind)


def _eligibility_for(account_kind: str) -> str:
    # 机制已退役：能力探测决定激活；字段仅为 DB 兼容保留
    return "confirmed"


def probe(provider: str, account_kind: str, api_key: str, store=None,
          model: str | None = None) -> dict:
    """执行探测（合成图）。返回 {probe_id, ok, capabilities, error}。

    注意：探测可能消耗少量服务额度；model 为可选覆盖（覆盖即唯一候选）。
    """
    catalog.require(provider, account_kind)   # 未知厂商/接入类型 → ValueError
    if not api_key or len(api_key) > 512:
        return {"ok": False, "error_code": "invalid_key",
                "error": "Key 为空或过长"}
    store = store or ss.default_store()
    eligibility = _eligibility_for(account_kind)
    adapter = _adapter_for(provider, account_kind)
    if model:
        adapter.model = model   # 属性注入（FakeProvider 猴补场景下被忽略）
    result: ProbeResult = adapter.probe(api_key)
    # 探测语境的用户文案（区别于任务中失败时的"已保留本地结果"措辞）
    probe_messages = {"quota_exhausted": "服务额度不足，请充值后重试",
                      "timeout_unknown": "探测超时，请稍后重试",
                      "plan_limit_reached": result.detail}
    payload = {
        "ok": result.ok,
        "capabilities": _caps_dict(result.capabilities),
        "error_code": result.error_code,
        "error": probe_messages.get(result.error_code)
                 or USER_MESSAGES.get(result.error_code, result.detail) if not result.ok else "",
        "eligibility": eligibility,
    }
    if result.ok:
        probe_id = secrets.token_hex(16)
        _pending_probes[probe_id] = {
            "key": api_key, "expires_at": time.time() + PROBE_TTL_S,
            "provider": provider, "account_kind": account_kind,
            "model": model or "",
            "capabilities": payload["capabilities"],
        }
        payload["probe_id"] = probe_id
        payload["key_alias"] = ss.alias_for(api_key)
        payload["key_mask"] = ss.mask_key(api_key)
    # 立即从入参层面忘记 Key（调用方负责）
    return payload


def _caps_dict(caps) -> dict:
    return {"protocol": caps.protocol, "text_ok": caps.text_ok,
            "vision_ok": caps.vision_ok, "multi_image": caps.multi_image,
            "structured_output": caps.structured_output,
            "cancel_supported": caps.cancel_supported,
            "model_id": caps.model_id, "usage_source": caps.usage_source,
            "notes": caps.notes}


def create_from_probe(probe_id: str, *, enable: bool, skill_id: str = "default", store=None) -> dict:
    """用 probe_id 建立连接（保存凭据 + 记录能力 + 同意版本）。"""
    entry = _pending_probes.get(probe_id)
    if entry is None or entry["expires_at"] < time.time():
        raise ValueError("探测已过期，请重新测试")
    store = store or ss.default_store()
    from app.model_connections import skills
    skills.load(skill_id)
    entry["capabilities"]["skill_id"] = skill_id
    secret_ref = ss.new_secret_ref()
    store.save(secret_ref, entry["key"])
    conn_id = secrets.token_hex(8)
    now = time.time()
    from app.store import db
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO model_connections (id, provider, account_kind, protocol, "
            "display_name, model_id, secret_ref, capabilities_json, eligibility_status, "
            "enabled, consent_version, consented_at, config_revision, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (conn_id, entry["provider"], entry["account_kind"],
             entry["capabilities"]["protocol"],
             catalog.display_name(entry["provider"], entry["account_kind"],
                                  ss.mask_key(entry["key"])),
             entry["capabilities"].get("model_id") or entry.get("model") or "", secret_ref,
             db.j(entry["capabilities"]), _eligibility_for(entry["account_kind"]),
             1 if enable else 0, CONSENT_VERSION, now, 1, now, now))
    _pending_probes.pop(probe_id, None)
    return {"id": conn_id, "enabled": enable}


def list_connections() -> list[dict]:
    from app.store import db
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM model_connections ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        caps = db.uj(r["capabilities_json"]) or {}
        out.append({
            "id": r["id"], "provider": r["provider"],
            "account_kind": r["account_kind"], "protocol": r["protocol"],
            "display_name": r["display_name"],
            "model_id": r["model_id"],
            "skill_id": caps.get("skill_id", "default"),
            "has_key": True,
            "vision_ok": bool(caps.get("vision_ok")),
            "text_ok": bool(caps.get("text_ok")),
            "eligibility_status": r["eligibility_status"],
            "enabled": bool(r["enabled"]),
            "consent_version": r["consent_version"],
            "config_revision": r["config_revision"],
        })
    return out


def get_connection(conn_row) -> dict:
    from app.store import db
    caps = db.uj(conn_row["capabilities_json"]) or {}
    return {"id": conn_row["id"], "provider": conn_row["provider"],
            "account_kind": conn_row["account_kind"],
            "model_id": conn_row["model_id"],
            "capabilities": caps, "enabled": bool(conn_row["enabled"]),
            "config_revision": conn_row["config_revision"],
            "secret_ref": conn_row["secret_ref"]}


def enabled_connection(kind: str | None = None):
    """取当前唯一启用的连接（供任务使用）；无则 None。kind 可过滤 api/coding_plan。"""
    from app.store import db
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM model_connections WHERE enabled=1").fetchall()
    for r in rows:
        if kind is None or r["account_kind"] == kind:
            return r
    return None


def patch(conn_id: str, *, expected_revision: int, enabled: bool | None = None,
          reverify: bool = False, skill_id: str | None = None, store=None) -> dict:
    """启停 / 重新验证。乐观锁：revision 不匹配返回冲突。"""
    from app.store import db
    store = store or ss.default_store()
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM model_connections WHERE id=?", (conn_id,)).fetchone()
        if row is None:
            raise LookupError("连接不存在")
        if row["config_revision"] != expected_revision:
            return {"conflict": True, "current_revision": row["config_revision"]}
        updates, vals = ["updated_at=?", "config_revision=config_revision+1"], [time.time()]
        if skill_id is not None:
            from app.model_connections import skills
            skills.load(skill_id)
        if enabled and row["account_kind"] == "coding_plan" and row["protocol"] != "anthropic":
            reverify = True
        if enabled is not None:
            updates.append("enabled=?"); vals.append(1 if enabled else 0)
        if reverify:
            key = store.load(row["secret_ref"])
            adapter = _adapter_for(row["provider"], row["account_kind"])
            adapter.model = row["model_id"] or adapter.model   # 冻结模型：复验不换候选
            result = adapter.probe(key)
            caps = _caps_dict(result.capabilities)
            caps["skill_id"] = skill_id or (db.uj(row["capabilities_json"]) or {}).get("skill_id", "default")
            updates += ["capabilities_json=?", "protocol=?", "model_id=?"]; vals.extend([db.j(caps),caps["protocol"],caps["model_id"]])
            if not result.ok:
                # 重新验证失败：不自动禁用已有连接（保留本地可用性），仅报告
                return {"id": conn_id, "reverify_ok": False,
                        "error_code": result.error_code,
                        "error": USER_MESSAGES.get(result.error_code, "")}
        if skill_id is not None and not reverify:
            caps = db.uj(row["capabilities_json"]) or {}
            caps["skill_id"] = skill_id
            updates.append("capabilities_json=?"); vals.append(db.j(caps))
        vals.append(conn_id)
        conn.execute(f"UPDATE model_connections SET {', '.join(updates)} WHERE id=?", vals)
        new = conn.execute("SELECT config_revision, enabled FROM model_connections WHERE id=?",
                           (conn_id,)).fetchone()
    return {"id": conn_id, "reverify_ok": True,
            "config_revision": new["config_revision"], "enabled": bool(new["enabled"])}


def rotate(conn_id: str, api_key: str, *, expected_revision: int, store=None) -> dict:
    """轮换 Key：原子替换凭据并升版本。"""
    from app.store import db
    store = store or ss.default_store()
    if not api_key or len(api_key) > 512:
        raise ValueError("Key 为空或过长")
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM model_connections WHERE id=?", (conn_id,)).fetchone()
        if row is None:
            raise LookupError("连接不存在")
        if row["config_revision"] != expected_revision:
            return {"conflict": True, "current_revision": row["config_revision"]}
        adapter = _adapter_for(row["provider"], row["account_kind"])
        adapter.model = row["model_id"] or adapter.model   # 冻结模型：轮换不换候选
        result = adapter.probe(api_key)
        if not result.ok:
            return {"id": conn_id, "reverify_ok": False,
                    "error_code": result.error_code,
                    "error": USER_MESSAGES.get(result.error_code, "")}
        new_ref = ss.new_secret_ref()
        store.save(new_ref, api_key)
        conn.execute(
            "UPDATE model_connections SET secret_ref=?, display_name=?, "
            "config_revision=config_revision+1, updated_at=? WHERE id=?",
            (new_ref, catalog.display_name(row["provider"], row["account_kind"],
                                           ss.mask_key(api_key)),
             time.time(), conn_id))
        store.delete(row["secret_ref"])
    return {"id": conn_id, "reverify_ok": True}


def delete(conn_id: str, store=None) -> dict:
    """撤回连接：清理凭据；不影响本地分析结果。"""
    from app.store import db
    store = store or ss.default_store()
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM model_connections WHERE id=?", (conn_id,)).fetchone()
        if row is None:
            raise LookupError("连接不存在")
        conn.execute("DELETE FROM model_connections WHERE id=?", (conn_id,))
    try:
        store.delete(row["secret_ref"])
    except ss.SecretStoreError:
        pass  # 凭据清理失败不阻塞连接删除；DB 引用已移除
    return {"id": conn_id, "deleted": True}
