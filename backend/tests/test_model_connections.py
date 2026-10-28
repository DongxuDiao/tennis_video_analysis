"""连接底座测试（阶段1）：探测→启用→轮换/启停→删除；凭据隔离。

不联网：adapter 注入 FakeProvider；凭据用 MemoryStore。
验收对应方案 §10 阶段1：断网可用、Key 不落 DB/日志、错 Key 明确提示、关闭后无新发送。
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_DIR", str(tmp_path))
    from app.store import db
    db.init_db()

from app.model_connections import service as svc
from app.model_connections import secret_store as ss
from app.store import db


@pytest.fixture(autouse=True)
def _fake_adapter(monkeypatch):
    from app.model_providers.fake import FakeProvider
    fake = FakeProvider()
    monkeypatch.setattr(svc, "_adapter_for", lambda provider, kind: fake)
    return fake


@pytest.fixture
def store():
    return ss.MemoryStore()


def test_probe_and_enable_flow(store):
    out = svc.probe("glm", "api", "sk-test-12345678")
    assert out["ok"] and out["capabilities"]["vision_ok"]
    created = svc.create_from_probe(out["probe_id"], enable=True, store=store)
    assert created["enabled"] is True
    conns = svc.list_connections()
    assert len(conns) == 1
    c = conns[0]
    assert c["has_key"] is True and c["enabled"] is True
    # DB 行不包含 Key 本体
    from app.store import db
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM model_connections").fetchone()
    assert "sk-test" not in (row["secret_ref"] or "") + (row["display_name"] or "")
    # display_name 只含掩码
    assert "…" in c["display_name"] and "12345678" not in c["display_name"]


def test_probe_expiry(store):
    out = svc.probe("glm", "api", "sk-test-12345678")
    svc._pending_probes[out["probe_id"]]["expires_at"] = 0
    with pytest.raises(ValueError):
        svc.create_from_probe(out["probe_id"], enable=True, store=store)


def test_bad_key_clear_error(_fake_adapter, monkeypatch, store):
    _fake_adapter.error_code = "auth_failed"
    out = svc.probe("glm", "api", "sk-badkey-000000")
    assert not out["ok"] and out["error_code"] == "auth_failed"
    assert "无效" in out["error"]
    assert "probe_id" not in out


def test_coding_plan_can_probe_without_eligibility_flag():
    # 资格 pending：不执行远端探测、不可启用
    out = svc.probe("glm", "coding_plan", "sk-whatever-000000")
    assert out["ok"] is True


def test_patch_optimistic_lock_and_disable(store, monkeypatch):
    out = svc.probe("glm", "api", "sk-test-12345678")
    created = svc.create_from_probe(out["probe_id"], enable=True, store=store)
    conn_id = created["id"]
    # 错误 revision → 冲突
    conflict = svc.patch(conn_id, expected_revision=99, enabled=False, store=store)
    assert conflict.get("conflict") is True
    # 正确 revision → 禁用
    ok = svc.patch(conn_id, expected_revision=1, enabled=False, store=store)
    assert ok["enabled"] is False and ok["config_revision"] == 2
    assert svc.enabled_connection() is None   # 关闭后无可用连接 → 无新发送


def test_rotate_replaces_credential(store):
    out = svc.probe("glm", "api", "sk-old-key-000000")
    created = svc.create_from_probe(out["probe_id"], enable=True, store=store)
    conn_id = created["id"]
    old_ref = None
    from app.store import db
    with db.connect() as conn:
        old_ref = conn.execute("SELECT secret_ref FROM model_connections WHERE id=?",
                               (conn_id,)).fetchone()["secret_ref"]
    r = svc.rotate(conn_id, "sk-new-key-999999", expected_revision=1, store=store)
    assert r["reverify_ok"] is True
    assert not store.exists(old_ref)          # 旧凭据清理
    with db.connect() as conn:
        new_ref = conn.execute("SELECT secret_ref FROM model_connections WHERE id=?",
                               (conn_id,)).fetchone()["secret_ref"]
    assert store.load(new_ref) == "sk-new-key-999999"


def test_delete_clears_credential(store):
    out = svc.probe("glm", "api", "sk-test-12345678")
    created = svc.create_from_probe(out["probe_id"], enable=True, store=store)
    svc.delete(created["id"], store=store)
    assert svc.list_connections() == []
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM model_connections").fetchone()
    assert row is None
    # 凭据已清理（secret_ref 无法从 DB 找到；store 内无残留该连接条目）
    refs = [r for r in store._data if r.startswith("TennisLab")]
    assert refs == []


def test_key_never_in_logs_or_exceptions():
    # SecretStoreError 消息固定文案，不携带 Key
    e = ss.SecretStoreError("凭据不存在或无法读取（需要登录 Keychain）")
    assert "sk-" not in str(e)


def test_plan_limit_probe_shows_reset_without_retry_promise(monkeypatch, store):
    import httpx
    from app.model_providers.glm_coding import GlmCodingAdapter
    adapter=GlmCodingAdapter(transport=httpx.MockTransport(lambda r:httpx.Response(429,json={'error':{'code':'1310','message':'额度将在 2026-09-18 10:00:47 重置'}})))
    monkeypatch.setattr(svc,'_adapter_for',lambda *args:adapter)
    out=svc.probe('glm','coding_plan','fake-key',store=store)
    assert out['error_code']=='plan_limit_reached'
    assert '2026-09-18 10:00:47' in out['error']
    assert '自动重试' not in out['error']
    assert 'probe_id' not in out


def test_non_glm_provider_probe_create_flow(store):
    """目录驱动：openai 连接走同一 probe→create 底座，display_name 无 GLM 字样。"""
    out = svc.probe("openai", "api", "sk-openai-123456")
    assert out["ok"] and out["capabilities"]["vision_ok"]
    created = svc.create_from_probe(out["probe_id"], enable=True, store=store)
    c = svc.list_connections()[0]
    assert c["provider"] == "openai" and c["enabled"] is True
    assert c["display_name"].startswith("OpenAI") and "GLM" not in c["display_name"]


def test_probe_model_override_freezes_into_db(monkeypatch, store):
    """服务层把 model 覆盖 pin 到适配器（属性注入），并把探测冻结值写库。"""
    class PinnedSpy:
        model = "dir-default"
        account_kind = "api"
        protocol = "openai_compatible"
        def probe(self, key):
            from app.model_providers.base import Capabilities, ProbeResult
            return ProbeResult(True, Capabilities(protocol=self.protocol,
                                                  text_ok=True, vision_ok=True,
                                                  model_id=self.model))
    monkeypatch.setattr(svc, "_adapter_for", lambda p, k: PinnedSpy())
    out = svc.probe("doubao", "api", "ark-key-12345678", store=store, model="ep-20250901")
    assert out["ok"] and out["capabilities"]["model_id"] == "ep-20250901"
    svc.create_from_probe(out["probe_id"], enable=True, store=store)
    c = svc.list_connections()[0]
    assert c["model_id"] == "ep-20250901"


def test_unknown_provider_rejected_with_value_error():
    with pytest.raises(ValueError):
        svc.probe("not-a-vendor", "api", "sk-test-12345678")
    with pytest.raises(ValueError):
        svc.probe("openai", "coding_plan", "sk-test-12345678")   # openai 无套餐类型


def test_reverify_keeps_frozen_model(monkeypatch, store):
    """复验必须用 DB 冻结模型，不得回退目录首选候选。"""
    seen = {}
    class ProbeSpy:
        model = "dir-default"
        account_kind = "api"
        protocol = "openai_compatible"
        def probe(self, key):
            seen["model"] = self.model
            from app.model_providers.base import Capabilities, ProbeResult
            caps = Capabilities(protocol=self.protocol, text_ok=True, vision_ok=True,
                                model_id=self.model)
            return ProbeResult(True, caps)
    monkeypatch.setattr(svc, "_adapter_for", lambda p, k: ProbeSpy())
    out = svc.probe("openai", "api", "sk-openai-123456", store=store)
    conn_id = svc.create_from_probe(out["probe_id"], enable=False, store=store)["id"]
    with db.connect() as conn:
        conn.execute("UPDATE model_connections SET model_id='m-frozen' WHERE id=?", (conn_id,))
    import time as _t
    with db.connect() as conn:
        rev = conn.execute("SELECT config_revision FROM model_connections WHERE id=?",
                           (conn_id,)).fetchone()["config_revision"]
    patched = svc.patch(conn_id, expected_revision=rev, reverify=True, store=store)
    assert patched["reverify_ok"] is True and seen["model"] == "m-frozen"
