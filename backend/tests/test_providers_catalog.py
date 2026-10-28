"""多厂商目录契约测试：8 厂商条目完整性、工厂可构造、payload 可序列化。"""
import json

import httpx
import pytest

from app.model_providers import catalog


def test_catalog_has_all_providers_with_unique_ids():
    ids = [s["id"] for s in catalog.catalog_payload()]
    assert ids == ["glm", "openai", "anthropic", "gemini",
                   "deepseek", "moonshot", "qwen", "doubao"]
    assert len(ids) == len(set(ids))


def test_every_provider_kind_constructs_matching_protocol():
    for spec in catalog.catalog_payload():
        for kind in spec["account_kinds"]:
            adapter = catalog.adapter_for(spec["id"], kind["id"],
                                          transport=httpx.MockTransport(
                                              lambda r: httpx.Response(200, json={})))
            assert adapter.protocol in ("openai_compatible", "anthropic")
            assert adapter.model   # 有默认候选模型


def test_glm_entries_match_legacy_constants():
    from app.model_providers.glm_api import BASE_URL as API_URL, VISION_MODEL_CANDIDATES
    from app.model_providers.glm_coding import BASE_URL as CODING_URL, MODEL
    spec = next(s for s in catalog.catalog_payload() if s["id"] == "glm")
    kinds = {k["id"]: k for k in spec["account_kinds"]}
    assert set(kinds) == {"api", "coding_plan"}
    assert kinds["api"]["default_model"] in VISION_MODEL_CANDIDATES
    assert kinds["coding_plan"]["default_model"] == MODEL
    a = catalog.adapter_for("glm", "api", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})))
    c = catalog.adapter_for("glm", "coding_plan", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})))
    assert a._base == API_URL and c._base == CODING_URL


def test_adapter_for_freezes_model_override():
    a = catalog.adapter_for("openai", "api", model_id="gpt-4o-mini",
                            transport=httpx.MockTransport(
                                lambda r: httpx.Response(200, json={})))
    assert a.model == "gpt-4o-mini"


def test_unknown_provider_or_kind_raises_value_error():
    with pytest.raises(ValueError):
        catalog.adapter_for("unknown", "api")
    with pytest.raises(ValueError):
        catalog.adapter_for("openai", "coding_plan")   # openai 无套餐类型


def test_catalog_payload_is_json_serializable():
    payload = catalog.catalog_payload()
    text = json.dumps(payload, ensure_ascii=False)
    assert "GLM" in text and "gpt-4o" in text
    for entry in payload:
        assert entry["label"] and entry["docs_url"] and "key_hint" in entry


def test_display_name_matches_legacy_glm_format():
    from app.model_connections.secret_store import mask_key
    masked = mask_key("sk-test-12345678")
    assert catalog.display_name("glm", "api", masked) == f"GLM 普通API · {masked}"
    assert catalog.display_name("glm", "coding_plan", masked) == f"GLM Coding Plan · {masked}"
    assert catalog.display_name("openai", "api", masked).startswith("OpenAI API · ")
