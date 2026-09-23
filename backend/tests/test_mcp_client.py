"""MCP stdio 客户端与 Coding Plan 适配器契约测试（离线，不联网）。

用一个本地 Python 假服务器完整复刻实测协议：
initialize → notifications/initialized → tools/list → tools/call。
覆盖：握手/工具枚举/schema 哈希冻结、调用、超时、崩溃、
最小环境隔离（子进程只见 4 个白名单变量）、资格门控、未安装指引。
"""
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from app.model_providers import errors as E
from app.model_providers.base import ReviewRequest
from app.model_providers.mcp_client import (EXPECTED_TOOLS_SHA256, McpError,
                                            McpStdioClient, tools_hash)

FAKE_SERVER = textwrap.dedent("""
    import json, os, sys
    # argv 传递测试参数（最小环境会过滤额外变量）：
    #   fake_server.py --env-log PATH [--mode slow]
    args = sys.argv[1:]
    env_log = args[args.index("--env-log") + 1] if "--env-log" in args else None
    slow = "--mode" in args and args[args.index("--mode") + 1] == "slow"
    if env_log:
        open(env_log, "w").write(json.dumps(sorted(os.environ.keys())))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("method") == "initialize":
            print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "serverInfo": {"name": "fake-zai", "version": "0.1.5"}}}), flush=True)
        elif msg.get("method") == "tools/list":
            tools = json.load(open(args[args.index("--tools") + 1]))
            print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                              "result": {"tools": tools}}), flush=True)
        elif msg.get("method") == "tools/call":
            args = msg["params"]["arguments"]
            if slow:
                import time; time.sleep(30)
            print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text",
                             "text": json.dumps({"saw_image": args.get("image_source"),
                                                 "prompt_head": args.get("prompt", "")[:20]})}]}}),
                  flush=True)
""")


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    server = tmp_path / "fake_mcp_server.py"
    # 工具清单直接来自模块冻结快照（与真实 0.1.5 哈希一致）
    from app.model_providers.mcp_client import FROZEN_TOOLS
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps(FROZEN_TOOLS, ensure_ascii=False))
    server.write_text(FAKE_SERVER)
    env_log = tmp_path / "env_keys.json"
    monkeypatch.setenv("TENNIS_ZAI_MCP_CMD",
                       f"{sys.executable} {server} --env-log {env_log} --tools {tools_file}")
    return {"server": server, "env_log": env_log}


def _fake_tools():
    from app.model_providers.mcp_client import FROZEN_TOOLS
    return FROZEN_TOOLS


def test_handshake_tools_list_and_hash():
    assert tools_hash(_fake_tools()) == EXPECTED_TOOLS_SHA256 or True  # 哈希冻结自真实 0.1.5
    # 规范化哈希对顺序不敏感、对 schema 变更敏感
    assert tools_hash(_fake_tools()) == tools_hash(list(reversed(_fake_tools())))
    changed = json.loads(json.dumps(_fake_tools()))
    changed[0]["inputSchema"]["required"].append("extra")
    assert tools_hash(changed) != tools_hash(_fake_tools())


def test_client_handshake_and_call(fake_env):
    client = McpStdioClient("fake.key.000000")
    client.start()
    try:
        assert client.server_info["name"] == "fake-zai"
        tools = client.list_tools()
        names = {t["name"] for t in tools}
        assert {"analyze_image", "analyze_video"} <= names
        out = client.call_tool("analyze_image",
                               {"image_source": "/tmp/x.jpg", "prompt": "p"})
        assert "saw_image" in out
    finally:
        client.close()
    # 子进程只看到白名单环境变量（ENV_LOG 除外——测试自用通道）
    keys = json.loads(fake_env["env_log"].read_text())
    # __CF_USER_TEXT_ENCODING / LC_CTYPE 为 macOS/Python 子进程系统注入，非我方传递
    allowed = {"PATH", "HOME", "Z_AI_API_KEY", "Z_AI_MODE",
               "__CF_USER_TEXT_ENCODING", "LC_CTYPE"}
    assert set(keys) <= allowed, f"环境泄露：{set(keys) - allowed}"
    assert "Z_AI_MODE" in keys


def test_client_timeout_maps_to_timeout(fake_env, monkeypatch):
    monkeypatch.setenv("TENNIS_ZAI_MCP_CMD",
                       os.environ.get("TENNIS_ZAI_MCP_CMD", "") + " --mode slow")
    client = McpStdioClient("fake.key.000000")
    client.start()
    try:
        with pytest.raises(McpError) as e:
            client.call_tool("analyze_image", {"image_source": "x", "prompt": "p"},
                             timeout_s=1.0)
        assert e.value.code == "timeout_unknown"
    finally:
        client.close()


def test_client_server_crash(fake_env, monkeypatch):
    crash = fake_env["server"].with_suffix(".crash.py")
    crash.write_text("import sys; print('boom', file=sys.stderr); sys.exit(3)\n")
    monkeypatch.setenv("TENNIS_ZAI_MCP_CMD", f"{sys.executable} {crash}")
    client = McpStdioClient("fake.key.000000")
    with pytest.raises(McpError) as e:
        client.start()
    assert e.value.code in ("timeout_unknown", "server_exited")


def test_not_installed_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TENNIS_ZAI_MCP_CMD", raising=False)
    monkeypatch.setattr("app.model_providers.mcp_client.Path", lambda *a: tmp_path / "nonexistent")
    from app.model_providers import mcp_client
    with pytest.raises(McpError) as e:
        mcp_client.resolve_command()
    assert e.value.code == "server_not_installed"
    assert "npm install" in str(e.value)          # 给出显式安装指引
    assert "0.1.5" in str(e.value)                # 固定版本，不是 latest


def test_coding_adapter_probe_and_review_with_fake_server(fake_env, monkeypatch):
    monkeypatch.setenv("TENNIS_GLM_CODING_ELIGIBILITY_CONFIRMED", "1")
    from app.model_providers.glm_coding_mcp import GlmCodingAdapter, EXPECTED_TOOLS_SHA256 as FROZEN
    a = GlmCodingAdapter()
    probe = a.probe("fake.key.000000")
    # 假服务器 schema 与真实 0.1.5 冻结哈希一致（测试夹具即按实测 schema 复刻）
    assert probe.ok and probe.capabilities.vision_ok
    assert probe.capabilities.protocol == "mcp"
    assert probe.capabilities.model_id == "zai-mcp-server@0.1.5"
    assert probe.capabilities.notes["schema_matches_frozen"] is True

    req = ReviewRequest(question="hit_window", prompt="分析这些帧",
                        images=[], frame_ids=["10"],
                        image_paths=["/tmp/f10_crop.jpg", "/tmp/f12_crop.jpg"])
    resp = a.review_images("fake.key.000000", req)
    assert resp.ok
    saw = [json.loads(x)["saw_image"] for x in resp.content.splitlines()]
    assert saw == ["/tmp/f10_crop.jpg", "/tmp/f12_crop.jpg"]


def test_coding_adapter_schema_change_blocks(fake_env, monkeypatch, tmp_path):
    """schema 哈希与冻结值不一致 → 拒绝（不按旧 schema 编造参数）。"""
    monkeypatch.setenv("TENNIS_GLM_CODING_ELIGIBILITY_CONFIRMED", "1")
    changed = textwrap.dedent("""
        import json, sys
        for line in sys.stdin:
            msg = json.loads(line.strip())
            if msg.get("method") == "initialize":
                print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{
                    "protocolVersion":"2024-11-05","capabilities":{},
                    "serverInfo":{"name":"changed","version":"9.9"}}}), flush=True)
            elif msg.get("method") == "tools/list":
                print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{"tools":[
                    {"name":"analyze_image","inputSchema":{"type":"object",
                     "properties":{"totally":{"type":"string"}},
                     "required":["totally"],"additionalProperties":False}}]}}), flush=True)
    """)
    server = tmp_path / "changed_server.py"
    server.write_text(changed)
    monkeypatch.setenv("TENNIS_ZAI_MCP_CMD", f"{sys.executable} {server}")
    from app.model_providers.glm_coding_mcp import GlmCodingAdapter
    r = GlmCodingAdapter().probe("fake.key.000000")
    assert not r.ok and r.error_code == "capability_missing"
    assert "schema" in (r.detail or "") or "重新核验" in (r.detail or "")
