"""MCP stdio 客户端（@z_ai/mcp-server，官方 Claude 式接入）。

实测协议（2026-09-17，@z_ai/mcp-server@0.1.5）：
- newline 分隔的 JSON-RPC 2.0；initialize → notifications/initialized → tools/list → tools/call。
- 服务器启动即校验 Z_AI_API_KEY 格式；tools/list 不消耗模型额度。
- 视觉工具实名 analyze_image / analyze_video（不是设计文档假设的 image_analysis），
  输入为本地文件路径或 URL（不是 data URL）。

安全与运行约定（方案 §5.2）：
- 固定版本 0.1.5；不做运行时自动下载安装（npx -y latest 禁止）。
- 子进程仅获得 PATH/HOME/Z_AI_API_KEY/Z_AI_MODE 四个环境变量，不读用户 MCP 配置。
- 关闭 stdin → 有界等待 → 清理整个进程组。
- 工具 schema 哈希冻结；变更 → capability 待重测，禁止按旧 schema 编造参数。
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

PINNED_VERSION = "0.1.5"
PACKAGE = "@z_ai/mcp-server"
PROTOCOL_VERSION = "2024-11-05"

# 2026-09-17 实测 @z_ai/mcp-server@0.1.5 的 tools/list（忠实快照，供哈希与测试复用）
FROZEN_TOOLS: list[dict] = [
 {
  "name": "analyze_data_visualization",
  "inputSchema": {
   "type": "object",
   "properties": {
    "image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "prompt": {
     "type": "string",
     "description": "What insights or information you want to extract from this visualization."
    },
    "analysis_focus": {
     "type": "string",
     "description": "Optional: specify what to focus on (e.g., 'trends', 'anomalies', 'comparisons', 'performance metrics'). Leave empty for comprehensive analysis."
    }
   },
   "required": [
    "image_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "analyze_image",
  "inputSchema": {
   "type": "object",
   "properties": {
    "image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "prompt": {
     "type": "string",
     "description": "Detailed description of what you want to analyze, extract, or understand from the image. Be specific about your requirements."
    }
   },
   "required": [
    "image_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "analyze_video",
  "inputSchema": {
   "type": "object",
   "properties": {
    "video_source": {
     "type": "string",
     "description": "Local file path or remote URL to the video (supports MP4, MOV, M4V)"
    },
    "prompt": {
     "type": "string",
     "description": "Detailed text prompt describing what to analyze, extract, or understand from the video"
    }
   },
   "required": [
    "video_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "diagnose_error_screenshot",
  "inputSchema": {
   "type": "object",
   "properties": {
    "image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "prompt": {
     "type": "string",
     "description": "Description of what you need help with regarding this error. Include any relevant context about when it occurred."
    },
    "context": {
     "type": "string",
     "description": "Optional: additional context about when the error occurred (e.g., 'during npm install', 'when running the app', 'after deployment'). Helps with more accurate diagnosis."
    }
   },
   "required": [
    "image_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "extract_text_from_screenshot",
  "inputSchema": {
   "type": "object",
   "properties": {
    "image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "prompt": {
     "type": "string",
     "description": "Instructions for text extraction. Specify what type of text to extract and any formatting requirements."
    },
    "programming_language": {
     "type": "string",
     "description": "Optional: specify the programming language if the screenshot contains code (e.g., 'python', 'javascript', 'java'). Leave empty for auto-detection or non-code text."
    }
   },
   "required": [
    "image_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "ui_diff_check",
  "inputSchema": {
   "type": "object",
   "properties": {
    "expected_image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "actual_image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "prompt": {
     "type": "string",
     "description": "Instructions for the comparison. Specify what aspects to focus on or what level of detail is needed."
    }
   },
   "required": [
    "expected_image_source",
    "actual_image_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "ui_to_artifact",
  "inputSchema": {
   "type": "object",
   "properties": {
    "image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "output_type": {
     "type": "string",
     "enum": [
      "code",
      "prompt",
      "spec",
      "description"
     ],
     "description": "Type of output to generate. Options: 'code' (generate frontend code), 'prompt' (generate AI prompt for recreating this UI), 'spec' (generate design specification document), 'description' (natural language description of the UI)."
    },
    "prompt": {
     "type": "string",
     "description": "Detailed instructions describing what to generate from this UI image. Should clearly state the desired output and any specific requirements."
    }
   },
   "required": [
    "image_source",
    "output_type",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 },
 {
  "name": "understand_technical_diagram",
  "inputSchema": {
   "type": "object",
   "properties": {
    "image_source": {
     "type": "string",
     "description": "Local file path or remote URL to the image"
    },
    "prompt": {
     "type": "string",
     "description": "What you want to understand or extract from this diagram."
    },
    "diagram_type": {
     "type": "string",
     "description": "Optional: specify the diagram type if known (e.g., 'architecture', 'flowchart', 'uml', 'er-diagram', 'sequence'). Leave empty for auto-detection."
    }
   },
   "required": [
    "image_source",
    "prompt"
   ],
   "additionalProperties": False,
   "$schema": "http://json-schema.org/draft-07/schema#"
  }
 }
]



_ALLOWED_ENV = ("PATH", "HOME", "Z_AI_API_KEY", "Z_AI_MODE")


class McpError(RuntimeError):
    def __init__(self, message: str, code: str = "mcp_error"):
        super().__init__(message)
        self.code = code


def _minimal_env(api_key: str) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env["Z_AI_API_KEY"] = api_key
    env["Z_AI_MODE"] = "ZHIPU"
    return env


def resolve_command() -> list[str]:
    """解析服务器启动命令。优先级：
    1) TENNIS_ZAI_MCP_CMD（显式覆盖，如 "node /path/index.js"）
    2) 仓库显式安装：backend/node_modules/@z_ai/mcp-server/build/index.js
       （scripts/install_mcp.sh 显式安装，运行时不自动下载）
    找不到 → 明确报错与安装指引。
    """
    override = os.environ.get("TENNIS_ZAI_MCP_CMD")
    if override:
        return shlex.split(override)
    repo_install = Path(__file__).resolve().parents[2] / "node_modules" / \
        "@z_ai" / "mcp-server" / "build" / "index.js"
    if repo_install.is_file():
        return ["node", str(repo_install)]
    raise McpError(
        f"未找到 {PACKAGE}@{PINNED_VERSION}。请先显式安装：cd backend && npm install "
        f"{PACKAGE}@{PINNED_VERSION} --no-save（或设置 TENNIS_ZAI_MCP_CMD 指向已安装的 "
        "index.js）。运行期间不会自动下载。", code="server_not_installed")


def canonical_tools_blob(tools: list[dict]) -> str:
    slim = [{"name": t.get("name", ""),
             "inputSchema": t.get("inputSchema", {})}
            for t in sorted(tools, key=lambda t: t.get("name", ""))]
    return json.dumps(slim, ensure_ascii=False, sort_keys=True)


def tools_hash(tools: list[dict]) -> str:
    return hashlib.sha256(canonical_tools_blob(tools).encode()).hexdigest()


EXPECTED_TOOLS_SHA256 = tools_hash(FROZEN_TOOLS)  # 2026-09-17 实测 @z_ai/mcp-server@0.1.5


class McpStdioClient:
    """极简同步 JSON-RPC/stdio 客户端（无 SDK 依赖）。"""

    def __init__(self, api_key: str, *, command: list[str] | None = None,
                 start_timeout_s: float = 15.0):
        self._cmd = command if command is not None else resolve_command()
        self._api_key = api_key
        self._start_timeout_s = start_timeout_s
        self.proc: subprocess.Popen | None = None
        self._next_id = 0
        self._responses: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stderr_tail: list[str] = []
        self.server_info: dict = {}

    # ---- 生命周期 ----
    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(
                self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=_minimal_env(self._api_key),
                start_new_session=True, cwd=str(Path(__file__).resolve().parents[2]))
        except FileNotFoundError as e:
            raise McpError(f"无法启动 MCP 服务器：{e}", code="server_spawn_failed") from None
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        # 启动失败（如 Key 格式非法）会在 initialize 超时/进程退出中暴露
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "tennis-lab", "version": "0.1"}},
            timeout_s=self._start_timeout_s)
        self.server_info = result.get("serverInfo", {})
        self._notify("notifications/initialized")

    def close(self, grace_s: float = 5.0) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()

    # ---- IO ----
    def _read_stdout(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg:
                with self._cond:
                    self._responses[int(msg["id"])] = msg
                    self._cond.notify_all()

    def _read_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())
            del self._stderr_tail[:-20]

    def _request(self, method: str, params: dict, timeout_s: float) -> dict:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
        payload = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                              "params": params})
        if self.proc is None or self.proc.stdin is None:
            raise McpError("MCP 会话未启动", code="not_started")
        try:
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            raise McpError("MCP 服务器已退出：" +
                           " ".join(self._stderr_tail[-3:])[:200],
                           code="server_exited") from None
        deadline = time.time() + timeout_s
        with self._cond:
            while rid not in self._responses:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise McpError(f"MCP 请求超时：{method}",
                                   code="timeout_unknown")
                self._cond.wait(remaining)
            msg = self._responses.pop(rid)
        if "error" in msg and msg["error"]:
            raise McpError(f"MCP 错误：{str(msg['error'])[:200]}",
                           code="mcp_error")
        return msg.get("result", {})

    def _notify(self, method: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    # ---- 协议操作 ----
    def list_tools(self, timeout_s: float = 20.0) -> list[dict]:
        return self._request("tools/list", {}, timeout_s).get("tools", [])

    def call_tool(self, name: str, arguments: dict,
                  timeout_s: float = 90.0) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments},
                               timeout_s)
        parts = []
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if result.get("isError"):
            raise McpError("工具执行失败：" + " ".join(parts)[:200],
                           code="tool_failed")
        return "\n".join(parts)
