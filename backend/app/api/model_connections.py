"""模型连接 API：唯一设置入口的后端。

安全基线（方案 §6.2/§6.3）：
- 本机 loopback：校验 Host；变更类请求校验 Origin（若提供）与本机会话 token。
- GET 只回 has_key/掩码别名，绝不回 Key。
- 日志不记录 HTTP body / Authorization。
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Header, HTTPException, Request, UploadFile, File
from pydantic import BaseModel, Field

from app.model_connections import service as svc
from app.model_connections import secret_store as ss

router = APIRouter(prefix="/api/model-connections", tags=["model-connections"])

# 进程级会话 token：浏览器首次 GET 获得，变更请求必须携带（防跨站写）
_session_token: str | None = None
_ALLOWED_HOSTS = {"127.0.0.1:8765", "localhost:8765", "127.0.0.1:8766", "localhost:8766"}


def check_origin(request: Request) -> "JSONResponse | None":
    """中间件用：loopback Host + 同源 Origin（在任何 body 解析之前执行）。
    中间件中不能抛 HTTPException（会变 500），直接返回 403 响应。"""
    from fastapi.responses import JSONResponse
    host = (request.headers.get("host") or "").lower()
    if host not in _ALLOWED_HOSTS:
        return JSONResponse({"detail": "仅限本机访问"}, status_code=403)
    origin = request.headers.get("origin")
    if origin:
        from urllib.parse import urlparse
        o = urlparse(origin)
        if f"{o.hostname}:{o.port or 80}" not in _ALLOWED_HOSTS:
            return JSONResponse({"detail": "跨站请求被拒绝"}, status_code=403)
    return None


def _guard(request: Request, token: str | None):
    """会话 token 校验（Origin/Host 已由中间件先行检查）。"""
    global _session_token
    if _session_token is None:
        _session_token = secrets.token_hex(24)
    if token != _session_token:
        raise HTTPException(403, "会话校验失败，请刷新页面重试")


@router.get("")
def list_all(request: Request):
    global _session_token
    if _session_token is None:
        _session_token = secrets.token_hex(24)
    # 读取类请求不强制 token（同源浏览器自动携带会话），但仍限本机
    host = (request.headers.get("host") or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise HTTPException(403, "仅限本机访问")
    return {"connections": svc.list_connections(), "session_token": _session_token,
            "consent_version": svc.CONSENT_VERSION,
            "coding_plan_doc": "https://docs.bigmodel.cn/cn/coding-plan/quick-start"}


class ProbeIn(BaseModel):
    provider: str = "glm"
    account_kind: str = Field(pattern="^(api|coding_plan)$")
    api_key: str = Field(min_length=8, max_length=512)


@router.post("/probe")
def probe(body: ProbeIn, request: Request,
          x_session_token: str | None = Header(default=None)):
    _guard(request, x_session_token)
    try:
        out = svc.probe(body.provider, body.account_kind, body.api_key)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except ss.SecretStoreError as e:
        raise HTTPException(503, str(e))
    body.api_key = ""   # 请求对象即刻失效
    return out


class CreateIn(BaseModel):
    probe_id: str
    enable: bool = False
    skill_id: str = "default"


@router.post("")
def create(body: CreateIn, request: Request,
           x_session_token: str | None = Header(default=None)):
    _guard(request, x_session_token)
    try:
        return svc.create_from_probe(body.probe_id, enable=body.enable, skill_id=body.skill_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except ss.SecretStoreError as e:
        raise HTTPException(503, str(e))


class PatchIn(BaseModel):
    enabled: bool | None = None
    reverify: bool = False
    skill_id: str | None = None
    rotate_key: str | None = Field(default=None, max_length=512)
    expected_revision: int = Field(ge=1)


@router.patch("/{conn_id}")
def patch(conn_id: str, body: PatchIn, request: Request,
          x_session_token: str | None = Header(default=None)):
    _guard(request, x_session_token)
    try:
        if body.rotate_key is not None:
            return svc.rotate(conn_id, body.rotate_key, expected_revision=body.expected_revision)
        out = svc.patch(conn_id, enabled=body.enabled, reverify=body.reverify, expected_revision=body.expected_revision, skill_id=body.skill_id)
    except LookupError:
        raise HTTPException(404, "连接不存在")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except ss.SecretStoreError as e:
        raise HTTPException(503, str(e))
    body.rotate_key = ""
    if out.get("conflict"):
        raise HTTPException(409, {"message": "配置已被其他页面修改，请刷新",
                                  "current_revision": out["current_revision"]})
    return out


@router.delete("/{conn_id}")
def remove(conn_id: str, request: Request,
           x_session_token: str | None = Header(default=None)):
    _guard(request, x_session_token)
    try:
        return svc.delete(conn_id)
    except LookupError:
        raise HTTPException(404, "连接不存在")


@router.get("/skills/default")
def default_skill():
    from app.model_connections import skills
    return skills.info(skills.load())


@router.get("/skills/{skill_id}")
def skill_info(skill_id: str):
    from app.model_connections import skills
    try:
        return skills.info(skills.load(skill_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.post("/skills")
async def upload_skill(request: Request, file: UploadFile = File(...),
                       x_session_token: str | None = Header(default=None)):
    _guard(request, x_session_token)
    from app.model_connections import skills
    data = await file.read(skills.MAX_BYTES + 1)
    try:
        return skills.save(file.filename or '', data)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
