"""增强编排与审计（方案 §5/§7/§8）。

enhance(work, ...)：
  plan → evidence → provider 调用（预算台账 + DB 去重）→ 严格解析 →
  本地验证 → 应用受限修改 → 脱敏审计产物。
任何失败路径都保留本地结果；状态机字段写入 enhancement_summary.json。
不联网于 expert_engine 子进程内——本模块只在 worker 的 expert 服务层运行。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from app.model_connections import service as conn_svc
from app.model_connections import secret_store as ss
from app.model_providers import errors as E
from app.model_providers.base import ReviewRequest
from app.skill_runtime import policy as P
from app.skill_runtime.contracts import EvidenceManifest, parse_model_response
from app.enhancement import evidence as ev
from app.enhancement import planner
from app.enhancement.validator import validate_decision


def _adapter_for_connection(conn_row):
    if conn_row["account_kind"] == "api":
        from app.model_providers.glm_api import GlmApiAdapter
        return GlmApiAdapter()
    from app.model_providers.glm_coding import GlmCodingAdapter
    return GlmCodingAdapter()


def enhance(work: Path, *, project_id: str, job_id: str,
            provider=None, store=None) -> dict:
    """在 automatic 本地分析后执行模型复核。provider 可注入 FakeProvider 离线测试。

    返回 enhancement_summary dict 并落盘；本地结果永远保留。
    """
    summary = {"status": "disabled", "rounds": [], "requests": 0,
               "accepted": 0, "rejected": 0, "unresolved": 0,
               "source_line": "本地分析", "provider": None, "error": None}
    budget = P.Budget()
    ledger = P.BudgetLedger()

    if provider is None:
        conn = conn_svc.enabled_connection()
        if conn is None:
            _write(work, summary)
            return summary
        summary["provider"] = {"id": conn["id"], "kind": conn["account_kind"],
                               "model_id": conn["model_id"],
                               "config_revision": conn["config_revision"]}
        caps = json.loads(conn["capabilities_json"])
        if not caps.get("vision_ok"):
            summary["status"] = "failed"
            summary["error"] = "capability_missing"
            _write(work, summary)
            return summary
        store = store or ss.default_store()
        try:
            api_key = store.load(conn["secret_ref"])
        except ss.SecretStoreError as e:
            summary["status"], summary["error"] = "failed", str(e)
            _write(work, summary)
            return summary
        adapter = _adapter_for_connection(conn)
        conn_row = conn
    else:
        adapter = provider
        conn_row = None
        api_key = "fake"

    plans = planner.plan_anomalies(work, budget)
    if not plans:
        summary["status"] = "skipped"
        summary["source_line"] = "本地分析（本次无需增强复核）"
        _write(work, summary)
        return summary

    summary["status"] = "succeeded"
    summary["source_line"] = "本地分析"
    (work / "enhancement").mkdir(exist_ok=True)
    requests_log = []

    for plan in plans:
        if not budget.allows_call(ledger.used_calls):
            summary["status"] = "partial"
            summary.setdefault("limit_notes", []).append("vision_call_budget_exhausted")
            break
        t0 = time.time()
        try:
            person_ids, ball_ids, audio = ev.candidate_ids_for(work, plan["frames"])
            manifest, data_urls, image_paths = ev.build_evidence(
                work, project_id=project_id, job_id=job_id,
                question=plan["question"], frames=plan["frames"],
                allowed_person_ids=person_ids, allowed_ball_ids=ball_ids,
                audio_candidates=audio)
            prompt = ev.build_prompt(manifest, plan)
        except (ValueError, KeyError) as e:
            summary["rounds"].append({"kind": plan["kind"], "status": "skipped",
                                      "error": str(e)[:120]})
            continue

        request = ReviewRequest(question=plan["question"], prompt=prompt,
                                images=data_urls,
                                frame_ids=[str(f.frame) for f in manifest.frames],
                                image_paths=image_paths,
                                max_tokens=budget.max_tokens_per_request,
                                timeout_s=budget.request_timeout_s)
        status = "failed"
        error_code = None
        response_text = ""
        try:
            resp = adapter.review_images(api_key, request)
            ledger.spend(time.time() - t0, kind=plan["question"], ok=resp.ok)
            if not resp.ok:
                error_code = resp.error_code
                if resp.error_code == "timeout_unknown":
                    status = "timeout_unknown"
            else:
                response_text = resp.content
                status = "succeeded"
        except E.NotActivated:
            ledger.spend(time.time() - t0, kind=plan["question"], ok=False)
            error_code = status = "not_activated"
        except E.ProviderError as e:
            ledger.spend(time.time() - t0, kind=plan["question"], ok=False)
            error_code = e.code

        round_rec = {"kind": plan["kind"], "question": plan["question"],
                     "evidence_id": manifest.evidence_id, "status": status,
                     "error_code": error_code, "reason": plan["reason"]}
        # DB 去重记录（本地重复调用幂等）
        _record_request(job_id, manifest, conn_row, status)
        requests_log.append({
            "evidence_id": manifest.evidence_id, "question": plan["question"],
            "frames": [f.frame for f in manifest.frames],
            "status": status, "error_code": error_code,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "usage": None, "elapsed_s": round(time.time() - t0, 2)})

        if status != "succeeded":
            summary["rounds"].append(round_rec)
            if status in ("timeout_unknown", "not_activated"):
                summary["status"] = "partial"
                summary.setdefault("limit_notes", []).append(error_code)
            continue

        # 严格解析 + 本地验证
        try:
            parsed = parse_model_response(response_text, manifest)
        except ValueError as e:
            summary["rejected"] += 1
            round_rec["parse_error"] = str(e)[:160]
            summary["rounds"].append(round_rec)
            continue
        proposals = []
        for d in parsed.decisions:
            verdict = validate_decision(d, manifest, work)
            key = {"accepted": "accepted", "rejected": "rejected"}.get(verdict.status,
                                                                      "unresolved")
            summary[key] += 1
            proposals.append({"decision": d.model_dump(),
                              "verdict": {"status": verdict.status,
                                          "reason_code": verdict.reason_code,
                                          "detail": verdict.detail[:200]}})
        round_rec["proposals"] = proposals
        round_rec["needs_more_evidence"] = parsed.needs_more_evidence
        summary["rounds"].append(round_rec)

    # 任何失败/超时轮次都算 partial（本地结果保留，提示部分不可用）
    if summary["status"] == "succeeded" and any(
            r.get("status") not in ("succeeded",) for r in summary["rounds"]):
        summary["status"] = "partial"

    summary["requests"] = ledger.used_calls
    summary["remote_seconds"] = round(ledger.remote_seconds, 1)
    summary["budget"] = {"max_vision_calls": budget.max_vision_calls,
                         "used": ledger.used_calls}
    n_windows = len([r for r in summary["rounds"] if r.get("status") == "succeeded"])
    summary["source_line"] = (f"本地分析 · GLM 复核 {n_windows} 段"
                              if n_windows else "本地分析")
    if summary["status"] in ("failed", "partial") and any(
            r.get("error_code") for r in summary["rounds"]):
        summary["source_line"] += "（增强服务部分不可用，已保留本地结果）"
    (work / "enhancement" / "requests.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in requests_log))
    (work / "enhancement" / "proposals.json").write_text(
        json.dumps({"rounds": [r for r in summary["rounds"] if r.get("proposals")]},
                   ensure_ascii=False, indent=1))
    _write(work, summary)
    return summary


def _write(work: Path, summary: dict):
    (work / "enhancement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))


def _record_request(job_id: str, manifest: EvidenceManifest, conn_row, status: str):
    """写 enhancement_requests 行（唯一键去重；重复→deduped 状态标记）。"""
    from app.store import db
    skill = P.__name__  # 占位：真实 skill hash 由 manifest 模块提供
    from app.skill_runtime.manifest import skill_hash
    conn_id = conn_row["id"] if conn_row is not None else "fake"
    rev = conn_row["config_revision"] if conn_row is not None else 0
    evidence_hash = hashlib.sha256(
        manifest.model_dump_json().encode()).hexdigest()
    rid = db.new_id()
    with db.connect() as conn:
        try:
            conn.execute(
                "INSERT INTO enhancement_requests (id, job_id, connection_id, "
                "config_revision, evidence_hash, skill_hash, prompt_version, round, "
                "status, started_at, finished_at, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,1,?,?,?)",
                (rid, job_id, conn_id, rev, evidence_hash, skill_hash(),
                 P.PROMPT_VERSION, status, time.time(), time.time(), time.time()))
        except Exception:
            conn.execute(
                "UPDATE enhancement_requests SET status='deduped' WHERE job_id=? AND "
                "evidence_hash=? AND connection_id=? AND config_revision=? AND "
                "prompt_version=?",
                (job_id, evidence_hash, conn_id, rev, P.PROMPT_VERSION))
