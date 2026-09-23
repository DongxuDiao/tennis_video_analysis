"""人工复核：追加式 correction 日志 → 物化 reviewed 视图。自动结果不可变。"""
from __future__ import annotations

import json
import time
from typing import Any

from app.store import db, db as dbm

OP_TYPES = ("add_event", "modify_event", "exclude_event", "modify_ball_point")


def current_revision(conn, run_id: str) -> int:
    row = conn.execute("SELECT COALESCE(MAX(revision),0) AS r FROM corrections WHERE run_id=?",
                       (run_id,)).fetchone()
    return int(row["r"])


def list_corrections(conn, run_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM corrections WHERE run_id=? ORDER BY revision", (run_id,)).fetchall()
    return [{"id": r["id"], "revision": r["revision"], "created_at": r["created_at"],
             "op": json.loads(r["op_json"])} for r in rows]


def append_correction(conn, run_id: str, op: dict, expected_revision: int) -> dict:
    cur_rev = current_revision(conn, run_id)
    if expected_revision != cur_rev:
        raise RevisionConflict(cur_rev)
    op = _validate_op(op)
    revision = cur_rev + 1
    conn.execute("INSERT INTO corrections (run_id, revision, op_json, created_at) VALUES (?,?,?,?)",
                 (run_id, revision, json.dumps(op, ensure_ascii=False), time.time()))
    return {"run_id": run_id, "revision": revision}


class RevisionConflict(Exception):
    def __init__(self, current: int):
        self.current = current
        super().__init__(f"版本冲突：当前 revision={current}")


def _validate_op(op: dict) -> dict:
    t = op.get("op")
    if t not in OP_TYPES:
        raise ValueError(f"未知操作类型 {t}，允许：{OP_TYPES}")
    if t in ("modify_event", "exclude_event") and not op.get("event_id"):
        raise ValueError("modify/exclude 需要事件 id")
    if t == "add_event" and not op.get("event", {}).get("type"):
        raise ValueError("add_event 需要事件类型")
    if t == "modify_ball_point" and "point" not in op:
        raise ValueError("modify_ball_point 需要 point")
    return op


def apply_ops(events: list[dict], ops: list[dict]) -> list[dict]:
    """把 correction 日志物化为 reviewed 事件视图（不改动 automatic 数据）。"""
    out = json.loads(json.dumps(events))  # 深拷贝
    by_op_id: dict[int, dict] = {}
    undone: set[int] = set()
    # 先找出被撤销的操作
    for o in ops:
        op = o["op"]
        if op.get("op") == "undo" and op.get("target_op_id"):
            undone.add(op["target_op_id"])
    for o in ops:
        if o["id"] in undone:
            continue
        op = o["op"]
        by_op_id[o["id"]] = op
        if op["op"] == "add_event":
            ev = dict(op["event"])
            ev.update({"origin": "human_added", "review_status": "human",
                       "added_by_op": o["id"], "id": f"manual-{o['id']}"})
            out.append(ev)
        elif op["op"] == "modify_event":
            for ev in out:
                if ev["id"] == op["event_id"]:
                    for k, v in op.get("changes", {}).items():
                        if k in ("id", "run_id"):
                            continue
                        ev[k] = v
                    ev["origin"] = "reviewed"
                    ev["review_status"] = "reviewed"
                    ev["modified_by_op"] = o["id"]
        elif op["op"] == "exclude_event":
            for ev in out:
                if ev["id"] == op["event_id"]:
                    ev["review_status"] = "excluded"
                    ev["excluded_by_op"] = o["id"]
                    ev["exclude_reason"] = op.get("reason", "")
    out.sort(key=lambda e: e.get("source_time_ms", 0))
    return out


def undo_correction(conn, run_id: str, target_op_id: int) -> dict:
    row = conn.execute("SELECT * FROM corrections WHERE run_id=? AND id=?", (run_id, target_op_id)).fetchone()
    if row is None:
        raise ValueError("目标操作不存在")
    cur_rev = current_revision(conn, run_id)
    op = {"op": "undo", "target_op_id": target_op_id}
    revision = cur_rev + 1
    conn.execute("INSERT INTO corrections (run_id, revision, op_json, created_at) VALUES (?,?,?,?)",
                 (run_id, revision, json.dumps(op, ensure_ascii=False), time.time()))
    return {"run_id": run_id, "revision": revision, "undid": target_op_id}


def reviewed_summary(auto_summary: dict, reviewed_events: list[dict]) -> dict:
    hits = [e for e in reviewed_events
            if e["type"] == "player_hit" and e.get("review_status") != "excluded"]
    auto_hits = [e for e in reviewed_events
                 if e["type"] == "player_hit" and e.get("origin") == "automatic"
                 and e.get("review_status") != "excluded"]
    reviewed_marked = [e for e in hits if e.get("review_status") in ("reviewed", "human")]
    return {
        "player_hit_total_reviewed": len(hits),
        "player_hit_from_automatic": len(auto_hits),
        "player_hit_human_added": len(hits) - len(auto_hits),
        "player_hit_marked_reviewed": len(reviewed_marked),
        "stroke_counts_reviewed": {
            "forehand": sum(1 for e in hits if e["stroke_type"] == "forehand"),
            "backhand": sum(1 for e in hits if e["stroke_type"] == "backhand"),
            "unknown": sum(1 for e in hits if e["stroke_type"] == "unknown"),
        },
        "note": "摘要仅使用支持该结论的数据；unknown 不从分母中消失",
    }
