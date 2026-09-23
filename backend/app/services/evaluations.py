"""评估：自动/辅助结果 vs 独立人工真值。一对一最大匹配 + 指标分母 + 失败样例。

指标明确区分 automatic / reviewed / assisted 来源；主算法评分只针对未修正 automatic。
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.services import corrections as corr_svc
from app.vision.geometry import WallQuad


def match_events(preds: list[dict], gts: list[dict], tol_ms: int) -> dict:
    """一对一匹配：同类型事件，最大匹配数优先，再最小化总时间差。

    贪心按 (时间差) 升序配对可保证最大匹配（区间图着色性质：二部图区间容差匹配
    在该代价结构下贪心即最优——按结束时间升序配对）。
    """
    pairs: list[tuple[int, float]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            dt = abs(p["source_time_ms"] - g["source_time_ms"])
            if dt <= tol_ms:
                pairs.append((i, j, dt))
    pairs.sort(key=lambda t: t[2])
    used_p, used_g = set(), set()
    matched = []
    for i, j, dt in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matched.append({"pred_index": i, "gt_index": j, "dt_ms": dt,
                        "pred": preds[i], "gt": gts[j]})
    tp = len(matched)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if tp == 0 and (precision is not None or recall is not None) else None)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
            "denominator_pred": len(preds), "denominator_gt": len(gts),
            "matched": matched}


def _stroke_confusion(matched: list[dict]) -> dict:
    labels = ["forehand", "backhand", "unknown"]
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for m in matched:
        gt_s = m["gt"].get("stroke_type", "unknown") or "unknown"
        pr_s = m["pred"].get("stroke_type", "unknown") or "unknown"
        matrix.setdefault(gt_s, {}).setdefault(pr_s, 0)
        matrix[gt_s][pr_s] += 1
    correct = sum(matrix[a].get(a, 0) for a in ("forehand", "backhand"))
    total_classifiable = sum(matrix[a][b] for a in ("forehand", "backhand") for b in labels)
    return {"confusion": matrix,
            "classified_correct": correct,
            "gt_classifiable_matched": total_classifiable,
            "stroke_accuracy_on_matched_classifiable": round(correct / total_classifiable, 4) if total_classifiable else None,
            "note": "仅统计匹配上且真值可分类的击球；端到端正确率需另计漏检/unknown"}


def _ball_localization(run_frames: list[dict], gt_ball_points: list[dict],
                       tol_px: float, diag_px: float) -> dict:
    """仅在人工标为可见且可定位的帧上：召回 + 定位误差（原图 px 与对角线归一化）。"""
    by_t = {f["source_pts_ms"]: f for f in run_frames}
    n, errs, misses = 0, [], []
    for g in gt_ball_points:
        if g.get("unobservable"):
            continue
        frame = by_t.get(g["source_time_ms"])
        sel = (frame.get("ball") or {}).get("selected") if frame else None
        if not sel or not sel.get("raw"):
            misses.append(g)
            continue
        err = ((sel["point"][0] - g["point"][0]) ** 2 + (sel["point"][1] - g["point"][1]) ** 2) ** 0.5
        if err <= tol_px:
            n += 1
            errs.append(err)
        else:
            misses.append({**g, "error_px": round(err, 1)})
    total = sum(1 for g in gt_ball_points if not g.get("unobservable"))
    return {
        "gt_frames_total": len(gt_ball_points),
        "gt_frames_observable": total,
        "gt_frames_unobservable": sum(1 for g in gt_ball_points if g.get("unobservable")),
        "within_tolerance": n,
        "recall": round(n / total, 4) if total else None,
        "tolerance_px": tol_px,
        "mean_error_px": round(sum(errs) / len(errs), 1) if errs else None,
        "error_normalized_by_diagonal": round(sum(errs) / len(errs) / diag_px, 5) if errs and diag_px else None,
        "failures": misses[:50],
        "note": "分母=人工标记为可见且可定位的帧；未观测计入未命中",
    }


def _wall_points(run_events: list[dict], gt_wall_points: list[dict], wall: WallQuad | None,
                 tol_unit: float) -> dict:
    preds = [e for e in run_events if e["type"] == "wall_contact_candidate"]
    m = match_events(
        [{**e, "source_time_ms": e["source_time_ms"]} for e in preds], gt_wall_points, 200)
    errs = []
    for pair in m["matched"]:
        pu, gu = None, None
        if wall:
            pu = wall.map_point_to_wall_unit((pair["pred"]["point"]["x"], pair["pred"]["point"]["y"])) \
                if pair["pred"].get("point") else None
            gu = wall.map_point_to_wall_unit((pair["gt"]["point"][0], pair["gt"]["point"][1]))
        if pu and gu:
            errs.append(((pu[0] - gu[0]) ** 2 + (pu[1] - gu[1]) ** 2) ** 0.5)
    within = sum(1 for e in errs if e <= tol_unit)
    return {
        "matched": m["tp"], "false_positives": m["fp"], "missed": m["fn"],
        "match_rate": round(m["tp"] / len(gt_wall_points), 4) if gt_wall_points else None,
        "mean_error_wall_unit": round(sum(errs) / len(errs), 4) if errs else None,
        "within_tolerance_unit": within,
        "note": "仅比较可确认撞墙事件；单位=墙面归一化坐标",
    }


def evaluate(conn, run_row, gt_set_row, params: dict) -> dict:
    run_id = run_row["id"]
    run_dir = run_row["run_dir"]
    from app.vision.pipeline import load_frames_window
    from pathlib import Path
    frames = load_frames_window(Path(run_dir) / "frames")
    events = json.loads((Path(run_dir) / "events.json").read_text(encoding="utf-8"))["events"]
    view = params.get("view", "automatic")
    source_label = run_row["mode"]  # automatic | assisted
    if view == "reviewed":
        ops = corr_svc.list_corrections(conn, run_id)
        events = corr_svc.apply_ops(events, ops)
        events = [e for e in events if e.get("review_status") != "excluded"]

    gt = json.loads(gt_set_row["data_json"])
    tol_ms = int(params.get("time_tolerance_ms", 100))
    tol_ball_px = float(params.get("ball_position_tolerance_px", 40))
    tol_wall_unit = float(params.get("wall_unit_tolerance", 0.08))
    cfg = json.loads(run_row["config_json"])
    wall = WallQuad.from_dict(cfg.get("wall"))
    # 分析图像对角线：来自 run 落盘的分析尺寸（缺失时由参数补充）
    dims = _run_analysis_dims(run_row)
    diag = (dims["w"] ** 2 + dims["h"] ** 2) ** 0.5 if dims else None

    gt_hits = [g for g in gt.get("events", []) if g["type"] == "player_hit"]
    gt_wall = [g for g in gt.get("events", []) if g["type"] == "wall_contact"]
    pred_hits = [e for e in events if e["type"] == "player_hit"]
    hit_match = match_events(pred_hits, gt_hits, tol_ms)
    matched_pred_ids = {m["pred"]["id"] for m in hit_match["matched"]}
    matched_gt_times = {m["gt"]["source_time_ms"] for m in hit_match["matched"]}

    result = {
        "id": uuid.uuid4().hex,
        "run_id": run_id, "gt_set_id": gt_set_row["id"],
        "gt_name": gt_set_row["name"], "gt_version": gt_set_row["version"],
        "view": view, "run_mode": source_label,
        "params": {"time_tolerance_ms": tol_ms, "ball_position_tolerance_px": tol_ball_px,
                   "wall_unit_tolerance": tol_wall_unit},
        "hit_events": {k: v for k, v in hit_match.items() if k != "matched"},
        "hit_matches_detail": [
            {"pred_id": m["pred"]["id"], "pred_time_ms": m["pred"]["source_time_ms"],
             "gt_time_ms": m["gt"]["source_time_ms"], "dt_ms": m["dt_ms"],
             "pred_stroke": m["pred"].get("stroke_type"), "gt_stroke": m["gt"].get("stroke_type")}
            for m in hit_match["matched"]],
        "false_positives": [{"id": p["id"], "source_time_ms": p["source_time_ms"],
                             "evidence": p.get("evidence")}
                            for p in pred_hits if p["id"] not in matched_pred_ids],
        "false_negatives": [{"source_time_ms": g["source_time_ms"], "stroke_type": g.get("stroke_type")}
                            for g in gt_hits if g["source_time_ms"] not in matched_gt_times],
        "stroke": _stroke_confusion(hit_match["matched"]),
        "ball_localization": _ball_localization(frames, gt.get("ball_points", []), tol_ball_px, diag or 1.0),
        "wall_points": _wall_points(events, gt_wall, wall, tol_wall_unit),
        "note": "评分基于独立人工真值；confidence/heuristic_score 不参与判定；失败样例可回跳原视频",
    }
    return result


def _run_analysis_dims(run_row) -> dict | None:
    from pathlib import Path
    p = Path(run_row["run_dir"]) / "analysis_dims.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None
