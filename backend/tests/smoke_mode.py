"""在新检测模式上做短片段直跑冒烟（不经 HTTP），对比覆盖率。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
from app.store import db
from app.worker import main as worker


def main():
    proj_id = sys.argv[1]
    start_ms, end_ms = int(sys.argv[2]), int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "tiled_local"
    tiles = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {"cols": 3, "rows": 2, "overlap": 0.25}

    with db.connect() as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id=?", (proj_id,)).fetchone()
    assert proj, "项目不存在"
    print(f"video: {proj['original_filename']} {proj['width']}x{proj['height']} @ {proj['nominal_fps']:.1f}fps rot={proj['rotation']}")

    rid = db.new_id()
    run_dir = config.projects_dir() / proj_id / "runs" / rid
    cfg = {"handedness": "right",
           "processing": {"detection_mode": mode, "tiles": tiles},
           "tracking": {"max_prediction_gap_ms": 80, "association_max_dist_px": 160,
                        "base_search_radius_px": 260, "radius_per_lost_ms": 1.0}}
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, project_id, mode, status, config_json, code_version, "
            "segment_start_ms, segment_end_ms, cache_key, run_dir, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rid, proj_id, "automatic", "queued", json.dumps(cfg), config.CODE_VERSION,
             start_ms, end_ms, None, str(run_dir), time.time()))
        job = db.create_job(conn, jtype="analysis", payload={"run_id": rid},
                            project_id=proj_id, run_id=rid)
    import time as _t0
    t0 = _t0.time()
    with db.connect() as conn:
        # 先原子占位（防止独立 worker 重复认领同一任务），再直跑
        cur = conn.execute(
            "UPDATE jobs SET status='running', worker_id='smoke-direct', started_at=?, updated_at=? "
            "WHERE id=? AND status='queued'", (_t0.time(), _t0.time(), job["id"]))
        assert cur.rowcount == 1, "任务占位失败"
    with db.connect() as conn:
        worker.handle_analysis({"run_id": rid}, {"id": job["id"], "worker_id": "smoke"}, conn)
        # 直跑模式：任务标记完成，防止之后启动的 worker 重复执行同一 run
        import time as _t
        conn.execute("UPDATE jobs SET status='succeeded', finished_at=?, worker_id='smoke-direct' WHERE id=?",
                     (_t.time(), job["id"]))
    dt = time.time() - t0
    with db.connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    perf = json.loads(run["performance_json"])
    summary = json.loads(run["summary_json"])
    print(f"mode={mode} tiles={tiles}")
    print(f"wall={perf['wall_clock_ms']/1000:.1f}s for {(end_ms-start_ms)/1000:.1f}s segment "
          f"({perf['wall_clock_ms']/(end_ms-start_ms):.2f}x) device={perf['effective_device']} "
          f"rss={perf.get('peak_rss_mb')}MB")
    print("stage_ms:", perf["stage_ms"])
    print("search_mode_counts:", perf.get("search_mode_counts"))
    print(f"ball_observed={summary['frames_ball_observed']}/{summary['frames_total']} "
          f"coverage={summary['ball_observation_coverage']}")
    print("counts:", summary["counts"], "swings:", summary.get("swing_segments"))
    print("person_tracking:", summary.get("person_tracking"))
    events = json.loads((Path(run["run_dir"]) / "events.json").read_text())["events"]
    for ev in events[:20]:
        print(f"  t={ev['source_time_ms']}ms {ev['type']:22s} score={ev['heuristic_score']:.2f} "
              f"stroke={ev['stroke_type']}")


if __name__ == "__main__":
    main()
