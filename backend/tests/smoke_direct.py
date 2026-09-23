"""直接调用流水线的 smoke 脚本（不经 HTTP），快速验证导入→分析→事件全链路。"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 独立数据目录：本脚本在宿主机运行，若沿用仓库内 data/ 会与
# tennis-lab compose 容器共写同一 SQLite（Docker bind mount 的锁翻译
# 造成跨 VM 偶发 database is locked，等满 busy timeout 后失败）
os.environ.setdefault("TENNIS_DATA_DIR", "/tmp/tennis-smoke-data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
from app.store import db
from app.worker import main as worker

# 模型文件体积大，smoke 数据目录用软链复用仓库内已下载的模型
_models_src = Path(__file__).resolve().parents[2] / "data" / "models"
_models_dst = Path(os.environ["TENNIS_DATA_DIR"]) / "models"
if _models_src.is_dir() and not _models_dst.exists():
    _models_dst.parent.mkdir(parents=True, exist_ok=True)
    _models_dst.symlink_to(_models_src, target_is_directory=True)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic_wall_practice.mp4"


def main() -> None:
    # 清空数据目录（仅 smoke 用）
    if config.db_path().exists():
        config.db_path().unlink()
    for p in config.projects_dir().glob("*"):
        shutil.rmtree(p)
    db.init_db()

    pid = db.new_id()
    proj_dir = config.projects_dir() / pid
    proj_dir.mkdir(parents=True)
    dst = proj_dir / "original.mp4"
    shutil.copy(FIXTURE, dst)
    now = time.time()
    with db.connect() as conn:
        conn.execute("INSERT INTO projects (id, created_at, original_filename, original_path, video_hash, size_bytes, proxy_status) VALUES (?,?,?,?,?,?,?)",
                     (pid, now, "synthetic.mp4", str(dst), "smoke", dst.stat().st_size, "pending"))
        job = db.create_job(conn, jtype="import", payload={"project_id": pid}, project_id=pid)
    t0 = time.time()
    with db.connect() as conn:
        worker.handle_import({"project_id": pid}, {"id": job["id"], "worker_id": "smoke"}, conn)
    print(f"[import] {time.time()-t0:.1f}s")

    with db.connect() as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    print("[probe]", {k: proj[k] for k in ("width", "height", "duration_ms", "nominal_fps", "rotation", "proxy_status")})
    assert Path(proj["original_path"]).parent.joinpath("proxy.mp4").exists(), "代理未生成"

    # 创建 run：全片段，墙四点取画面中墙面区域
    wall = {"points": [[1040, 60], [1270, 60], [1270, 719], [1040, 719]]}
    with db.connect() as conn:
        rid = db.new_id()
        run_dir = config.projects_dir() / pid / "runs" / rid
        conn.execute("INSERT INTO runs (id, project_id, mode, status, config_json, code_version, segment_start_ms, segment_end_ms, cache_key, run_dir, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (rid, pid, "automatic", "queued", json.dumps({"handedness": "right", "wall": wall}), "smoke", 0, proj["duration_ms"], None, str(run_dir), now))
        job2 = db.create_job(conn, jtype="analysis", payload={"run_id": rid}, project_id=pid, run_id=rid)
    t0 = time.time()
    with db.connect() as conn:
        worker.handle_analysis({"run_id": rid}, {"id": job2["id"], "worker_id": "smoke"}, conn)
    print(f"[analysis] {time.time()-t0:.1f}s")

    with db.connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    perf = json.loads(run["performance_json"])
    print("[perf]", json.dumps({k: perf[k] for k in ("stage_ms", "frames_processed", "effective_device", "wall_clock_ms")}, ensure_ascii=False))
    summary = json.loads(run["summary_json"])
    print("[summary]", json.dumps(summary["counts"], ensure_ascii=False),
          "coverage:", summary["ball_observation_coverage"])
    events = json.loads((Path(run["run_dir"]) / "events.json").read_text())["events"]
    print(f"[events] {len(events)} 条")
    for ev in events[:10]:
        print(f"  t={ev['source_time_ms']}ms {ev['type']:24s} score={ev['heuristic_score']} pt={ev['point'] and (ev['point']['x'], ev['point']['y'])}")


def _conn_factory():
    def f():
        return db.connect()
    return f


if __name__ == "__main__":
    # worker 处理器直接接收 conn；适配签名
    class _Ctx:
        pass
    main()
