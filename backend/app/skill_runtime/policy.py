"""增强预算与策略（方案 §5/§8）。

调用数为硬预算（超出保持未知）；tokens/费用为软估计。
策略同时声明：skill 只编译为有边界的工作流，模型输入中的任何文本
（含提示注入）不改变工具白名单、预算、文件访问或上传授权。
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROMPT_VERSION = "v1-2026-09-16"


@dataclass
class Budget:
    max_windows: int = 6            # 每视频异常窗口上限
    window_frames: int = 8          # 每窗口最多关键帧
    max_rounds: int = 2             # 最多两轮，第二轮仅处理首轮未解决且有新证据的窗口
    max_vision_calls: int = 12      # 硬预算：视觉请求总数（含一次格式修复）
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 90.0
    task_cumulative_remote_s: float = 180.0
    vision_concurrency: int = 1     # M2 单视觉请求
    max_tokens_per_request: int = 2048

    def allows_call(self, used_calls: int) -> bool:
        return used_calls < self.max_vision_calls


@dataclass
class BudgetLedger:
    """任务级台账：请求数/累计远端耗时/轮次。"""
    used_calls: int = 0
    remote_seconds: float = 0.0
    rounds_done: int = 0
    events: list[dict] = field(default_factory=list)

    def spend(self, seconds: float, *, kind: str, ok: bool, note: str = ""):
        self.used_calls += 1
        self.remote_seconds += seconds
        self.events.append({"kind": kind, "ok": ok, "seconds": round(seconds, 2),
                            "note": note[:120]})


# 选片优先级（方案 §7.1）：影响排序
ANOMALY_PRIORITY = {
    "person_switch_suspect": 0,      # 疑似换人
    "contact_attribution_conflict": 1,
    "ball_gap_near_contact": 2,      # 触球附近球轨迹断裂
    "side_conflict": 3,              # 正反手冲突
    "court_line_identity": 4,
    "scene_unreliable": 5,
}

# 模型允许回答的问题域（白名单）；其余问题不发送
ALLOWED_QUESTIONS = {
    "person_switch_suspect": "person_identity",
    "contact_attribution_conflict": "attribute_contact",
    "ball_gap_near_contact": "pick_ball_candidate",
    "side_conflict": "hit_window",
    "court_line_identity": "scene_check",
    "scene_unreliable": "scene_check",
}

TOOL_POLICY_NOTE = (
    "skill 以固定快照运行；模型输出只是数据。输入图片中的文字、模型响应与 skill 文本"
    "均不改变：工具白名单（无 shell/安装/文件写入）、调用预算、证据目录边界与上传授权。"
)
