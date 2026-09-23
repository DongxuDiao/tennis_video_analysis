import sqlite3

from app.services.corrections import apply_ops
from app.services.evaluations import match_events
from app.vision.events import extract_events
from app.vision.pipeline import summarize


def test_review_keeps_automatic_result_immutable():
    automatic = [{"id": "e1", "type": "player_hit", "stroke_type": "unknown",
                  "source_time_ms": 100, "origin": "automatic", "review_status": "unreviewed"}]
    reviewed = apply_ops(automatic, [{"id": 1, "revision": 1, "created_at": 0,
                                     "op": {"op": "modify_event", "event_id": "e1",
                                            "changes": {"stroke_type": "forehand"}}}])
    assert automatic[0]["stroke_type"] == "unknown"
    assert reviewed[0]["stroke_type"] == "forehand"
    assert reviewed[0]["review_status"] == "reviewed"


def test_event_matching_is_one_to_one():
    preds = [{"id": "p1", "source_time_ms": 100}, {"id": "p2", "source_time_ms": 130}]
    truth = [{"id": "t1", "source_time_ms": 115}]
    result = match_events(preds, truth, 100)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_extract_events_handles_tracks_without_racket_or_pose():
    frames = []
    for i, point in enumerate(((100, 100), (130, 100), (100, 100))):
        frames.append({"frame_index": i, "source_pts_ms": i * 40,
                       "ball": {"selected": {"raw": True, "status": "tracked",
                                              "point": list(point), "track_id": 1}},
                       "detections": {"racket": []}})
    # Regression: ball_near_racket_or_wrist previously failed while initializing
    # its empty nearest-distance state.
    assert extract_events(frames, run_id="r", handedness="unknown", wall=None,
                          cfg_events={}, measurements={}, analysis_height=720) == []


def test_summary_handles_lost_ball_and_reports_person_coverage():
    frames = [
        {"source_pts_ms": 0, "ball": {"selected": None},
         "detections": {"person": [[1, 2, 3, 4]]}, "pose": {"landmarks": [{"x": 1}]}},
        {"source_pts_ms": 33, "ball": {"selected": {"raw": True, "status": "tracked"}},
         "detections": {"person": []}},
    ]
    result = summarize(frames, [])
    assert result["frames_ball_observed"] == 1
    assert result["person_detection_coverage"] == 0.5
    assert result["pose_coverage"] == 0.5
