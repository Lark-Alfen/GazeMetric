from core.report_exports import build_summary_payload
from core.timeline import (
    TIMELINE_FIELDNAMES,
    create_face_timeline_record,
    create_no_face_timeline_record,
    create_pose_failed_timeline_record,
    video_frame_timestamp,
)


def face_record(frame_index, face_index=0, engagement="Engaged"):
    return create_face_timeline_record(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10,
        face_index=face_index,
        x_angle=1.0,
        y_angle=2.0,
        gaze_direction="Forward",
        engagement_label=engagement,
        head_direction="Looking Straight",
        head_pose_score=90.0,
        gaze_center_score=80.0,
        engagement_score_raw=85.0,
        engagement_score_smooth=85.0,
        quality_confidence_score=70.0,
        quality_confidence_label="High",
    )


def test_pose_failed_record_has_every_timeline_column():
    record = create_pose_failed_timeline_record(frame_index=12, timestamp_seconds=1.2, face_index=1)
    assert list(record) == TIMELINE_FIELDNAMES


def test_pose_failed_record_belongs_to_a_detected_face():
    record = create_pose_failed_timeline_record(frame_index=12, timestamp_seconds=1.2, face_index=1)
    assert record["frame_index"] == 12
    assert record["timestamp_seconds"] == 1.2
    assert record["face_index"] == 1
    assert record["gaze_direction"] == record["engagement_label"] == record["head_direction"] == "Pose Unavailable"
    assert record["x_angle"] is None and record["y_angle"] is None
    assert record["engagement_score_smooth"] is None


def test_pose_failed_record_is_distinct_from_a_no_face_record():
    pose_failed = create_pose_failed_timeline_record(frame_index=3, timestamp_seconds=0.3, face_index=0)
    no_face = create_no_face_timeline_record(frame_index=3, timestamp_seconds=0.3)
    assert no_face["face_index"] == -1
    assert pose_failed["face_index"] == 0
    assert pose_failed["engagement_label"] != no_face["engagement_label"]


def test_summary_counts_pose_unavailable_as_its_own_bucket():
    records = [
        face_record(1),
        face_record(2),
        create_pose_failed_timeline_record(frame_index=3, timestamp_seconds=0.3, face_index=0),
        create_no_face_timeline_record(frame_index=4, timestamp_seconds=0.4),
    ]
    payload = build_summary_payload(records, source_type="video")
    assert payload["kpis"]["pose_unavailable_records"] == 1
    assert payload["kpis"]["no_face_records"] == 1
    assert payload["kpis"]["face_records"] == 3
    counts = payload["distributions"]["engagement_label"]["counts"]
    assert counts == {"Engaged": 2, "Pose Unavailable": 1}
    # failed-pose frames have no score, so they must not drag the average toward zero
    assert payload["kpis"]["engagement_score"]["samples"] == 2
    assert payload["kpis"]["engagement_score"]["average"] == 85.0


# ---------------------------------------------------------------- video timestamps
def test_video_timestamp_counts_frames_when_no_position_is_reported():
    assert video_frame_timestamp(1, 25.0) == 0.0
    assert video_frame_timestamp(26, 25.0, position_msec=0.0) == 1.0
    assert video_frame_timestamp(91, 30.0) == 3.0


def test_video_timestamp_prefers_the_decoder_position():
    # variable frame rate: the 10th frame sits at 0.9s although 10 frames at 30fps would be 0.3s
    assert video_frame_timestamp(10, 30.0, position_msec=900.4) == 0.9


def test_video_timestamp_ignores_a_zero_or_negative_position_after_the_first_frame():
    assert video_frame_timestamp(31, 30.0, position_msec=0.0) == 1.0
    assert video_frame_timestamp(31, 30.0, position_msec=-1.0) == 1.0


def test_video_timestamp_does_not_depend_on_wall_clock_time():
    import time

    first = video_frame_timestamp(50, 25.0)
    time.sleep(0.05)
    assert video_frame_timestamp(50, 25.0) == first
