import json

from core.report_exports import build_summary_payload
from core.timeline import (
    TIMELINE_FIELDNAMES,
    TimelineWriter,
    create_face_timeline_record,
    create_no_face_timeline_record,
    create_pose_failed_timeline_record,
    export_timeline_records,
    read_timeline_csv,
    video_frame_timestamp,
    write_timeline_json,
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


# ---------------------------------------------------------------- TimelineWriter
def visible_rows(csv_path):
    """What another process would see in the file right now (header excluded)."""
    with open(csv_path, encoding="utf-8") as handle:
        return len(handle.read().splitlines()) - 1


def test_writer_creates_the_file_with_a_header_straight_away(tmp_path):
    writer = TimelineWriter(str(tmp_path / "session"))
    with open(writer.csv_path, encoding="utf-8") as handle:
        assert handle.read().strip() == ",".join(TIMELINE_FIELDNAMES)
    writer.close()


def test_rows_are_flushed_every_thirty_so_a_killed_run_keeps_them(tmp_path):
    writer = TimelineWriter(str(tmp_path))
    for i in range(1, 30):
        writer.write(face_record(i))
    assert visible_rows(writer.csv_path) == 0  # not flushed yet

    writer.write(face_record(30))
    assert visible_rows(writer.csv_path) == 30

    for i in range(31, 60):
        writer.write(face_record(i))
    assert visible_rows(writer.csv_path) == 30
    writer.write(face_record(60))
    assert visible_rows(writer.csv_path) == 60
    writer.close()


def test_close_writes_the_remaining_rows_and_can_be_called_twice(tmp_path):
    writer = TimelineWriter(str(tmp_path))
    for i in range(1, 8):
        writer.write(face_record(i))
    writer.close()
    writer.close()
    assert visible_rows(writer.csv_path) == 7
    assert writer.rows_written == 7


def test_writer_holds_no_records_in_memory(tmp_path):
    writer = TimelineWriter(str(tmp_path))
    for i in range(1, 200):
        writer.write(face_record(i))
    assert not any(isinstance(value, list) and len(value) > 5 for value in vars(writer).values())
    writer.close()


def test_every_record_type_survives_a_csv_round_trip(tmp_path):
    records = [
        face_record(1),
        face_record(2, face_index=1, engagement="Not Engaged"),
        create_pose_failed_timeline_record(frame_index=3, timestamp_seconds=0.3, face_index=0),
        create_no_face_timeline_record(frame_index=4, timestamp_seconds=0.4),
        create_face_timeline_record(
            frame_index=5,
            timestamp_seconds=0.5,
            face_index=0,
            x_angle=-12.3456,
            y_angle=7.0,
            gaze_direction="Left",
            engagement_label="Not Engaged",
            head_direction="Looking Left",
            head_pose_score=12.5,
            gaze_center_score=0.0,
            engagement_score_raw=33.33,
            engagement_score_smooth=40.1,
            face_visibility_score=0.8123,
            lighting_score=0.5,
            landmark_stability_score=0.9,
            quality_confidence_score=61.25,
            quality_confidence_label="Medium",
            quality_warnings=["Low light", "Frequent movement"],
        ),
    ]
    writer = TimelineWriter(str(tmp_path))
    for record in records:
        writer.write(record)
    writer.close()

    assert read_timeline_csv(writer.csv_path) == records


def test_streamed_timeline_matches_the_old_all_at_once_export(tmp_path):
    records = [face_record(i) for i in range(1, 11)] + [create_no_face_timeline_record(11, 1.1)]

    old_dir = tmp_path / "old"
    old_csv, old_json = export_timeline_records(records, str(old_dir))

    new_dir = tmp_path / "new"
    writer = TimelineWriter(str(new_dir))
    for record in records:
        writer.write(record)
    writer.close()
    write_timeline_json(read_timeline_csv(writer.csv_path), writer.json_path)

    with open(old_csv, "rb") as old_file, open(writer.csv_path, "rb") as new_file:
        assert old_file.read() == new_file.read()
    assert json.loads(open(old_json, encoding="utf-8").read()) == json.loads(open(writer.json_path, encoding="utf-8").read())


def test_summary_from_the_re_read_csv_equals_the_summary_from_memory(tmp_path):
    records = [face_record(i) for i in range(1, 6)] + [
        create_pose_failed_timeline_record(frame_index=6, timestamp_seconds=0.6, face_index=0),
        create_no_face_timeline_record(frame_index=7, timestamp_seconds=0.7),
    ]
    writer = TimelineWriter(str(tmp_path))
    for record in records:
        writer.write(record)
    writer.close()

    from_memory = build_summary_payload(records, source_type="video")
    from_csv = build_summary_payload(read_timeline_csv(writer.csv_path), source_type="video")
    from_memory.pop("generated_at")
    from_csv.pop("generated_at")
    assert from_csv == from_memory
