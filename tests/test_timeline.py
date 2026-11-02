import csv
import json
import os

import pytest

from core import timeline
from core.report_exports import build_summary_payload
from core.timeline import (
    TIMELINE_FIELDNAMES,
    TimelineWriter,
    create_face_timeline_record,
    create_no_face_timeline_record,
    create_pose_failed_timeline_record,
    export_timeline_records,
    read_timeline_csv,
    resolve_timeline_output_dir,
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


# ---------------------------------------------------------------- session folder resolution
class FrozenDatetime:
    @staticmethod
    def now():
        from datetime import datetime

        return datetime(2026, 3, 4, 5, 6, 7)


@pytest.fixture
def frozen_clock(monkeypatch, tmp_path):
    monkeypatch.setattr(timeline, "datetime", FrozenDatetime)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_folder_is_runs_session_timestamp(frozen_clock):
    assert resolve_timeline_output_dir(None) == os.path.join("runs", "session_20260304_050607")
    assert resolve_timeline_output_dir("") == os.path.join("runs", "session_20260304_050607")


def test_a_base_directory_gets_a_session_folder_inside_it(frozen_clock):
    assert resolve_timeline_output_dir("results") == os.path.join("results", "session_20260304_050607")
    assert resolve_timeline_output_dir("results/") == os.path.join("results", "session_20260304_050607")


def test_an_explicit_session_path_is_kept_as_given(frozen_clock):
    assert resolve_timeline_output_dir(os.path.join("out", "session_mine")) == os.path.join("out", "session_mine")


def test_an_existing_explicit_session_folder_gets_a_numeric_suffix(frozen_clock):
    path = os.path.join("out", "session_mine")
    os.makedirs(path)
    assert resolve_timeline_output_dir(path) == path + "_01"
    os.makedirs(path + "_01")
    assert resolve_timeline_output_dir(path) == path + "_02"


def test_two_runs_in_the_same_second_do_not_share_a_folder(frozen_clock):
    first = resolve_timeline_output_dir("results")
    os.makedirs(first)
    second = resolve_timeline_output_dir("results")
    assert second == first + "_01"


def test_resolving_a_folder_does_not_create_it(frozen_clock):
    resolve_timeline_output_dir("results")
    assert not os.path.exists("results")


# ---------------------------------------------------------------- record builders
def build_face_record(**overrides):
    values = dict(
        frame_index=3,
        timestamp_seconds=0.123,
        face_index=0,
        x_angle=1.234567,
        y_angle=-2.345678,
        gaze_direction="Forward",
        engagement_label="Engaged",
        head_direction="Looking Straight",
        head_pose_score=88.123456,
        gaze_center_score=77.987654,
        engagement_score_raw=66.666666,
        engagement_score_smooth=55.55,
    )
    values.update(overrides)
    return create_face_timeline_record(**values)


def test_face_record_rounds_numbers_and_keeps_every_column():
    record = build_face_record(
        face_visibility_score=0.123456,
        lighting_score=0.987654,
        landmark_stability_score=0.555555,
        quality_confidence_score=71.4567,
        quality_confidence_label="High",
    )
    assert list(record) == TIMELINE_FIELDNAMES
    assert record["x_angle"] == 1.2346 and record["y_angle"] == -2.3457
    assert record["head_pose_score"] == 88.1235 and record["gaze_center_score"] == 77.9877
    assert record["engagement_score_raw"] == 66.67
    assert record["engagement_score_smooth"] == 55.55
    assert record["face_visibility_score"] == 0.1235 and record["lighting_score"] == 0.9877
    assert record["landmark_stability_score"] == 0.5556
    assert record["quality_confidence_score"] == 71.46


def test_face_record_without_quality_data_uses_none_and_unknown():
    record = build_face_record()
    for key in ("face_visibility_score", "lighting_score", "landmark_stability_score", "quality_confidence_score"):
        assert record[key] is None
    assert record["quality_confidence_label"] == "Unknown"
    assert record["quality_warnings"] == ""


@pytest.mark.parametrize(
    "warnings, text",
    [
        (None, ""),
        ([], ""),
        (["Low light"], "Low light"),
        (["Low light", "", None, "Frequent movement"], "Low light; Frequent movement"),
        ("Already text", "Already text"),
    ],
)
def test_quality_warnings_are_joined_into_one_cell(warnings, text):
    assert build_face_record(quality_warnings=warnings)["quality_warnings"] == text


def test_face_record_accepts_numpy_numbers():
    import numpy as np

    record = build_face_record(x_angle=np.float32(2.5), y_angle=np.float64(-1.25), head_pose_score=np.float32(50))
    assert record["x_angle"] == 2.5 and record["y_angle"] == -1.25 and record["head_pose_score"] == 50.0
    json.dumps(record)  # must stay serializable for timeline.json


def test_no_face_record_matches_the_column_list_and_marks_every_label():
    record = create_no_face_timeline_record(frame_index=9, timestamp_seconds=0.9)
    assert list(record) == TIMELINE_FIELDNAMES
    assert record["face_index"] == -1
    assert {record[key] for key in ("gaze_direction", "engagement_label", "head_direction", "quality_warnings")} == {"No Face"}
    assert all(record[key] is None for key in ("x_angle", "y_angle", "head_pose_score", "engagement_score_smooth"))


# ---------------------------------------------------------------- export_timeline_records
def test_export_writes_csv_and_json_with_a_matching_header(tmp_path):
    records = [face_record(1), create_no_face_timeline_record(2, 0.2)]
    csv_path, json_path = export_timeline_records(records, str(tmp_path / "deep" / "session"))

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == TIMELINE_FIELDNAMES
    assert len(rows) == 3
    assert rows[2][TIMELINE_FIELDNAMES.index("engagement_label")] == "No Face"
    assert rows[2][TIMELINE_FIELDNAMES.index("x_angle")] == ""  # None is an empty cell

    with open(json_path, encoding="utf-8") as handle:
        assert json.load(handle) == records


def test_export_of_nothing_still_writes_a_header_and_an_empty_list(tmp_path):
    csv_path, json_path = export_timeline_records([], str(tmp_path))
    with open(csv_path, encoding="utf-8") as handle:
        assert handle.read().strip() == ",".join(TIMELINE_FIELDNAMES)
    with open(json_path, encoding="utf-8") as handle:
        assert json.load(handle) == []
