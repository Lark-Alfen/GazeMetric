import json

import pytest

import core.calibration_runtime
import core.quality
from tests.conftest import FakeFaceMesh


def test_a_normal_run_exits_zero_and_writes_every_frame(main_run, video):
    assert main_run.run(main_run.args(video)) == 0
    rows = main_run.timeline_rows()
    assert len(rows) == 40
    assert {row["engagement_label"] for row in rows} <= {"Engaged", "Not Engaged"}
    assert main_run.face_mesh.closed
    assert main_run.captures[0].released


def test_a_run_without_faces_still_exits_zero(main_run, video):
    FakeFaceMesh.faces = 0
    assert main_run.run(main_run.args(video)) == 0
    assert {row["engagement_label"] for row in main_run.timeline_rows()} == {"No Face"}


def test_unopenable_source_exits_nonzero_and_closes_the_face_mesh(main_run, tmp_path):
    assert main_run.run(main_run.args(str(tmp_path / "missing.mp4"))) == 1
    assert "Unable to open source" in main_run.output
    assert main_run.face_mesh.closed
    assert main_run.captures[0].released
    assert main_run.session_dir() is None  # nothing was analysed, so no empty report


def test_one_bad_frame_is_skipped_with_a_traceback_and_the_run_continues(main_run, video):
    real = core.quality.compute_quality_metrics
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 10:
            raise ValueError("bad frame")
        return real(*args, **kwargs)

    main_run.monkeypatch.setattr(core.quality, "compute_quality_metrics", flaky)
    assert main_run.run(main_run.args(video)) == 0
    assert "Traceback (most recent call last)" in main_run.output
    assert "ValueError: bad frame" in main_run.output
    assert "1 frame(s) were skipped" in main_run.output
    assert len(main_run.timeline_rows()) == 39


def test_a_persistent_failure_stops_the_run_but_still_writes_reports_and_exits_one(main_run, video):
    def broken(*_args, **_kwargs):
        raise RuntimeError("quality broke")

    main_run.monkeypatch.setattr(core.quality, "compute_quality_metrics", broken)
    assert main_run.run(main_run.args(video)) == 1
    assert "Traceback (most recent call last)" in main_run.output
    assert "frames in a row failed" in main_run.output
    assert "Saved timeline CSV" in main_run.output
    assert main_run.face_mesh.closed
    assert main_run.captures[0].released


def test_failure_during_calibration_releases_the_camera_and_exits_one(main_run, video):
    def broken(**_kwargs):
        raise RuntimeError("calibration broke")

    main_run.monkeypatch.setattr(core.calibration_runtime, "run_calibration_phase", broken)
    assert main_run.run(main_run.args(video, skip_calibration=False)) == 1
    assert "Traceback (most recent call last)" in main_run.output
    assert "calibration broke" in main_run.output
    assert main_run.captures[0].released
    assert main_run.face_mesh.closed


def test_quitting_during_calibration_exits_zero_and_cleans_up(main_run, video):
    main_run.keys = [ord("q")]
    assert main_run.run(main_run.args(video, skip_calibration=False)) == 0
    assert main_run.captures[0].released
    assert main_run.face_mesh.closed


@pytest.mark.parametrize("script", ["engagement_analysis_live.py", "engagement_analysis_recorded.py"])
def test_entry_scripts_pass_the_exit_code_to_the_shell(script):
    with open(script, encoding="utf-8") as handle:
        assert "sys.exit(main(parse_arguments()))" in handle.read()


def test_a_corrupt_profile_is_kept_as_bad_and_replaced_by_a_fresh_calibration(main_run, video, tmp_path):
    profile_path = tmp_path / "profile.json"
    corrupt = '{"head_center": {"x": '
    profile_path.write_text(corrupt, encoding="utf-8")

    assert main_run.run(main_run.args(video, skip_calibration=False, calibration_profile=str(profile_path))) == 0

    assert "Calibration profile is unusable" in main_run.output
    assert (tmp_path / "profile.json.bad").read_text(encoding="utf-8") == corrupt
    saved = json.loads(profile_path.read_text(encoding="utf-8"))
    assert saved["sample_count"] > 0


def test_a_corrupt_profile_with_skipped_calibration_is_moved_not_overwritten(main_run, video, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("garbage", encoding="utf-8")

    assert main_run.run(main_run.args(video, calibration_profile=str(profile_path))) == 0

    assert not profile_path.exists()
    assert (tmp_path / "profile.json.bad").read_text(encoding="utf-8") == "garbage"
    assert "Skipping calibration" in main_run.output


def test_a_valid_profile_is_loaded_and_left_alone(main_run, video, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"head_center": {"x": 3.0, "y": 1.0}}), encoding="utf-8")
    before = profile_path.read_text(encoding="utf-8")

    assert main_run.run(main_run.args(video, calibration_profile=str(profile_path))) == 0

    assert "Loaded calibration profile" in main_run.output
    assert profile_path.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("profile.json.*")) == []


def test_timeline_json_and_csv_are_both_written_and_agree(main_run, video):
    assert main_run.run(main_run.args(video)) == 0
    session = main_run.session_dir()
    with open(f"{session}/timeline.json", encoding="utf-8") as handle:
        records = json.load(handle)
    rows = main_run.timeline_rows()
    assert len(records) == len(rows) == 40
    assert [str(r["frame_index"]) for r in records] == [row["frame_index"] for row in rows]
    assert isinstance(records[0]["timestamp_seconds"], float)
    assert isinstance(records[0]["face_index"], int)


def test_timeline_rows_are_on_disk_while_the_run_is_still_going(main_run, video):
    import core.quality

    real = core.quality.compute_quality_metrics
    seen = {}

    def spy(*args, **kwargs):
        if "rows" not in seen and len(seen.setdefault("calls", [])) == 70:
            seen["rows"] = main_run.timeline_rows()
        seen["calls"].append(1)
        return real(*args, **kwargs)

    longer = video.replace("clip.mp4", "longer.mp4")
    from tests.conftest import make_video

    make_video(longer, frames=100)
    main_run.monkeypatch.setattr(core.quality, "compute_quality_metrics", spy)
    assert main_run.run(main_run.args(longer)) == 0
    assert seen["rows"] is not None
    assert len(seen["rows"]) == 60  # 70 frames done, flushed in blocks of 30


def test_closing_the_window_ends_the_run_and_still_writes_the_reports(main_run, video):
    state = {"frames": 0}
    original_wait = main_run._wait_key

    def count_frames(delay):
        state["frames"] += 1
        return original_wait(delay)

    main_run.monkeypatch.setattr("cv2.waitKey", count_frames)
    main_run.window_visible = lambda: state["frames"] < 12  # the user clicks X during frame 12

    assert main_run.run(main_run.args(video)) == 0

    assert len(main_run.timeline_rows()) == 12
    assert "Saved session summary JSON" in main_run.output
    assert main_run.captures[0].released
    assert main_run.face_mesh.closed


def test_closing_the_window_during_calibration_stops_before_analysis(main_run, video):
    main_run.window_visible = lambda: False

    assert main_run.run(main_run.args(video, skip_calibration=False)) == 0

    assert "Window closed" in main_run.output
    assert main_run.session_dir() is None
    assert main_run.captures[0].released
    assert main_run.face_mesh.closed


def test_the_banner_is_printed_before_calibration_starts(main_run, video):
    main_run.keys = [ord("q")]
    assert main_run.run(main_run.args(video, skip_calibration=False)) == 0
    output = main_run.output
    assert "EngageVision - Full Engagement Analysis System" in output
    assert output.index("Full Engagement Analysis System") < output.index("Calibration step started")
    assert output.count("Full Engagement Analysis System") == 1


def test_the_banner_names_the_video_source(main_run, video):
    assert main_run.run(main_run.args(video)) == 0
    assert "video file is analyzed" in main_run.output
    assert "camera" not in main_run.output.lower()


def test_ctrl_c_still_prints_the_summary_and_writes_the_reports(main_run, video):
    calls = {"count": 0}
    original_wait = main_run._wait_key

    def interrupt_on_frame_10(delay):
        calls["count"] += 1
        if calls["count"] == 10:
            raise KeyboardInterrupt
        return original_wait(delay)

    main_run.monkeypatch.setattr("cv2.waitKey", interrupt_on_frame_10)
    assert main_run.run(main_run.args(video)) == 0
    assert "Interrupted by user (Ctrl+C). Generating summary" in main_run.output
    assert "ENGAGEMENT LEVEL ANALYSIS" in main_run.output
    assert "Saved session summary JSON" in main_run.output
    assert len(main_run.timeline_rows()) == 10


def test_two_faces_keep_their_identity_when_the_detection_order_flips(main_run, video):
    FakeFaceMesh.people = [(-0.25, -25), (0.25, 25)]  # one person turned left on the left, one turned right on the right

    assert main_run.run(main_run.args(video)) == 0

    rows = main_run.timeline_rows()
    assert len(rows) == 80
    by_face = {}
    for row in rows:
        by_face.setdefault(row["face_index"], []).append(float(row["y_angle"]))
    assert sorted(by_face) == ["0", "1"]
    for angles in by_face.values():
        assert len(angles) == 40
        assert max(angles) - min(angles) < 1.0  # a mix-up would alternate between the two people's angles
    assert abs(by_face["0"][0] - by_face["1"][0]) > 20


def test_each_face_smooths_its_own_score(main_run, video):
    FakeFaceMesh.people = [(-0.25, -25), (0.25, 25)]
    assert main_run.run(main_run.args(video)) == 0
    scores = {}
    for row in main_run.timeline_rows():
        scores.setdefault(row["face_index"], []).append(float(row["engagement_score_smooth"]))
    # constant input per person gives a constant smoothed score; shared state would drift between the two values
    for values in scores.values():
        assert max(values[3:]) - min(values[3:]) < 0.5
