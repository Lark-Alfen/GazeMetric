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
