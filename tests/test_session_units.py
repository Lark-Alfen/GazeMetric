"""The pieces main() is built from, tested without running a whole session."""

import argparse
from types import SimpleNamespace

import cv2
import pytest

import engagement_analysis_full as app
from core.analysis_logic import get_default_calibration_profile
from core.calibration_runtime import CALIBRATION_COMPLETED, CALIBRATION_INSUFFICIENT, CALIBRATION_QUIT, CALIBRATION_SKIPPED


def make_args(**overrides):
    values = dict(
        source="video",
        input_path="clip.mp4",
        output_video_path=None,
        timeline_dir=None,
        skip_calibration=False,
        calibration_seconds=5.0,
        calibration_profile=None,
        no_display=False,
        camera_index=0,
        max_faces=2,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeCap:
    def __init__(self, props=None, frames=0):
        self.props = props or {}
        self.frames = frames
        self.reads = 0
        self.seeks = []
        self.released = False

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        self.seeks.append((prop, value))

    def isOpened(self):
        return not self.released

    def read(self):
        if self.reads >= self.frames:
            return False, None
        self.reads += 1
        return True, object()

    def release(self):
        self.released = True


def make_state(**overrides):
    values = dict(
        args=make_args(),
        cap=FakeCap(),
        models=None,
        calibration_profile=get_default_calibration_profile(),
        source_fps=25.0,
        total_frames=0,
        timeline_writer=None,
        output_writer=None,
        session_start_time=100.0,
    )
    values.update(overrides)
    return app.SessionState(**values)


# ---------------------------------------------------------------- source properties and output writer
def test_video_properties_come_from_the_file():
    cap = FakeCap({cv2.CAP_PROP_FPS: 29.97, cv2.CAP_PROP_FRAME_COUNT: 900.0})
    assert app.read_source_properties(cap, make_args(source="video")) == (29.97, 900)


def test_unknown_frame_rate_falls_back_to_20():
    assert app.read_source_properties(FakeCap({cv2.CAP_PROP_FPS: 0.0}), make_args()) == (20.0, 0)
    assert app.read_source_properties(FakeCap({cv2.CAP_PROP_FPS: -1.0}), make_args()) == (20.0, 0)


def test_a_webcam_has_no_frame_total():
    cap = FakeCap({cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 123.0})
    assert app.read_source_properties(cap, make_args(source="webcam")) == (30.0, 0)


def test_no_output_writer_without_an_output_path():
    assert app.create_output_writer(make_args(), 25.0) is None


def test_video_output_uses_the_source_rate_and_webcam_output_measures_its_own():
    video = app.create_output_writer(make_args(output_video_path="o.mp4", source="video"), 25.0)
    webcam = app.create_output_writer(make_args(output_video_path="o.mp4", source="webcam"), 30.0)
    assert (video.fps, video.fallback_fps) == (25.0, 25.0)
    assert (webcam.fps, webcam.fallback_fps) == (None, 30.0)


# ---------------------------------------------------------------- timestamps
def test_video_timestamps_follow_the_video(monkeypatch):
    state = make_state(cap=FakeCap({cv2.CAP_PROP_POS_MSEC: 1200.0}))
    state.frame_index = 31
    assert app.frame_timestamp(state) == 1.2


def test_webcam_timestamps_follow_the_wall_clock(monkeypatch):
    monkeypatch.setattr(app.time, "perf_counter", lambda: 102.3456)
    state = make_state(args=make_args(source="webcam"))
    assert app.frame_timestamp(state) == 2.346


# ---------------------------------------------------------------- calibration choices
@pytest.fixture
def fail_if_calibrating(monkeypatch):
    def refuse(**_kwargs):
        raise AssertionError("calibration must not run")

    monkeypatch.setattr(app, "run_calibration_phase", refuse)


def test_skip_calibration_uses_the_defaults(fail_if_calibrating, capsys):
    profile, stop = app.load_or_run_calibration(make_args(skip_calibration=True), FakeCap(), None, 25.0)
    assert profile == get_default_calibration_profile()
    assert stop is False
    assert "Skipping calibration" in capsys.readouterr().out


def test_a_saved_profile_is_used_without_calibrating(fail_if_calibrating, tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_text('{"head_center": {"x": 4.0, "y": 0.0}}', encoding="utf-8")
    profile, stop = app.load_or_run_calibration(make_args(calibration_profile=str(path)), FakeCap(), None, 25.0)
    assert profile["head_center"]["x"] == 4.0 and stop is False
    assert "Loaded calibration profile" in capsys.readouterr().out


def run_calibration_with(monkeypatch, generated, status, args=None, cap=None):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return generated, status

    monkeypatch.setattr(app, "run_calibration_phase", fake)
    models = SimpleNamespace(face_mesh="fm", mp_face_mesh="mpfm", mp_drawing="draw", leye_3d="l", reye_3d="r")
    result = app.load_or_run_calibration(args or make_args(), cap or FakeCap(), models, 25.0)
    return result, seen


def test_a_fresh_calibration_is_used_and_saved(monkeypatch, tmp_path, capsys):
    fresh = get_default_calibration_profile()
    fresh["sample_count"] = 80
    target = tmp_path / "saved.json"
    cap = FakeCap()
    (profile, stop), seen = run_calibration_with(
        monkeypatch, fresh, CALIBRATION_COMPLETED, make_args(calibration_profile=str(target)), cap
    )
    assert profile is fresh and stop is False
    assert target.is_file()
    assert "Saved calibration profile" in capsys.readouterr().out
    assert seen["calibration_seconds"] == 5.0 and seen["show_window"] is True and seen["source_type"] == "video"
    assert cap.seeks == [(cv2.CAP_PROP_POS_FRAMES, 0)]  # a video is rewound so the whole clip is analysed


def test_a_webcam_is_not_rewound_after_calibration(monkeypatch):
    cap = FakeCap()
    run_calibration_with(monkeypatch, None, CALIBRATION_SKIPPED, make_args(source="webcam"), cap)
    assert cap.seeks == []


@pytest.mark.parametrize(
    "status, message",
    [(CALIBRATION_SKIPPED, "Using default thresholds."), (CALIBRATION_INSUFFICIENT, "No usable calibration. Using default thresholds.")],
)
def test_calibration_without_a_profile_falls_back_to_the_defaults(monkeypatch, capsys, status, message):
    (profile, stop), _ = run_calibration_with(monkeypatch, None, status)
    assert profile == get_default_calibration_profile() and stop is False
    assert message in capsys.readouterr().out


def test_quitting_during_calibration_stops_the_run(monkeypatch):
    (profile, stop), _ = run_calibration_with(monkeypatch, None, CALIBRATION_QUIT)
    assert (profile, stop) == (None, True)


def test_no_display_is_passed_on_to_calibration(monkeypatch):
    _, seen = run_calibration_with(monkeypatch, None, CALIBRATION_SKIPPED, make_args(no_display=True))
    assert seen["show_window"] is False


# ---------------------------------------------------------------- showing frames
def test_show_frame_without_a_window_only_reports_progress(monkeypatch, capsys):
    monkeypatch.setattr(app.cv2, "imshow", lambda *_: pytest.fail("no window expected"))
    state = make_state(args=make_args(no_display=True), total_frames=300)
    state.frame_index = 99
    assert app.show_frame(state, object()) is False
    assert capsys.readouterr().out == ""
    state.frame_index = 100
    assert app.show_frame(state, object()) is False
    assert "Processed 100/300 frames" in capsys.readouterr().out


@pytest.mark.parametrize("key, closed, stops", [(-1, False, False), (ord("q"), False, True), (ord("Q"), False, True), (-1, True, True)])
def test_show_frame_stops_on_q_or_a_closed_window(monkeypatch, key, closed, stops):
    monkeypatch.setattr(app.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda *_: key)
    monkeypatch.setattr(app, "window_was_closed", lambda: closed)
    assert app.show_frame(make_state(), object()) is stops


# ---------------------------------------------------------------- the frame loop
def run_stream(monkeypatch, outcomes, frames=None):
    """Run analyse_stream where process_frame follows `outcomes` (True = ok, False = raises)."""
    calls = iter(outcomes)

    def fake_process(state, raw_frame):
        state.frame_index += 1
        if not next(calls):
            raise ValueError(f"bad frame {state.frame_index}")
        return "frame"

    monkeypatch.setattr(app, "process_frame", fake_process)
    monkeypatch.setattr(app, "show_frame", lambda state, frame: False)
    state = make_state(cap=FakeCap(frames=frames if frames is not None else len(outcomes)))
    return state


def test_every_frame_is_read_until_the_source_ends(monkeypatch):
    state = run_stream(monkeypatch, [True] * 10)
    app.analyse_stream(state)
    assert state.frame_index == 10 and state.failed_frames == 0


def test_a_bad_frame_is_skipped_and_counted(monkeypatch, capsys):
    state = run_stream(monkeypatch, [True, True, False, True, True])
    app.analyse_stream(state)
    assert state.frame_index == 5 and state.failed_frames == 1
    out = capsys.readouterr().out
    assert "Frame 3 could not be processed" in out


def test_only_the_first_of_a_series_of_failures_prints_a_traceback(monkeypatch, capsys):
    state = run_stream(monkeypatch, [False, False, False, True])
    app.analyse_stream(state)
    out = capsys.readouterr()
    assert (out.out + out.err).count("Traceback") == 1
    assert "Frame 2 skipped again (ValueError: bad frame 2)" in out.out


def test_failures_that_are_not_in_a_row_never_stop_the_run(monkeypatch):
    state = run_stream(monkeypatch, ([False] * (app.MAX_CONSECUTIVE_FRAME_ERRORS - 1) + [True]) * 3)
    app.analyse_stream(state)
    assert state.failed_frames == 3 * (app.MAX_CONSECUTIVE_FRAME_ERRORS - 1)


def test_too_many_failures_in_a_row_stop_the_run(monkeypatch):
    state = run_stream(monkeypatch, [False] * 100, frames=100)
    with pytest.raises(RuntimeError, match="frames in a row failed"):
        app.analyse_stream(state)
    assert state.failed_frames == app.MAX_CONSECUTIVE_FRAME_ERRORS


def test_the_user_stopping_ends_the_loop_without_a_failure(monkeypatch):
    state = run_stream(monkeypatch, [True] * 10)
    monkeypatch.setattr(app, "show_frame", lambda s, f: s.frame_index == 4)
    app.analyse_stream(state)
    assert state.frame_index == 4 and state.failed_frames == 0


def test_processed_frames_are_written_to_the_output_video(monkeypatch):
    written = []
    state = run_stream(monkeypatch, [True] * 3)
    state.output_writer = SimpleNamespace(write=written.append)
    app.analyse_stream(state)
    assert written == ["frame"] * 3


def test_the_end_of_a_video_prints_the_stopping_banner_but_a_webcam_does_not(monkeypatch, capsys):
    app.analyse_stream(run_stream(monkeypatch, [True]))
    assert "GENERATING ENGAGEMENT ANALYSIS SUMMARY" in capsys.readouterr().out
    state = run_stream(monkeypatch, [True])
    state.args = make_args(source="webcam")
    app.analyse_stream(state)
    assert "GENERATING ENGAGEMENT ANALYSIS SUMMARY" not in capsys.readouterr().out


# ---------------------------------------------------------------- session state and cleanup
def test_each_session_state_has_its_own_collections_and_tracker():
    first, second = make_state(), make_state()
    first.statements.append("x")
    first.face_tracker.match(1, [(0.5, 0.5)])
    assert second.statements == [] and second.face_tracker.tracks == []
    assert first.frame_index == 0 and first.frames_with_faces == 0 and first.failed_frames == 0


def test_release_resources_releases_everything_even_if_the_video_close_fails(monkeypatch):
    closed = {"mesh": False, "windows": 0}
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: closed.__setitem__("windows", closed["windows"] + 1))
    models = SimpleNamespace(face_mesh=SimpleNamespace(close=lambda: closed.__setitem__("mesh", True)))
    cap = FakeCap()

    class BadWriter:
        def close(self):
            raise OSError("disk full")

    with pytest.raises(OSError):
        app.release_resources(make_args(), cap, BadWriter(), models)
    assert cap.released and closed == {"mesh": True, "windows": 1}


def test_release_resources_skips_windows_without_a_display(monkeypatch):
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: pytest.fail("no window to destroy"))
    models = SimpleNamespace(face_mesh=SimpleNamespace(close=lambda: None))
    app.release_resources(make_args(no_display=True), FakeCap(), None, models)
