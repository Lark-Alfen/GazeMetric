from types import SimpleNamespace

import numpy as np
import pytest

from core import calibration_runtime
from core.analysis_logic import get_default_calibration_profile
from core.calibration_runtime import (
    CALIBRATION_COMPLETED,
    CALIBRATION_INSUFFICIENT,
    CALIBRATION_QUIT,
    CALIBRATION_SKIPPED,
    get_calibration_plan,
    run_calibration_phase,
)


# ---------------------------------------------------------------- get_calibration_plan
def test_webcam_plan_is_wall_clock_based():
    seconds, frame_limit, min_samples = get_calibration_plan("webcam", 10.0, 30.0)
    assert (seconds, frame_limit, min_samples) == (10.0, None, 80)


def test_webcam_plan_enforces_the_minimums():
    assert get_calibration_plan("webcam", 0.2, 30.0) == (1.0, None, 20)


def test_video_plan_counts_frames_of_the_video():
    seconds, frame_limit, min_samples = get_calibration_plan("video", 5.0, 30.0)
    assert seconds == 5.0
    assert frame_limit == 150
    assert min_samples == 60  # 40% of the frames in the window


@pytest.mark.parametrize("fps", [12.0, 25.0, 60.0])
def test_video_plan_scales_with_the_frame_rate(fps):
    _, frame_limit, min_samples = get_calibration_plan("video", 5.0, fps)
    assert frame_limit == int(5.0 * fps)
    assert min_samples == max(20, int(0.4 * frame_limit))


def test_video_min_samples_never_exceeds_the_frames_available():
    _, frame_limit, min_samples = get_calibration_plan("video", 1.0, 10.0)
    assert frame_limit == 10
    assert min_samples == 10


# ---------------------------------------------------------------- run_calibration_phase
class FakeCapture:
    def __init__(self, frames):
        self.remaining = frames
        self.read_count = 0

    def isOpened(self):
        return True

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        self.read_count += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)


class FakeFaceMesh:
    def __init__(self, face_present):
        self.face_present = face_present

    def process(self, _frame):
        return SimpleNamespace(multi_face_landmarks=[object()] if self.face_present else None)


class FakeDrawing:
    @staticmethod
    def draw_landmarks(**_kwargs):
        pass


def run(frames, face_present, source_type="video", seconds=1.0, fps=25.0, build_profile=None):
    captured = {}

    def fake_build_profile(samples, min_samples):
        captured["samples"] = len(samples)
        captured["min_samples"] = min_samples
        profile = get_default_calibration_profile()
        profile["sample_count"] = len(samples)
        return profile

    cap = FakeCapture(frames)
    result = run_calibration_phase(
        cap=cap,
        face_mesh=FakeFaceMesh(face_present),
        mp_face_mesh=SimpleNamespace(FACEMESH_CONTOURS=None),
        mp_drawing=FakeDrawing,
        source_type=source_type,
        calibration_seconds=seconds,
        source_fps=fps,
        threshold=0.3,
        leye_3d=None,
        reye_3d=None,
        extract_landmark_arrays=lambda *_: (np.zeros((478, 2)), np.zeros((6, 2))),
        compute_gaze_scores=lambda *_: {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5},
        build_camera_and_distortion=lambda *_: (None, None),
        solve_pose_vectors=lambda *_: {"x_angle": 0.0, "y_angle": 0.0},
        build_calibration_profile=build_profile or fake_build_profile,
    )
    return result, cap, captured


@pytest.fixture(autouse=True)
def no_window(monkeypatch):
    monkeypatch.setattr(calibration_runtime.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(calibration_runtime.cv2, "waitKey", lambda *_: -1)
    monkeypatch.setattr(calibration_runtime.cv2, "putText", lambda *_args, **_kwargs: None)


def test_video_calibration_stops_after_the_frames_of_the_window():
    (profile, status), cap, captured = run(frames=500, face_present=True, seconds=2.0, fps=25.0)
    assert status == CALIBRATION_COMPLETED
    assert cap.read_count == 50  # 2 seconds of a 25 fps video, not the whole file
    assert captured["samples"] == 50
    assert captured["min_samples"] == 20
    assert profile["sample_count"] == 50


def test_video_calibration_uses_the_same_frames_whatever_the_processing_speed(monkeypatch):
    clock = iter(range(0, 10_000, 100))  # a very slow machine: 100 s pass per frame
    monkeypatch.setattr(calibration_runtime.time, "perf_counter", lambda: next(clock))
    (_, status), cap, _ = run(frames=500, face_present=True, seconds=2.0, fps=25.0)
    assert status == CALIBRATION_COMPLETED
    assert cap.read_count == 50


def test_webcam_calibration_still_ends_on_wall_clock_time(monkeypatch):
    ticks = iter([0.0, 0.5, 1.0, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
    monkeypatch.setattr(calibration_runtime.time, "perf_counter", lambda: next(ticks))
    (_, status), cap, _ = run(frames=500, face_present=True, source_type="webcam", seconds=2.0, fps=30.0)
    assert status == CALIBRATION_COMPLETED
    assert cap.read_count == 4


def test_short_video_uses_the_frames_it_has():
    (_, status), cap, captured = run(frames=30, face_present=True, seconds=5.0, fps=25.0)
    assert cap.read_count == 30
    assert captured["samples"] == 30
    assert status == CALIBRATION_COMPLETED


def test_no_face_gives_insufficient():
    (profile, status), _, _ = run(frames=100, face_present=False, seconds=2.0)
    assert profile is None
    assert status == CALIBRATION_INSUFFICIENT


def test_pressing_s_reports_skipped_not_insufficient(monkeypatch):
    monkeypatch.setattr(calibration_runtime.cv2, "waitKey", lambda *_: ord("s"))
    (profile, status), _, _ = run(frames=100, face_present=True)
    assert profile is None
    assert status == CALIBRATION_SKIPPED


def test_pressing_q_reports_quit(monkeypatch):
    monkeypatch.setattr(calibration_runtime.cv2, "waitKey", lambda *_: ord("Q"))
    (profile, status), _, _ = run(frames=100, face_present=True)
    assert profile is None
    assert status == CALIBRATION_QUIT


def test_rejected_calibration_reports_insufficient():
    def reject(samples, min_samples):
        return get_default_calibration_profile()  # sample_count 0, what a rejected calibration returns

    (profile, status), _, _ = run(frames=100, face_present=True, build_profile=reject)
    assert profile is None
    assert status == CALIBRATION_INSUFFICIENT
