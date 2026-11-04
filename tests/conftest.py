"""Fixtures that run the real main() on a generated video, with a fake face tracker and no windows."""

import argparse
import contextlib
import csv
import glob
import io
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests.test_pose import project_head

WIDTH, HEIGHT = 640, 360
FPS = 25.0


def make_video(path, frames=40):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), (40 + i) % 256, dtype=np.uint8))
    writer.release()
    return str(path)


def fake_landmarks(shift_x=0.0, yaw=-10):
    """478 normalized landmarks whose six head landmarks form a plausible, slightly turned face."""
    head = project_head(5, yaw)  # pixels in a 1280x720 frame
    points = np.full((478, 2), 0.5)
    for index, row in zip((1, 199, 33, 263, 61, 291), head):
        points[index] = (row[0] / 1280.0, row[1] / 720.0)
    # open eyes with the iris in the middle, so the gaze scores are well defined
    for iris, (x0, x1, lid_up, lid_low, corner_in, corner_out), eye_x in (
        (468, (130, 243, 27, 23, 133, 33), 0.40),
        (473, (463, 359, 257, 253, 362, 263), 0.58),
    ):
        left, right = eye_x - 0.04, eye_x + 0.04
        points[x0] = (left, 0.45)
        points[x1] = (right, 0.45)
        points[iris] = (eye_x, 0.45)
        points[lid_up] = (eye_x, 0.43)
        points[lid_low] = (eye_x, 0.47)
        points[corner_in] = (right, 0.45)
        points[corner_out] = (left, 0.45)
    points[159], points[145] = (0.40, 0.43), (0.40, 0.47)
    points[386], points[374] = (0.58, 0.43), (0.58, 0.47)
    points[:, 0] += shift_x
    return [SimpleNamespace(x=float(x), y=float(y), z=0.0) for x, y in points]


class FakeDetector:
    """Stands in for core.face_landmarker.FaceLandmarkDetector: same interface, faces made up by the test."""

    instances = []
    faces = 1
    people = None  # optional [(shift_x, yaw), ...]; the detection order flips on every other frame
    calls = 0
    contour_connections = [(0, 1), (1, 2), (2, 3)]

    def __init__(self, model_path, max_faces=2, **_kwargs):
        self.model_path = model_path
        self.max_faces = max_faces
        self.closed = False
        self.restarts = 0
        self.timestamps = []
        FakeDetector.instances.append(self)

    def detect(self, _frame, timestamp_ms):
        self.timestamps.append(int(timestamp_ms))
        if FakeDetector.people:
            FakeDetector.calls += 1
            people = FakeDetector.people if FakeDetector.calls % 2 else FakeDetector.people[::-1]
            return [fake_landmarks(shift, yaw) for shift, yaw in people]
        return [fake_landmarks() for _ in range(FakeDetector.faces)]

    def restart(self):
        self.restarts += 1

    def close(self):
        self.closed = True


class MainRun:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.keys = []  # scripted waitKey return values, then -1
        self.window_visible = lambda: True
        self.captures = []
        self.output = ""

        FakeDetector.instances = []
        FakeDetector.faces = 1
        FakeDetector.people = None
        FakeDetector.calls = 0
        monkeypatch.setattr("engagement_analysis_full.FaceLandmarkDetector", FakeDetector)
        self.model_path = str(tmp_path / "face_landmarker.task")
        with open(self.model_path, "wb") as handle:
            handle.write(b"stand-in for the model file; the fake detector never reads it")

        real_capture = cv2.VideoCapture
        run = self

        class TrackedCapture:
            def __init__(self, *args):
                self._capture = real_capture(*args)
                self.released = False
                run.captures.append(self)

            def release(self):
                self.released = True
                self._capture.release()

            def __getattr__(self, name):
                return getattr(self._capture, name)

        monkeypatch.setattr(cv2, "VideoCapture", TrackedCapture)
        monkeypatch.setattr(cv2, "imshow", lambda *_: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(cv2, "waitKey", self._wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 1.0 if self.window_visible() else 0.0)

    def _wait_key(self, _delay):
        return self.keys.pop(0) if self.keys else -1

    @property
    def detector(self):
        return FakeDetector.instances[-1]

    def args(self, video, **overrides):
        values = dict(
            source="video",
            input_path=video,
            output_video_path=None,
            timeline_dir=str(self.tmp_path / "runs"),
            skip_calibration=True,
            calibration_seconds=1.0,
            calibration_profile=None,
            no_display=False,
            camera_index=0,
            max_faces=2,
            face_model=self.model_path,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def run(self, args):
        from engagement_analysis_full import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exit_code = main(args)
        self.output = out.getvalue() + err.getvalue()
        return exit_code

    def session_dir(self):
        sessions = glob.glob(os.path.join(str(self.tmp_path), "runs", "session_*"))
        return sessions[-1] if sessions else None

    def timeline_rows(self):
        session = self.session_dir()
        if session is None or not os.path.exists(os.path.join(session, "timeline.csv")):
            return None
        with open(os.path.join(session, "timeline.csv"), newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


@pytest.fixture
def main_run(tmp_path, monkeypatch):
    return MainRun(tmp_path, monkeypatch)


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "clip.mp4")
