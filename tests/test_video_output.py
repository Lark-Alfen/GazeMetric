import cv2
import numpy as np
import pytest

from core.video_output import MAX_FPS, MIN_FPS, AnnotatedVideoWriter, estimate_fps


class FakeWriter:
    def __init__(self, path, fps, frame_size):
        self.path, self.fps, self.frame_size = path, fps, frame_size
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class Recorder:
    """Stands in for open_video_writer and remembers what it opened."""

    def __init__(self):
        self.writers = []

    def __call__(self, path, fps, frame_size):
        writer = FakeWriter(path, fps, frame_size)
        self.writers.append(writer)
        return writer


class Clock:
    def __init__(self, step):
        self.now = 100.0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def frame(value=0, width=64, height=48):
    return np.full((height, width, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------- estimate_fps
def test_estimate_fps_from_frame_times():
    times = [10.0 + i * 0.1 for i in range(30)]
    assert estimate_fps(times, fallback_fps=30.0) == pytest.approx(10.0)


def test_estimate_fps_falls_back_without_enough_data():
    assert estimate_fps([], 25.0) == 25.0
    assert estimate_fps([1.0], 25.0) == 25.0
    assert estimate_fps([1.0, 1.0], 25.0) == 25.0


def test_estimate_fps_is_clamped():
    assert estimate_fps([0.0, 100.0], 25.0) == MIN_FPS
    assert estimate_fps([0.0, 0.0001], 25.0) == MAX_FPS


# ---------------------------------------------------------------- AnnotatedVideoWriter
def test_fixed_fps_opens_immediately_with_that_rate():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=20.0, fps=29.97, open_writer=opened)
    writer.write(frame())
    assert len(opened.writers) == 1
    assert opened.writers[0].fps == 29.97
    assert opened.writers[0].frame_size == (64, 48)
    assert len(opened.writers[0].frames) == 1


def test_measured_fps_is_used_when_none_is_given():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=30.0, probe_frames=30, clock=Clock(0.1), open_writer=opened)
    for i in range(29):
        writer.write(frame(i))
    assert opened.writers == []  # still measuring
    writer.write(frame(29))
    assert len(opened.writers) == 1
    assert opened.writers[0].fps == pytest.approx(10.0)  # not the camera's nominal 30


def test_buffered_frames_are_written_first_and_in_order():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=30.0, probe_frames=5, clock=Clock(0.1), open_writer=opened)
    for i in range(8):
        writer.write(frame(i))
    assert [int(f[0][0][0]) for f in opened.writers[0].frames] == list(range(8))


def test_session_shorter_than_the_probe_is_still_saved():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=30.0, probe_frames=30, clock=Clock(0.05), open_writer=opened)
    for i in range(10):
        writer.write(frame(i))
    assert writer.close() is True
    assert opened.writers[0].fps == pytest.approx(20.0)
    assert len(opened.writers[0].frames) == 10
    assert opened.writers[0].released


def test_single_frame_session_uses_the_fallback_rate():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=24.0, open_writer=opened)
    writer.write(frame())
    assert writer.close() is True
    assert opened.writers[0].fps == 24.0


def test_close_without_frames_writes_nothing():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=24.0, open_writer=opened)
    assert writer.close() is False
    assert opened.writers == []


def test_close_releases_the_file():
    opened = Recorder()
    writer = AnnotatedVideoWriter("out.mp4", fallback_fps=24.0, fps=24.0, open_writer=opened)
    writer.write(frame())
    assert writer.close() is True
    assert opened.writers[0].released


def test_saved_file_plays_back_at_the_measured_rate(tmp_path):
    """Real encoder: a loop that managed 10 fps must give a 4 second file, whatever the camera claims."""
    path = str(tmp_path / "nested" / "out.mp4")
    writer = AnnotatedVideoWriter(path, fallback_fps=30.0, probe_frames=30, clock=Clock(0.1))
    for i in range(40):
        writer.write(frame(i, width=160, height=120))
    assert writer.close() is True

    capture = cv2.VideoCapture(path)
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.01)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frames == 40
    assert frames / capture.get(cv2.CAP_PROP_FPS) == pytest.approx(4.0, abs=0.05)
    capture.release()
