import cv2
import pytest

from core import capture
from core.capture import open_capture, open_failure_message


@pytest.fixture
def opened(monkeypatch):
    calls = []

    def fake_capture(*args):
        calls.append(args)
        return object()

    monkeypatch.setattr(capture.cv2, "VideoCapture", fake_capture)
    return calls


def test_webcam_index_is_passed_through_with_directshow_on_windows(opened, monkeypatch):
    monkeypatch.setattr(capture.os, "name", "nt")
    open_capture("webcam", camera_index=3)
    assert opened == [(3, cv2.CAP_DSHOW)]


def test_webcam_uses_the_default_backend_elsewhere(opened, monkeypatch):
    monkeypatch.setattr(capture.os, "name", "posix")
    open_capture("webcam", camera_index=1)
    assert opened == [(1,)]


def test_first_camera_is_the_default(opened, monkeypatch):
    monkeypatch.setattr(capture.os, "name", "posix")
    open_capture("webcam")
    assert opened == [(0,)]


@pytest.mark.parametrize("os_name", ["nt", "posix"])
def test_video_files_ignore_the_camera_index_and_backend(opened, monkeypatch, os_name):
    monkeypatch.setattr(capture.os, "name", os_name)
    open_capture("video", "clip.mp4", camera_index=5)
    assert opened == [("clip.mp4",)]


def test_failure_message_for_a_camera_names_the_index_and_the_option():
    message = open_failure_message("webcam", camera_index=4)
    assert "camera 4" in message
    assert "--camera-index" in message


def test_failure_message_for_a_video_names_the_file():
    message = open_failure_message("video", "missing.mp4")
    assert "missing.mp4" in message
    assert "--camera-index" not in message
