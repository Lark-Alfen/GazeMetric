"""Opening the video source: a webcam by index, or a recorded file."""

import os

import cv2


def open_capture(source_type, input_path=None, camera_index=0):
    """Open the webcam (DirectShow on Windows, which starts much faster than the default backend) or a video file."""
    if source_type == "webcam":
        if os.name == "nt":
            return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(camera_index)
    return cv2.VideoCapture(input_path)


def open_failure_message(source_type, input_path=None, camera_index=0):
    if source_type == "webcam":
        return (
            f"Unable to open camera {camera_index}. Check that it is connected and not used by another program, "
            "or choose another one with --camera-index."
        )
    return f"Unable to open video file: {input_path}"
