"""Annotated output video writer that matches the playback speed to the real processing rate."""

import os
import time

import cv2

PROBE_FRAMES = 30
MIN_FPS = 1.0
MAX_FPS = 120.0


def open_video_writer(path, fps, frame_size):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {path}")
    return writer


def estimate_fps(frame_times, fallback_fps):
    """Frames per second actually achieved, from the wall-clock time each frame was handed over."""
    if len(frame_times) >= 2:
        elapsed = frame_times[-1] - frame_times[0]
        if elapsed > 0:
            return min(MAX_FPS, max(MIN_FPS, (len(frame_times) - 1) / elapsed))
    return fallback_fps


class AnnotatedVideoWriter:
    """Writes annotated frames to a video file.

    With a fixed `fps` (recorded input, which has a real frame rate) the file is created right away. Without
    one (webcam), many cameras report a frame rate the processing loop cannot reach, so the first
    `probe_frames` frames are held back, the achieved rate is measured over them, and the file is created
    with that rate. A frame must not be modified after it has been passed to `write`.
    """

    def __init__(self, path, fallback_fps, fps=None, probe_frames=PROBE_FRAMES, clock=time.perf_counter, open_writer=open_video_writer):
        self.path = path
        self.fallback_fps = fallback_fps
        self.fps = fps
        self.probe_frames = probe_frames
        self._clock = clock
        self._open_writer = open_writer
        self._writer = None
        self._pending_frames = []
        self._pending_times = []

    def write(self, frame):
        if self._writer is None and self.fps is not None:
            self._open(self.fps, frame)

        if self._writer is not None:
            self._writer.write(frame)
            return

        self._pending_frames.append(frame)
        self._pending_times.append(self._clock())
        if len(self._pending_frames) >= self.probe_frames:
            self._open_from_pending_frames()

    def close(self):
        """Finish the file. Returns True when a video was written."""
        if self._writer is None and self._pending_frames:
            self._open_from_pending_frames()
        if self._writer is None:
            return False
        self._writer.release()
        return True

    def _open_from_pending_frames(self):
        self.fps = estimate_fps(self._pending_times, self.fallback_fps)
        self._open(self.fps, self._pending_frames[0])
        for frame in self._pending_frames:
            self._writer.write(frame)
        self._pending_frames = []
        self._pending_times = []

    def _open(self, fps, first_frame):
        frame_h, frame_w = first_frame.shape[:2]
        self._writer = self._open_writer(self.path, fps, (frame_w, frame_h))
