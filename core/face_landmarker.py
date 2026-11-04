"""Face landmark detection with the MediaPipe Face Landmarker (Tasks API) and its model file."""

import hashlib
import os
import tempfile
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "face_landmarker.task")  # the Models folder is git-ignored


def missing_model_message(model_path):
    return (
        f"Face model not found: {model_path}\n"
        "Download it once with:  python download_face_model.py\n"
        f"or get it from {MODEL_URL} and pass its location with --face-model."
    )


def download_model(model_path=DEFAULT_MODEL_PATH, url=MODEL_URL, expected_sha256=MODEL_SHA256, opener=urllib.request.urlopen):
    """Download the model to model_path, check it against the known SHA-256 and move it into place.

    The file is only put at model_path when the checksum matches, so a failed or tampered download never
    leaves something that looks like a model.
    """
    target_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(target_dir, exist_ok=True)

    digest = hashlib.sha256()
    handle, temp_path = tempfile.mkstemp(dir=target_dir, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as temp_file, opener(url) as response:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                digest.update(chunk)
                temp_file.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Downloaded file does not match the expected checksum ({digest.hexdigest()}); not saved.")
        os.replace(temp_path, model_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return model_path


class FaceLandmarkDetector:
    """Finds the faces and their 478 landmarks in a stream of RGB frames.

    Wraps the Face Landmarker in video mode, which tracks faces from frame to frame and needs timestamps that
    only ever increase. `detect` guarantees that, whatever clock the caller uses.
    """

    def __init__(self, model_path, max_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.model_path = model_path
        self.max_faces = max_faces
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker = None
        self._last_timestamp_ms = -1
        self._connections = None
        self._create()

    def _create(self):
        # imported here because MediaPipe is slow to load and --help does not need it
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
            min_face_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        if self._connections is None:
            self._connections = [(c.start, c.end) for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]

    @property
    def contour_connections(self):
        """(start, end) landmark index pairs that outline the eyes, brows, lips and face oval."""
        return self._connections

    def detect(self, rgb_frame, timestamp_ms):
        """Landmarks for every face in the frame: a list with one list of landmarks (.x, .y, .z) per face."""
        import mediapipe as mp

        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(image, timestamp_ms).face_landmarks

    def restart(self):
        """Forget the tracking state, e.g. when a video is played again from its first frame."""
        self.close()
        self._create()

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
