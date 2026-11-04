import hashlib
import io
import os

import numpy as np
import pytest

import download_face_model
from core import face_landmarker
from core.face_landmarker import (
    DEFAULT_MODEL_PATH,
    MODEL_URL,
    FaceLandmarkDetector,
    download_model,
    missing_model_message,
)

PAYLOAD = b"pretend this is a model file" * 1000
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def opener_for(data, fail_after=None):
    def opener(url):
        assert url == MODEL_URL
        response = FakeResponse(data)
        if fail_after is not None:
            original_read = response.read
            state = {"served": 0}

            def read(size=-1):
                if state["served"] >= fail_after:
                    raise ConnectionError("connection dropped")
                chunk = original_read(size)
                state["served"] += len(chunk)
                return chunk

            response.read = read
        return response

    return opener


# ---------------------------------------------------------------- model path and message
def test_default_model_path_is_in_the_ignored_models_folder_of_the_project():
    assert DEFAULT_MODEL_PATH == os.path.join(face_landmarker.PROJECT_ROOT, "Models", "face_landmarker.task")
    assert os.path.isabs(DEFAULT_MODEL_PATH)
    with open(".gitignore", encoding="utf-8") as handle:
        assert "Models/" in handle.read().split()


def test_missing_model_message_says_how_to_get_the_model():
    message = missing_model_message("somewhere/model.task")
    assert "somewhere/model.task" in message
    assert "python download_face_model.py" in message
    assert "--face-model" in message and MODEL_URL in message


# ---------------------------------------------------------------- download_model
def test_download_saves_the_file_when_the_checksum_matches(tmp_path):
    target = tmp_path / "Models" / "face_landmarker.task"
    result = download_model(str(target), expected_sha256=PAYLOAD_SHA256, opener=opener_for(PAYLOAD))
    assert result == str(target)
    assert target.read_bytes() == PAYLOAD
    assert [p.name for p in target.parent.iterdir()] == ["face_landmarker.task"]  # no temporary file left


def test_a_wrong_checksum_saves_nothing(tmp_path):
    target = tmp_path / "model.task"
    with pytest.raises(ValueError, match="checksum"):
        download_model(str(target), expected_sha256="0" * 64, opener=opener_for(PAYLOAD))
    assert list(tmp_path.iterdir()) == []


def test_a_failed_download_keeps_an_existing_model_untouched(tmp_path):
    target = tmp_path / "model.task"
    target.write_bytes(b"the model that already works")
    with pytest.raises(ValueError):
        download_model(str(target), expected_sha256="0" * 64, opener=opener_for(b"corrupted"))
    assert target.read_bytes() == b"the model that already works"
    assert [p.name for p in tmp_path.iterdir()] == ["model.task"]


def test_a_dropped_connection_leaves_no_partial_file(tmp_path):
    target = tmp_path / "model.task"
    with pytest.raises(ConnectionError):
        download_model(str(target), expected_sha256=PAYLOAD_SHA256, opener=opener_for(PAYLOAD, fail_after=1000))
    assert list(tmp_path.iterdir()) == []


def test_the_pinned_checksum_is_a_sha256():
    assert len(face_landmarker.MODEL_SHA256) == 64 and int(face_landmarker.MODEL_SHA256, 16) >= 0


# ---------------------------------------------------------------- download_face_model.py
def test_download_script_reports_success(tmp_path, monkeypatch, capsys):
    saved = []
    monkeypatch.setattr(download_face_model, "download_model", lambda path: saved.append(path) or path)
    monkeypatch.setattr("sys.argv", ["download_face_model.py", "--output", str(tmp_path / "m.task")])
    assert download_face_model.main() == 0
    assert saved == [str(tmp_path / "m.task")]
    assert "Saved the model" in capsys.readouterr().out


def test_download_script_reports_failure_with_a_non_zero_exit_code(monkeypatch, capsys):
    def fail(path):
        raise OSError("no network")

    monkeypatch.setattr(download_face_model, "download_model", fail)
    monkeypatch.setattr("sys.argv", ["download_face_model.py"])
    assert download_face_model.main() == 1
    assert "Download failed: no network" in capsys.readouterr().out


# ---------------------------------------------------------------- the real detector (needs the model and the new MediaPipe)
def real_model_path():
    path = os.environ.get("FACE_MODEL", DEFAULT_MODEL_PATH)
    if not os.path.isfile(path):
        pytest.skip(f"face model not downloaded ({path}); run python download_face_model.py")
    try:
        from mediapipe.tasks.python import vision

        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS  # only in current MediaPipe releases
    except (ImportError, AttributeError):
        pytest.skip("installed MediaPipe is too old for the Face Landmarker port; pip install -r requirements.txt")
    return path


@pytest.fixture
def detector():
    created = FaceLandmarkDetector(real_model_path(), max_faces=2)
    yield created
    created.close()


def blank(height=240, width=320):
    return np.full((height, width, 3), 127, dtype=np.uint8)


def test_a_frame_without_a_face_gives_no_faces(detector):
    assert list(detector.detect(blank(), 0)) == []


def test_timestamps_that_repeat_or_go_backwards_are_accepted(detector):
    for stamp in (100, 100, 100, 50, 0, 200.7, 200.2):
        assert list(detector.detect(blank(), stamp)) == []


def test_restart_can_start_the_timeline_again_from_zero(detector):
    for stamp in (0, 40, 80):
        detector.detect(blank(), stamp)
    detector.restart()
    for stamp in (0, 40):
        assert list(detector.detect(blank(), stamp)) == []


def test_contour_connections_are_landmark_index_pairs(detector):
    pairs = detector.contour_connections
    assert len(pairs) > 100
    assert all(isinstance(a, int) and isinstance(b, int) and 0 <= a < 478 and 0 <= b < 478 for a, b in pairs)


def test_close_is_safe_to_call_twice():
    created = FaceLandmarkDetector(real_model_path(), max_faces=1)
    created.close()
    created.close()
