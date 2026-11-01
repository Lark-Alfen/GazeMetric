from types import SimpleNamespace

import numpy as np
import pytest

from core.analysis_logic import compute_gaze_scores, extract_landmark_arrays

LANDMARK_COUNT = 478
CENTER_GAZE = {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5}


def fake_face(overrides=None, width=1000, height=800):
    """Normalized landmarks laid out so the eye ratios are easy to predict."""
    points = {index: (0.5, 0.5) for index in range(LANDMARK_COUNT)}
    points.update(overrides or {})
    landmarks = [SimpleNamespace(x=points[i][0], y=points[i][1], z=0.0) for i in range(LANDMARK_COUNT)]
    return SimpleNamespace(landmark=landmarks), width, height


def eye_points(inner, outer, iris, top, bottom, iris_y):
    return {
        130: (inner, 0.5),
        243: (outer, 0.5),
        468: (iris, iris_y),
        27: (0.5, top),
        23: (0.5, bottom),
    }


def test_landmarks_keep_sub_pixel_coordinates():
    face, w, h = fake_face({33: (0.12345, 0.54321)})
    face_2d, face_2d_head = extract_landmark_arrays(face, w, h)
    assert face_2d.dtype == np.float64
    assert face_2d[33] == pytest.approx([123.45, 434.568])
    assert face_2d_head[2] == pytest.approx([123.45, 434.568])


def test_all_landmarks_are_in_pixels_not_truncated():
    face, w, h = fake_face({i: (i / 1000.0 + 0.0004, 0.3 + i / 100000.0) for i in range(LANDMARK_COUNT)})
    face_2d, _ = extract_landmark_arrays(face, w, h)
    assert np.any(face_2d != np.floor(face_2d))
    assert face_2d[:, 0] == pytest.approx([(i / 1000.0 + 0.0004) * w for i in range(LANDMARK_COUNT)])


def test_horizontal_gaze_ratio_uses_sub_pixel_positions():
    # inner corner 100.3px, outer 140.8px, iris 114.8px: true ratio 14.5 / 40.5
    overrides = eye_points(inner=0.1003, outer=0.1408, iris=0.1148, top=0.45, bottom=0.55, iris_y=0.5)
    face, w, h = fake_face(overrides, width=1000)
    face_2d, _ = extract_landmark_arrays(face, w, h)
    scores = compute_gaze_scores(face_2d, CENTER_GAZE, threshold=0.0)
    assert scores["lx"] == pytest.approx(14.5 / 40.5)
    # truncating the three x values to whole pixels would give 14 / 40 = 0.35, a 0.008 error
    assert abs(scores["lx"] - 14 / 40) > 0.005


def test_vertical_gaze_ratio_uses_sub_pixel_positions():
    overrides = eye_points(inner=0.4, outer=0.5, iris=0.45, top=0.4005, bottom=0.4505, iris_y=0.4155)
    face, w, h = fake_face(overrides, height=800)
    face_2d, _ = extract_landmark_arrays(face, w, h)
    scores = compute_gaze_scores(face_2d, CENTER_GAZE, threshold=0.0)
    assert scores["ly"] == pytest.approx((0.4155 - 0.4005) / (0.4505 - 0.4005))
