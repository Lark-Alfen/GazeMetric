from types import SimpleNamespace

import numpy as np
import pytest

from core.analysis_logic import (
    classify_gaze_direction,
    compute_eye_openness,
    compute_gaze_scores,
    extract_landmark_arrays,
    get_default_calibration_profile,
)

LANDMARK_COUNT = 478
CENTER_GAZE = {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5}


def fake_face(overrides=None, width=1000, height=800):
    """Normalized landmarks laid out so the eye ratios are easy to predict."""
    points = {index: (0.5, 0.5) for index in range(LANDMARK_COUNT)}
    points.update(overrides or {})
    landmarks = [SimpleNamespace(x=points[i][0], y=points[i][1], z=0.0) for i in range(LANDMARK_COUNT)]
    return landmarks, width, height


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
    overrides.update({33: (0.3, 0.5), 133: (0.4, 0.5), 159: (0.35, 0.4005), 145: (0.35, 0.4505)})  # an open eye
    face, w, h = fake_face(overrides, height=800)
    face_2d, _ = extract_landmark_arrays(face, w, h)
    scores = compute_gaze_scores(face_2d, CENTER_GAZE, threshold=0.0)
    assert scores["ly"] == pytest.approx((0.4155 - 0.4005) / (0.4505 - 0.4005))


# ---------------------------------------------------------------- blinks
def face_with_eyes(gap, iris_y_fraction=0.8, width=1000, height=800):
    """Both eyes 40 px wide with the given lid gap (px) and the iris `iris_y_fraction` of the way down the lids."""
    top = 400.0
    overrides = {}
    for outer, inner, upper, lower, iris, lid_up, lid_low, x_in, x_out in (
        (33, 133, 159, 145, 468, 27, 23, 130, 243),
        (263, 362, 386, 374, 473, 257, 253, 463, 359),
    ):
        left_x = 300.0 if outer == 33 else 600.0
        overrides[outer] = (left_x / width, 0.5)
        overrides[inner] = ((left_x + 40.0) / width, 0.5)
        overrides[upper] = ((left_x + 20.0) / width, top / height)
        overrides[lower] = ((left_x + 20.0) / width, (top + gap) / height)
        overrides[lid_up] = ((left_x + 20.0) / width, top / height)
        overrides[lid_low] = ((left_x + 20.0) / width, (top + gap) / height)
        overrides[x_in] = (left_x / width, 0.5)
        overrides[x_out] = ((left_x + 40.0) / width, 0.5)
        overrides[iris] = ((left_x + 20.0) / width, (top + iris_y_fraction * gap) / height)
    face, w, h = fake_face(overrides, width, height)
    return extract_landmark_arrays(face, w, h)[0]


def test_eye_openness_is_lid_gap_over_eye_width():
    assert compute_eye_openness(face_with_eyes(gap=12.0)) == pytest.approx((0.3, 0.3))
    assert compute_eye_openness(face_with_eyes(gap=2.0)) == pytest.approx((0.05, 0.05))


def test_eye_openness_of_zero_width_eye_is_zero():
    face, w, h = fake_face()
    assert compute_eye_openness(extract_landmark_arrays(face, w, h)[0]) == (0.0, 0.0)


def test_open_eyes_update_the_vertical_scores():
    scores = compute_gaze_scores(face_with_eyes(gap=12.0, iris_y_fraction=0.8), CENTER_GAZE, threshold=0.0)
    assert scores["ly"] == pytest.approx(0.8)
    assert scores["ry"] == pytest.approx(0.8)


def test_vertical_scores_are_held_while_both_eyes_are_closed():
    previous = {"lx": 0.5, "ly": 0.55, "rx": 0.5, "ry": 0.52}
    scores = compute_gaze_scores(face_with_eyes(gap=2.0, iris_y_fraction=3.0), previous, threshold=0.0)
    assert scores["ly"] == 0.55
    assert scores["ry"] == 0.52


def test_horizontal_scores_still_update_while_blinking():
    scores = compute_gaze_scores(face_with_eyes(gap=2.0), CENTER_GAZE, threshold=0.0)
    assert scores["lx"] == pytest.approx(0.5)
    assert scores["rx"] == pytest.approx(0.5)


def test_one_closed_eye_is_not_a_blink():
    face_2d = face_with_eyes(gap=12.0, iris_y_fraction=0.8)
    face_2d[386, 1] = face_2d[374, 1]  # second eye shut, first eye open
    assert compute_eye_openness(face_2d)[1] == 0.0
    scores = compute_gaze_scores(face_2d, CENTER_GAZE, threshold=0.0)
    assert scores["ly"] == pytest.approx(0.8)


def test_a_blink_does_not_become_an_up_or_down_gaze():
    profile = get_default_calibration_profile()
    looking_forward = {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5}
    # collapsed lids put the iris far outside the lid span; the held score keeps the result Forward
    blink_frame = face_with_eyes(gap=1.0, iris_y_fraction=9.0)
    held = compute_gaze_scores(blink_frame, looking_forward, threshold=0.0)
    assert classify_gaze_direction(held, calibration_profile=profile) == "Forward"
