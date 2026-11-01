import cv2
import numpy as np
import pytest

from core.analysis_logic import (
    HEAD_MODEL_3D,
    build_camera_and_distortion,
    build_eye_reference_points,
    solve_pose_vectors,
)

WIDTH, HEIGHT = 1280, 720


@pytest.mark.parametrize("width, height", [(1280, 720), (640, 480), (720, 1280), (500, 500)])
def test_principal_point_is_the_frame_centre(width, height):
    cam_matrix, _ = build_camera_and_distortion(width, height)
    assert cam_matrix[0][2] == width / 2
    assert cam_matrix[1][2] == height / 2


def test_camera_matrix_shape_and_focal_length():
    cam_matrix, dist_coeffs = build_camera_and_distortion(1280, 720)
    assert cam_matrix.shape == (3, 3)
    assert cam_matrix[0][0] == cam_matrix[1][1] == 1280
    assert np.array_equal(cam_matrix[2], [0, 0, 1])
    assert not dist_coeffs.any()


def project_head(pitch_deg, yaw_deg, roll_deg=0.0, distance=1500.0):
    """Project the head model, rotated by a known pose, onto the image like a pinhole camera would."""
    cam_matrix, _ = build_camera_and_distortion(WIDTH, HEIGHT)
    rotation, _ = cv2.Rodrigues(np.radians([pitch_deg, yaw_deg, roll_deg]))
    points = HEAD_MODEL_3D @ rotation.T + np.array([0.0, 0.0, distance])
    return np.stack(
        [
            cam_matrix[0][0] * points[:, 0] / points[:, 2] + cam_matrix[0][2],
            cam_matrix[1][1] * points[:, 1] / points[:, 2] + cam_matrix[1][2],
        ],
        axis=1,
    )


def solve(face_2d_head):
    cam_matrix, dist_coeffs = build_camera_and_distortion(WIDTH, HEIGHT)
    leye_3d, reye_3d = build_eye_reference_points()
    return solve_pose_vectors(face_2d_head, cam_matrix, dist_coeffs, leye_3d, reye_3d)


def test_frontal_face_has_zero_angles():
    pose = solve(project_head(0, 0))
    assert pose["x_angle"] == pytest.approx(0, abs=0.5)
    assert pose["y_angle"] == pytest.approx(0, abs=0.5)


@pytest.mark.parametrize("magnitude", [10, 20, 30, 45])
def test_head_turn_is_reported_in_real_degrees(magnitude):
    # The nose moves to the right of the image for a positive y_angle, to the left for a negative one.
    right = solve(project_head(0, -magnitude))
    left = solve(project_head(0, magnitude))
    assert right["y_angle"] == pytest.approx(magnitude, abs=1.0)
    assert left["y_angle"] == pytest.approx(-magnitude, abs=1.0)
    assert right["x_angle"] == pytest.approx(0, abs=1.0)


@pytest.mark.parametrize("magnitude", [10, 20, 30])
def test_head_tilt_is_reported_in_real_degrees(magnitude):
    up = solve(project_head(-magnitude, 0))
    down = solve(project_head(magnitude, 0))
    assert up["x_angle"] == pytest.approx(magnitude, abs=1.0)
    assert down["x_angle"] == pytest.approx(-magnitude, abs=1.0)
    assert up["y_angle"] == pytest.approx(0, abs=1.0)


def test_angle_signs_match_what_happens_on_screen():
    def nose_offset_from_eyes(points):
        # rows: nose, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
        eye_midpoint = (points[2] + points[3]) / 2
        return points[0] - eye_midpoint

    straight = nose_offset_from_eyes(project_head(0, 0))
    turned_right = project_head(0, -20)
    tilted_up = project_head(-20, 0)
    assert nose_offset_from_eyes(turned_right)[0] > straight[0]
    assert nose_offset_from_eyes(tilted_up)[1] < straight[1]
    assert solve(turned_right)["y_angle"] > 0
    assert solve(tilted_up)["x_angle"] > 0


def test_angles_do_not_depend_on_face_distance():
    near = solve(project_head(5, -25, distance=900.0))
    far = solve(project_head(5, -25, distance=3000.0))
    assert near["y_angle"] == pytest.approx(far["y_angle"], abs=1.0)
    assert near["x_angle"] == pytest.approx(far["x_angle"], abs=1.0)
