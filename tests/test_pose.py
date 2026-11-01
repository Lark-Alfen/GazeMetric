import numpy as np
import pytest

from core.analysis_logic import build_camera_and_distortion


@pytest.mark.parametrize("width, height", [(1280, 720), (640, 480), (720, 1280), (500, 500)])
def test_principal_point_is_the_frame_centre(width, height):
    cam_matrix, _, _ = build_camera_and_distortion(width, height)
    assert cam_matrix[0][2] == width / 2
    assert cam_matrix[1][2] == height / 2


def test_camera_matrix_shape_and_focal_length():
    cam_matrix, dist_coeffs, dist_matrix = build_camera_and_distortion(1280, 720)
    assert cam_matrix.shape == (3, 3)
    assert cam_matrix[0][0] == cam_matrix[1][1] == 1280
    assert np.array_equal(cam_matrix[2], [0, 0, 1])
    assert not dist_coeffs.any() and not dist_matrix.any()
