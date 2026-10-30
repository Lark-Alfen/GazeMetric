import pytest

from core.scoring import (
    clamp01,
    compute_composite_engagement_score,
    compute_gaze_center_score,
    compute_head_pose_score,
)

GAZE_CENTERED = {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5}
GAZE_EDGE = {"lx": 1.0, "ly": 1.0, "rx": 1.0, "ry": 1.0}


@pytest.mark.parametrize(
    "value, expected",
    [(-3, 0.0), (0, 0.0), (0.25, 0.25), (1, 1.0), (7.5, 1.0), ("0.4", 0.4)],
)
def test_clamp01_bounds(value, expected):
    assert clamp01(value) == pytest.approx(expected)


def test_head_pose_score_is_one_at_center():
    assert compute_head_pose_score(0, 0) == pytest.approx(1.0)


def test_head_pose_score_is_zero_at_max_angle():
    assert compute_head_pose_score(30, 30) == pytest.approx(0.0)
    assert compute_head_pose_score(-30, 30) == pytest.approx(0.0)


def test_head_pose_score_is_linear_between_center_and_max():
    assert compute_head_pose_score(15, 15) == pytest.approx(0.5)
    assert compute_head_pose_score(15, 0) == pytest.approx(0.75)


def test_head_pose_score_does_not_go_below_zero_past_max_angle():
    assert compute_head_pose_score(90, -90) == pytest.approx(0.0)


def test_head_pose_score_respects_calibrated_center():
    assert compute_head_pose_score(10, -5, center_x=10, center_y=-5) == pytest.approx(1.0)


def test_gaze_center_score_defaults_to_half_center():
    assert compute_gaze_center_score(GAZE_CENTERED) == pytest.approx(1.0)
    assert compute_gaze_center_score(GAZE_EDGE) == pytest.approx(0.0)


def test_gaze_center_score_uses_profile_center():
    profile = {"gaze_center": {"lx": 0.6, "ly": 0.4, "rx": 0.6, "ry": 0.4}}
    looking_at_center = {"lx": 0.6, "ly": 0.4, "rx": 0.6, "ry": 0.4}
    assert compute_gaze_center_score(looking_at_center, calibration_profile=profile) == pytest.approx(1.0)
    assert compute_gaze_center_score(GAZE_CENTERED, calibration_profile=profile) < 1.0


def test_composite_score_is_100_when_fully_centered():
    head, gaze, score = compute_composite_engagement_score(GAZE_CENTERED, 0, 0)
    assert head == pytest.approx(1.0)
    assert gaze == pytest.approx(1.0)
    assert score == pytest.approx(100.0)


def test_composite_score_weights_head_55_and_gaze_45():
    _, _, only_gaze = compute_composite_engagement_score(GAZE_CENTERED, 30, 30)
    _, _, only_head = compute_composite_engagement_score(GAZE_EDGE, 0, 0)
    assert only_gaze == pytest.approx(45.0)
    assert only_head == pytest.approx(55.0)


def test_composite_score_stays_within_0_and_100():
    for angle in (-200, -30, 0, 30, 200):
        _, _, score = compute_composite_engagement_score(GAZE_EDGE, angle, -angle)
        assert 0.0 <= score <= 100.0


def test_composite_score_uses_calibrated_head_center():
    profile = {
        "head_center": {"x": 10.0, "y": 5.0},
        "gaze_center": {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5},
    }
    _, _, score = compute_composite_engagement_score(GAZE_CENTERED, 10, 5, calibration_profile=profile)
    assert score == pytest.approx(100.0)
