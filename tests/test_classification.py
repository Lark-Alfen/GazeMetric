import pytest

from core.analysis_logic import (
    GAZE_FORWARD_MAX_DELTA_H,
    GAZE_FORWARD_MAX_DELTA_V,
    GAZE_REFERENCE_TOLERANCE,
    classify_engagement,
    classify_gaze_direction,
    classify_head_direction,
    get_default_calibration_profile,
)


def gaze(dx=0.0, dy=0.0):
    # Gaze scores offset from the 0.5 default center, same offset for both eyes.
    return {"lx": 0.5 + dx, "rx": 0.5 + dx, "ly": 0.5 + dy, "ry": 0.5 + dy}


# ---------------------------------------------------------------- classify_gaze_direction
def test_gaze_centered_is_forward():
    assert classify_gaze_direction(gaze()) == "Forward"


def test_gaze_small_jitter_is_still_forward():
    assert classify_gaze_direction(gaze(0.01, 0.01)) == "Forward"


@pytest.mark.parametrize(
    "dx, dy, expected",
    [
        (0.10, 0.0, "Right"),
        (-0.10, 0.0, "Left"),
        (0.0, 0.10, "Down"),
        (0.0, -0.10, "Up"),
    ],
)
def test_gaze_large_offsets_give_a_direction(dx, dy, expected):
    assert classify_gaze_direction(gaze(dx, dy)) == expected


def test_gaze_prefers_the_dominant_axis():
    assert classify_gaze_direction(gaze(0.10, 0.01)) == "Right"


def test_gaze_without_profile_matches_default_profile():
    scores = gaze(0.10, 0.0)
    assert classify_gaze_direction(scores) == classify_gaze_direction(
        scores, calibration_profile=get_default_calibration_profile()
    )


def test_gaze_is_relative_to_the_calibrated_center():
    profile = get_default_calibration_profile()
    profile["gaze_center"] = {"lx": 0.62, "rx": 0.62, "ly": 0.5, "ry": 0.5}
    looking_at_calibrated_center = {"lx": 0.62, "rx": 0.62, "ly": 0.5, "ry": 0.5}
    assert classify_gaze_direction(looking_at_calibrated_center, calibration_profile=profile) == "Forward"
    assert classify_gaze_direction(gaze(), calibration_profile=profile) != "Forward"


def profile_with_tolerance(tolerance):
    profile = get_default_calibration_profile()
    profile["gaze_tolerance"] = {key: tolerance for key in ("lx", "ly", "rx", "ry")}
    return profile


@pytest.mark.parametrize("tolerance", [0.08, 0.16, 0.25])
def test_wider_calibrated_tolerance_widens_the_forward_band_horizontally(tolerance):
    scale = tolerance / GAZE_REFERENCE_TOLERANCE
    inside = 0.9 * GAZE_FORWARD_MAX_DELTA_H * scale
    profile = profile_with_tolerance(tolerance)
    assert classify_gaze_direction(gaze(inside, 0.0), calibration_profile=profile) == "Forward"
    assert classify_gaze_direction(gaze(-inside, 0.0), calibration_profile=profile) == "Forward"


@pytest.mark.parametrize("tolerance", [0.08, 0.16, 0.25])
def test_wider_calibrated_tolerance_widens_the_forward_band_vertically(tolerance):
    scale = tolerance / GAZE_REFERENCE_TOLERANCE
    inside = 0.9 * GAZE_FORWARD_MAX_DELTA_V * scale
    profile = profile_with_tolerance(tolerance)
    assert classify_gaze_direction(gaze(0.0, inside), calibration_profile=profile) == "Forward"
    assert classify_gaze_direction(gaze(0.0, -inside), calibration_profile=profile) == "Forward"


def test_the_same_offset_is_forward_with_a_wide_tolerance_and_not_with_the_default():
    scores = gaze(0.05, 0.0)
    assert classify_gaze_direction(scores, calibration_profile=profile_with_tolerance(0.08)) != "Forward"
    assert classify_gaze_direction(scores, calibration_profile=profile_with_tolerance(0.25)) == "Forward"


@pytest.mark.parametrize("tolerance", [0.04, 0.08, 0.16, 0.25])
@pytest.mark.parametrize(
    "dx, dy, expected",
    [(2.0, 0.0, "Right"), (-2.0, 0.0, "Left"), (0.0, 2.0, "Down"), (0.0, -2.0, "Up")],
)
def test_offsets_of_twice_the_tolerance_are_always_a_direction(tolerance, dx, dy, expected):
    profile = profile_with_tolerance(tolerance)
    assert classify_gaze_direction(gaze(dx * tolerance, dy * tolerance), calibration_profile=profile) == expected


@pytest.mark.parametrize("tolerance", [0.04, 0.06, 0.08])
def test_tolerances_at_or_below_the_default_keep_the_original_forward_band(tolerance):
    profile = profile_with_tolerance(tolerance)
    assert classify_gaze_direction(gaze(0.02, 0.0), calibration_profile=profile) == "Forward"
    assert classify_gaze_direction(gaze(0.03, 0.0), calibration_profile=profile) != "Forward"


# ---------------------------------------------------------------- classify_head_direction
@pytest.mark.parametrize(
    "x_angle, y_angle, expected",
    [
        (0, 0, "Looking Straight"),
        (3, 3, "Looking Straight"),
        (0, -20, "Looking Left"),
        (0, 20, "Looking Right"),
        (-20, 0, "Looking Down"),
        (20, 0, "Looking Up"),
    ],
)
def test_head_direction_default_thresholds(x_angle, y_angle, expected):
    assert classify_head_direction(x_angle, y_angle) == expected


def test_head_direction_horizontal_wins_when_both_axes_exceed():
    assert classify_head_direction(20, 20) == "Looking Right"
    assert classify_head_direction(-20, -20) == "Looking Left"


def test_head_direction_is_relative_to_the_calibrated_center():
    profile = get_default_calibration_profile()
    profile["head_center"] = {"x": 10.0, "y": -10.0}
    assert classify_head_direction(10, -10, calibration_profile=profile) == "Looking Straight"
    assert classify_head_direction(0, 0, calibration_profile=profile) == "Looking Straight"
    assert classify_head_direction(10, 10, calibration_profile=profile) == "Looking Right"


def test_head_direction_threshold_boundary_is_exclusive():
    # default threshold is 15: exactly 15 degrees away is still straight
    assert classify_head_direction(0, 15) == "Looking Straight"
    assert classify_head_direction(0, 15.1) == "Looking Right"


# ---------------------------------------------------------------- classify_engagement
def test_engaged_requires_forward_gaze_and_head_within_threshold():
    assert classify_engagement("Forward", 0, 0) == "Engaged"
    assert classify_engagement("Forward", 11.9, 11.9) == "Engaged"


@pytest.mark.parametrize("direction", ["Left", "Right", "Up", "Down", "Not Forward"])
def test_not_engaged_when_gaze_is_not_forward(direction):
    assert classify_engagement(direction, 0, 0) == "Not Engaged"


@pytest.mark.parametrize("x_angle, y_angle", [(12, 0), (0, 12), (-13, 0), (0, -25)])
def test_not_engaged_when_head_is_turned_past_threshold(x_angle, y_angle):
    assert classify_engagement("Forward", x_angle, y_angle) == "Not Engaged"


def test_engagement_uses_calibrated_head_center_and_thresholds():
    profile = get_default_calibration_profile()
    profile["head_center"] = {"x": 20.0, "y": 0.0}
    profile["engagement_angle_threshold"] = {"x": 6.0, "y": 6.0}
    assert classify_engagement("Forward", 20, 0, calibration_profile=profile) == "Engaged"
    assert classify_engagement("Forward", 0, 0, calibration_profile=profile) == "Not Engaged"
    assert classify_engagement("Forward", 27, 0, calibration_profile=profile) == "Not Engaged"
