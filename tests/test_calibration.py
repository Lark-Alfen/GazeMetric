import pytest

from core.analysis_logic import (
    build_calibration_profile,
    get_default_calibration_profile,
    normalize_calibration_profile,
)


def make_sample(lx=0.5, ly=0.5, rx=0.5, ry=0.5, x_angle=0.0, y_angle=0.0):
    return {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "x_angle": x_angle, "y_angle": y_angle}


# ---------------------------------------------------------------- normalize_calibration_profile
@pytest.mark.parametrize("bad_input", [None, "profile", 42, [], ()])
def test_normalize_returns_defaults_for_non_dict(bad_input):
    assert normalize_calibration_profile(bad_input) == get_default_calibration_profile()


def test_normalize_is_idempotent_on_defaults():
    defaults = get_default_calibration_profile()
    assert normalize_calibration_profile(defaults) == defaults


def test_normalize_fills_missing_sections_from_defaults():
    result = normalize_calibration_profile({"head_center": {"x": 4.0}})
    expected = get_default_calibration_profile()
    expected["head_center"]["x"] = 4.0
    assert result == expected


def test_normalize_ignores_sections_with_wrong_type():
    result = normalize_calibration_profile({"gaze_center": "oops", "head_center": [1, 2]})
    assert result == get_default_calibration_profile()


def test_normalize_clamps_out_of_range_values():
    result = normalize_calibration_profile(
        {
            "sample_count": -4,
            "gaze_center": {"lx": 5, "ly": -1},
            "gaze_tolerance": {"lx": 0, "ry": 1},
            "head_center": {"x": 100, "y": -100},
            "engagement_angle_threshold": {"x": 1, "y": 100},
            "head_direction_threshold": {"x": 1, "y": 100},
        }
    )
    assert result["sample_count"] == 0
    assert result["gaze_center"]["lx"] == pytest.approx(0.8)
    assert result["gaze_center"]["ly"] == pytest.approx(0.2)
    assert result["gaze_tolerance"]["lx"] == pytest.approx(0.04)
    assert result["gaze_tolerance"]["ry"] == pytest.approx(0.25)
    assert result["head_center"] == {"x": 30.0, "y": -30.0}
    assert result["engagement_angle_threshold"] == {"x": 6.0, "y": 20.0}
    assert result["head_direction_threshold"] == {"x": 10.0, "y": 30.0}


# ---------------------------------------------------------------- build_calibration_profile
def test_build_with_no_samples_returns_defaults():
    assert build_calibration_profile([]) == get_default_calibration_profile()


def test_build_ignores_invalid_samples():
    junk = [1, None, {"lx": 1}, "x"]
    assert build_calibration_profile(junk) == get_default_calibration_profile()


def test_build_with_enough_identical_samples_matches_the_samples():
    samples = [make_sample(lx=0.55, ly=0.45, rx=0.58, ry=0.47, x_angle=3.0, y_angle=-2.0)] * 40
    profile = build_calibration_profile(samples, min_samples=20)
    assert profile["sample_count"] == 40
    assert profile["gaze_center"] == pytest.approx({"lx": 0.55, "ly": 0.45, "rx": 0.58, "ry": 0.47})
    assert profile["head_center"] == pytest.approx({"x": 3.0, "y": -2.0})
    # zero spread gives the minimum tolerance and the base thresholds
    assert profile["gaze_tolerance"]["lx"] == pytest.approx(0.04)
    assert profile["engagement_angle_threshold"] == pytest.approx({"x": 8.0, "y": 8.0})
    assert profile["head_direction_threshold"] == pytest.approx({"x": 12.0, "y": 12.0})


def test_build_blends_toward_defaults_when_samples_are_few():
    samples = [make_sample(lx=0.6)] * 10
    profile = build_calibration_profile(samples, min_samples=20)  # 50% complete
    assert profile["sample_count"] == 10
    assert profile["gaze_center"]["lx"] == pytest.approx(0.55)  # halfway between 0.6 and 0.5
    assert profile["gaze_tolerance"]["lx"] == pytest.approx(0.06)  # halfway between 0.04 and 0.08


def test_build_thresholds_grow_with_head_movement_spread():
    steady = [make_sample(x_angle=0.0)] * 100
    wobbly = [make_sample(x_angle=angle) for angle in ([2.0, -2.0] * 50)]
    steady_profile = build_calibration_profile(steady, min_samples=20)
    wobbly_profile = build_calibration_profile(wobbly, min_samples=20)
    assert wobbly_profile["engagement_angle_threshold"]["x"] == pytest.approx(13.0)  # 8 + 2.5 * std(2)
    assert wobbly_profile["head_direction_threshold"]["x"] == pytest.approx(18.0)  # 12 + 3 * std(2)
    assert wobbly_profile["engagement_angle_threshold"]["x"] > steady_profile["engagement_angle_threshold"]["x"]
    # y did not move, so its thresholds stay at the base values
    assert wobbly_profile["engagement_angle_threshold"]["y"] == pytest.approx(8.0)


def test_build_output_is_always_normalized():
    extreme = [make_sample(lx=0.99, x_angle=a) for a in ([40.0, -40.0] * 30)]
    profile = build_calibration_profile(extreme, min_samples=20)
    assert normalize_calibration_profile(profile) == profile
    assert profile["engagement_angle_threshold"]["x"] <= 20.0
    assert profile["head_direction_threshold"]["x"] <= 30.0
