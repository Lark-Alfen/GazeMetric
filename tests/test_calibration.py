import numpy as np
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
    wobbly = [make_sample(x_angle=float(angle)) for angle in np.random.default_rng(0).normal(0.0, 2.0, 2000)]
    steady_profile = build_calibration_profile(steady, min_samples=20)
    wobbly_profile = build_calibration_profile(wobbly, min_samples=20)
    # for roughly normal data the median-based spread matches the standard deviation (2)
    assert wobbly_profile["engagement_angle_threshold"]["x"] == pytest.approx(13.0, abs=0.5)  # 8 + 2.5 * 2
    assert wobbly_profile["head_direction_threshold"]["x"] == pytest.approx(18.0, abs=0.6)  # 12 + 3 * 2
    assert wobbly_profile["engagement_angle_threshold"]["x"] > steady_profile["engagement_angle_threshold"]["x"]
    # y did not move, so its thresholds stay at the base values
    assert wobbly_profile["engagement_angle_threshold"]["y"] == pytest.approx(8.0)


def test_build_spread_is_the_scaled_median_absolute_deviation():
    # median 10, absolute deviations 0,1,1,2,2 -> MAD 1 -> spread 1.4826
    samples = [make_sample(x_angle=angle) for angle in (8.0, 9.0, 10.0, 11.0, 12.0)]
    profile = build_calibration_profile(samples, min_samples=5)
    assert profile["head_center"]["x"] == pytest.approx(10.0)
    assert profile["engagement_angle_threshold"]["x"] == pytest.approx(8.0 + 2.5 * 1.4826)
    assert profile["head_direction_threshold"]["x"] == pytest.approx(12.0 + 3.0 * 1.4826)


def tight_samples(count=100):
    rng = np.random.default_rng(1)
    return [
        make_sample(
            lx=0.5 + rng.normal(0, 0.01),
            ly=0.5 + rng.normal(0, 0.01),
            rx=0.5 + rng.normal(0, 0.01),
            ry=0.5 + rng.normal(0, 0.01),
            x_angle=2.0 + rng.normal(0, 0.5),
            y_angle=-1.0 + rng.normal(0, 0.5),
        )
        for _ in range(count)
    ]


def test_a_few_outlier_frames_barely_change_the_profile():
    clean = build_calibration_profile(tight_samples(), min_samples=20)
    glances = [make_sample(lx=0.8, ly=0.2, rx=0.8, ry=0.2, x_angle=35.0, y_angle=40.0)] * 5
    dirty = build_calibration_profile(tight_samples() + glances, min_samples=20)

    for key in ("lx", "ly", "rx", "ry"):
        assert dirty["gaze_center"][key] == pytest.approx(clean["gaze_center"][key], rel=0.01)
        assert dirty["gaze_tolerance"][key] == pytest.approx(clean["gaze_tolerance"][key], abs=0.005)
    for axis in ("x", "y"):
        assert dirty["head_center"][axis] == pytest.approx(clean["head_center"][axis], abs=0.3)
        assert dirty["engagement_angle_threshold"][axis] == pytest.approx(clean["engagement_angle_threshold"][axis], abs=0.5)
        assert dirty["head_direction_threshold"][axis] == pytest.approx(clean["head_direction_threshold"][axis], abs=0.6)


def test_a_fifth_of_the_frames_looking_away_still_does_not_move_the_centre():
    samples = tight_samples(80) + [make_sample(lx=0.8, x_angle=35.0, y_angle=40.0)] * 20
    profile = build_calibration_profile(samples, min_samples=20)
    assert profile["gaze_center"]["lx"] == pytest.approx(0.5, abs=0.02)
    assert profile["head_center"]["x"] == pytest.approx(2.0, abs=1.0)


def test_calibration_is_rejected_when_the_head_never_settles(capsys):
    samples = [make_sample(x_angle=float(angle)) for angle in np.random.default_rng(2).uniform(-60, 60, 200)]
    profile = build_calibration_profile(samples, min_samples=20)
    assert profile == get_default_calibration_profile()
    assert "rejected" in capsys.readouterr().out


def test_build_does_not_print_for_a_good_calibration(capsys):
    build_calibration_profile(tight_samples(), min_samples=20)
    assert capsys.readouterr().out == ""


def test_a_profile_saved_by_the_old_code_still_loads():
    old_profile = {
        "sample_count": 123,
        "gaze_center": {"lx": 0.52, "ly": 0.48, "rx": 0.51, "ry": 0.49},
        "gaze_tolerance": {"lx": 0.07, "ly": 0.09, "rx": 0.06, "ry": 0.1},
        "head_center": {"x": 1.5, "y": -2.5},
        "engagement_angle_threshold": {"x": 9.0, "y": 10.0},
        "head_direction_threshold": {"x": 13.0, "y": 14.0},
    }
    assert normalize_calibration_profile(old_profile) == old_profile


def test_build_output_is_always_normalized():
    extreme = [make_sample(lx=0.99, x_angle=a) for a in ([40.0, -40.0] * 30)]
    profile = build_calibration_profile(extreme, min_samples=20)
    assert normalize_calibration_profile(profile) == profile
    assert profile["engagement_angle_threshold"]["x"] <= 20.0
    assert profile["head_direction_threshold"]["x"] <= 30.0
