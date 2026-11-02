import numpy as np
import pytest

from core.quality import (
    clamp01,
    compute_quality_metrics,
    get_confidence_label,
    get_default_quality_state,
)

W, H = 640, 480


def textured_frame(mean=135, spread=45, seed=0):
    """A frame with enough variation to pass the contrast check."""
    noise = np.random.default_rng(seed).normal(mean, spread, (H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


def flat_frame(value):
    return np.full((H, W, 3), value, dtype=np.uint8)


def face_in(x_min, y_min, x_max, y_max):
    """478 landmarks laid out over a rectangle, so the face bounding box is exactly that rectangle."""
    xs = np.linspace(x_min, x_max, 478)
    ys = np.linspace(y_min, y_max, 478)
    points = np.stack([xs, ys], axis=1)
    points[::2, 1] = y_min + (y_max - y_min) * 0.3  # spread the rows so it is not a diagonal line
    points[0] = (x_min, y_min)
    points[1] = (x_max, y_max)
    return points


GOOD_FACE = (220, 110, 420, 370)  # 200 x 260 px, centred, about 17% of the frame


def metrics(frame=None, face=GOOD_FACE, state=None):
    frame = textured_frame() if frame is None else frame
    state = get_default_quality_state() if state is None else state
    face_2d = face if isinstance(face, np.ndarray) else face_in(*face)
    return compute_quality_metrics(frame, face_2d, W, H, state)


# ---------------------------------------------------------------- a good frame
def test_a_well_lit_centred_face_has_no_warnings_and_a_high_label():
    result = metrics()
    assert result["warnings"] == []
    assert result["confidence_label"] == "High"
    assert 70 <= result["confidence_score"] <= 100
    assert set(result) == {
        "face_visibility_score",
        "lighting_score",
        "landmark_stability_score",
        "confidence_score",
        "confidence_label",
        "warnings",
    }


def test_confidence_is_the_weighted_sum_of_the_three_scores():
    result = metrics()
    expected = 100 * (
        0.45 * result["face_visibility_score"] + 0.30 * result["lighting_score"] + 0.25 * result["landmark_stability_score"]
    )
    assert result["confidence_score"] == pytest.approx(expected)


# ---------------------------------------------------------------- lighting
def test_dark_frame_warns_low_light():
    warnings = metrics(frame=flat_frame(20))["warnings"]
    assert "Low light" in warnings
    assert "Overexposed frame" not in warnings


def test_bright_frame_warns_overexposed():
    warnings = metrics(frame=flat_frame(240))["warnings"]
    assert "Overexposed frame" in warnings
    assert "Low light" not in warnings


# the grey image averages three independent colour channels, so its std is about 0.67 x the noise spread
@pytest.mark.parametrize("spread, warns", [(8, True), (20, True), (25, True), (33, False), (45, False)])
def test_contrast_threshold_is_a_standard_deviation_of_20(spread, warns):
    frame = textured_frame(mean=135, spread=spread)
    assert ("Low contrast" in metrics(frame=frame)["warnings"]) is warns


def test_flat_mid_grey_frame_only_warns_low_contrast():
    warnings = metrics(frame=flat_frame(135))["warnings"]
    assert warnings == ["Low contrast"]


@pytest.mark.parametrize("mean, expected", [(60, True), (69, True), (75, False), (200, False), (210, True)])
def test_brightness_thresholds_around_70_and_205(mean, expected):
    warnings = metrics(frame=textured_frame(mean=mean, spread=10))["warnings"]
    assert (("Low light" in warnings) or ("Overexposed frame" in warnings)) is expected


def test_lighting_is_judged_on_the_face_region_not_the_whole_frame():
    frame = flat_frame(20)
    x0, y0, x1, y1 = GOOD_FACE
    frame[y0:y1, x0:x1] = textured_frame(seed=3)[y0:y1, x0:x1]  # bright, varied face in a dark room
    assert metrics(frame=frame)["warnings"] == []


def test_good_lighting_scores_higher_than_bad_lighting():
    good = metrics()["lighting_score"]
    assert metrics(frame=flat_frame(20))["lighting_score"] < good
    assert metrics(frame=flat_frame(240))["lighting_score"] < good


# ---------------------------------------------------------------- face size and position
def test_tiny_face_warns_too_far():
    assert "Face too far from camera" in metrics(face=(300, 220, 330, 250))["warnings"]


def test_face_filling_the_frame_warns_too_close():
    warnings = metrics(face=(10, 10, 630, 470))["warnings"]
    assert "Face too close to camera" in warnings
    assert "Face too far from camera" not in warnings


def test_face_in_a_corner_warns_off_center():
    assert "Face off-center" in metrics(face=(0, 0, 110, 110))["warnings"]


@pytest.mark.parametrize(
    "centre_x, warns",
    [(320, False), (531, False), (580, True), (601, True)],
)
def test_off_center_threshold(centre_x, warns):
    # face 160 x 200 px with its centre moved right: horizontal offset of 0.33 does not warn, 0.41 and more do
    face = (centre_x - 80, 140, centre_x + 80, 340) if centre_x + 80 <= W else (centre_x - 80, 140, W - 1, 340)
    assert ("Face off-center" in metrics(face=face)["warnings"]) is warns


def test_centred_face_of_normal_size_gets_no_position_warning():
    warnings = metrics()["warnings"]
    assert not {"Face too far from camera", "Face too close to camera", "Face off-center"} & set(warnings)


def centred_face(width, height):
    cx, cy = W // 2, H // 2
    return (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2)


@pytest.mark.parametrize(
    "side, warns_far",
    [(100, True), (120, True), (150, False), (200, False)],
)
def test_face_area_threshold_is_five_percent_of_the_frame(side, warns_far):
    # a square face of `side` px: 120 -> 4.7% of 640x480, 150 -> 7.3%
    assert ("Face too far from camera" in metrics(face=centred_face(side, side))["warnings"]) is warns_far


def face_with_area_ratio(ratio):
    """A centred 4:3 face covering `ratio` of the frame."""
    height = (0.75 * ratio * W * H) ** 0.5
    return centred_face(int(height * 4 / 3), int(height))


@pytest.mark.parametrize("ratio, warns", [(0.045, True), (0.049, True), (0.052, False), (0.07, False)])
def test_too_far_starts_just_below_five_percent_of_the_frame(ratio, warns):
    assert ("Face too far from camera" in metrics(face=face_with_area_ratio(ratio))["warnings"]) is warns


@pytest.mark.parametrize("ratio, warns", [(0.40, False), (0.50, False), (0.54, False), (0.57, True), (0.70, True)])
def test_too_close_starts_just_above_55_percent_of_the_frame(ratio, warns):
    assert ("Face too close to camera" in metrics(face=face_with_area_ratio(ratio))["warnings"]) is warns



def test_visibility_score_drops_for_bad_framing():
    good = metrics()["face_visibility_score"]
    assert metrics(face=(300, 220, 330, 250))["face_visibility_score"] < good
    assert metrics(face=(0, 0, 110, 110))["face_visibility_score"] < good
    assert metrics(face=(10, 10, 630, 470))["face_visibility_score"] < good


def test_sudden_jump_in_face_size_lowers_visibility():
    state = get_default_quality_state()
    steady = metrics(state=state)["face_visibility_score"]
    steady_again = metrics(state=state)["face_visibility_score"]
    jump = metrics(face=(170, 60, 470, 420), state=state)["face_visibility_score"]
    assert steady_again == pytest.approx(steady)
    assert jump < steady


def test_empty_landmarks_give_zero_visibility():
    result = compute_quality_metrics(textured_frame(), np.zeros((0, 2)), W, H, get_default_quality_state())
    assert result["face_visibility_score"] == 0.0


# ---------------------------------------------------------------- landmark stability
def test_first_frame_has_a_neutral_stability_score():
    result = metrics()
    assert result["landmark_stability_score"] == pytest.approx(0.90)


def test_identical_landmarks_twice_are_perfectly_stable():
    state = get_default_quality_state()
    metrics(state=state)
    second = metrics(state=state)
    assert second["landmark_stability_score"] == pytest.approx(1.0)
    assert not {"Unstable tracking", "Frequent movement"} & set(second["warnings"])


def test_random_jitter_of_five_percent_is_unstable():
    state = get_default_quality_state()
    face = face_in(*GOOD_FACE)
    metrics(face=face, state=state)
    jitter = np.random.default_rng(2).uniform(-0.05, 0.05, face.shape) * np.array([W, H])
    second = metrics(face=face + jitter, state=state)
    assert "Unstable tracking" in second["warnings"]
    assert second["landmark_stability_score"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "shift_fraction, expected",
    [(0.005, []), (0.017, []), (0.025, ["Frequent movement"]), (0.035, ["Unstable tracking"])],
)
def test_movement_thresholds(shift_fraction, expected):
    state = get_default_quality_state()
    face = face_in(*GOOD_FACE)
    metrics(face=face, state=state)
    moved = face + np.array([shift_fraction * W, 0.0])
    warnings = metrics(face=moved, state=state)["warnings"]
    assert [w for w in warnings if w in ("Frequent movement", "Unstable tracking")] == expected


def test_a_different_landmark_count_restarts_the_comparison():
    state = get_default_quality_state()
    metrics(state=state)
    shorter = face_in(*GOOD_FACE)[:100]
    result = compute_quality_metrics(textured_frame(), shorter, W, H, state)
    assert result["landmark_stability_score"] == pytest.approx(0.90)


def test_two_people_do_not_share_stability_state():
    first, second = get_default_quality_state(), get_default_quality_state()
    metrics(face=(60, 100, 260, 360), state=first)
    metrics(face=(380, 100, 580, 360), state=second)
    again = metrics(face=(60, 100, 260, 360), state=first)
    assert again["landmark_stability_score"] == pytest.approx(1.0)
    assert "Unstable tracking" not in again["warnings"]


# ---------------------------------------------------------------- confidence label and helpers
@pytest.mark.parametrize(
    "score, label",
    [(100, "High"), (70.0, "High"), (69.99, "Medium"), (45.0, "Medium"), (44.99, "Low"), (0, "Low")],
)
def test_confidence_label_boundaries(score, label):
    assert get_confidence_label(score) == label


@pytest.mark.parametrize("value, clamped", [(-5, 0.0), (0, 0.0), (0.4, 0.4), (1, 1.0), (7, 1.0), ("0.5", 0.5)])
def test_clamp01(value, clamped):
    assert clamp01(value) == clamped


def test_confidence_stays_between_0_and_100_for_extreme_inputs():
    rng = np.random.default_rng(9)
    state = get_default_quality_state()
    for value in (0, 255):
        for face in (GOOD_FACE, (0, 0, 5, 5), (0, 0, 640, 480), (600, 440, 640, 480)):
            result = metrics(frame=flat_frame(value), face=face, state=state)
            assert 0.0 <= result["confidence_score"] <= 100.0
    for _ in range(30):
        x0, y0 = rng.integers(0, 400), rng.integers(0, 300)
        face = (x0, y0, x0 + rng.integers(5, 240), y0 + rng.integers(5, 180))
        result = metrics(frame=textured_frame(seed=int(rng.integers(0, 1000))), face=face, state=state)
        assert 0.0 <= result["confidence_score"] <= 100.0


def test_bad_conditions_lower_the_overall_confidence_and_label():
    good = metrics()
    bad = metrics(frame=flat_frame(20), face=(0, 0, 40, 40))
    assert bad["confidence_score"] < good["confidence_score"]
    assert bad["confidence_label"] in ("Medium", "Low")


def test_warnings_are_not_repeated_and_keep_a_stable_order():
    warnings = metrics(frame=flat_frame(20), face=(0, 0, 40, 40))["warnings"]
    assert len(warnings) == len(set(warnings))
    assert warnings.index("Face too far from camera") < warnings.index("Low light")  # visibility first, then lighting


def test_stability_compares_each_frame_with_the_one_before_not_with_the_first():
    state = get_default_quality_state()
    face = face_in(*GOOD_FACE)
    for step in range(5):  # a slow drift of 1% of the width per frame, 4% in total
        result = metrics(face=face + np.array([step * 0.01 * W, 0.0]), state=state)
        assert not {"Frequent movement", "Unstable tracking"} & set(result["warnings"])
    assert result["landmark_stability_score"] == pytest.approx(0.5)
