"""Head pose, gaze scoring, and calibration profile helpers."""

import cv2
import numpy as np

FACE_3D_TEMPLATE = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)

# FACE_3D_TEMPLATE is written with y up and z toward the camera. OpenCV camera axes have y down and
# z away from the camera, so head pose is solved against the template with y and z flipped.
HEAD_MODEL_3D = FACE_3D_TEMPLATE * np.array([1.0, -1.0, -1.0])
NOSE_AXIS_LENGTH = 300.0

# Gaze classification works on the raw difference from the calibrated gaze centre. These raw-difference
# limits were tuned for GAZE_REFERENCE_TOLERANCE (the default profile's tolerance); a profile with a
# wider tolerance scales them up by tolerance / GAZE_REFERENCE_TOLERANCE, a narrower one keeps them.
GAZE_REFERENCE_TOLERANCE = 0.08
GAZE_FORWARD_MAX_DELTA_H = 0.022
GAZE_FORWARD_MAX_DELTA_V = 0.03
GAZE_SIDE_MIN_DELTA = 0.028
GAZE_UP_MIN_DELTA = 0.045
GAZE_DOWN_MIN_DELTA = 0.03

# Both eyes below this eye-aspect ratio (lid gap / eye width) count as a blink.
EYE_CLOSED_THRESHOLD = 0.15

# Calibration is rejected when the (robust) spread of the head angles is larger than this many degrees:
# the person was not looking at the camera for most of it.
CALIBRATION_MAX_HEAD_STD = 10.0
MAD_TO_STD = 1.4826

# A head pose is rejected when the model, projected back with the solved pose, misses the detected
# landmarks by more than this fraction of the distance between the outer eye corners.
MAX_POSE_REPROJECTION_RATIO = 0.25


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, float(value)))


def get_default_calibration_profile():
    return {
        "sample_count": 0,
        "gaze_center": {
            "lx": 0.5,
            "ly": 0.5,
            "rx": 0.5,
            "ry": 0.5,
        },
        "gaze_tolerance": {
            "lx": 0.08,
            "ly": 0.08,
            "rx": 0.08,
            "ry": 0.08,
        },
        "head_center": {
            "x": 0.0,
            "y": 0.0,
        },
        "engagement_angle_threshold": {
            "x": 12.0,
            "y": 12.0,
        },
        "head_direction_threshold": {
            "x": 15.0,
            "y": 15.0,
        },
    }


def normalize_calibration_profile(profile):
    normalized = get_default_calibration_profile()
    if not isinstance(profile, dict):
        return normalized

    normalized["sample_count"] = max(0, int(profile.get("sample_count", normalized["sample_count"])))

    source_gaze_center = profile.get("gaze_center", {})
    if isinstance(source_gaze_center, dict):
        for key in ("lx", "ly", "rx", "ry"):
            normalized["gaze_center"][key] = _clamp(source_gaze_center.get(key, normalized["gaze_center"][key]), 0.2, 0.8)

    source_gaze_tolerance = profile.get("gaze_tolerance", {})
    if isinstance(source_gaze_tolerance, dict):
        for key in ("lx", "ly", "rx", "ry"):
            normalized["gaze_tolerance"][key] = _clamp(
                source_gaze_tolerance.get(key, normalized["gaze_tolerance"][key]),
                0.04,
                0.25,
            )

    source_head_center = profile.get("head_center", {})
    if isinstance(source_head_center, dict):
        normalized["head_center"]["x"] = _clamp(source_head_center.get("x", normalized["head_center"]["x"]), -30.0, 30.0)
        normalized["head_center"]["y"] = _clamp(source_head_center.get("y", normalized["head_center"]["y"]), -30.0, 30.0)

    source_engagement_threshold = profile.get("engagement_angle_threshold", {})
    if isinstance(source_engagement_threshold, dict):
        normalized["engagement_angle_threshold"]["x"] = _clamp(
            source_engagement_threshold.get("x", normalized["engagement_angle_threshold"]["x"]),
            6.0,
            20.0,
        )
        normalized["engagement_angle_threshold"]["y"] = _clamp(
            source_engagement_threshold.get("y", normalized["engagement_angle_threshold"]["y"]),
            6.0,
            20.0,
        )

    source_head_direction_threshold = profile.get("head_direction_threshold", {})
    if isinstance(source_head_direction_threshold, dict):
        normalized["head_direction_threshold"]["x"] = _clamp(
            source_head_direction_threshold.get("x", normalized["head_direction_threshold"]["x"]),
            10.0,
            30.0,
        )
        normalized["head_direction_threshold"]["y"] = _clamp(
            source_head_direction_threshold.get("y", normalized["head_direction_threshold"]["y"]),
            10.0,
            30.0,
        )

    return normalized


def _robust_center_and_spread(values):
    """Median and a standard-deviation estimate from the median absolute deviation.

    A few frames where the person glanced away or blinked move neither number, unlike mean and std.
    """
    center = float(np.median(values))
    spread = float(MAD_TO_STD * np.median(np.abs(values - center)))
    return center, spread


def build_calibration_profile(calibration_samples, min_samples=20):
    if not calibration_samples:
        return get_default_calibration_profile()

    valid_samples = []
    for sample in calibration_samples:
        if not isinstance(sample, dict):
            continue
        keys = ("lx", "ly", "rx", "ry", "x_angle", "y_angle")
        if all(key in sample for key in keys):
            valid_samples.append(sample)

    if not valid_samples:
        return get_default_calibration_profile()

    lx = np.asarray([float(sample["lx"]) for sample in valid_samples], dtype=np.float64)
    ly = np.asarray([float(sample["ly"]) for sample in valid_samples], dtype=np.float64)
    rx = np.asarray([float(sample["rx"]) for sample in valid_samples], dtype=np.float64)
    ry = np.asarray([float(sample["ry"]) for sample in valid_samples], dtype=np.float64)
    x_angle = np.asarray([float(sample["x_angle"]) for sample in valid_samples], dtype=np.float64)
    y_angle = np.asarray([float(sample["y_angle"]) for sample in valid_samples], dtype=np.float64)

    centers = {}
    spreads = {}
    for name, values in (("lx", lx), ("ly", ly), ("rx", rx), ("ry", ry), ("x_angle", x_angle), ("y_angle", y_angle)):
        centers[name], spreads[name] = _robust_center_and_spread(values)

    worst_head_spread = max(spreads["x_angle"], spreads["y_angle"])
    if worst_head_spread > CALIBRATION_MAX_HEAD_STD:
        print(
            f"Calibration rejected: the head moved too much ({worst_head_spread:.1f} degrees spread, "
            f"limit {CALIBRATION_MAX_HEAD_STD:.0f}). Look straight at the camera and try again."
        )
        return get_default_calibration_profile()

    sample_count = len(valid_samples)
    completion_ratio = min(1.0, sample_count / float(max(1, min_samples)))

    base_profile = get_default_calibration_profile()
    raw_profile = {
        "sample_count": sample_count,
        "gaze_center": {key: centers[key] for key in ("lx", "ly", "rx", "ry")},
        "gaze_tolerance": {key: 0.04 + (2.0 * spreads[key]) for key in ("lx", "ly", "rx", "ry")},
        "head_center": {
            "x": centers["x_angle"],
            "y": centers["y_angle"],
        },
        "engagement_angle_threshold": {
            "x": 8.0 + (2.5 * spreads["x_angle"]),
            "y": 8.0 + (2.5 * spreads["y_angle"]),
        },
        "head_direction_threshold": {
            "x": 12.0 + (3.0 * spreads["x_angle"]),
            "y": 12.0 + (3.0 * spreads["y_angle"]),
        },
    }

    # Blend toward defaults when sample count is low to reduce overfitting.
    blended_profile = get_default_calibration_profile()
    blended_profile["sample_count"] = sample_count
    for key in ("lx", "ly", "rx", "ry"):
        blended_profile["gaze_center"][key] = (
            (completion_ratio * raw_profile["gaze_center"][key])
            + ((1.0 - completion_ratio) * base_profile["gaze_center"][key])
        )
        blended_profile["gaze_tolerance"][key] = (
            (completion_ratio * raw_profile["gaze_tolerance"][key])
            + ((1.0 - completion_ratio) * base_profile["gaze_tolerance"][key])
        )

    for axis in ("x", "y"):
        blended_profile["head_center"][axis] = (
            (completion_ratio * raw_profile["head_center"][axis])
            + ((1.0 - completion_ratio) * base_profile["head_center"][axis])
        )
        blended_profile["engagement_angle_threshold"][axis] = (
            (completion_ratio * raw_profile["engagement_angle_threshold"][axis])
            + ((1.0 - completion_ratio) * base_profile["engagement_angle_threshold"][axis])
        )
        blended_profile["head_direction_threshold"][axis] = (
            (completion_ratio * raw_profile["head_direction_threshold"][axis])
            + ((1.0 - completion_ratio) * base_profile["head_direction_threshold"][axis])
        )

    return normalize_calibration_profile(blended_profile)


def build_eye_reference_points():
    leye_3d = np.array(FACE_3D_TEMPLATE)
    leye_3d[:, 0] += 225
    leye_3d[:, 1] -= 175
    leye_3d[:, 2] += 135

    reye_3d = np.array(FACE_3D_TEMPLATE)
    reye_3d[:, 0] -= 225
    reye_3d[:, 1] -= 175
    reye_3d[:, 2] += 135
    return leye_3d, reye_3d


def extract_landmark_arrays(face_landmarks, img_w, img_h):
    """Pixel coordinates of the 478 landmarks of one face (a list of landmarks with normalized .x and .y)."""
    # Keep sub-pixel precision: the eye ratios divide by eye widths of only a few tens of pixels, so
    # truncating to whole pixels would add visible jitter. Cast to int only where drawing needs it.
    face_2d = np.asarray(
        [(lm.x * img_w, lm.y * img_h) for lm in face_landmarks],
        dtype=np.float64,
    )

    face_2d_head = np.array(
        [
            face_2d[1],
            face_2d[199],
            face_2d[33],
            face_2d[263],
            face_2d[61],
            face_2d[291],
        ],
        dtype=np.float64,
    )
    return face_2d, face_2d_head


def _eye_aspect_ratio(face_2d, upper_lid, lower_lid, outer_corner, inner_corner):
    width = abs(face_2d[outer_corner, 0] - face_2d[inner_corner, 0])
    if width == 0:
        return 0.0
    return float(abs(face_2d[upper_lid, 1] - face_2d[lower_lid, 1]) / width)


def compute_eye_openness(face_2d):
    """Eye aspect ratios (lid gap over eye width) for the eye with iris 468 and the eye with iris 473."""
    first_eye = _eye_aspect_ratio(face_2d, upper_lid=159, lower_lid=145, outer_corner=33, inner_corner=133)
    second_eye = _eye_aspect_ratio(face_2d, upper_lid=386, lower_lid=374, outer_corner=263, inner_corner=362)
    return first_eye, second_eye


def compute_gaze_scores(face_2d, previous_scores, threshold):
    """Smoothed iris position ratios for both eyes.

    During a blink the lid landmarks collapse onto each other, which makes the vertical ratios meaningless,
    so ly and ry keep their previous values until at least one eye is open again.
    """
    last_lx = previous_scores["lx"]
    last_ly = previous_scores["ly"]
    last_rx = previous_scores["rx"]
    last_ry = previous_scores["ry"]

    lx_score = last_lx
    ly_score = last_ly
    rx_score = last_rx
    ry_score = last_ry

    if (face_2d[243, 0] - face_2d[130, 0]) != 0:
        current = (face_2d[468, 0] - face_2d[130, 0]) / (face_2d[243, 0] - face_2d[130, 0])
        if abs(current - last_lx) < threshold:
            current = (current + last_lx) / 2
        lx_score = current

    blinking = max(compute_eye_openness(face_2d)) < EYE_CLOSED_THRESHOLD

    if not blinking and (face_2d[23, 1] - face_2d[27, 1]) != 0:
        current = (face_2d[468, 1] - face_2d[27, 1]) / (face_2d[23, 1] - face_2d[27, 1])
        if abs(current - last_ly) < threshold:
            current = (current + last_ly) / 2
        ly_score = current

    if (face_2d[359, 0] - face_2d[463, 0]) != 0:
        current = (face_2d[473, 0] - face_2d[463, 0]) / (face_2d[359, 0] - face_2d[463, 0])
        if abs(current - last_rx) < threshold:
            current = (current + last_rx) / 2
        rx_score = current

    if not blinking and (face_2d[253, 1] - face_2d[257, 1]) != 0:
        current = (face_2d[473, 1] - face_2d[257, 1]) / (face_2d[253, 1] - face_2d[257, 1])
        if abs(current - last_ry) < threshold:
            current = (current + last_ry) / 2
        ry_score = current

    return {
        "lx": lx_score,
        "ly": ly_score,
        "rx": rx_score,
        "ry": ry_score,
    }


def build_camera_and_distortion(img_w, img_h):
    focal_length = 1 * img_w
    cam_matrix = np.array(
        [[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return cam_matrix, dist_coeffs


def _is_plausible_head_pose(face_2d_head, rot_vec, trans_vec, cam_matrix, dist_coeffs):
    # solvePnP reports success for almost any input, so check the result instead: finite values, the
    # whole model in front of the camera, and a projection that matches the detected landmarks.
    if not (np.all(np.isfinite(rot_vec)) and np.all(np.isfinite(trans_vec))):
        return False

    rmat, _ = cv2.Rodrigues(rot_vec)
    model_in_camera = HEAD_MODEL_3D @ rmat.T + trans_vec.reshape(1, 3)
    if np.any(model_in_camera[:, 2] <= 0):
        return False

    projected, _ = cv2.projectPoints(HEAD_MODEL_3D, rot_vec, trans_vec, cam_matrix, dist_coeffs)
    mean_error = np.mean(np.linalg.norm(projected.reshape(-1, 2) - face_2d_head, axis=1))
    eye_distance = np.linalg.norm(face_2d_head[2] - face_2d_head[3])
    return bool(mean_error <= MAX_POSE_REPROJECTION_RATIO * eye_distance)


def _initial_head_pose_guess(face_2d_head, cam_matrix):
    # Start the iterative solve from a frontal pose at the depth implied by the eye distance. Without a
    # starting point it sometimes converges on a mirrored solution behind the camera for distant faces.
    eye_distance_px = float(np.linalg.norm(face_2d_head[2] - face_2d_head[3]))
    if not np.isfinite(eye_distance_px) or eye_distance_px < 1.0:
        return None

    focal_length = cam_matrix[0][0]
    model_eye_distance = float(np.linalg.norm(HEAD_MODEL_3D[2] - HEAD_MODEL_3D[3]))
    depth = focal_length * model_eye_distance / eye_distance_px
    nose_x, nose_y = face_2d_head[0]
    tvec = np.array(
        [
            [(nose_x - cam_matrix[0][2]) * depth / focal_length],
            [(nose_y - cam_matrix[1][2]) * depth / focal_length],
            [depth],
        ],
        dtype=np.float64,
    )
    return np.zeros((3, 1), dtype=np.float64), tvec


def solve_pose_vectors(face_2d_head, cam_matrix, dist_coeffs, leye_3d, reye_3d):
    """Solve head pose from the six head landmarks.

    x_angle and y_angle are in degrees: positive x_angle is the head tilted up, positive
    y_angle is the head turned toward the right of the image. Returns None when the head pose cannot
    be solved. "eye_axes_valid" is False when only one of the per-eye solves failed; the head angles
    are still usable then, but the eye and gaze axes must not be drawn.
    """
    guess = _initial_head_pose_guess(face_2d_head, cam_matrix)
    if guess is None:
        return None
    rvec_guess, tvec_guess = guess

    try:
        ok_l, l_rvec, l_tvec = cv2.solvePnP(
            leye_3d,
            face_2d_head,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        ok_r, r_rvec, r_tvec = cv2.solvePnP(
            reye_3d,
            face_2d_head,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        success_pnp, rot_vec, trans_vec = cv2.solvePnP(
            HEAD_MODEL_3D,
            face_2d_head,
            cam_matrix,
            dist_coeffs,
            rvec_guess,
            tvec_guess,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not success_pnp or not _is_plausible_head_pose(face_2d_head, rot_vec, trans_vec, cam_matrix, dist_coeffs):
        return None

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return {
        "l_rvec": l_rvec,
        "l_tvec": l_tvec,
        "r_rvec": r_rvec,
        "r_tvec": r_tvec,
        "rot_vec": rot_vec,
        "trans_vec": trans_vec,
        "eye_axes_valid": bool(ok_l and ok_r),
        "x_angle": -float(angles[0]),
        "y_angle": -float(angles[1]),
    }


def build_projection_data(pose_data, cam_matrix, dist_coeffs, gaze_scores, x_score_multiplier, y_score_multiplier):
    if not pose_data.get("eye_axes_valid", True):
        return None

    l_gaze_rvec = np.array(pose_data["l_rvec"])
    l_gaze_rvec[2][0] -= (gaze_scores["lx"] - 0.5) * x_score_multiplier
    l_gaze_rvec[0][0] += (gaze_scores["ly"] - 0.5) * y_score_multiplier

    r_gaze_rvec = np.array(pose_data["r_rvec"])
    r_gaze_rvec[2][0] -= (gaze_scores["rx"] - 0.5) * x_score_multiplier
    r_gaze_rvec[0][0] += (gaze_scores["ry"] - 0.5) * y_score_multiplier

    axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)

    l_axis, _ = cv2.projectPoints(axis, pose_data["l_rvec"], pose_data["l_tvec"], cam_matrix, dist_coeffs)
    l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, pose_data["l_tvec"], cam_matrix, dist_coeffs)

    r_axis, _ = cv2.projectPoints(axis, pose_data["r_rvec"], pose_data["r_tvec"], cam_matrix, dist_coeffs)
    r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, pose_data["r_tvec"], cam_matrix, dist_coeffs)

    return {
        "l_axis": l_axis,
        "l_gaze_axis": l_gaze_axis,
        "r_axis": r_axis,
        "r_gaze_axis": r_gaze_axis,
    }


def draw_pose_overlay(img, face_2d_head, projection_data, draw_headpose, draw_gaze, draw_full_axis):
    if projection_data is None:
        return

    l_corner = face_2d_head[2].astype(np.int32)
    r_corner = face_2d_head[3].astype(np.int32)

    if draw_headpose:
        if draw_full_axis:
            cv2.line(img, l_corner, tuple(np.ravel(projection_data["l_axis"][0]).astype(np.int32)), (200, 200, 0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(projection_data["l_axis"][1]).astype(np.int32)), (0, 200, 0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(projection_data["r_axis"][0]).astype(np.int32)), (200, 200, 0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(projection_data["r_axis"][1]).astype(np.int32)), (0, 200, 0), 3)

        cv2.line(img, l_corner, tuple(np.ravel(projection_data["l_axis"][2]).astype(np.int32)), (0, 200, 200), 3)
        cv2.line(img, r_corner, tuple(np.ravel(projection_data["r_axis"][2]).astype(np.int32)), (0, 200, 200), 3)

    if draw_gaze:
        if draw_full_axis:
            cv2.line(img, l_corner, tuple(np.ravel(projection_data["l_gaze_axis"][0]).astype(np.int32)), (255, 0, 0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(projection_data["l_gaze_axis"][1]).astype(np.int32)), (0, 255, 0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(projection_data["r_gaze_axis"][0]).astype(np.int32)), (255, 0, 0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(projection_data["r_gaze_axis"][1]).astype(np.int32)), (0, 255, 0), 3)

        cv2.line(img, l_corner, tuple(np.ravel(projection_data["l_gaze_axis"][2]).astype(np.int32)), (0, 0, 255), 3)
        cv2.line(img, r_corner, tuple(np.ravel(projection_data["r_gaze_axis"][2]).astype(np.int32)), (0, 0, 255), 3)


def draw_nose_direction(img, rot_vec, trans_vec, cam_matrix, dist_coeffs):
    # The model's nose is its origin; the axis points out of the face toward the camera.
    nose_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -NOSE_AXIS_LENGTH]], dtype=np.float64)
    projected, _ = cv2.projectPoints(nose_points, rot_vec, trans_vec, cam_matrix, dist_coeffs)
    p1 = (int(projected[0][0][0]), int(projected[0][0][1]))
    p2 = (int(projected[1][0][0]), int(projected[1][0][1]))
    cv2.line(img, p1, p2, (255, 0, 0), 3)


def classify_gaze_direction(gaze_scores, calibration_profile=None):
    profile = calibration_profile or get_default_calibration_profile()

    gaze_center = profile["gaze_center"]
    gaze_tolerance = profile["gaze_tolerance"]

    lx_delta = float(gaze_scores["lx"]) - gaze_center["lx"]
    ly_delta = float(gaze_scores["ly"]) - gaze_center["ly"]
    rx_delta = float(gaze_scores["rx"]) - gaze_center["rx"]
    ry_delta = float(gaze_scores["ry"]) - gaze_center["ry"]

    lx_offset = (float(gaze_scores["lx"]) - gaze_center["lx"]) / max(gaze_tolerance["lx"], 1e-6)
    ly_offset = (float(gaze_scores["ly"]) - gaze_center["ly"]) / max(gaze_tolerance["ly"], 1e-6)
    rx_offset = (float(gaze_scores["rx"]) - gaze_center["rx"]) / max(gaze_tolerance["rx"], 1e-6)
    ry_offset = (float(gaze_scores["ry"]) - gaze_center["ry"]) / max(gaze_tolerance["ry"], 1e-6)

    h_scale = max(1.0, (gaze_tolerance["lx"] + gaze_tolerance["rx"]) / 2.0 / GAZE_REFERENCE_TOLERANCE)
    v_scale = max(1.0, (gaze_tolerance["ly"] + gaze_tolerance["ry"]) / 2.0 / GAZE_REFERENCE_TOLERANCE)

    horizontal_offset = (lx_offset + rx_offset) / 2.0
    vertical_offset = (ly_offset + ry_offset) / 2.0
    horizontal_raw_delta = (lx_delta + rx_delta) / 2.0
    vertical_raw_delta = (ly_delta + ry_delta) / 2.0

    x_centered = abs(lx_offset) <= 0.55 and abs(rx_offset) <= 0.55
    y_centered = abs(ly_offset) <= 0.95 and abs(ry_offset) <= 0.95
    centered = (
        x_centered
        and y_centered
        and abs(horizontal_raw_delta) <= GAZE_FORWARD_MAX_DELTA_H * h_scale
        and abs(vertical_raw_delta) <= GAZE_FORWARD_MAX_DELTA_V * v_scale
    )
    if centered:
        return "Forward"

    horizontal_strength = abs(horizontal_offset)
    vertical_strength = abs(vertical_offset)

    right_candidate = (
        horizontal_raw_delta > GAZE_SIDE_MIN_DELTA * h_scale
        and (
            horizontal_offset > 0.35
            or max(lx_offset, rx_offset) > 0.70
        )
    )
    left_candidate = (
        horizontal_raw_delta < -GAZE_SIDE_MIN_DELTA * h_scale
        and (
            horizontal_offset < -0.35
            or min(lx_offset, rx_offset) < -0.70
        )
    )
    up_candidate = vertical_raw_delta < -GAZE_UP_MIN_DELTA * v_scale and vertical_offset < -0.85
    down_candidate = vertical_raw_delta > GAZE_DOWN_MIN_DELTA * v_scale and vertical_offset > 0.42

    if (up_candidate or down_candidate) and vertical_strength > (horizontal_strength + 0.08):
        return "Up" if up_candidate else "Down"

    if right_candidate or left_candidate:
        if not (up_candidate or down_candidate) or horizontal_strength >= vertical_strength:
            return "Right" if right_candidate else "Left"

    if up_candidate:
        return "Up"
    if down_candidate:
        return "Down"
    return "Not Forward"


def classify_engagement(gaze_direction, x_angle, y_angle, calibration_profile=None):
    profile = calibration_profile or get_default_calibration_profile()
    head_center_x = profile["head_center"]["x"]
    head_center_y = profile["head_center"]["y"]
    x_threshold = profile["engagement_angle_threshold"]["x"]
    y_threshold = profile["engagement_angle_threshold"]["y"]

    if (
        gaze_direction == "Forward"
        and abs(float(x_angle) - head_center_x) < x_threshold
        and abs(float(y_angle) - head_center_y) < y_threshold
    ):
        return "Engaged"
    return "Not Engaged"


def classify_head_direction(x_angle, y_angle, calibration_profile=None):
    profile = calibration_profile or get_default_calibration_profile()
    x_delta = float(x_angle) - profile["head_center"]["x"]
    y_delta = float(y_angle) - profile["head_center"]["y"]

    x_threshold = profile["head_direction_threshold"]["x"]
    y_threshold = profile["head_direction_threshold"]["y"]

    if y_delta < -y_threshold:
        return "Looking Left"
    if y_delta > y_threshold:
        return "Looking Right"
    if x_delta < -x_threshold:
        return "Looking Down"
    if x_delta > x_threshold:
        return "Looking Up"
    return "Looking Straight"
