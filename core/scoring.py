"""Composite engagement scoring from head pose and gaze."""

def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def compute_head_pose_score(x_angle, y_angle, max_angle=30.0, center_x=0.0, center_y=0.0):
    x_component = 1.0 - min(abs(float(x_angle) - float(center_x)), max_angle) / max_angle
    y_component = 1.0 - min(abs(float(y_angle) - float(center_y)), max_angle) / max_angle
    return clamp01((x_component + y_component) / 2.0)


def compute_gaze_center_score(gaze_scores, calibration_profile=None):
    gaze_center = {"lx": 0.5, "ly": 0.5, "rx": 0.5, "ry": 0.5}
    if calibration_profile is not None:
        gaze_center = calibration_profile["gaze_center"]

    center_scores = []
    for key in ("lx", "ly", "rx", "ry"):
        distance_from_center = min(abs(float(gaze_scores[key]) - float(gaze_center[key])), 0.5)
        center_scores.append(clamp01(1.0 - (distance_from_center / 0.5)))
    return sum(center_scores) / len(center_scores)


def compute_composite_engagement_score(gaze_scores, x_angle, y_angle, calibration_profile=None):
    center_x = 0.0
    center_y = 0.0
    if calibration_profile is not None:
        center_x = calibration_profile["head_center"]["x"]
        center_y = calibration_profile["head_center"]["y"]

    head_pose_score = compute_head_pose_score(
        x_angle,
        y_angle,
        center_x=center_x,
        center_y=center_y,
    )
    gaze_center_score = compute_gaze_center_score(gaze_scores, calibration_profile=calibration_profile)

    normalized_score = (0.55 * head_pose_score) + (0.45 * gaze_center_score)
    return head_pose_score, gaze_center_score, clamp01(normalized_score) * 100.0