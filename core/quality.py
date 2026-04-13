import cv2
import numpy as np


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def get_default_quality_state():
    return {
        "previous_face_area_ratio": None,
        "previous_landmarks": None,
    }


def get_confidence_label(confidence_score):
    if confidence_score >= 70.0:
        return "High"
    if confidence_score >= 45.0:
        return "Medium"
    return "Low"


def _get_face_bbox(face_2d):
    if face_2d is None or len(face_2d) == 0:
        return None

    x_min = float(np.min(face_2d[:, 0]))
    x_max = float(np.max(face_2d[:, 0]))
    y_min = float(np.min(face_2d[:, 1]))
    y_max = float(np.max(face_2d[:, 1]))
    return x_min, y_min, x_max, y_max


def _compute_face_visibility_score(face_2d, img_w, img_h, quality_state):
    warnings = []
    face_bbox = _get_face_bbox(face_2d)
    if face_bbox is None:
        return 0.0, warnings, None

    x_min, y_min, x_max, y_max = face_bbox
    face_width = max(1.0, x_max - x_min)
    face_height = max(1.0, y_max - y_min)
    frame_area = float(max(1, img_w * img_h))
    face_area_ratio = (face_width * face_height) / frame_area

    if face_area_ratio < 0.05:
        area_score = clamp01(face_area_ratio / 0.05)
        warnings.append("Face too far from camera")
    elif face_area_ratio > 0.55:
        area_score = clamp01(1.0 - ((face_area_ratio - 0.55) / 0.30))
        warnings.append("Face too close to camera")
    else:
        area_score = 1.0

    face_center_x = (x_min + x_max) / 2.0
    face_center_y = (y_min + y_max) / 2.0
    center_delta_x = abs(face_center_x - (img_w / 2.0)) / float(max(1, img_w))
    center_delta_y = abs(face_center_y - (img_h / 2.0)) / float(max(1, img_h))
    center_score = clamp01(1.0 - ((center_delta_x + center_delta_y) / 0.55))
    if center_score < 0.30:
        warnings.append("Face off-center")

    previous_face_area_ratio = quality_state.get("previous_face_area_ratio")
    if previous_face_area_ratio is None:
        area_stability_score = 1.0
    else:
        area_stability_score = clamp01(1.0 - (min(abs(face_area_ratio - previous_face_area_ratio), 0.14) / 0.14))

    quality_state["previous_face_area_ratio"] = face_area_ratio
    visibility_score = (0.55 * area_score) + (0.30 * center_score) + (0.15 * area_stability_score)
    return clamp01(visibility_score), warnings, face_bbox


def _extract_face_roi(gray_frame, face_bbox):
    if face_bbox is None:
        return gray_frame

    x_min, y_min, x_max, y_max = face_bbox
    x1 = int(max(0, min(gray_frame.shape[1] - 1, np.floor(x_min))))
    y1 = int(max(0, min(gray_frame.shape[0] - 1, np.floor(y_min))))
    x2 = int(max(x1 + 1, min(gray_frame.shape[1], np.ceil(x_max))))
    y2 = int(max(y1 + 1, min(gray_frame.shape[0], np.ceil(y_max))))
    roi = gray_frame[y1:y2, x1:x2]
    return roi if roi.size else gray_frame


def _compute_lighting_score(frame, face_bbox):
    warnings = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi = _extract_face_roi(gray_frame, face_bbox)

    mean_brightness = float(np.mean(face_roi))
    contrast = float(np.std(face_roi))

    # Best brightness is near center of dynamic range.
    brightness_score = clamp01(1.0 - (min(abs(mean_brightness - 135.0), 110.0) / 110.0))
    contrast_score = clamp01(contrast / 55.0)
    lighting_score = (0.70 * brightness_score) + (0.30 * contrast_score)

    if mean_brightness < 70.0:
        warnings.append("Low light")
    elif mean_brightness > 205.0:
        warnings.append("Overexposed frame")

    if contrast < 20.0:
        warnings.append("Low contrast")

    return clamp01(lighting_score), warnings


def _compute_landmark_stability_score(face_2d, img_w, img_h, quality_state):
    warnings = []
    previous_landmarks = quality_state.get("previous_landmarks")
    current_landmarks = np.asarray(face_2d, dtype=np.float64)

    if previous_landmarks is None or previous_landmarks.shape != current_landmarks.shape:
        quality_state["previous_landmarks"] = current_landmarks
        return 0.90, warnings

    normalization_vector = np.array([max(1, img_w), max(1, img_h)], dtype=np.float64)
    landmark_deltas = (current_landmarks - previous_landmarks) / normalization_vector
    mean_delta = float(np.mean(np.linalg.norm(landmark_deltas, axis=1)))

    stability_score = clamp01(1.0 - (mean_delta / 0.020))
    if mean_delta > 0.030:
        warnings.append("Unstable tracking")
    elif mean_delta > 0.018:
        warnings.append("Frequent movement")

    quality_state["previous_landmarks"] = current_landmarks
    return stability_score, warnings


def compute_quality_metrics(frame, face_2d, img_w, img_h, quality_state):
    visibility_score, visibility_warnings, face_bbox = _compute_face_visibility_score(
        face_2d,
        img_w,
        img_h,
        quality_state,
    )
    lighting_score, lighting_warnings = _compute_lighting_score(frame, face_bbox)
    stability_score, stability_warnings = _compute_landmark_stability_score(
        face_2d,
        img_w,
        img_h,
        quality_state,
    )

    confidence_score = clamp01(
        (0.45 * visibility_score)
        + (0.30 * lighting_score)
        + (0.25 * stability_score)
    ) * 100.0

    warnings = []
    for warning in visibility_warnings + lighting_warnings + stability_warnings:
        if warning not in warnings:
            warnings.append(warning)

    return {
        "face_visibility_score": visibility_score,
        "lighting_score": lighting_score,
        "landmark_stability_score": stability_score,
        "confidence_score": confidence_score,
        "confidence_label": get_confidence_label(confidence_score),
        "warnings": warnings,
    }