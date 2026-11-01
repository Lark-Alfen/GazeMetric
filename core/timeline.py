"""Per-frame timeline logging to CSV/JSON."""

import csv
import json
import os
from datetime import datetime

TIMELINE_FIELDNAMES = [
    "frame_index",
    "timestamp_seconds",
    "face_index",
    "x_angle",
    "y_angle",
    "gaze_direction",
    "engagement_label",
    "head_direction",
    "head_pose_score",
    "gaze_center_score",
    "engagement_score_raw",
    "engagement_score_smooth",
    "face_visibility_score",
    "lighting_score",
    "landmark_stability_score",
    "quality_confidence_score",
    "quality_confidence_label",
    "quality_warnings",
]

POSE_UNAVAILABLE_LABEL = "Pose Unavailable"


def video_frame_timestamp(frame_index, source_fps, position_msec=None):
    """Seconds into a recorded video for a 1-based frame index.

    Uses the position the decoder reports when there is one (this follows variable-frame-rate files) and
    otherwise counts frames at source_fps. The first frame reports a position of 0, which is also what
    the frame count gives.
    """
    if position_msec is not None and position_msec > 0:
        return round(position_msec / 1000.0, 3)
    return round((frame_index - 1) / source_fps, 3)


def get_default_timeline_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("runs", f"session_{timestamp}")


def _ensure_unique_path(path):
    candidate = path
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path}_{suffix:02d}"
        suffix += 1
    return candidate


def resolve_timeline_output_dir(timeline_dir=None):
    if timeline_dir:
        normalized_dir = os.path.normpath(timeline_dir)
        base_name = os.path.basename(normalized_dir)

        # If user already provided an explicit session path, preserve it and only
        # add a numeric suffix when that exact folder already exists.
        if base_name.startswith("session_"):
            return _ensure_unique_path(normalized_dir)

        return get_default_timeline_dir_in_base(normalized_dir)

    return get_default_timeline_dir()


def get_default_timeline_dir_in_base(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(base_dir, f"session_{timestamp}")
    return _ensure_unique_path(candidate)


def export_timeline_records(timeline_records, timeline_dir):
    os.makedirs(timeline_dir, exist_ok=True)

    csv_path = os.path.join(timeline_dir, "timeline.csv")
    json_path = os.path.join(timeline_dir, "timeline.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TIMELINE_FIELDNAMES)
        writer.writeheader()
        for record in timeline_records:
            writer.writerow(record)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(timeline_records, json_file, indent=2)

    return csv_path, json_path


def _format_quality_warnings(quality_warnings):
    if quality_warnings is None:
        return ""
    if isinstance(quality_warnings, list):
        return "; ".join(str(item) for item in quality_warnings if item)
    return str(quality_warnings)


def create_face_timeline_record(
    frame_index,
    timestamp_seconds,
    face_index,
    x_angle,
    y_angle,
    gaze_direction,
    engagement_label,
    head_direction,
    head_pose_score,
    gaze_center_score,
    engagement_score_raw,
    engagement_score_smooth,
    face_visibility_score=None,
    lighting_score=None,
    landmark_stability_score=None,
    quality_confidence_score=None,
    quality_confidence_label=None,
    quality_warnings=None,
):
    warnings_text = _format_quality_warnings(quality_warnings)
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "face_index": face_index,
        "x_angle": round(float(x_angle), 4),
        "y_angle": round(float(y_angle), 4),
        "gaze_direction": gaze_direction,
        "engagement_label": engagement_label,
        "head_direction": head_direction,
        "head_pose_score": round(float(head_pose_score), 4),
        "gaze_center_score": round(float(gaze_center_score), 4),
        "engagement_score_raw": round(float(engagement_score_raw), 2),
        "engagement_score_smooth": float(engagement_score_smooth),
        "face_visibility_score": round(float(face_visibility_score), 4) if face_visibility_score is not None else None,
        "lighting_score": round(float(lighting_score), 4) if lighting_score is not None else None,
        "landmark_stability_score": (
            round(float(landmark_stability_score), 4)
            if landmark_stability_score is not None
            else None
        ),
        "quality_confidence_score": (
            round(float(quality_confidence_score), 2)
            if quality_confidence_score is not None
            else None
        ),
        "quality_confidence_label": quality_confidence_label if quality_confidence_label is not None else "Unknown",
        "quality_warnings": warnings_text,
    }


def create_pose_failed_timeline_record(frame_index, timestamp_seconds, face_index):
    """A face was detected but its head pose could not be solved."""
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "face_index": face_index,
        "x_angle": None,
        "y_angle": None,
        "gaze_direction": POSE_UNAVAILABLE_LABEL,
        "engagement_label": POSE_UNAVAILABLE_LABEL,
        "head_direction": POSE_UNAVAILABLE_LABEL,
        "head_pose_score": None,
        "gaze_center_score": None,
        "engagement_score_raw": None,
        "engagement_score_smooth": None,
        "face_visibility_score": None,
        "lighting_score": None,
        "landmark_stability_score": None,
        "quality_confidence_score": None,
        "quality_confidence_label": "Unknown",
        "quality_warnings": POSE_UNAVAILABLE_LABEL,
    }


def create_no_face_timeline_record(frame_index, timestamp_seconds):
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "face_index": -1,
        "x_angle": None,
        "y_angle": None,
        "gaze_direction": "No Face",
        "engagement_label": "No Face",
        "head_direction": "No Face",
        "head_pose_score": None,
        "gaze_center_score": None,
        "engagement_score_raw": None,
        "engagement_score_smooth": None,
        "face_visibility_score": None,
        "lighting_score": None,
        "landmark_stability_score": None,
        "quality_confidence_score": None,
        "quality_confidence_label": "No Face",
        "quality_warnings": "No Face",
    }