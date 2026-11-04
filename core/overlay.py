"""On-screen text drawn next to each detected face."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
LINE_HEIGHT = 22
MARGIN = 6
OUTLINE_OFFSETS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


def score_colour(score):
    if score < 45:
        return RED
    if score < 70:
        return YELLOW
    return GREEN


def engagement_colour(engagement_text):
    return GREEN if engagement_text == "Engaged" else RED


def build_hud_lines(
    face_id,
    engagement_text,
    head_text,
    engagement_score,
    quality_label,
    quality_score,
    quality_warnings,
    gaze_direction,
):
    """(text, colour) for every line of one face's panel, top to bottom."""
    lines = [
        (f"Face {face_id}: {engagement_text}", engagement_colour(engagement_text)),
        (head_text, GREEN),
        (f"Score: {engagement_score:6.2f}/100", score_colour(engagement_score)),
        (f"Quality: {quality_label} ({quality_score:5.1f}/100)", score_colour(quality_score)),
    ]
    lines.extend((f"Warning: {warning}", ORANGE) for warning in quality_warnings[:2])
    lines.append((f"Gaze: {gaze_direction}", GREEN if gaze_direction == "Forward" else YELLOW))
    return lines


def face_bbox(face_2d):
    return (
        float(np.min(face_2d[:, 0])),
        float(np.min(face_2d[:, 1])),
        float(np.max(face_2d[:, 0])),
        float(np.max(face_2d[:, 1])),
    )


def panel_origin(bbox, panel_w, panel_h, frame_w, frame_h):
    """Top-left corner of the panel: above the face when it fits, otherwise below it, kept inside the frame."""
    x_min, y_min, _x_max, y_max = bbox
    x = int(max(0, min(x_min, frame_w - panel_w)))
    y = y_min - panel_h - MARGIN
    if y < 0:
        y = y_max + MARGIN
    y = int(max(0, min(y, frame_h - panel_h)))
    return x, y


def draw_face_contours(frame, face_2d, connections, colour=(224, 224, 224), thickness=1):
    """Outline the eyes, brows, lips and face oval by joining the landmark pairs in `connections`."""
    if not connections:
        return
    points = np.round(face_2d).astype(np.int32)
    segments = list(points[np.asarray(connections)])
    cv2.polylines(frame, segments, False, colour, thickness, cv2.LINE_AA)


def draw_face_hud(frame, face_2d, lines):
    """Draw the lines in a panel attached to the face, so several faces never share the same text position."""
    frame_h, frame_w = frame.shape[:2]
    panel_w = max(cv2.getTextSize(text, FONT, FONT_SCALE, 1)[0][0] for text, _ in lines)
    panel_h = LINE_HEIGHT * len(lines)
    x, y = panel_origin(face_bbox(face_2d), panel_w, panel_h, frame_w, frame_h)

    for row, (text, colour) in enumerate(lines):
        origin = (x, y + (row + 1) * LINE_HEIGHT - 6)
        # A dark outline keeps the text readable on any background. It is the same thin text drawn one pixel
        # away in every direction: a thicker font would have different letter spacing in some OpenCV versions.
        for dx, dy in OUTLINE_OFFSETS:
            cv2.putText(frame, text, (origin[0] + dx, origin[1] + dy), FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text, origin, FONT, FONT_SCALE, colour, 1, cv2.LINE_AA)
