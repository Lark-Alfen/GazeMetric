import numpy as np
import pytest

from core.overlay import (
    GREEN,
    LINE_HEIGHT,
    ORANGE,
    RED,
    YELLOW,
    build_hud_lines,
    draw_face_hud,
    engagement_colour,
    face_bbox,
    panel_origin,
    score_colour,
)


def face_points(x_min, y_min, x_max, y_max):
    return np.array([[x_min, y_min], [x_max, y_max], [(x_min + x_max) / 2, (y_min + y_max) / 2]], dtype=float)


def hud(face_id=0, engagement="Engaged", warnings=(), gaze="Forward", score=80.0, quality=90.0):
    return build_hud_lines(face_id, engagement, "Looking Straight", score, "High", quality, list(warnings), gaze)


def inked_area(frame):
    """Bounding box (x0, y0, x1, y1) of everything that was drawn on a black frame."""
    ys, xs = np.nonzero(frame.any(axis=2))
    return xs.min(), ys.min(), xs.max(), ys.max()


# ---------------------------------------------------------------- content and colours
def test_panel_names_the_face_and_its_engagement_state():
    lines = hud(face_id=3, engagement="Not Engaged")
    assert lines[0] == ("Face 3: Not Engaged", RED)


def test_engagement_label_colour_follows_the_state():
    assert engagement_colour("Engaged") == GREEN
    assert engagement_colour("Not Engaged") == RED


@pytest.mark.parametrize("score, colour", [(100, GREEN), (70, GREEN), (69.9, YELLOW), (45, YELLOW), (44.9, RED), (0, RED)])
def test_score_colours(score, colour):
    assert score_colour(score) == colour


def test_panel_has_one_line_per_item_and_at_most_two_warnings():
    assert len(hud()) == 5
    texts = [text for text, _ in hud(warnings=["a", "b", "c"])]
    assert len(texts) == 7
    assert texts.count("Warning: a") == texts.count("Warning: b") == 1 and "Warning: c" not in texts
    assert all(colour == ORANGE for text, colour in hud(warnings=["a"]) if text.startswith("Warning"))


def test_gaze_line_is_green_only_when_forward():
    assert hud(gaze="Forward")[-1] == ("Gaze: Forward", GREEN)
    assert hud(gaze="Left")[-1] == ("Gaze: Left", YELLOW)


# ---------------------------------------------------------------- placement
def test_panel_sits_above_the_face_when_there_is_room():
    x, y = panel_origin((300, 300, 400, 420), panel_w=200, panel_h=110, frame_w=800, frame_h=600)
    assert x == 300
    assert y + 110 <= 300


def test_panel_goes_below_the_face_when_there_is_no_room_above():
    x, y = panel_origin((300, 20, 400, 140), panel_w=200, panel_h=110, frame_w=800, frame_h=600)
    assert y >= 140


def test_panel_is_kept_inside_the_frame():
    x, y = panel_origin((700, 450, 790, 590), panel_w=250, panel_h=150, frame_w=800, frame_h=600)
    assert 0 <= x and x + 250 <= 800
    assert 0 <= y and y + 150 <= 600


def test_face_bbox_covers_all_points():
    assert face_bbox(face_points(10, 20, 110, 220)) == (10.0, 20.0, 110.0, 220.0)


# ---------------------------------------------------------------- drawing
def test_two_faces_get_separate_panels():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    left = np.zeros_like(frame)
    right = np.zeros_like(frame)
    draw_face_hud(left, face_points(100, 300, 250, 480), hud(face_id=0))
    draw_face_hud(right, face_points(650, 300, 800, 480), hud(face_id=1))
    left_box, right_box = inked_area(left), inked_area(right)
    assert left_box[2] < right_box[0]  # no shared columns, so nothing can overprint
    assert left_box[3] <= 300 and right_box[3] <= 300  # above each face

    draw_face_hud(frame, face_points(100, 300, 250, 480), hud(face_id=0))
    draw_face_hud(frame, face_points(650, 300, 800, 480), hud(face_id=1))
    assert np.array_equal(frame, np.maximum(left, right))


def test_panel_moves_with_the_face():
    first = np.zeros((600, 1000, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    draw_face_hud(first, face_points(100, 300, 250, 480), hud())
    draw_face_hud(second, face_points(400, 300, 550, 480), hud())
    assert inked_area(second)[0] - inked_area(first)[0] == 300


def test_drawing_stays_inside_a_small_frame_and_inside_its_own_panel():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    draw_face_hud(frame, face_points(250, 150, 299, 199), hud(warnings=["Low light", "Face off-center"]))
    x0, y0, x1, y1 = inked_area(frame)
    assert x0 >= 0 and y0 >= 0 and x1 < 300 and y1 < 200
    assert y1 - y0 <= LINE_HEIGHT * 7
