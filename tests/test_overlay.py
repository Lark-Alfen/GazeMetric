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


# ---------------------------------------------------------------- face contours
from core.overlay import draw_face_contours  # noqa: E402


def test_contours_are_drawn_along_the_connections_only():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    face_2d = np.zeros((478, 2))
    face_2d[3] = (10, 20)
    face_2d[4] = (90, 20)
    face_2d[5] = (50, 80)  # not connected to anything
    draw_face_contours(frame, face_2d, [(3, 4)])
    ys, xs = np.nonzero(frame.any(axis=2))
    assert 20 in set(ys) and set(ys) <= {19, 20, 21}  # a horizontal line at y = 20, softened by anti-aliasing
    assert abs(xs.min() - 10) <= 1 and abs(xs.max() - 90) <= 1
    assert not frame[80, 50].any()


def test_no_connections_draws_nothing():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_face_contours(frame, np.zeros((478, 2)), [])
    draw_face_contours(frame, np.zeros((478, 2)), None)
    assert not frame.any()


def test_contour_points_outside_the_frame_do_not_crash():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    face_2d = np.zeros((478, 2))
    face_2d[0] = (-30, -30)
    face_2d[1] = (500, 400)
    draw_face_contours(frame, face_2d, [(0, 1)])
    assert frame.any()  # the visible part of the line


def test_many_connections_use_float_landmarks():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    face_2d = np.random.default_rng(1).uniform(5, 55, (478, 2))
    draw_face_contours(frame, face_2d, [(i, i + 1) for i in range(100)], colour=(0, 255, 0))
    assert frame[..., 1].any() and not frame[..., 0].any()


# ---------------------------------------------------------------- the outline stays attached to the text
def text_boxes(frame):
    """Bounding boxes of the coloured text pixels and of the dark outline pixels, on a mid-grey frame."""
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    coloured = (abs(g - r) > 120) | (abs(g - b) > 120)
    dark = (b < 60) & (g < 60) & (r < 60)

    def box(mask):
        ys, xs = np.nonzero(mask)
        return xs.min(), ys.min(), xs.max(), ys.max()

    return box(coloured), box(dark)


def test_the_dark_outline_hugs_the_text_whatever_the_opencv_version():
    # With OpenCV 5 a thicker outline font was about 7% wider than the text, so the two drifted apart.
    frame = np.full((300, 600, 3), 127, dtype=np.uint8)
    draw_face_hud(frame, face_points(200, 150, 400, 290), hud(engagement="Not Engaged", warnings=["Low light", "Face off-center"]))
    (cx0, cy0, cx1, cy1), (dx0, dy0, dx1, dy1) = text_boxes(frame)
    assert cx0 - 3 <= dx0 <= cx0 and cx1 <= dx1 <= cx1 + 3
    assert cy0 - 3 <= dy0 <= cy0 and cy1 <= dy1 <= cy1 + 3
