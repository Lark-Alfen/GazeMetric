from core.tracking import FaceTracker

LEFT, RIGHT = (0.25, 0.5), (0.75, 0.5)


def tracker(**kwargs):
    return FaceTracker(score_window_size=15, **kwargs)


def test_first_faces_get_new_ids_in_order():
    tracks = tracker().match(1, [LEFT, RIGHT])
    assert [t.track_id for t in tracks] == [0, 1]


def test_swapped_detection_order_keeps_the_ids():
    t = tracker()
    first = t.match(1, [LEFT, RIGHT])
    second = t.match(2, [RIGHT, LEFT])
    assert [x.track_id for x in second] == [first[1].track_id, first[0].track_id]
    assert second[0] is first[1] and second[1] is first[0]


def test_faces_that_move_a_little_keep_their_id():
    t = tracker()
    t.match(1, [LEFT])
    moved = t.match(2, [(0.30, 0.52)])
    assert moved[0].track_id == 0
    assert moved[0].centre == (0.30, 0.52)


def test_new_face_gets_a_new_id_and_existing_one_is_kept():
    t = tracker()
    t.match(1, [LEFT])
    tracks = t.match(2, [LEFT, RIGHT])
    assert [x.track_id for x in tracks] == [0, 1]


def test_face_that_jumps_too_far_is_a_new_person():
    t = tracker()
    t.match(1, [(0.1, 0.1)])
    tracks = t.match(2, [(0.9, 0.9)])
    assert tracks[0].track_id == 1


def test_two_detections_near_one_track_do_not_share_it():
    t = tracker()
    t.match(1, [LEFT])
    tracks = t.match(2, [(0.27, 0.5), (0.24, 0.5)])
    assert sorted(x.track_id for x in tracks) == [0, 1]
    assert tracks[1].track_id == 0  # the closer detection keeps the id
    assert tracks[0].track_id == 1


def test_track_survives_a_short_gap_with_its_state():
    t = tracker()
    first = t.match(1, [LEFT])[0]
    first.last_scores = {"lx": 0.6, "ly": 0.4, "rx": 0.6, "ry": 0.4}
    first.score_window.append(80.0)
    for frame in range(2, 12):
        assert t.match(frame, []) == []
    back = t.match(12, [LEFT])[0]
    assert back is first
    assert back.last_scores["lx"] == 0.6
    assert list(back.score_window) == [80.0]


def test_track_expires_after_30_missing_frames_and_state_is_not_reused():
    t = tracker()
    first = t.match(1, [LEFT])[0]
    first.last_scores = {"lx": 0.6, "ly": 0.4, "rx": 0.6, "ry": 0.4}
    first.score_window.append(80.0)

    for frame in range(2, 32):  # still allowed: 30 frames since it was last seen
        t.match(frame, [])
    assert t.match(31, [LEFT])[0] is first

    t = tracker()
    first = t.match(1, [LEFT])[0]
    for frame in range(2, 33):
        t.match(frame, [])
    returned = t.match(32, [LEFT])[0]
    assert returned is not first
    assert returned.track_id == 1
    assert returned.last_scores == {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
    assert len(returned.score_window) == 0


def test_each_track_has_its_own_state_objects():
    a, b = tracker().match(1, [LEFT, RIGHT])
    a.last_scores["lx"] = 0.9
    a.score_window.append(50.0)
    a.quality_state["previous_face_area_ratio"] = 0.3
    assert b.last_scores["lx"] == 0.0
    assert len(b.score_window) == 0
    assert b.quality_state["previous_face_area_ratio"] is None


def test_score_window_is_limited_to_the_configured_size():
    track = FaceTracker(score_window_size=3).match(1, [LEFT])[0]
    for value in range(10):
        track.score_window.append(value)
    assert list(track.score_window) == [7, 8, 9]


def test_no_faces_gives_no_tracks():
    assert tracker().match(1, []) == []
