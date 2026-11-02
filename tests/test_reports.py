import json

import pytest

from core.report_exports import build_summary_payload, export_session_reports
from core.reporting import print_summary
from core.timeline import create_face_timeline_record, create_no_face_timeline_record


def record(frame, face, score, quality, label="Engaged"):
    return create_face_timeline_record(
        frame_index=frame,
        timestamp_seconds=frame / 10,
        face_index=face,
        x_angle=0.0,
        y_angle=0.0,
        gaze_direction="Forward",
        engagement_label=label,
        head_direction="Looking Straight",
        head_pose_score=90.0,
        gaze_center_score=90.0,
        engagement_score_raw=score,
        engagement_score_smooth=score,
        quality_confidence_score=quality,
        quality_confidence_label="High",
    )


@pytest.fixture
def two_faces_three_frames():
    """Face 0 engaged around 80, face 1 not engaged around 20; face 1 is missing in the last frame."""
    return [
        record(1, 0, 80.0, 90.0),
        record(1, 1, 20.0, 50.0, "Not Engaged"),
        record(2, 0, 100.0, 70.0),
        record(2, 1, 40.0, 60.0, "Not Engaged"),
        record(3, 0, 60.0, 80.0),
        create_no_face_timeline_record(4, 0.4),
    ]


# ---------------------------------------------------------------- summary payload
def test_counts_distinguish_frames_faces_and_observations(two_faces_three_frames):
    kpis = build_summary_payload(two_faces_three_frames, "video")["kpis"]
    assert kpis["total_frames"] == 4
    assert kpis["face_records"] == 5
    assert kpis["frames_with_face"] == 3
    assert kpis["faces_seen"] == 2
    assert kpis["no_face_records"] == 1


def test_per_face_blocks_do_not_mix_people(two_faces_three_frames):
    per_face = build_summary_payload(two_faces_three_frames, "video")["kpis"]["per_face"]
    assert sorted(per_face) == ["0", "1"]
    assert per_face["0"]["records"] == 3 and per_face["1"]["records"] == 2
    assert per_face["0"]["frames"] == 3 and per_face["1"]["frames"] == 2
    assert per_face["0"]["engagement_score"] == {"average": 80.0, "min": 60.0, "max": 100.0, "samples": 3}
    assert per_face["1"]["engagement_score"] == {"average": 30.0, "min": 20.0, "max": 40.0, "samples": 2}
    assert per_face["0"]["quality_confidence"]["average"] == 80.0
    assert per_face["1"]["engagement_label"]["counts"] == {"Not Engaged": 2}
    assert per_face["0"]["engagement_label"]["counts"] == {"Engaged": 3}


def test_top_level_scores_stay_the_combined_numbers(two_faces_three_frames):
    score = build_summary_payload(two_faces_three_frames, "video")["kpis"]["engagement_score"]
    assert score == {"average": 60.0, "min": 20.0, "max": 100.0, "samples": 5}


def test_single_face_summary_keeps_every_existing_key():
    records = [record(i, 0, 70.0 + i, 80.0) for i in range(1, 4)]
    payload = build_summary_payload(records, "webcam")
    assert {"total_frames", "duration_seconds", "face_records", "no_face_records", "engagement_score",
            "quality_confidence", "warning_records"} <= set(payload["kpis"])
    assert {"engagement_label", "head_direction", "gaze_direction", "quality_confidence_label"} == set(payload["distributions"])
    assert list(payload["kpis"]["per_face"]) == ["0"]
    assert payload["kpis"]["per_face"]["0"]["engagement_score"] == payload["kpis"]["engagement_score"]


def test_summary_with_no_faces_has_empty_per_face():
    payload = build_summary_payload([create_no_face_timeline_record(1, 0.1)], "video")
    assert payload["kpis"]["per_face"] == {}
    assert payload["kpis"]["faces_seen"] == 0
    assert payload["kpis"]["engagement_score"]["average"] is None


# ---------------------------------------------------------------- pdf
def test_pdf_is_written_for_two_faces(two_faces_three_frames, tmp_path):
    summary_json, summary_pdf, error = export_session_reports(two_faces_three_frames, str(tmp_path), "video")
    assert error is None
    assert summary_pdf is not None and (tmp_path / "summary.pdf").stat().st_size > 1000
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["kpis"]["faces_seen"] == 2


def test_pdf_draws_one_line_per_face(two_faces_three_frames, tmp_path, monkeypatch):
    import matplotlib.axes

    plotted = []
    original = matplotlib.axes.Axes.plot

    def spy(self, *args, **kwargs):
        plotted.append((tuple(args[0]), tuple(args[1]), kwargs.get("label")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
    export_session_reports(two_faces_three_frames, str(tmp_path), "video")

    by_label = {}
    for xs, ys, label in plotted:
        by_label.setdefault(label, []).append((xs, ys))
    assert set(by_label) == {"Face 0", "Face 1"}
    assert ((0.1, 0.2, 0.3), (80.0, 100.0, 60.0)) in by_label["Face 0"]
    assert ((0.1, 0.2), (20.0, 40.0)) in by_label["Face 1"]
    # no point appears twice at one timestamp inside a line, which is what the old merged line did
    for lines in by_label.values():
        for xs, _ys in lines:
            assert len(xs) == len(set(xs))


# ---------------------------------------------------------------- console summary
def test_console_summary_labels_observations_and_reports_frames(capsys):
    statements = ["Looking Straight"] * 4 + ["Looking Left"] * 2
    states = ["Engaged"] * 5 + ["Not Engaged"]
    print_summary(statements, states, frames_with_faces=3)
    out = capsys.readouterr().out
    assert "Total frames analyzed" not in out
    assert out.count("Face observations analyzed: 6") == 2
    assert out.count("Frames with at least one face: 3") == 2


def test_console_summary_without_a_frame_count_still_works(capsys):
    print_summary(["Looking Straight"], ["Engaged"])
    out = capsys.readouterr().out
    assert "Face observations analyzed: 1" in out
    assert "Frames with at least one face" not in out


# ---------------------------------------------------------------- warnings, distributions and run context
def warned(frame, face, warnings, label="Engaged", head="Looking Straight"):
    built = create_face_timeline_record(
        frame_index=frame,
        timestamp_seconds=frame / 10,
        face_index=face,
        x_angle=0.0,
        y_angle=0.0,
        gaze_direction="Forward",
        engagement_label=label,
        head_direction=head,
        head_pose_score=90.0,
        gaze_center_score=90.0,
        engagement_score_raw=50.0,
        engagement_score_smooth=50.0,
        quality_confidence_score=50.0,
        quality_confidence_label="Medium",
        quality_warnings=warnings,
    )
    return built


def test_warnings_are_counted_per_individual_warning_and_ranked():
    records = [
        warned(1, 0, ["Low light", "Frequent movement"]),
        warned(2, 0, ["Low light"]),
        warned(3, 0, ["Low light", "Face off-center"]),
        warned(4, 0, []),
        create_no_face_timeline_record(5, 0.5),
    ]
    payload = build_summary_payload(records, "video")
    assert payload["warnings"]["total_unique"] == 3
    assert payload["warnings"]["top"][0] == {"warning": "Low light", "count": 3}
    assert {item["warning"]: item["count"] for item in payload["warnings"]["top"]} == {
        "Low light": 3,
        "Frequent movement": 1,
        "Face off-center": 1,
    }
    assert payload["kpis"]["warning_records"] == 3  # records that carry at least one warning


def test_the_no_face_marker_is_not_a_warning():
    payload = build_summary_payload([create_no_face_timeline_record(1, 0.1)], "video")
    assert payload["warnings"] == {"total_unique": 0, "top": []}
    assert payload["kpis"]["warning_records"] == 0


def test_only_the_ten_most_common_warnings_are_listed():
    records = [warned(i, 0, [f"warning {i}"]) for i in range(1, 15)]
    assert len(build_summary_payload(records, "video")["warnings"]["top"]) == 10


def test_distributions_have_counts_and_percentages_that_add_up():
    records = [
        warned(1, 0, [], "Engaged", "Looking Straight"),
        warned(2, 0, [], "Engaged", "Looking Left"),
        warned(3, 0, [], "Not Engaged", "Looking Left"),
        warned(4, 0, [], "Engaged", "Looking Straight"),
    ]
    distributions = build_summary_payload(records, "video")["distributions"]
    assert distributions["engagement_label"]["counts"] == {"Engaged": 3, "Not Engaged": 1}
    assert distributions["engagement_label"]["percentages"] == {"Engaged": 75.0, "Not Engaged": 25.0}
    assert distributions["engagement_label"]["total"] == 4
    assert distributions["head_direction"]["percentages"] == {"Looking Straight": 50.0, "Looking Left": 50.0}
    assert distributions["gaze_direction"]["counts"] == {"Forward": 4}


def test_quality_label_distribution_includes_frames_without_a_face():
    records = [warned(1, 0, []), create_no_face_timeline_record(2, 0.2)]
    counts = build_summary_payload(records, "video")["distributions"]["quality_confidence_label"]["counts"]
    assert counts == {"Medium": 1, "No Face": 1}


def test_duration_is_the_last_timestamp_and_frames_are_distinct():
    records = [warned(1, 0, []), warned(1, 1, []), warned(7, 0, [])]
    kpis = build_summary_payload(records, "video")["kpis"]
    assert kpis["total_frames"] == 2
    assert kpis["duration_seconds"] == 0.7


def test_empty_timeline_gives_an_empty_but_complete_summary():
    payload = build_summary_payload([], "webcam")
    assert payload["kpis"]["total_frames"] == 0
    assert payload["kpis"]["duration_seconds"] == 0.0
    assert payload["kpis"]["engagement_score"]["average"] is None
    assert payload["distributions"]["engagement_label"] == {"total": 0, "counts": {}, "percentages": {}}


def test_run_context_and_timestamp_are_recorded():
    payload = build_summary_payload([], "video", input_path="in.mp4", output_video_path="out.mp4")
    assert payload["run_context"] == {"source_type": "video", "input_path": "in.mp4", "output_video_path": "out.mp4"}
    assert payload["generated_at"].endswith("Z")


# ---------------------------------------------------------------- export_session_reports
def test_export_writes_summary_json_and_pdf(tmp_path):
    json_path, pdf_path, error = export_session_reports([warned(1, 0, [])], str(tmp_path / "out"), "video", input_path="in.mp4")
    assert error is None
    assert json_path.endswith("summary.json") and pdf_path.endswith("summary.pdf")
    saved = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert saved["run_context"]["input_path"] == "in.mp4"
    assert "pdf_generation_error" not in saved


def test_a_failing_pdf_is_reported_and_the_json_survives(tmp_path, monkeypatch):
    from core import report_exports

    def explode(*_args, **_kwargs):
        raise RuntimeError("matplotlib is unhappy")

    monkeypatch.setattr(report_exports, "create_summary_pdf", explode)
    json_path, pdf_path, error = export_session_reports([warned(1, 0, [])], str(tmp_path), "video")

    assert pdf_path is None
    assert error == "matplotlib is unhappy"
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["pdf_generation_error"] == "matplotlib is unhappy"
    assert saved["kpis"]["face_records"] == 1
    assert not (tmp_path / "summary.pdf").exists()


def test_export_creates_the_output_folder(tmp_path):
    target = tmp_path / "a" / "b"
    export_session_reports([], str(target), "video")
    assert (target / "summary.json").is_file()
