import argparse
import sys

import pytest

import engagement_analysis_full
import engagement_analysis_live
import engagement_analysis_recorded
from core.cli import add_common_arguments, validate_common_arguments

COMMON_DESTS = {"output_video_path", "timeline_dir", "skip_calibration", "calibration_seconds", "calibration_profile", "no_display"}


def parse(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return module.parse_arguments()


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not a real video, only has to exist")
    return str(path)


# ---------------------------------------------------------------- the shared helpers
def test_common_arguments_and_defaults():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args([])
    assert set(vars(args)) == COMMON_DESTS
    assert args.output_video_path is None and args.timeline_dir is None and args.calibration_profile is None
    assert args.skip_calibration is False
    assert args.calibration_seconds == 10.0
    assert args.no_display is False


def test_common_arguments_accept_every_option():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args(
        [
            "--output-video", "o.mp4",
            "--timeline-dir", "d",
            "--skip-calibration",
            "--calibration-seconds", "2.5",
            "--calibration-profile", "p.json",
        ]
    )
    assert (args.output_video_path, args.timeline_dir, args.calibration_profile) == ("o.mp4", "d", "p.json")
    assert args.skip_calibration is True and args.calibration_seconds == 2.5


@pytest.mark.parametrize("seconds", ["0", "-3"])
def test_non_positive_calibration_seconds_are_rejected(seconds, capsys):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args(["--calibration-seconds", seconds])
    with pytest.raises(SystemExit) as exit_info:
        validate_common_arguments(parser, args)
    assert exit_info.value.code == 2
    assert "--calibration-seconds must be greater than 0" in capsys.readouterr().err


# ---------------------------------------------------------------- the three entry points
@pytest.mark.parametrize("module", [engagement_analysis_full, engagement_analysis_live, engagement_analysis_recorded])
def test_every_entry_point_has_the_common_options(module, monkeypatch, clip):
    argv = ["--input", clip] if module is not engagement_analysis_live else []
    if module is engagement_analysis_full:
        argv += ["--source", "video"]
    args = parse(module, monkeypatch, *argv, "--skip-calibration", "--calibration-seconds", "3")
    assert COMMON_DESTS <= set(vars(args))
    assert args.skip_calibration is True and args.calibration_seconds == 3.0


@pytest.mark.parametrize("module", [engagement_analysis_full, engagement_analysis_live, engagement_analysis_recorded])
def test_every_entry_point_rejects_zero_calibration_seconds(module, monkeypatch, clip, capsys):
    argv = ["--input", clip] if module is not engagement_analysis_live else []
    if module is engagement_analysis_full:
        argv += ["--source", "video"]
    with pytest.raises(SystemExit) as exit_info:
        parse(module, monkeypatch, *argv, "--calibration-seconds", "0")
    assert exit_info.value.code == 2


@pytest.mark.parametrize("module", [engagement_analysis_full, engagement_analysis_live, engagement_analysis_recorded])
def test_no_display_flag_is_available_everywhere(module, monkeypatch, clip):
    argv = ["--input", clip] if module is not engagement_analysis_live else []
    if module is engagement_analysis_full:
        argv += ["--source", "video"]
    assert parse(module, monkeypatch, *argv).no_display is False
    assert parse(module, monkeypatch, *argv, "--no-display").no_display is True


def test_live_mode_is_a_webcam_without_an_input_file(monkeypatch):
    args = parse(engagement_analysis_live, monkeypatch)
    assert (args.source, args.input_path) == ("webcam", None)


def test_recorded_mode_is_a_video_and_needs_an_existing_input(monkeypatch, clip, capsys):
    args = parse(engagement_analysis_recorded, monkeypatch, "--input", clip)
    assert (args.source, args.input_path) == ("video", clip)

    with pytest.raises(SystemExit):
        parse(engagement_analysis_recorded, monkeypatch)
    with pytest.raises(SystemExit):
        parse(engagement_analysis_recorded, monkeypatch, "--input", clip + ".missing")
    assert "Input video file not found" in capsys.readouterr().err


def test_full_script_defaults_to_the_webcam_and_checks_video_input(monkeypatch, clip, capsys):
    args = parse(engagement_analysis_full, monkeypatch)
    assert (args.source, args.input_path) == ("webcam", None)

    args = parse(engagement_analysis_full, monkeypatch, "--source", "video", "--input", clip)
    assert (args.source, args.input_path) == ("video", clip)

    with pytest.raises(SystemExit):
        parse(engagement_analysis_full, monkeypatch, "--source", "video")
    assert "--input is required when --source video" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse(engagement_analysis_full, monkeypatch, "--source", "video", "--input", clip + ".missing")


def test_the_option_definitions_exist_in_one_place_only():
    for path in ("engagement_analysis_full.py", "engagement_analysis_live.py", "engagement_analysis_recorded.py"):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for option in ("--output-video", "--timeline-dir", "--skip-calibration", "--calibration-seconds", "--calibration-profile"):
            assert f'"{option}"' not in text, f"{option} is still defined in {path}"
