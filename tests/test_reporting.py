import pytest

from core.reporting import print_startup_banner


def banner(capsys, **kwargs):
    print_startup_banner(**kwargs)
    return capsys.readouterr().out


def test_banner_no_longer_warns_against_ctrl_c(capsys):
    text = banner(capsys)
    assert "DO NOT use Ctrl+C" not in text
    assert "skip the summary" not in text
    assert "Ctrl+C" in text and "see the summary" in text


def test_webcam_banner_talks_about_the_camera(capsys):
    text = banner(capsys, source_type="webcam")
    assert "camera" in text
    assert "video file" not in text
    assert "Starting camera" not in text


def test_video_banner_does_not_mention_a_camera(capsys):
    text = banner(capsys, source_type="video")
    assert "video file" in text
    assert "camera" not in text.lower().replace("webcam", "")


@pytest.mark.parametrize("source_type", ["webcam", "video"])
def test_window_controls_are_listed_only_when_there_is_a_window(capsys, source_type):
    assert "video window" in banner(capsys, source_type=source_type, show_window=True)
    assert "video window" not in banner(capsys, source_type=source_type, show_window=False)
