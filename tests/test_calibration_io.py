import json

from core.analysis_logic import get_default_calibration_profile
from core.calibration_io import load_calibration_profile, save_calibration_profile


def test_load_returns_none_without_a_path():
    assert load_calibration_profile(None) is None
    assert load_calibration_profile("") is None


def test_load_returns_none_for_missing_file(tmp_path):
    assert load_calibration_profile(str(tmp_path / "missing.json")) is None


def test_save_then_load_round_trips(tmp_path):
    profile = get_default_calibration_profile()
    profile["sample_count"] = 120
    profile["head_center"] = {"x": 4.5, "y": -2.5}
    profile["gaze_center"]["lx"] = 0.57
    path = tmp_path / "profile.json"

    save_calibration_profile(profile, str(path))

    assert load_calibration_profile(str(path)) == profile


def test_save_creates_missing_parent_folders(tmp_path):
    path = tmp_path / "nested" / "deeper" / "profile.json"
    save_calibration_profile(get_default_calibration_profile(), str(path))
    assert path.is_file()


def test_save_without_a_path_does_nothing(tmp_path):
    save_calibration_profile(get_default_calibration_profile(), None)
    save_calibration_profile(get_default_calibration_profile(), "")
    assert list(tmp_path.iterdir()) == []


def test_load_normalizes_out_of_range_values(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"head_center": {"x": 999, "y": 0}, "gaze_center": {"lx": -4}}), encoding="utf-8")

    loaded = load_calibration_profile(str(path))

    assert loaded["head_center"]["x"] == 30.0
    assert loaded["gaze_center"]["lx"] == 0.2
    assert loaded["sample_count"] == 0
