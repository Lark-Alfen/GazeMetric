import json
import os

import pytest

from core import calibration_io
from core.analysis_logic import get_default_calibration_profile
from core.calibration_io import (
    ProfileLoadError,
    load_calibration_profile,
    quarantine_unreadable_profile,
    save_calibration_profile,
)


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


# ---------------------------------------------------------------- unreadable profiles
@pytest.mark.parametrize(
    "content",
    ["", "{not json", '{"head_center": ', "[1, 2, 3]", '"just a string"', "42", "null"],
    ids=["empty", "invalid", "truncated", "list", "string", "number", "null"],
)
def test_load_raises_for_an_unreadable_profile(tmp_path, content):
    path = tmp_path / "profile.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProfileLoadError):
        load_calibration_profile(str(path))
    assert path.read_text(encoding="utf-8") == content  # loading never touches the file


def test_load_raises_for_a_file_that_is_not_utf8(tmp_path):
    path = tmp_path / "profile.json"
    path.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ProfileLoadError):
        load_calibration_profile(str(path))


def test_quarantine_moves_the_file_to_bad(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{broken", encoding="utf-8")

    bad_path = quarantine_unreadable_profile(str(path))

    assert bad_path == str(path) + ".bad"
    assert not path.exists()
    assert open(bad_path, encoding="utf-8").read() == "{broken"


def test_quarantine_never_overwrites_an_earlier_bad_copy(tmp_path):
    path = tmp_path / "profile.json"
    for attempt in ("first", "second", "third"):
        path.write_text(attempt, encoding="utf-8")
        quarantine_unreadable_profile(str(path))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["profile.json.bad", "profile.json.bad.1", "profile.json.bad.2"]
    assert (tmp_path / "profile.json.bad").read_text(encoding="utf-8") == "first"
    assert (tmp_path / "profile.json.bad.2").read_text(encoding="utf-8") == "third"


# ---------------------------------------------------------------- atomic saves
def test_save_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "profile.json"
    save_calibration_profile(get_default_calibration_profile(), str(path))
    assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]


def test_interrupted_save_keeps_the_original_profile(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    original = get_default_calibration_profile()
    original["sample_count"] = 77
    save_calibration_profile(original, str(path))

    def die_halfway(profile, handle, **_kwargs):
        handle.write('{"sample_count": 1, "trunc')
        raise OSError("disk full")

    monkeypatch.setattr(calibration_io.json, "dump", die_halfway)
    with pytest.raises(OSError):
        save_calibration_profile(get_default_calibration_profile(), str(path))

    monkeypatch.undo()
    assert load_calibration_profile(str(path))["sample_count"] == 77
    assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]


def test_failed_replace_keeps_the_original_profile(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    save_calibration_profile(get_default_calibration_profile(), str(path))
    before = path.read_text(encoding="utf-8")

    def refuse(*_args):
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        save_calibration_profile({"sample_count": 5}, str(path))

    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]
