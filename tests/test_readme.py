import re

import pytest

import engagement_analysis_full
import engagement_analysis_live
import engagement_analysis_recorded
from core.timeline import TIMELINE_FIELDNAMES

with open("README.md", encoding="utf-8") as handle:
    README = handle.read()


def option_strings(module, monkeypatch, tmp_path):
    """Every --option a script's parser accepts, found through its --help text."""
    import io
    import sys
    from contextlib import redirect_stdout

    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    out = io.StringIO()
    with redirect_stdout(out), pytest.raises(SystemExit):
        module.parse_arguments()
    return set(re.findall(r"--[a-z][a-z-]+", out.getvalue())) - {"--help"}


@pytest.mark.parametrize("module", [engagement_analysis_full, engagement_analysis_live, engagement_analysis_recorded])
def test_every_command_line_option_is_in_the_readme(module, monkeypatch, tmp_path):
    options = option_strings(module, monkeypatch, tmp_path)
    assert options, "no options found"
    missing = [option for option in sorted(options) if f"`{option}" not in README]
    assert missing == []


@pytest.mark.parametrize("column", TIMELINE_FIELDNAMES)
def test_every_timeline_column_is_documented(column):
    assert f"| `{column}` |" in README


def test_the_documented_columns_are_only_real_columns():
    section = README.split("### `timeline.csv` columns")[1].split("\n## ")[0]
    documented = re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.MULTILINE)
    assert documented == TIMELINE_FIELDNAMES  # same columns, same order


def test_the_score_weights_in_the_readme_match_the_code():
    assert "0.55 x head pose score + 0.45 x gaze centre score" in README
    assert "0.45 x visibility + 0.30 x lighting + 0.25 x stability" in README


def test_the_python_versions_match_the_ci_matrix():
    with open(".github/workflows/ci.yml", encoding="utf-8") as handle:
        ci = handle.read()
    versions = re.search(r"python-version: \[(.*?)\]", ci).group(1).replace('"', "").split(", ")
    assert len(versions) >= 2
    expected = ", ".join(versions[:-1]) + " or " + versions[-1]
    assert f"Python {expected}" in README


def test_the_model_download_step_is_documented():
    assert "python download_face_model.py" in README
    assert "Models/face_landmarker.task" in README


def test_the_obsolete_encoding_workaround_is_gone():
    assert "requirements_utf8" not in README
    assert "-Encoding Unicode" not in README
