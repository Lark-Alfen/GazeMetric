import cv2
import numpy as np
import pytest

from core import quality
from core.quality import _compute_lighting_score


def reference_lighting_score(frame, face_bbox):
    """The original implementation: convert the whole frame, then crop."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = quality._extract_face_roi(gray, face_bbox)
    return float(np.mean(roi)), float(np.std(roi))


@pytest.fixture
def frame():
    return np.random.default_rng(5).integers(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "bbox",
    [
        (50.4, 30.2, 200.9, 180.1),
        (0.0, 0.0, 319.0, 239.0),
        (-20.0, -10.0, 60.0, 50.0),
        (300.5, 220.5, 400.0, 300.0),
        (100.0, 100.0, 100.0, 100.0),
        None,
    ],
    ids=["inside", "whole-frame", "off-top-left", "off-bottom-right", "empty", "no-face"],
)
def test_lighting_score_is_unchanged_by_cropping_before_converting(frame, bbox):
    mean, std = reference_lighting_score(frame, bbox)
    # rebuild the expected score from the reference statistics with the same formula the module uses
    brightness = quality.clamp01(1.0 - (min(abs(mean - 135.0), 110.0) / 110.0))
    contrast = quality.clamp01(std / 55.0)
    expected = quality.clamp01((0.70 * brightness) + (0.30 * contrast))

    score, warnings = _compute_lighting_score(frame, bbox)

    assert score == pytest.approx(expected, abs=1e-9)
    assert ("Low light" in warnings) == (mean < 70.0)
    assert ("Overexposed frame" in warnings) == (mean > 205.0)
    assert ("Low contrast" in warnings) == (std < 20.0)


def test_only_the_face_region_is_converted_to_grayscale(frame, monkeypatch):
    converted = []
    real = cv2.cvtColor

    def spy(image, code, *args, **kwargs):
        converted.append(image.shape)
        return real(image, code, *args, **kwargs)

    monkeypatch.setattr(quality.cv2, "cvtColor", spy)
    _compute_lighting_score(frame, (50.0, 40.0, 150.0, 120.0))

    assert converted == [(80, 100, 3)]
    assert converted[0][0] * converted[0][1] < frame.shape[0] * frame.shape[1] / 5
