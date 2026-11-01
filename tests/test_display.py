import cv2
import pytest

from core import display
from core.display import WINDOW_NAME, window_was_closed


@pytest.mark.parametrize("visible, closed", [(1.0, False), (0.0, True), (-1.0, True)])
def test_window_closed_follows_the_visibility_property(monkeypatch, visible, closed):
    seen = {}

    def fake_property(name, prop):
        seen["args"] = (name, prop)
        return visible

    monkeypatch.setattr(display.cv2, "getWindowProperty", fake_property)
    assert window_was_closed() is closed
    assert seen["args"] == (WINDOW_NAME, cv2.WND_PROP_VISIBLE)


def test_window_that_no_longer_exists_counts_as_closed(monkeypatch):
    def gone(*_args):
        raise cv2.error("window not found")

    monkeypatch.setattr(display.cv2, "getWindowProperty", gone)
    assert window_was_closed() is True
