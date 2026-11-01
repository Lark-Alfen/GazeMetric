"""The single OpenCV window both the calibration step and the analysis loop draw into."""

import cv2

WINDOW_NAME = "EngageVision - Full Analysis"


def window_was_closed():
    """True once the user has closed the window with its X button.

    Call it after cv2.waitKey: the window only exists after the first imshow, and closing is only noticed
    while events are being processed. Without this check the next imshow would silently recreate the window.
    """
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
