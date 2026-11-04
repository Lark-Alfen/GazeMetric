"""Download the MediaPipe Face Landmarker model that GazeMetric needs (about 4 MB, once)."""

import argparse
import sys

from core.face_landmarker import DEFAULT_MODEL_PATH, MODEL_URL, download_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=DEFAULT_MODEL_PATH,
        help="Where to save the model (default: Models/face_landmarker.task in the project folder).",
    )
    args = parser.parse_args()

    print(f"Downloading {MODEL_URL}")
    try:
        path = download_model(args.output)
    except Exception as error:
        print(f"Download failed: {error}")
        return 1
    print(f"Saved the model to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
