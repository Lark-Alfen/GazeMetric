"""Entry point: engagement analysis from a live webcam feed."""

import argparse
import sys

from core.cli import add_camera_arguments, add_common_arguments, validate_common_arguments
from engagement_analysis_full import main


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run engagement analysis in live webcam mode.",
    )
    add_camera_arguments(parser)
    add_common_arguments(parser)

    args = parser.parse_args()
    validate_common_arguments(parser, args)

    args.source = "webcam"
    args.input_path = None
    return args


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))