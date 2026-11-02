"""Entry point: engagement analysis on a recorded video file."""

import argparse
import os
import sys

from core.cli import add_common_arguments, validate_common_arguments
from engagement_analysis_full import main


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run engagement analysis on a recorded video file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Path to input video file.",
    )
    add_common_arguments(parser)

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        parser.error(f"Input video file not found: {args.input_path}")
    validate_common_arguments(parser, args)

    args.source = "video"
    return args


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))