"""Command-line options shared by the three entry points."""


def add_common_arguments(parser):
    parser.add_argument(
        "--output-video",
        dest="output_video_path",
        default=None,
        help="Optional path to save annotated output video.",
    )
    parser.add_argument(
        "--timeline-dir",
        dest="timeline_dir",
        default=None,
        help="Base directory for timeline/report artifacts; each run creates a unique session_<timestamp> folder (default base: runs).",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip baseline calibration and use default thresholds.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=10.0,
        help="Baseline calibration duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--calibration-profile",
        default=None,
        help="Optional JSON path to load/save calibration profile.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window (for batch runs and machines without a display). Prints progress instead.",
    )


def validate_common_arguments(parser, args):
    if args.calibration_seconds <= 0:
        parser.error("--calibration-seconds must be greater than 0.")
