import argparse

from engagement_analysis_full import main


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run engagement analysis in live webcam mode.",
    )
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

    args = parser.parse_args()
    if args.calibration_seconds <= 0:
        parser.error("--calibration-seconds must be greater than 0.")

    args.source = "webcam"
    args.input_path = None
    return args


if __name__ == "__main__":
    main(parse_arguments())