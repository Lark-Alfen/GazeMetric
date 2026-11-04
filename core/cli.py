"""Command-line options shared by the three entry points."""

from core.face_landmarker import DEFAULT_MODEL_PATH


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
        "--face-model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe face_landmarker.task model (default: Models/face_landmarker.task; get it with download_face_model.py).",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=2,
        help="Maximum number of faces to track at the same time (default: 2).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window (for batch runs and machines without a display). Prints progress instead.",
    )


def add_camera_arguments(parser):
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Which webcam to use, 0 is the first one (default: 0).",
    )


def validate_common_arguments(parser, args):
    if args.calibration_seconds <= 0:
        parser.error("--calibration-seconds must be greater than 0.")
    if args.max_faces < 1:
        parser.error("--max-faces must be at least 1.")
    if getattr(args, "camera_index", 0) < 0:
        parser.error("--camera-index must be 0 or greater.")
