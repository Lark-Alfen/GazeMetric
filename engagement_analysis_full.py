"""Main engagement analysis pipeline shared by live and recorded modes."""

from __future__ import division

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import cv2

from core.analysis_logic import (
    build_calibration_profile,
    build_camera_and_distortion,
    build_eye_reference_points,
    build_projection_data,
    classify_engagement,
    classify_gaze_direction,
    classify_head_direction,
    compute_gaze_scores,
    draw_nose_direction,
    draw_pose_overlay,
    extract_landmark_arrays,
    get_default_calibration_profile,
    solve_pose_vectors,
)
from core.calibration_io import (
    ProfileLoadError,
    load_calibration_profile,
    print_calibration_profile_summary,
    quarantine_unreadable_profile,
    save_calibration_profile,
)
from core.calibration_runtime import CALIBRATION_QUIT, CALIBRATION_SKIPPED, run_calibration_phase
from core.capture import open_capture, open_failure_message
from core.cli import add_camera_arguments, add_common_arguments, validate_common_arguments
from core.display import WINDOW_NAME, window_was_closed
from core.face_landmarker import FaceLandmarkDetector, missing_model_message
from core.overlay import build_hud_lines, draw_face_contours, draw_face_hud
from core.quality import compute_quality_metrics
from core.report_exports import export_session_reports
from core.reporting import (
    print_interrupted_banner,
    print_score_summary,
    print_startup_banner,
    print_stopping_banner,
    print_summary,
)
from core.scoring import compute_composite_engagement_score
from core.timeline import (
    TimelineWriter,
    create_face_timeline_record,
    create_no_face_timeline_record,
    create_pose_failed_timeline_record,
    read_timeline_csv,
    resolve_timeline_output_dir,
    video_frame_timestamp,
    write_timeline_json,
)
from core.tracking import FaceTracker
from core.video_output import AnnotatedVideoWriter

############## PARAMETERS #######################################################

# Set these values to show/hide certain vectors of the estimation
draw_gaze = True
draw_full_axis = True
draw_headpose = False

# Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
x_score_multiplier = 4
y_score_multiplier = 4

# Threshold of how close scores should be to average between frames
threshold = .3

#################################################################################

SCORE_SMOOTHING_WINDOW = 15
PROGRESS_EVERY_FRAMES = 100
MAX_CONSECUTIVE_FRAME_ERRORS = 30


def format_progress(frames_done, total_frames):
    """Progress line for --no-display; the total is only known for recorded video."""
    if total_frames and total_frames > 0:
        return f"Processed {frames_done}/{total_frames} frames ({100.0 * frames_done / total_frames:.0f}%)"
    return f"Processed {frames_done} frames"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run engagement analysis on webcam or an input video file.",
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "video"],
        default="webcam",
        help="Input source type. Use 'webcam' for camera or 'video' for a file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to input video file (required when --source video).",
    )
    add_camera_arguments(parser)
    add_common_arguments(parser)

    args = parser.parse_args()

    if args.source == "video":
        if not args.input_path:
            parser.error("--input is required when --source video.")
        if not os.path.exists(args.input_path):
            parser.error(f"Input video file not found: {args.input_path}")

    validate_common_arguments(parser, args)

    return args


@dataclass
class FaceModels:
    """The MediaPipe objects and the fixed eye model that every frame is analysed with."""

    detector: object
    leye_3d: object
    reye_3d: object


@dataclass
class SessionState:
    """Everything that changes while the frames of one session are analysed."""

    args: argparse.Namespace
    cap: object
    models: FaceModels
    calibration_profile: dict
    source_fps: float
    total_frames: int
    timeline_writer: TimelineWriter
    output_writer: object
    session_start_time: float
    face_tracker: FaceTracker = field(default_factory=lambda: FaceTracker(SCORE_SMOOTHING_WINDOW))
    statements: list = field(default_factory=list)
    engagement_states: list = field(default_factory=list)
    engagement_score_values: list = field(default_factory=list)
    frame_index: int = 0
    frames_with_faces: int = 0
    failed_frames: int = 0
    camera_model_size: tuple = None
    cam_matrix: object = None
    dist_coeffs: object = None


def create_face_models(model_path, max_faces):
    detector = FaceLandmarkDetector(model_path, max_faces=max_faces)
    leye_3d, reye_3d = build_eye_reference_points()
    return FaceModels(detector, leye_3d, reye_3d)


def read_source_properties(cap, args):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.source == "video" else 0
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 20.0
    return source_fps, total_frames


def create_output_writer(args, source_fps):
    if not args.output_video_path:
        return None
    # recorded input has a real frame rate; a webcam's is measured from the first frames
    return AnnotatedVideoWriter(
        args.output_video_path,
        fallback_fps=source_fps,
        fps=source_fps if args.source == "video" else None,
    )


def load_or_run_calibration(args, cap, models, source_fps):
    """Return (calibration_profile, should_stop): a saved profile, a fresh calibration or the defaults."""
    try:
        loaded_profile = load_calibration_profile(args.calibration_profile)
    except ProfileLoadError as load_error:
        bad_profile_path = quarantine_unreadable_profile(args.calibration_profile)
        print(f"Calibration profile is unusable: {load_error}")
        print(f"Moved it to {bad_profile_path} and continuing without it.")
        loaded_profile = None

    if loaded_profile is not None:
        print(f"Loaded calibration profile from: {args.calibration_profile}")
        return loaded_profile, False

    if args.skip_calibration:
        print("Skipping calibration and using default thresholds.")
        return get_default_calibration_profile(), False

    generated_profile, calibration_status = run_calibration_phase(
        cap=cap,
        detector=models.detector,
        source_type=args.source,
        calibration_seconds=args.calibration_seconds,
        source_fps=source_fps,
        threshold=threshold,
        leye_3d=models.leye_3d,
        reye_3d=models.reye_3d,
        extract_landmark_arrays=extract_landmark_arrays,
        compute_gaze_scores=compute_gaze_scores,
        build_camera_and_distortion=build_camera_and_distortion,
        solve_pose_vectors=solve_pose_vectors,
        build_calibration_profile=build_calibration_profile,
        show_window=not args.no_display,
    )
    if calibration_status == CALIBRATION_QUIT:
        return None, True
    models.detector.restart()  # the analysis starts fresh, with its own timestamps

    calibration_profile = get_default_calibration_profile()
    if generated_profile is not None:
        calibration_profile = generated_profile
        if args.calibration_profile:
            save_calibration_profile(calibration_profile, args.calibration_profile)
            print(f"Saved calibration profile to: {args.calibration_profile}")
    elif calibration_status == CALIBRATION_SKIPPED:
        print("Using default thresholds.")
    else:
        print("No usable calibration. Using default thresholds.")

    if args.source == "video":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # analyse the whole clip, including the frames used for calibration
    return calibration_profile, False


def print_session_info(args, timeline_output_dir, calibration_profile):
    if args.source == "video":
        print(f"Using video file: {args.input_path}")
    else:
        print("Using webcam source")

    if args.output_video_path:
        print(f"Annotated output will be saved to: {args.output_video_path}")
    print(f"Timeline outputs will be saved to: {timeline_output_dir}")
    print(f"Engagement score smoothing window: {SCORE_SMOOTHING_WINDOW} frames")
    print_calibration_profile_summary(calibration_profile)


def frame_timestamp(state):
    if state.args.source == "video":
        return video_frame_timestamp(state.frame_index, state.source_fps, state.cap.get(cv2.CAP_PROP_POS_MSEC))
    return round(time.perf_counter() - state.session_start_time, 3)


def analyse_face(state, frame, face_2d, face_2d_head, track, timestamp_seconds):
    """Analyse, record and draw one face; every face writes exactly one timeline record."""
    models = state.models
    face_index = track.track_id
    draw_face_contours(frame, face_2d, models.detector.contour_connections)

    img_h, img_w, _ = frame.shape
    quality_metrics = compute_quality_metrics(
        frame=frame,
        face_2d=face_2d,
        img_w=img_w,
        img_h=img_h,
        quality_state=track.quality_state,
    )

    track.last_scores = compute_gaze_scores(face_2d, track.last_scores, threshold)
    pose_data = solve_pose_vectors(
        face_2d_head,
        state.cam_matrix,
        state.dist_coeffs,
        models.leye_3d,
        models.reye_3d,
    )
    if pose_data is None:
        state.timeline_writer.write(
            create_pose_failed_timeline_record(
                frame_index=state.frame_index,
                timestamp_seconds=timestamp_seconds,
                face_index=face_index,
            )
        )
        return

    projection_data = build_projection_data(
        pose_data=pose_data,
        cam_matrix=state.cam_matrix,
        dist_coeffs=state.dist_coeffs,
        gaze_scores=track.last_scores,
        x_score_multiplier=x_score_multiplier,
        y_score_multiplier=y_score_multiplier,
    )
    draw_pose_overlay(
        frame,
        face_2d_head,
        projection_data,
        draw_headpose=draw_headpose,
        draw_gaze=draw_gaze,
        draw_full_axis=draw_full_axis,
    )
    draw_nose_direction(
        frame,
        pose_data["rot_vec"],
        pose_data["trans_vec"],
        state.cam_matrix,
        state.dist_coeffs,
    )

    profile = state.calibration_profile
    gaze_direction = classify_gaze_direction(track.last_scores, calibration_profile=profile)
    engagement_text = classify_engagement(
        gaze_direction,
        pose_data["x_angle"],
        pose_data["y_angle"],
        calibration_profile=profile,
    )
    head_text = classify_head_direction(
        pose_data["x_angle"],
        pose_data["y_angle"],
        calibration_profile=profile,
    )

    (
        head_pose_score,
        gaze_center_score,
        engagement_score_raw,
    ) = compute_composite_engagement_score(
        gaze_scores=track.last_scores,
        x_angle=pose_data["x_angle"],
        y_angle=pose_data["y_angle"],
        calibration_profile=profile,
    )
    track.score_window.append(engagement_score_raw)
    engagement_score_smooth = round(sum(track.score_window) / len(track.score_window), 2)
    state.engagement_score_values.append(engagement_score_smooth)

    state.engagement_states.append(engagement_text)
    state.statements.append(head_text)
    state.timeline_writer.write(
        create_face_timeline_record(
            frame_index=state.frame_index,
            timestamp_seconds=timestamp_seconds,
            face_index=face_index,
            x_angle=pose_data["x_angle"],
            y_angle=pose_data["y_angle"],
            gaze_direction=gaze_direction,
            engagement_label=engagement_text,
            head_direction=head_text,
            head_pose_score=head_pose_score,
            gaze_center_score=gaze_center_score,
            engagement_score_raw=engagement_score_raw,
            engagement_score_smooth=engagement_score_smooth,
            face_visibility_score=quality_metrics["face_visibility_score"],
            lighting_score=quality_metrics["lighting_score"],
            landmark_stability_score=quality_metrics["landmark_stability_score"],
            quality_confidence_score=quality_metrics["confidence_score"],
            quality_confidence_label=quality_metrics["confidence_label"],
            quality_warnings=quality_metrics["warnings"],
        )
    )

    draw_face_hud(
        frame,
        face_2d,
        build_hud_lines(
            face_id=face_index,
            engagement_text=engagement_text,
            head_text=head_text,
            engagement_score=engagement_score_smooth,
            quality_label=quality_metrics["confidence_label"],
            quality_score=quality_metrics["confidence_score"],
            quality_warnings=quality_metrics["warnings"],
            gaze_direction=gaze_direction,
        ),
    )


def process_frame(state, raw_frame):
    """Analyse one camera or video frame, write its timeline records and return the annotated frame."""
    state.frame_index += 1
    timestamp_seconds = frame_timestamp(state)

    processing_frame = cv2.flip(raw_frame, 1) if state.args.source == "webcam" else raw_frame

    rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
    detected_faces = state.models.detector.detect(rgb_frame, timestamp_seconds * 1000)

    # the frame is already BGR; it is only drawn on, so no copy or conversion back is needed
    frame = processing_frame
    img_h, img_w, _ = frame.shape
    if (img_w, img_h) != state.camera_model_size:
        state.cam_matrix, state.dist_coeffs = build_camera_and_distortion(img_w, img_h)
        state.camera_model_size = (img_w, img_h)

    landmark_arrays = [extract_landmark_arrays(face_landmarks, img_w, img_h) for face_landmarks in detected_faces]
    # called every frame, also without faces, so that tracks of people who left expire
    face_tracks = state.face_tracker.match(
        state.frame_index,
        [(face_2d[:, 0].mean() / img_w, face_2d[:, 1].mean() / img_h) for face_2d, _ in landmark_arrays],
    )

    for (face_2d, face_2d_head), track in zip(landmark_arrays, face_tracks):
        analyse_face(state, frame, face_2d, face_2d_head, track, timestamp_seconds)

    if detected_faces:
        state.frames_with_faces += 1
    else:
        state.timeline_writer.write(
            create_no_face_timeline_record(
                frame_index=state.frame_index,
                timestamp_seconds=timestamp_seconds,
            )
        )
    return frame


def show_frame(state, frame):
    """Show the frame (or print progress without a window). Returns True when the user asked to stop."""
    if state.args.no_display:
        if state.frame_index % PROGRESS_EVERY_FRAMES == 0:
            print(format_progress(state.frame_index, state.total_frames))
        return False

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q") or key == ord("Q") or window_was_closed():
        print_stopping_banner()
        return True
    return False


def analyse_stream(state):
    """Read frames until the source ends or the user stops. A bad frame is skipped, too many in a row stop the run."""
    consecutive_failures = 0
    while state.cap.isOpened():
        success, raw_frame = state.cap.read()
        if not success:
            if state.args.source == "video":
                print_stopping_banner()
            break

        try:
            frame = process_frame(state, raw_frame)
            if state.output_writer is not None:
                state.output_writer.write(frame)
            if show_frame(state, frame):
                break
        except Exception as exc:
            state.failed_frames += 1
            consecutive_failures += 1
            if consecutive_failures == 1:
                print(f"\nFrame {state.frame_index} could not be processed and was skipped:")
                traceback.print_exc()
            else:
                print(f"Frame {state.frame_index} skipped again ({type(exc).__name__}: {exc})")
            if consecutive_failures >= MAX_CONSECUTIVE_FRAME_ERRORS:
                raise RuntimeError(f"{consecutive_failures} frames in a row failed; stopping.") from exc
        else:
            consecutive_failures = 0


def export_session(state, timeline_output_dir):
    """Print the console summary and write timeline.json and the summary reports."""
    print_summary(state.statements, state.engagement_states, state.frames_with_faces)
    print_score_summary(state.engagement_score_values)
    try:
        timeline_writer = state.timeline_writer
        timeline_records = read_timeline_csv(timeline_writer.csv_path)
        write_timeline_json(timeline_records, timeline_writer.json_path)
        print(f"Saved timeline CSV to: {timeline_writer.csv_path}")
        print(f"Saved timeline JSON to: {timeline_writer.json_path}")

        summary_json_path, summary_pdf_path, summary_pdf_error = export_session_reports(
            timeline_records,
            output_dir=timeline_output_dir,
            source_type=state.args.source,
            input_path=state.args.input_path,
            output_video_path=state.args.output_video_path,
        )
        print(f"Saved session summary JSON to: {summary_json_path}")
        if summary_pdf_path is not None:
            print(f"Saved session summary PDF to: {summary_pdf_path}")
        else:
            print(f"Failed to export session summary PDF: {summary_pdf_error}")
    except Exception as timeline_error:
        print(f"Failed to export timeline files: {timeline_error}")


def release_resources(args, cap, output_writer, models):
    try:
        if cap is not None:
            cap.release()
        if output_writer is not None and output_writer.close():
            print(f"Saved annotated video to: {args.output_video_path}")
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        models.detector.close()


def main(args):
    """Run the analysis. Returns the process exit code: 0 on success, 1 if the run failed."""
    if not os.path.isfile(args.face_model):
        print(missing_model_message(args.face_model))
        return 1

    timeline_output_dir = resolve_timeline_output_dir(args.timeline_dir)
    models = create_face_models(args.face_model, args.max_faces)

    cap = None
    output_writer = None
    timeline_writer = None
    state = None
    exit_code = 0

    try:
        camera_index = args.camera_index if args.source == "webcam" else 0
        cap = open_capture(args.source, args.input_path, camera_index)
        if not cap.isOpened():
            print(open_failure_message(args.source, args.input_path, camera_index))
            return 1

        source_fps, total_frames = read_source_properties(cap, args)
        output_writer = create_output_writer(args, source_fps)

        print_startup_banner(args.source, show_window=not args.no_display)
        calibration_profile, should_stop = load_or_run_calibration(args, cap, models, source_fps)
        if should_stop:
            return 0
        print_session_info(args, timeline_output_dir, calibration_profile)

        timeline_writer = TimelineWriter(timeline_output_dir)
        state = SessionState(
            args=args,
            cap=cap,
            models=models,
            calibration_profile=calibration_profile,
            source_fps=source_fps,
            total_frames=total_frames,
            timeline_writer=timeline_writer,
            output_writer=output_writer,
            session_start_time=time.perf_counter(),
        )
        analyse_stream(state)

    except KeyboardInterrupt:
        print_interrupted_banner()
    except Exception:
        print("\nError occurred:")
        traceback.print_exc()
        print("Generating summary with collected data...\n")
        exit_code = 1
    finally:
        if state is not None and state.failed_frames:
            print(f"{state.failed_frames} frame(s) were skipped because of errors.")
        if timeline_writer is not None:
            timeline_writer.close()
        if state is not None:
            export_session(state, timeline_output_dir)
        release_resources(args, cap, output_writer, models)

    return exit_code


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
