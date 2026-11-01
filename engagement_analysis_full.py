"""Main engagement analysis pipeline shared by live and recorded modes."""

from __future__ import division

import argparse
import os
import sys
import time
import traceback
from collections import deque

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
MAX_CONSECUTIVE_FRAME_ERRORS = 30


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

    if args.source == "video":
        if not args.input_path:
            parser.error("--input is required when --source video.")
        if not os.path.exists(args.input_path):
            parser.error(f"Input video file not found: {args.input_path}")

    if args.calibration_seconds <= 0:
        parser.error("--calibration-seconds must be greater than 0.")

    return args


def main(args):
    """Run the analysis. Returns the process exit code: 0 on success, 1 if the run failed."""
    import cv2
    import mediapipe as mp

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
    from core.display import WINDOW_NAME, window_was_closed
    from core.quality import (
        compute_quality_metrics,
        get_default_quality_state,
    )
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
    from core.video_output import AnnotatedVideoWriter

    statements = []
    engagement_states = []
    timeline_writer = None
    engagement_score_values = []
    quality_states = {}
    last_scores = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
    score_window = deque(maxlen=SCORE_SMOOTHING_WINDOW)
    frame_index = 0
    timeline_output_dir = resolve_timeline_output_dir(args.timeline_dir)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = None
    output_writer = None
    session_started = False
    exit_code = 0
    failed_frames = 0
    consecutive_failures = 0

    try:
        source = 0 if args.source == "webcam" else args.input_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Unable to open source: {source}")
            return 1

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 20.0

        output_writer = None
        if args.output_video_path:
            # recorded input has a real frame rate; a webcam's is measured from the first frames
            output_writer = AnnotatedVideoWriter(
                args.output_video_path,
                fallback_fps=source_fps,
                fps=source_fps if args.source == "video" else None,
            )

        leye_3d, reye_3d = build_eye_reference_points()

        calibration_profile = get_default_calibration_profile()
        try:
            loaded_calibration_profile = load_calibration_profile(args.calibration_profile)
        except ProfileLoadError as load_error:
            bad_profile_path = quarantine_unreadable_profile(args.calibration_profile)
            print(f"Calibration profile is unusable: {load_error}")
            print(f"Moved it to {bad_profile_path} and continuing without it.")
            loaded_calibration_profile = None

        if loaded_calibration_profile is not None:
            calibration_profile = loaded_calibration_profile
            print(f"Loaded calibration profile from: {args.calibration_profile}")
        elif args.skip_calibration:
            print("Skipping calibration and using default thresholds.")
        else:
            generated_profile, calibration_status = run_calibration_phase(
                cap=cap,
                face_mesh=face_mesh,
                mp_face_mesh=mp_face_mesh,
                mp_drawing=mp_drawing,
                source_type=args.source,
                calibration_seconds=args.calibration_seconds,
                source_fps=source_fps,
                threshold=threshold,
                leye_3d=leye_3d,
                reye_3d=reye_3d,
                extract_landmark_arrays=extract_landmark_arrays,
                compute_gaze_scores=compute_gaze_scores,
                build_camera_and_distortion=build_camera_and_distortion,
                solve_pose_vectors=solve_pose_vectors,
                build_calibration_profile=build_calibration_profile,
            )
            if calibration_status == CALIBRATION_QUIT:
                return 0

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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if args.source == "video":
            print(f"Using video file: {args.input_path}")
        else:
            print("Using webcam source")

        if args.output_video_path:
            print(f"Annotated output will be saved to: {args.output_video_path}")
        print(f"Timeline outputs will be saved to: {timeline_output_dir}")
        print(f"Engagement score smoothing window: {SCORE_SMOOTHING_WINDOW} frames")
        print_calibration_profile_summary(calibration_profile)
        print_startup_banner()

        timeline_writer = TimelineWriter(timeline_output_dir)
        session_start_time = time.perf_counter()
        session_started = True

        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success:
                if args.source == "video":
                    print_stopping_banner()
                break

            try:
                frame_index += 1
                if args.source == "video":
                    timestamp_seconds = video_frame_timestamp(frame_index, source_fps, cap.get(cv2.CAP_PROP_POS_MSEC))
                else:
                    timestamp_seconds = round(time.perf_counter() - session_start_time, 3)

                processing_frame = cv2.flip(raw_frame, 1) if args.source == "webcam" else raw_frame

                rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True

                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                img_h, img_w, _ = frame.shape

                frame_has_record = False

                if results.multi_face_landmarks:
                    for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                        )

                        (
                            face_2d,
                            face_2d_head,
                        ) = extract_landmark_arrays(face_landmarks, img_w, img_h)

                        quality_state = quality_states.setdefault(face_index, get_default_quality_state())
                        quality_metrics = compute_quality_metrics(
                            frame=frame,
                            face_2d=face_2d,
                            img_w=img_w,
                            img_h=img_h,
                            quality_state=quality_state,
                        )

                        last_scores = compute_gaze_scores(face_2d, last_scores, threshold)
                        cam_matrix, dist_coeffs = build_camera_and_distortion(img_w, img_h)
                        pose_data = solve_pose_vectors(
                            face_2d_head,
                            cam_matrix,
                            dist_coeffs,
                            leye_3d,
                            reye_3d,
                        )
                        if pose_data is None:
                            timeline_writer.write(
                                create_pose_failed_timeline_record(
                                    frame_index=frame_index,
                                    timestamp_seconds=timestamp_seconds,
                                    face_index=face_index,
                                )
                            )
                            frame_has_record = True
                            continue

                        projection_data = build_projection_data(
                            pose_data=pose_data,
                            cam_matrix=cam_matrix,
                            dist_coeffs=dist_coeffs,
                            gaze_scores=last_scores,
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
                            cam_matrix,
                            dist_coeffs,
                        )

                        gaze_direction = classify_gaze_direction(
                            last_scores,
                            calibration_profile=calibration_profile,
                        )
                        engagement_text = classify_engagement(
                            gaze_direction,
                            pose_data["x_angle"],
                            pose_data["y_angle"],
                            calibration_profile=calibration_profile,
                        )
                        head_text = classify_head_direction(
                            pose_data["x_angle"],
                            pose_data["y_angle"],
                            calibration_profile=calibration_profile,
                        )

                        (
                            head_pose_score,
                            gaze_center_score,
                            engagement_score_raw,
                        ) = compute_composite_engagement_score(
                            gaze_scores=last_scores,
                            x_angle=pose_data["x_angle"],
                            y_angle=pose_data["y_angle"],
                            calibration_profile=calibration_profile,
                        )
                        score_window.append(engagement_score_raw)
                        engagement_score_smooth = round(sum(score_window) / len(score_window), 2)
                        engagement_score_values.append(engagement_score_smooth)

                        engagement_states.append(engagement_text)
                        statements.append(head_text)
                        timeline_writer.write(
                            create_face_timeline_record(
                                frame_index=frame_index,
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
                        frame_has_record = True

                        score_color = (0, 255, 0)
                        if engagement_score_smooth < 45:
                            score_color = (0, 0, 255)
                        elif engagement_score_smooth < 70:
                            score_color = (0, 255, 255)

                        cv2.putText(
                            frame,
                            engagement_text,
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            f"Score: {engagement_score_smooth:6.2f}/100",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            score_color,
                            2,
                            cv2.LINE_AA,
                        )
                        quality_score = quality_metrics["confidence_score"]
                        quality_label = quality_metrics["confidence_label"]
                        quality_color = (0, 255, 0)
                        if quality_score < 45:
                            quality_color = (0, 0, 255)
                        elif quality_score < 70:
                            quality_color = (0, 255, 255)

                        cv2.putText(
                            frame,
                            f"Quality: {quality_label} ({quality_score:5.1f}/100)",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            quality_color,
                            2,
                            cv2.LINE_AA,
                        )

                        warning_y = 150
                        for warning in quality_metrics["warnings"][:2]:
                            cv2.putText(
                                frame,
                                f"Warning: {warning}",
                                (10, warning_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 165, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            warning_y += 25

                        gaze_color = (0, 255, 0) if gaze_direction == "Forward" else (0, 255, 255)
                        cv2.putText(
                            frame,
                            f"Gaze: {gaze_direction}",
                            (10, warning_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            gaze_color,
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            frame,
                            head_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                if not frame_has_record:
                    timeline_writer.write(
                        create_no_face_timeline_record(
                            frame_index=frame_index,
                            timestamp_seconds=timestamp_seconds,
                        )
                    )

                if output_writer is not None:
                    output_writer.write(frame)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q") or key == ord("Q") or window_was_closed():
                    print_stopping_banner()
                    break
            except Exception as exc:
                failed_frames += 1
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print(f"\nFrame {frame_index} could not be processed and was skipped:")
                    traceback.print_exc()
                else:
                    print(f"Frame {frame_index} skipped again ({type(exc).__name__}: {exc})")
                if consecutive_failures >= MAX_CONSECUTIVE_FRAME_ERRORS:
                    raise RuntimeError(f"{consecutive_failures} frames in a row failed; stopping.") from exc
            else:
                consecutive_failures = 0

    except KeyboardInterrupt:
        print_interrupted_banner()
    except Exception:
        print("\nError occurred:")
        traceback.print_exc()
        print("Generating summary with collected data...\n")
        exit_code = 1
    finally:
        if failed_frames:
            print(f"{failed_frames} frame(s) were skipped because of errors.")
        if timeline_writer is not None:
            timeline_writer.close()
        if session_started:
            print_summary(statements, engagement_states)
            print_score_summary(engagement_score_values)
            try:
                timeline_records = read_timeline_csv(timeline_writer.csv_path)
                write_timeline_json(timeline_records, timeline_writer.json_path)
                print(f"Saved timeline CSV to: {timeline_writer.csv_path}")
                print(f"Saved timeline JSON to: {timeline_writer.json_path}")

                summary_json_path, summary_pdf_path, summary_pdf_error = export_session_reports(
                    timeline_records,
                    output_dir=timeline_output_dir,
                    source_type=args.source,
                    input_path=args.input_path,
                    output_video_path=args.output_video_path,
                )
                print(f"Saved session summary JSON to: {summary_json_path}")
                if summary_pdf_path is not None:
                    print(f"Saved session summary PDF to: {summary_pdf_path}")
                else:
                    print(f"Failed to export session summary PDF: {summary_pdf_error}")
            except Exception as timeline_error:
                print(f"Failed to export timeline files: {timeline_error}")

        try:
            if cap is not None:
                cap.release()
            if output_writer is not None and output_writer.close():
                print(f"Saved annotated video to: {args.output_video_path}")
        finally:
            cv2.destroyAllWindows()
            face_mesh.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
