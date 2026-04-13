from __future__ import division

import argparse
import os
import time
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
        load_calibration_profile,
        print_calibration_profile_summary,
        save_calibration_profile,
    )
    from core.calibration_runtime import run_calibration_phase
    from core.reporting import (
        print_interrupted_banner,
        print_score_summary,
        print_startup_banner,
        print_stopping_banner,
        print_summary,
    )
    from core.quality import (
        compute_quality_metrics,
        get_default_quality_state,
    )
    from core.report_exports import export_session_reports
    from core.scoring import compute_composite_engagement_score
    from core.timeline import (
        create_face_timeline_record,
        create_no_face_timeline_record,
        export_timeline_records,
        resolve_timeline_output_dir,
    )

    statements = []
    engagement_states = []
    timeline_records = []
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

    source = 0 if args.source == "webcam" else args.input_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Unable to open source: {source}")
        face_mesh.close()
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 20.0

    video_writer = None

    leye_3d, reye_3d = build_eye_reference_points()

    calibration_profile = get_default_calibration_profile()
    loaded_calibration_profile = load_calibration_profile(args.calibration_profile)

    if loaded_calibration_profile is not None:
        calibration_profile = loaded_calibration_profile
        print(f"Loaded calibration profile from: {args.calibration_profile}")
    elif args.skip_calibration:
        print("Skipping calibration and using default thresholds.")
    else:
        generated_profile, should_stop = run_calibration_phase(
            cap=cap,
            face_mesh=face_mesh,
            mp_face_mesh=mp_face_mesh,
            mp_drawing=mp_drawing,
            source_type=args.source,
            calibration_seconds=args.calibration_seconds,
            threshold=threshold,
            leye_3d=leye_3d,
            reye_3d=reye_3d,
            extract_landmark_arrays=extract_landmark_arrays,
            compute_gaze_scores=compute_gaze_scores,
            build_camera_and_distortion=build_camera_and_distortion,
            solve_pose_vectors=solve_pose_vectors,
            build_calibration_profile=build_calibration_profile,
        )
        if should_stop:
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            return

        if generated_profile is not None:
            calibration_profile = generated_profile
            if args.calibration_profile:
                save_calibration_profile(calibration_profile, args.calibration_profile)
                print(f"Saved calibration profile to: {args.calibration_profile}")
        else:
            print("Calibration could not collect enough data. Using default thresholds.")

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

    session_start_time = time.perf_counter()

    try:
        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success:
                if args.source == "video":
                    print_stopping_banner()
                break

            frame_index += 1
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
                        face_2d_one,
                        face_3d_one,
                        nose_2d,
                        nose_3d,
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
                    cam_matrix, dist_coeffs, dist_matrix = build_camera_and_distortion(img_w, img_h)
                    pose_data = solve_pose_vectors(
                        face_2d_head,
                        face_2d_one,
                        face_3d_one,
                        cam_matrix,
                        dist_coeffs,
                        dist_matrix,
                        leye_3d,
                        reye_3d,
                    )
                    if pose_data is None:
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
                        nose_2d,
                        nose_3d,
                        pose_data["rot_vec"],
                        pose_data["trans_vec"],
                        cam_matrix,
                        dist_matrix,
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
                    timeline_records.append(
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
                timeline_records.append(
                    create_no_face_timeline_record(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                    )
                )

            if args.output_video_path:
                if video_writer is None:
                    output_dir = os.path.dirname(args.output_video_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)

                    frame_h, frame_w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        args.output_video_path,
                        fourcc,
                        source_fps,
                        (frame_w, frame_h),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Unable to create output video: {args.output_video_path}")

                video_writer.write(frame)

            cv2.imshow("EngageVision - Full Analysis", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print_stopping_banner()
                break

    except KeyboardInterrupt:
        print_interrupted_banner()
    except Exception as exc:
        print(f"\nError occurred: {exc}")
        print("Generating summary with collected data...\n")
    finally:
        print_summary(statements, engagement_states)
        print_score_summary(engagement_score_values)
        try:
            timeline_csv_path, timeline_json_path = export_timeline_records(timeline_records, timeline_output_dir)
            print(f"Saved timeline CSV to: {timeline_csv_path}")
            print(f"Saved timeline JSON to: {timeline_json_path}")

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

        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved annotated video to: {args.output_video_path}")
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main(parse_arguments())
