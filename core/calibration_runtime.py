"""Interactive baseline calibration phase run before analysis."""

import time

import cv2

from core.display import WINDOW_NAME, window_was_closed

# Outcomes of run_calibration_phase
CALIBRATION_COMPLETED = "completed"
CALIBRATION_SKIPPED = "skipped"
CALIBRATION_INSUFFICIENT = "insufficient"
CALIBRATION_QUIT = "quit"

# Share of the frames in the calibration window that should produce a usable sample (video input).
VIDEO_MIN_SAMPLE_SHARE = 0.4


def get_calibration_plan(source_type, calibration_seconds, source_fps):
    """How long calibration runs and how many samples it should collect.

    A webcam is measured in wall-clock seconds. A recorded video is measured in frames of the video, so the
    result does not depend on how fast the machine happens to process it. Returns
    (calibration_seconds, frame_limit or None, min_samples).
    """
    calibration_seconds = max(1.0, float(calibration_seconds))
    if source_type != "video":
        return calibration_seconds, None, max(20, int(calibration_seconds * 8))

    frame_limit = max(1, int(calibration_seconds * source_fps))
    min_samples = min(frame_limit, max(20, int(frame_limit * VIDEO_MIN_SAMPLE_SHARE)))
    return calibration_seconds, frame_limit, min_samples


def run_calibration_phase(
    cap,
    face_mesh,
    mp_face_mesh,
    mp_drawing,
    source_type,
    calibration_seconds,
    source_fps,
    threshold,
    leye_3d,
    reye_3d,
    extract_landmark_arrays,
    compute_gaze_scores,
    build_camera_and_distortion,
    solve_pose_vectors,
    build_calibration_profile,
    show_window=True,
):
    calibration_samples = []
    calibration_scores = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
    calibration_seconds, frame_limit, min_samples = get_calibration_plan(source_type, calibration_seconds, source_fps)
    frames_read = 0
    camera_model_size = None
    start_time = time.perf_counter()

    print("\n" + "=" * 70)
    print("Calibration step started")
    print(f"Keep your face centered and look straight for {calibration_seconds:.1f} seconds.")
    print("Press 'S' to skip calibration or 'Q' to stop the run.")
    print("=" * 70 + "\n")

    while cap.isOpened():
        success, raw_frame = cap.read()
        if not success:
            break
        frames_read += 1

        processing_frame = cv2.flip(raw_frame, 1) if source_type == "webcam" else raw_frame

        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        frame = processing_frame
        img_h, img_w, _ = frame.shape

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
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

            calibration_scores = compute_gaze_scores(face_2d, calibration_scores, threshold)
            if (img_w, img_h) != camera_model_size:
                cam_matrix, dist_coeffs = build_camera_and_distortion(img_w, img_h)
                camera_model_size = (img_w, img_h)
            pose_data = solve_pose_vectors(
                face_2d_head,
                cam_matrix,
                dist_coeffs,
                leye_3d,
                reye_3d,
            )
            if pose_data is not None:
                calibration_samples.append(
                    {
                        "lx": calibration_scores["lx"],
                        "ly": calibration_scores["ly"],
                        "rx": calibration_scores["rx"],
                        "ry": calibration_scores["ry"],
                        "x_angle": pose_data["x_angle"],
                        "y_angle": pose_data["y_angle"],
                    }
                )

        if frame_limit is None:
            elapsed = time.perf_counter() - start_time
            remaining = max(0.0, calibration_seconds - elapsed)
            finished = elapsed >= calibration_seconds
        else:
            remaining = max(0.0, (frame_limit - frames_read) / source_fps)
            finished = frames_read >= frame_limit

        cv2.putText(
            frame,
            "Calibration: look straight at the camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Time left: {remaining:4.1f}s   Samples: {len(calibration_samples)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press S to skip | Press Q to stop",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        if show_window:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(5) & 0xFF
        else:
            key = -1
        if key == ord("s") or key == ord("S"):
            print("Calibration skipped by user input.")
            return None, CALIBRATION_SKIPPED
        if key == ord("q") or key == ord("Q"):
            print("Stopping before analysis.")
            return None, CALIBRATION_QUIT
        if show_window and window_was_closed():
            print("Window closed. Stopping before analysis.")
            return None, CALIBRATION_QUIT

        if finished:
            break

    if not calibration_samples:
        return None, CALIBRATION_INSUFFICIENT

    if len(calibration_samples) < min_samples:
        print(
            "Calibration collected fewer samples than recommended "
            f"({len(calibration_samples)}/{min_samples}). Using blended profile."
        )

    profile = build_calibration_profile(calibration_samples, min_samples=min_samples)
    if profile["sample_count"] == 0:
        # rejected: build_calibration_profile already said why and handed back the defaults
        return None, CALIBRATION_INSUFFICIENT
    return profile, CALIBRATION_COMPLETED