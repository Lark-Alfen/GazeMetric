"""Interactive baseline calibration phase run before analysis."""

import time

import cv2


def run_calibration_phase(
    cap,
    face_mesh,
    mp_face_mesh,
    mp_drawing,
    source_type,
    calibration_seconds,
    threshold,
    leye_3d,
    reye_3d,
    extract_landmark_arrays,
    compute_gaze_scores,
    build_camera_and_distortion,
    solve_pose_vectors,
    build_calibration_profile,
):
    calibration_samples = []
    calibration_scores = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
    calibration_seconds = max(1.0, float(calibration_seconds))
    min_samples = max(20, int(calibration_seconds * 8))
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

        processing_frame = cv2.flip(raw_frame, 1) if source_type == "webcam" else raw_frame

        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
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
                face_2d_one,
                face_3d_one,
                _nose_2d,
                _nose_3d,
            ) = extract_landmark_arrays(face_landmarks, img_w, img_h)

            calibration_scores = compute_gaze_scores(face_2d, calibration_scores, threshold)
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

        elapsed = time.perf_counter() - start_time
        remaining = max(0.0, calibration_seconds - elapsed)

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

        cv2.imshow("EngageVision - Full Analysis", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("s") or key == ord("S"):
            print("Calibration skipped by user input.")
            return None, False
        if key == ord("q") or key == ord("Q"):
            print("Stopping before analysis.")
            return None, True

        if elapsed >= calibration_seconds:
            break

    if not calibration_samples:
        return None, False

    if len(calibration_samples) < min_samples:
        print(
            "Calibration collected fewer samples than recommended "
            f"({len(calibration_samples)}/{min_samples}). Using blended profile."
        )

    return build_calibration_profile(calibration_samples, min_samples=min_samples), False