import json
import os


def load_calibration_profile(profile_path):
    if not profile_path or not os.path.exists(profile_path):
        return None

    try:
        from core.analysis_logic import normalize_calibration_profile

        with open(profile_path, "r", encoding="utf-8") as profile_file:
            profile_data = json.load(profile_file)
        return normalize_calibration_profile(profile_data)
    except Exception as exc:
        print(f"Failed to load calibration profile ({profile_path}): {exc}")
        return None


def save_calibration_profile(calibration_profile, profile_path):
    if not profile_path:
        return

    output_dir = os.path.dirname(profile_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(profile_path, "w", encoding="utf-8") as profile_file:
        json.dump(calibration_profile, profile_file, indent=2)


def print_calibration_profile_summary(calibration_profile):
    print("Calibration profile active:")
    print(f"  Samples: {calibration_profile['sample_count']}")
    print(
        "  Head center (x/y): "
        f"{calibration_profile['head_center']['x']:.2f} / {calibration_profile['head_center']['y']:.2f}"
    )
    print(
        "  Engagement angle thresholds (x/y): "
        f"{calibration_profile['engagement_angle_threshold']['x']:.2f} / "
        f"{calibration_profile['engagement_angle_threshold']['y']:.2f}"
    )
    print(
        "  Head direction thresholds (x/y): "
        f"{calibration_profile['head_direction_threshold']['x']:.2f} / "
        f"{calibration_profile['head_direction_threshold']['y']:.2f}"
    )