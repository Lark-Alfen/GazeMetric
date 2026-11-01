"""Load and save calibration profiles as JSON."""

import json
import os


class ProfileLoadError(Exception):
    """The profile file exists but cannot be used."""


def load_calibration_profile(profile_path):
    """Load a profile. Returns None when no path was given or the file does not exist yet.

    Raises ProfileLoadError when the file exists but is not a readable profile, so a damaged file is never
    mistaken for a missing one.
    """
    if not profile_path or not os.path.exists(profile_path):
        return None

    from core.analysis_logic import normalize_calibration_profile

    try:
        with open(profile_path, "r", encoding="utf-8") as profile_file:
            profile_data = json.load(profile_file)
    except (OSError, ValueError) as exc:
        raise ProfileLoadError(f"{profile_path} could not be read: {exc}") from exc

    if not isinstance(profile_data, dict):
        raise ProfileLoadError(f"{profile_path} does not contain a calibration profile")
    return normalize_calibration_profile(profile_data)


def quarantine_unreadable_profile(profile_path):
    """Move an unreadable profile aside to <path>.bad (or .bad.1, .bad.2 ...) and return the new path."""
    bad_path = profile_path + ".bad"
    suffix = 0
    while os.path.exists(bad_path):
        suffix += 1
        bad_path = f"{profile_path}.bad.{suffix}"
    os.replace(profile_path, bad_path)
    return bad_path


def save_calibration_profile(calibration_profile, profile_path):
    """Write the profile through a temporary file so an interrupted save cannot leave a truncated one."""
    if not profile_path:
        return

    output_dir = os.path.dirname(profile_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_path = profile_path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as profile_file:
            json.dump(calibration_profile, profile_file, indent=2)
        os.replace(temp_path, profile_path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


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