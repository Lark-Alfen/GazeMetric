"""Console banners and summary output for analysis runs."""

from collections import Counter

from core import APP_NAME

LINE = "=" * 70


def print_startup_banner(source_type="webcam", show_window=True):
    print(LINE)
    print(f"       {APP_NAME} - Full Engagement Analysis System")
    print(LINE)
    print("\nFeatures:")
    print("  - Head pose estimation (Looking Left/Right/Down/Straight)")
    print("  - Gaze direction tracking (Engaged/Not Engaged)")
    print("  - Real-time facial landmark detection")
    print("\nControls:")
    if show_window:
        print("  - Press 'Q' or 'q' in the video window, or close it, to stop and see the summary")
    print("  - Press Ctrl+C in this terminal to stop and see the summary")
    if source_type == "video":
        print("\nThe video file is analyzed from its first frame to the last.")
    else:
        print("\nThe camera is used until you stop the run.")
    print(LINE)
    print()


def print_stopping_banner():
    print("\n\n" + LINE)
    print("       GENERATING ENGAGEMENT ANALYSIS SUMMARY")
    print(LINE + "\n")


def print_interrupted_banner():
    print("\n\n" + LINE)
    print("   Interrupted by user (Ctrl+C). Generating summary...")
    print(LINE + "\n")


def print_summary(statements, engagement_states, frames_with_faces=None):
    print("\n" + LINE)
    print("                    HEAD POSITION ANALYSIS")
    print(LINE)

    if len(statements) > 0:
        print("\nPercentage of time spent in each head position:\n")
        for position, count in Counter(statements).most_common():
            percentage = (count / len(statements)) * 100
            print(f"  {position:20s}: {percentage:6.2f}%")
        print(f"\nFace observations analyzed: {len(statements)}")
        if frames_with_faces is not None:
            print(f"Frames with at least one face: {frames_with_faces}")
    else:
        print("\n  No head position data collected")

    print("\n" + LINE)
    print("                    ENGAGEMENT LEVEL ANALYSIS")
    print(LINE)

    if len(engagement_states) > 0:
        print("\nPercentage of time in each engagement state:\n")
        for engagement, count in Counter(engagement_states).most_common():
            percentage = (count / len(engagement_states)) * 100
            print(f"  {engagement:20s}: {percentage:6.2f}%")
        print(f"\nFace observations analyzed: {len(engagement_states)}")
        if frames_with_faces is not None:
            print(f"Frames with at least one face: {frames_with_faces}")
    else:
        print("\n  No engagement data collected")

    print("\n" + LINE)
    print("                    ANALYSIS COMPLETE")
    print(LINE)
    print()


def print_score_summary(engagement_scores):
    print("\n" + LINE)
    print("                    ENGAGEMENT SCORE SUMMARY")
    print(LINE)

    if len(engagement_scores) > 0:
        average_score = sum(engagement_scores) / len(engagement_scores)
        min_score = min(engagement_scores)
        max_score = max(engagement_scores)

        print(f"\nAverage score: {average_score:6.2f}/100")
        print(f"Minimum score: {min_score:6.2f}/100")
        print(f"Maximum score: {max_score:6.2f}/100")
        print(f"Score samples: {len(engagement_scores)}")
    else:
        print("\n  No engagement score data collected")

    print(LINE)
