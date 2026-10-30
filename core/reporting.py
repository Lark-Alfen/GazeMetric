"""Console banners and summary output for analysis runs."""

from collections import Counter

LINE = "=" * 70


def print_startup_banner():
    print(LINE)
    print("       EngageVision - Full Engagement Analysis System")
    print(LINE)
    print("\nFeatures:")
    print("  - Head pose estimation (Looking Left/Right/Down/Straight)")
    print("  - Gaze direction tracking (Engaged/Not Engaged)")
    print("  - Real-time facial landmark detection")
    print("\nControls:")
    print("  - Press 'Q' or 'q' in the video window to stop and see summary")
    print("  - DO NOT use Ctrl+C (it will skip the summary)")
    print("\nStarting camera...")
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


def print_summary(statements, engagement_states):
    print("\n" + LINE)
    print("                    HEAD POSITION ANALYSIS")
    print(LINE)

    if len(statements) > 0:
        print("\nPercentage of time spent in each head position:\n")
        for position, count in Counter(statements).most_common():
            percentage = (count / len(statements)) * 100
            print(f"  {position:20s}: {percentage:6.2f}%")
        print(f"\nTotal frames analyzed: {len(statements)}")
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
        print(f"\nTotal frames analyzed: {len(engagement_states)}")
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
