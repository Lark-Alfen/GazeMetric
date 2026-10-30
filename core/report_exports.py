"""Build and export end-of-session summary reports."""

import json
import os
from collections import Counter
from datetime import datetime, timezone


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def _distribution_from_values(values):
    filtered = [str(value) for value in values if value is not None and str(value).strip() != ""]
    counter = Counter(filtered)
    total = sum(counter.values())
    percentages = {}
    if total > 0:
        for key, count in counter.items():
            percentages[key] = round((count / float(total)) * 100.0, 2)
    return {
        "total": total,
        "counts": dict(counter),
        "percentages": percentages,
    }


def _extract_warning_counts(face_records):
    warning_counter = Counter()
    for record in face_records:
        warning_text = str(record.get("quality_warnings", "")).strip()
        if not warning_text or warning_text == "No Face":
            continue
        for token in warning_text.split(";"):
            token = token.strip()
            if token:
                warning_counter[token] += 1
    return warning_counter


def build_summary_payload(timeline_records, source_type, input_path=None, output_video_path=None):
    frame_ids = [record.get("frame_index") for record in timeline_records if record.get("frame_index") is not None]
    total_frames = len(set(frame_ids))

    timestamps = [_to_float(record.get("timestamp_seconds")) for record in timeline_records]
    timestamps = [value for value in timestamps if value is not None]
    duration_seconds = max(timestamps) if timestamps else 0.0

    face_records = [record for record in timeline_records if int(record.get("face_index", -1)) >= 0]
    no_face_records = [record for record in timeline_records if int(record.get("face_index", -1)) < 0]

    engagement_scores = [_to_float(record.get("engagement_score_smooth")) for record in face_records]
    engagement_scores = [value for value in engagement_scores if value is not None]

    quality_scores = [_to_float(record.get("quality_confidence_score")) for record in face_records]
    quality_scores = [value for value in quality_scores if value is not None]

    warning_counter = _extract_warning_counts(face_records)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_context": {
            "source_type": source_type,
            "input_path": input_path,
            "output_video_path": output_video_path,
        },
        "kpis": {
            "total_frames": total_frames,
            "duration_seconds": round(duration_seconds, 3),
            "face_records": len(face_records),
            "no_face_records": len(no_face_records),
            "engagement_score": {
                "average": round(_safe_mean(engagement_scores), 2) if engagement_scores else None,
                "min": round(min(engagement_scores), 2) if engagement_scores else None,
                "max": round(max(engagement_scores), 2) if engagement_scores else None,
                "samples": len(engagement_scores),
            },
            "quality_confidence": {
                "average": round(_safe_mean(quality_scores), 2) if quality_scores else None,
                "min": round(min(quality_scores), 2) if quality_scores else None,
                "max": round(max(quality_scores), 2) if quality_scores else None,
                "samples": len(quality_scores),
            },
            "warning_records": sum(1 for record in face_records if str(record.get("quality_warnings", "")).strip() != ""),
        },
        "distributions": {
            "engagement_label": _distribution_from_values(record.get("engagement_label") for record in face_records),
            "head_direction": _distribution_from_values(record.get("head_direction") for record in face_records),
            "gaze_direction": _distribution_from_values(record.get("gaze_direction") for record in face_records),
            "quality_confidence_label": _distribution_from_values(
                record.get("quality_confidence_label") for record in timeline_records
            ),
        },
        "warnings": {
            "total_unique": len(warning_counter),
            "top": [
                {"warning": warning, "count": count}
                for warning, count in warning_counter.most_common(10)
            ],
        },
    }
    return payload


def _build_time_series(face_records, key):
    series = []
    for record in face_records:
        timestamp = _to_float(record.get("timestamp_seconds"))
        value = _to_float(record.get(key))
        if timestamp is None or value is None:
            continue
        series.append((timestamp, value))
    series.sort(key=lambda item: item[0])
    return series


def _plot_distribution(ax, counts, title, color):
    if not counts:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    labels = list(counts.keys())
    values = list(counts.values())
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylabel("Count")


def create_summary_pdf(summary_payload, timeline_records, pdf_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    face_records = [record for record in timeline_records if int(record.get("face_index", -1)) >= 0]
    engagement_series = _build_time_series(face_records, "engagement_score_smooth")
    quality_series = _build_time_series(face_records, "quality_confidence_score")

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle("EngageVision Session Summary", fontsize=16, fontweight="bold")

    score_ax = axes[0][0]
    if engagement_series:
        score_ax.plot(
            [item[0] for item in engagement_series],
            [item[1] for item in engagement_series],
            color="#2a9d8f",
            linewidth=1.8,
        )
    else:
        score_ax.text(0.5, 0.5, "No score data", ha="center", va="center")
    score_ax.set_title("Engagement Score Over Time")
    score_ax.set_xlabel("Timestamp (s)")
    score_ax.set_ylabel("Score")
    score_ax.set_ylim(0, 100)
    score_ax.grid(alpha=0.25)

    quality_ax = axes[0][1]
    if quality_series:
        quality_ax.plot(
            [item[0] for item in quality_series],
            [item[1] for item in quality_series],
            color="#457b9d",
            linewidth=1.8,
        )
    else:
        quality_ax.text(0.5, 0.5, "No quality data", ha="center", va="center")
    quality_ax.set_title("Quality Confidence Over Time")
    quality_ax.set_xlabel("Timestamp (s)")
    quality_ax.set_ylabel("Confidence")
    quality_ax.set_ylim(0, 100)
    quality_ax.grid(alpha=0.25)

    _plot_distribution(
        axes[1][0],
        summary_payload["distributions"]["engagement_label"]["counts"],
        "Engagement Label Distribution",
        color="#8ecae6",
    )
    _plot_distribution(
        axes[1][1],
        summary_payload["distributions"]["head_direction"]["counts"],
        "Head Direction Distribution",
        color="#ffb703",
    )

    kpis = summary_payload["kpis"]
    score_stats = kpis["engagement_score"]
    quality_stats = kpis["quality_confidence"]
    footer_lines = [
        f"Frames: {kpis['total_frames']} | Duration: {kpis['duration_seconds']:.2f}s | Face records: {kpis['face_records']}",
        (
            "Engagement score avg/min/max: "
            f"{score_stats['average']} / {score_stats['min']} / {score_stats['max']}"
        ),
        (
            "Quality confidence avg/min/max: "
            f"{quality_stats['average']} / {quality_stats['min']} / {quality_stats['max']}"
        ),
        f"Warning records: {kpis['warning_records']}",
    ]
    figure.text(0.02, 0.02, "\n".join(footer_lines), fontsize=10)

    figure.tight_layout(rect=[0, 0.08, 1, 0.95])
    figure.savefig(pdf_path)
    plt.close(figure)


def export_session_reports(timeline_records, output_dir, source_type, input_path=None, output_video_path=None):
    os.makedirs(output_dir, exist_ok=True)

    summary_payload = build_summary_payload(
        timeline_records,
        source_type=source_type,
        input_path=input_path,
        output_video_path=output_video_path,
    )

    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)

    summary_pdf_path = os.path.join(output_dir, "summary.pdf")
    pdf_error = None
    try:
        create_summary_pdf(summary_payload, timeline_records, summary_pdf_path)
    except Exception as exc:
        pdf_error = str(exc)
        summary_payload["pdf_generation_error"] = pdf_error
        with open(summary_json_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary_payload, summary_file, indent=2)
        summary_pdf_path = None

    return summary_json_path, summary_pdf_path, pdf_error