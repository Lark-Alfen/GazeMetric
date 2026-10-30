# GazeMetric

GazeMetric is a real-time engagement analytics project for webcam and recorded video sessions.

Under the hood, it uses MediaPipe face landmarks, OpenCV-based head pose estimation, gaze-direction heuristics, and calibration-aware thresholds to classify attention behavior frame by frame. The system also computes a composite engagement score, tracks signal quality/confidence, and exports full session reports.

Core capabilities:
- head direction estimation (left, right, up, down, straight)
- gaze direction estimation (forward, left, right, up, down)
- engagement labeling and smoothed score (0-100)
- quality/confidence checks with runtime warnings
- run artifacts for analysis (`timeline.csv`, `timeline.json`, `summary.json`, `summary.pdf`)

## How It Works (Technical Overview)

1. Detect facial landmarks on each frame.
2. Estimate head pose angles from landmark geometry.
3. Compute gaze offsets and classify gaze direction.
4. Combine gaze + head pose into engagement labels and scores.
5. Track quality signals (visibility, lighting, landmark stability).
6. Save per-frame timeline data and end-of-run summary reports.

## What You Need

- Python 3.11 or 3.12
- A webcam (for live mode) or a video file (for recorded mode)

## Quick Setup (Windows)

1. Open PowerShell in the project folder.
2. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

3. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

If your machine complains about text encoding while installing, run:

```powershell
Get-Content requirements.txt -Encoding Unicode | Set-Content requirements_utf8.txt -Encoding utf8
pip install -r requirements_utf8.txt
```

## How To Run

### 1) Live Webcam Mode

```powershell
.\.venv\Scripts\python.exe engagement_analysis_live.py
```

### 2) Recorded Video Mode

Put your video in the `Video` folder (or use any full path), then run:

```powershell
.\.venv\Scripts\python.exe engagement_analysis_recorded.py --input "Video\your_video.mp4"
```

### 3) Full Script (Main Engine)

Webcam:

```powershell
.\.venv\Scripts\python.exe engagement_analysis_full.py
```

Video:

```powershell
.\.venv\Scripts\python.exe engagement_analysis_full.py --source video --input "Video\your_video.mp4"
```

## Useful Options

Skip calibration:

```powershell
.\.venv\Scripts\python.exe engagement_analysis_full.py --skip-calibration
```

Save annotated output video:

```powershell
.\.venv\Scripts\python.exe engagement_analysis_full.py --source video --input "Video\your_video.mp4" --output-video "runs\annotated.mp4"
```

Choose report base folder:

```powershell
.\.venv\Scripts\python.exe engagement_analysis_full.py --source video --input "Video\your_video.mp4" --timeline-dir "runs\my_runs"
```

## Output Files

After each run, the project saves:
- timeline.csv
- timeline.json
- summary.json
- summary.pdf

Each run now gets its own session folder, so old results are kept.

Example:
- runs/my_runs/session_20260414_101500/
- runs/my_runs/session_20260414_102030/

## Which Script Should I Use?

- `engagement_analysis_live.py`: easiest way to run webcam mode
- `engagement_analysis_recorded.py`: easiest way to run a saved video
- `engagement_analysis_full.py`: main script with all options

## Basic Controls

- Press Q in the video window to stop and save reports.

## Notes

- You can place multiple videos in `Video` and choose one with the `--input` path.
- File names can be anything.

## License

This project uses the MIT License. See `LICENSE` for details.
