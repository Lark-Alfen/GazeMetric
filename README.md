# GazeMetric

GazeMetric is a real-time engagement analytics project for webcam and recorded video sessions.

Under the hood, it uses MediaPipe Face Landmarker landmarks, OpenCV-based head pose estimation, gaze-direction heuristics, and calibration-aware thresholds to classify attention behavior frame by frame. The system also computes a composite engagement score, tracks signal quality/confidence, and exports full session reports.

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

- Python 3.11, 3.12 or 3.13 (the automated tests run on all three)
- A webcam (for live mode) or a video file (for recorded mode)

Tested on Windows (development) and Linux (the CI runs on Ubuntu). macOS is not tested.

## Setup

Run these from the project folder. After the virtual environment is activated, the same `python ...` commands
work on every system.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux (bash or zsh)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- On a fresh Linux machine OpenCV and MediaPipe need a few system libraries; on Ubuntu/Debian:
  `sudo apt-get install libgl1 libegl1 libgles2`. (`libgl1` for OpenCV and `libegl1` for MediaPipe were both
  missing on the CI runner; `libgles2` is added as a precaution and not confirmed to be required.)
- macOS asks for camera permission the first time: allow it for the terminal (or editor) you start the script
  from, in System Settings > Privacy & Security > Camera, then run the script again.
- Under WSL the webcam is not available by default; use recorded video mode there.
- Download the face model once (about 4 MB, checked against a known SHA-256):

  ```
  python download_face_model.py
  ```

  It is saved to `Models/face_landmarker.task`, which git ignores. To keep the model somewhere else, save it
  there and pass `--face-model PATH` when you run a script.
- To run the tests as well: `pip install -r requirements-dev.txt` and then `python -m pytest`.

## How To Run

Activate the virtual environment first (see above), then:

### 1) Live Webcam Mode

```
python engagement_analysis_live.py
```

### 2) Recorded Video Mode

Put your video in the `Video` folder (or use any full path), then run:

```
python engagement_analysis_recorded.py --input "Video/your_video.mp4"
```

### 3) Full Script (Main Engine)

Webcam:

```
python engagement_analysis_full.py
```

Video:

```
python engagement_analysis_full.py --source video --input "Video/your_video.mp4"
```

## Options

All three scripts accept these options (`engagement_analysis_full.py` also takes `--source` and `--input`;
`engagement_analysis_live.py` and `engagement_analysis_full.py` take `--camera-index`;
`engagement_analysis_recorded.py` requires `--input`). Run any script with `--help` for the same list.

| Option | Default | What it does |
| --- | --- | --- |
| `--source {webcam,video}` | `webcam` | Input type (`engagement_analysis_full.py` only). |
| `--input PATH` | none | Video file to analyse. Required for video input. |
| `--camera-index N` | `0` | Which webcam to use; `0` is the first one. |
| `--output-video PATH` | none | Also save the annotated video (face contours, gaze and head lines, per-face panel). |
| `--timeline-dir DIR` | `runs` | Base folder for the results. Every run creates its own `session_<timestamp>` folder inside it. |
| `--skip-calibration` | off | Don't calibrate; use the default thresholds. |
| `--calibration-seconds S` | `10` | Length of the calibration step. |
| `--calibration-profile FILE` | none | Load this calibration profile if the file exists, otherwise calibrate and save it there. |
| `--face-model PATH` | `Models/face_landmarker.task` | The MediaPipe Face Landmarker model file (see Setup). |
| `--max-faces N` | `2` | Maximum number of faces tracked at the same time. |
| `--no-display` | off | No preview window; prints progress instead. For batch runs and machines without a display. |

Examples:

```
python engagement_analysis_full.py --source video --input "Video/your_video.mp4" --output-video "runs/annotated.mp4"
python engagement_analysis_full.py --source video --input "Video/your_video.mp4" --timeline-dir "runs/my_runs"
python engagement_analysis_recorded.py --input "Video/your_video.mp4" --skip-calibration --no-display
python engagement_analysis_live.py --camera-index 1 --max-faces 1
```

## Calibration

Calibration measures what "looking straight at the camera" looks like for this person and camera, and sets the
thresholds from it (gaze centre and tolerance, head centre, and how far the head may turn before it counts as
not engaged).

- At the start of a run, look straight at the camera for `--calibration-seconds` (default 10). For a video file
  the first seconds of the video are used, and the whole video is analysed afterwards.
- In the window press **S** to skip it or **Q** (or close the window) to stop the run. Without a window
  (`--no-display`) there are no keys, it simply runs for its length.
- It uses the typical value (median) and spread of what it saw, so a few glances away or blinks do not change
  the result. If the head moved a lot during calibration (more than about 10 degrees of spread) it is rejected,
  a message says so, and the default thresholds are used.
- If too few frames had a face, the result is blended toward the defaults.

Reuse a calibration between runs:

```
python engagement_analysis_live.py --calibration-profile profiles/me.json
```

The first time this calibrates and saves `profiles/me.json`; later runs load it and skip calibration. If the file
exists but cannot be read, it is moved to `me.json.bad` (never overwritten) and the run continues without it.
A profile is specific to a person, camera position and resolution; delete it to calibrate again.

## Output Files

After each run the project saves, in `<timeline-dir>/session_<timestamp>/`:

- `timeline.csv`: one row per detected face per frame (and one row for every frame without a face). Written
  while the run is going, so a crash still leaves the rows so far.
- `timeline.json`: the same rows as JSON.
- `summary.json`: totals, averages, label distributions and the most common warnings, overall and per face.
- `summary.pdf`: charts of engagement score and quality over time (one line per face) and the label distributions.

Each run gets its own session folder, so old results are kept. Example:
`runs/my_runs/session_20260414_101500/`.

### `timeline.csv` columns

| Column | Meaning |
| --- | --- |
| `frame_index` | Frame number, starting at 1. |
| `timestamp_seconds` | Seconds into the video (video input) or since the run started (webcam). |
| `face_index` | Number that follows one person while they stay in view; `-1` means no face was found in this frame. |
| `x_angle` | Head up/down angle in degrees. Positive is looking up. |
| `y_angle` | Head left/right angle in degrees. Positive is looking toward the right of the image. |
| `gaze_direction` | `Forward`, `Left`, `Right`, `Up`, `Down`, `Not Forward`; `No Face` or `Pose Unavailable` when it could not be measured. |
| `engagement_label` | `Engaged` or `Not Engaged` (see below); `No Face` or `Pose Unavailable`. |
| `head_direction` | `Looking Straight`, `Looking Left`, `Looking Right`, `Looking Up`, `Looking Down`; or `No Face` / `Pose Unavailable`. |
| `head_pose_score` | 0 to 1, how close the head is to the calibrated straight position. |
| `gaze_center_score` | 0 to 1, how close the iris positions are to the calibrated centre. |
| `engagement_score_raw` | 0 to 100, the weighted combination of the two scores above for this frame. |
| `engagement_score_smooth` | 0 to 100, average of the last 15 raw scores of this face. |
| `face_visibility_score` | 0 to 1, from face size, position in the frame and sudden size changes. |
| `lighting_score` | 0 to 1, from the brightness and contrast of the face region. |
| `landmark_stability_score` | 0 to 1, how steady the face landmarks are from frame to frame. |
| `quality_confidence_score` | 0 to 100, overall trust in this frame's measurement. |
| `quality_confidence_label` | `High` (70 and above), `Medium` (45 and above) or `Low`; `Unknown` / `No Face`. |
| `quality_warnings` | Warnings for this frame, separated by `; ` (list below). Empty when there are none. |

Cells that could not be measured are empty (`null` in the JSON file).

## How The Score Works

- **Head pose score**: for the up/down and left/right angles, `1 - min(|angle - calibrated centre|, 30) / 30`,
  averaged. Straight is 1, 30 degrees or more away is 0.
- **Gaze centre score**: for each eye and direction, `1 - |iris position - calibrated centre| / 0.5`, averaged.
- **Engagement score**: `0.55 x head pose score + 0.45 x gaze centre score`, as a percentage; the smoothed
  value is the average of the last 15 frames of the same face.
- **Engaged**: the gaze direction is `Forward` and the head is within the engagement threshold of the calibrated
  centre in both directions (12 degrees by default, widened or narrowed by calibration). Otherwise
  `Not Engaged`.
- **Gaze direction** compares the iris position inside the eye with the calibrated centre. A wider calibrated
  tolerance widens the `Forward` band.
- **Quality confidence**: `0.45 x visibility + 0.30 x lighting + 0.25 x stability`.

Quality warnings:

| Warning | When |
| --- | --- |
| `Face too far from camera` | the face covers less than 5% of the frame |
| `Face too close to camera` | the face covers more than 55% of the frame |
| `Face off-center` | the face is far from the middle of the frame |
| `Low light` / `Overexposed frame` | the face region is darker than 70 or brighter than 205 (out of 255) on average |
| `Low contrast` | the face region has very little brightness variation |
| `Frequent movement` / `Unstable tracking` | the landmarks moved a lot between frames |

A face that was detected but whose head pose could not be solved is recorded as `Pose Unavailable`.

## Limitations And Privacy

- This tool analyses video of people's faces and writes data about each person. Only analyse video of people
  who know about it and agree to it, and follow the rules that apply to you (workplace, school, data
  protection law).
- Everything stays on your computer: results go to the `runs` folder (ignored by git), calibration profiles to
  the path you give, the annotated video only if you ask for it. The only thing in the project that connects to the
  internet is `download_face_model.py`, which you run yourself once to fetch the model; analysis never does. The annotated video contains the faces from the input.
- To delete results, delete the `session_<timestamp>` folder, or the whole `runs` folder. Delete profile files and
  annotated videos yourself; nothing is cleaned up automatically.
- The labels are heuristics from face landmarks, not a measurement of attention or of what a person is thinking.
  Head angles come from a generic face model and are approximate; glasses, bad light, a face turned far away or a
  low resolution make them worse (see the quality warnings). Do not use the output to make decisions about people
  without a human looking at the footage.

## Which Script Should I Use?

- `engagement_analysis_live.py`: easiest way to run webcam mode
- `engagement_analysis_recorded.py`: easiest way to run a saved video
- `engagement_analysis_full.py`: main script with all options

## Basic Controls

- **Q** in the video window, or closing the window, stops the run and saves the reports.
- **Ctrl+C** in the terminal does the same.
- During calibration, **S** skips it and **Q** (or closing the window) stops before the analysis starts.
- The scripts exit with code 0 when the run finished and 1 when it failed (for example a missing camera or
  an error during the run); the reports are still written after an error during analysis.

## Notes

- You can place multiple videos in `Video` and choose one with the `--input` path.
- File names can be anything.

## License

This project uses the MIT License. See `LICENSE` for details.
