"""Keeps the per-person state (gaze smoothing, score window, quality state) with the right face."""

from collections import deque
from dataclasses import dataclass, field

from core.quality import get_default_quality_state

MAX_MISSING_FRAMES = 30
# Largest jump of a face centre between two frames, as a fraction of the frame width/height.
MAX_MATCH_DISTANCE = 0.25


@dataclass
class FaceTrack:
    track_id: int
    centre: tuple
    last_seen_frame: int
    score_window: deque
    last_scores: dict = field(default_factory=lambda: {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0})
    quality_state: dict = field(default_factory=get_default_quality_state)


class FaceTracker:
    """Gives each detected face a stable id by matching it to the nearest face of the previous frames.

    MediaPipe does not keep its faces in the same order from frame to frame, so the position in the
    detection list cannot identify a person. Call `match` once per frame, also when no face was found,
    so that tracks which have been missing for more than `max_missing_frames` frames expire.
    """

    def __init__(self, score_window_size, max_missing_frames=MAX_MISSING_FRAMES, max_distance=MAX_MATCH_DISTANCE):
        self.score_window_size = score_window_size
        self.max_missing_frames = max_missing_frames
        self.max_distance = max_distance
        self.tracks = []
        self._next_id = 0

    def match(self, frame_index, centres):
        """Return one FaceTrack per centre (normalized x, y), in the same order as `centres`."""
        self.tracks = [t for t in self.tracks if frame_index - t.last_seen_frame <= self.max_missing_frames]

        candidates = []
        for track in self.tracks:
            for detection, centre in enumerate(centres):
                distance = ((track.centre[0] - centre[0]) ** 2 + (track.centre[1] - centre[1]) ** 2) ** 0.5
                if distance <= self.max_distance:
                    candidates.append((distance, track.track_id, detection, track))
        candidates.sort(key=lambda item: item[:3])

        assigned = {}
        used_tracks = set()
        for _distance, track_id, detection, track in candidates:
            if detection in assigned or track_id in used_tracks:
                continue
            assigned[detection] = track
            used_tracks.add(track_id)

        matched = []
        for detection, centre in enumerate(centres):
            track = assigned.get(detection)
            if track is None:
                track = FaceTrack(
                    track_id=self._next_id,
                    centre=tuple(centre),
                    last_seen_frame=frame_index,
                    score_window=deque(maxlen=self.score_window_size),
                )
                self._next_id += 1
                self.tracks.append(track)
            track.centre = tuple(centre)
            track.last_seen_frame = frame_index
            matched.append(track)
        return matched
