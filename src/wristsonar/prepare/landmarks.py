"""Generate auditable MediaPipe landmark sidecars from WatchHand videos."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from wristsonar.types import N_JOINTS

__all__ = [
    "LandmarkDetector",
    "LandmarkPreparationError",
    "LandmarkPreparationReport",
    "MediaPipeHandDetector",
    "prepare_landmarks",
]


class LandmarkPreparationError(RuntimeError):
    """A video could not yield a complete, trustworthy landmark sidecar."""


class LandmarkDetector(Protocol):
    """A camera-landmark implementation, injected to keep preparation testable."""

    version: str

    def detect(self, bgr_frame: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        """Return 21 landmarks in metres, or None when no hand is found."""


class MediaPipeHandDetector:
    """MediaPipe's world-landmark estimator behind the testable detector port.

    MediaPipe reports landmark coordinates in a hand-centred metric-like world
    frame. They are still video-fitted labels, not motion-capture ground truth;
    the metadata written by :func:`prepare_landmarks` preserves that fact.
    """

    version: str

    def __init__(self, *, max_hands: int = 1, min_confidence: float = 0.5) -> None:
        if max_hands != 1:
            raise ValueError("WatchHand sidecars support exactly one tracked hand")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise LandmarkPreparationError(
                "install preparation support with: pip install -e '.[prepare]'"
            ) from error
        self._hands: Any = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.version = f"mediapipe-{getattr(mp, '__version__', 'unknown')}/hands-1"

    def detect(self, bgr_frame: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise LandmarkPreparationError(
                f"expected BGR frame (height, width, 3), got {bgr_frame.shape}"
            )
        rgb = bgr_frame[..., ::-1]
        result = self._hands.process(rgb)
        world = result.multi_hand_world_landmarks
        if not world:
            return None
        landmarks = world[0].landmark
        if len(landmarks) != N_JOINTS:
            raise LandmarkPreparationError(
                f"MediaPipe returned {len(landmarks)} landmarks, expected {N_JOINTS}"
            )
        return np.asarray(
            [(point.x, point.y, point.z) for point in landmarks], dtype=np.float32
        )


@dataclass(frozen=True, slots=True)
class LandmarkPreparationReport:
    output: Path
    metadata: Path
    detector_version: str
    frames: int
    missing: int
    sha256: str


def prepare_landmarks(
    video: Path,
    frame_timestamps_s: NDArray[np.float64],
    output: Path,
    detector: LandmarkDetector,
) -> LandmarkPreparationReport:
    """Write one landmark per video frame and an inseparable provenance record.

    Missing detections fail preparation instead of being forward-filled. Filling
    produces attractive targets but turns detector failures into a hidden motion
    prior, which would make pose error against these labels uninterpretable.
    """
    try:
        import cv2
    except ImportError as error:
        raise LandmarkPreparationError(
            "install preparation support with: pip install -e '.[prepare]'"
        ) from error
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise LandmarkPreparationError(f"cannot open video {video}")
    poses: list[NDArray[np.float32]] = []
    missing = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pose = detector.detect(np.asarray(frame, dtype=np.uint8))
            if pose is None:
                missing += 1
                continue
            if pose.shape != (N_JOINTS, 3):
                raise LandmarkPreparationError(
                    f"detector returned {pose.shape}, expected ({N_JOINTS}, 3)"
                )
            poses.append(np.asarray(pose, dtype=np.float32))
    finally:
        capture.release()
    if len(poses) != len(frame_timestamps_s):
        raise LandmarkPreparationError(
            f"refusing incomplete labels: video has {len(poses)} detected of "
            f"{len(frame_timestamps_s)} timestamped frames ({missing} missing)"
        )
    array = np.stack(poses)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = output.with_suffix(".landmarks.json")
    metadata.write_text(
        json.dumps(
            {
                "schema": "wristsonar.landmarks/1",
                "ground_truth": "video-fitted",
                "detector": detector.version,
                "frames": len(poses),
                "sha256": digest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return LandmarkPreparationReport(
        output=output,
        metadata=metadata,
        detector_version=detector.version,
        frames=len(poses),
        missing=missing,
        sha256=digest,
    )
