"""Versioned, JSON-safe frames at the capture and inference boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.types import N_JOINTS

__all__ = ["CaptureFrame", "PoseFrame"]


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    """One two-channel, cropped echo window received from a watch."""

    samples: NDArray[np.float32]
    timestamp_s: float
    sample_rate: int
    protocol_version: int = 1

    def __post_init__(self) -> None:
        if self.protocol_version != 1:
            raise ValueError(f"unsupported capture protocol v{self.protocol_version}")
        if self.samples.ndim != 3 or self.samples.shape[0] != 2:
            raise ValueError(
                "capture samples must be (original+differential, bins, frames), "
                f"got {self.samples.shape}"
            )
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """A wrist-relative hand prediction suitable for any realtime sink."""

    joints: NDArray[np.float32]
    timestamp_s: float
    source_timestamp_s: float
    model_id: str
    calibration_minutes: float

    def __post_init__(self) -> None:
        if self.joints.shape != (N_JOINTS, 3):
            raise ValueError(
                f"expected ({N_JOINTS}, 3) joints, got {self.joints.shape}"
            )
        if self.calibration_minutes < 0:
            raise ValueError("calibration minutes cannot be negative")

    def jsonable(self) -> dict[str, object]:
        """Stable wire representation used by JSONL and WebSocket adapters."""
        return {
            "version": 1,
            "type": "wristsonar.pose",
            "timestamp_s": self.timestamp_s,
            "source_timestamp_s": self.source_timestamp_s,
            "model_id": self.model_id,
            "calibration_minutes": self.calibration_minutes,
            "frame": "wrist-relative",
            "joints": self.joints.tolist(),
        }
