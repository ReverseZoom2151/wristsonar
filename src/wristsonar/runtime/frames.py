"""Versioned, JSON-safe frames at the capture and inference boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wristsonar.types import N_JOINTS

__all__ = ["CaptureFrame", "PoseFrame", "WireFormatError"]


class WireFormatError(ValueError):
    """Raised when a wire frame is malformed or belongs to another protocol."""


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

    def jsonable(self) -> dict[str, object]:
        """Stable capture representation for a watch-to-host transport.

        Capture and pose frames deliberately have different type labels. A
        host that accidentally loops its own pose output back into its capture
        input must fail at the boundary rather than treating joints as an echo
        window and producing plausible looking nonsense.
        """
        return {
            "version": self.protocol_version,
            "type": "wristsonar.capture",
            "timestamp_s": self.timestamp_s,
            "sample_rate": self.sample_rate,
            "samples": self.samples.tolist(),
        }

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> CaptureFrame:
        """Parse exactly the version emitted by :meth:`jsonable`."""
        if value.get("type") != "wristsonar.capture":
            raise WireFormatError(
                "expected a wristsonar.capture frame, got "
                f"{value.get('type')!r}"
            )
        try:
            samples = np.asarray(value["samples"], dtype=np.float32)
            return cls(
                samples=samples,
                timestamp_s=float(value["timestamp_s"]),
                sample_rate=int(value["sample_rate"]),
                protocol_version=int(value["version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WireFormatError(f"invalid capture frame: {error}") from error


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
