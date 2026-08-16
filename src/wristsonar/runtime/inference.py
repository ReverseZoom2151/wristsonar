"""The strict boundary between live acoustic input, a model, and a sink.

The model does not get to decide whether a capture frame is compatible with
it.  The training chirp, crop and causal window define the model's input, so
this module validates those dimensions before inference.  A shape mismatch is
usually evidence of a watch app emitting a different DSP product, not a reason
to resize data until the exception disappears.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.runtime.frames import CaptureFrame, PoseFrame
from wristsonar.runtime.sinks import Sink
from wristsonar.types import N_JOINTS

__all__ = ["InferenceError", "RealtimeInference"]

PosePredictor = Callable[[NDArray[np.float32]], NDArray[np.floating]]


class InferenceError(RuntimeError):
    """The live data and the loaded model cannot truthfully be connected."""


@dataclass(slots=True)
class RealtimeInference:
    """Predict and emit a pose for every compatible capture window.

    ``predict`` receives a single unbatched window, shape ``(2, bins, frames)``,
    and must return 21 wrist-relative 3D joints in metres.  Keeping the model
    behind this narrow callable lets Torch, ONNX, TFLite and a test double share
    exactly the same validation and output semantics.
    """

    predict: PosePredictor
    sink: Sink
    model_id: str
    calibration_minutes: float
    expected_shape: tuple[int, int, int]
    expected_sample_rate: int = 48_000
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.calibration_minutes < 0:
            raise ValueError("calibration_minutes cannot be negative")
        if self.expected_shape[0] != 2 or any(n < 1 for n in self.expected_shape):
            raise ValueError(
                "expected_shape must be (2, positive bins, positive frames)"
            )
        if self.expected_sample_rate <= 0:
            raise ValueError("expected_sample_rate must be positive")

    def process(self, capture: CaptureFrame) -> PoseFrame:
        """Validate, predict, emit, and return a pose frame.

        The source timestamp stays attached to the result.  Measuring latency
        from the host's emission time alone would hide buffering delays in the
        watch transport, which is exactly the latency a user experiences.
        """
        if capture.sample_rate != self.expected_sample_rate:
            raise InferenceError(
                f"model expects {self.expected_sample_rate} Hz capture, got "
                f"{capture.sample_rate} Hz"
            )
        if capture.samples.shape != self.expected_shape:
            raise InferenceError(
                f"model expects capture shape {self.expected_shape}, got "
                f"{capture.samples.shape}"
            )
        if not np.isfinite(capture.samples).all():
            raise InferenceError("capture contains non-finite echo values")

        raw = np.asarray(self.predict(capture.samples), dtype=np.float32)
        if raw.shape != (N_JOINTS, 3):
            raise InferenceError(
                f"model returned {raw.shape}; expected ({N_JOINTS}, 3) joints"
            )
        if not np.isfinite(raw).all():
            raise InferenceError("model returned non-finite joint positions")

        # A wrist-mounted system has no evidence for global wrist translation.
        # Enforce the advertised reference frame even if a model drifts by a
        # few numerical units at the root output.
        joints = raw - raw[0:1]
        frame = PoseFrame(
            joints=joints,
            timestamp_s=float(self.clock()),
            source_timestamp_s=capture.timestamp_s,
            model_id=self.model_id,
            calibration_minutes=self.calibration_minutes,
        )
        self.sink.emit(frame)
        return frame
