"""Convert aligned microphone frames into causal model-ready echo windows.

Every convention this module used to hold privately now arrives in a
`PreprocessingDescriptor`, the same object the training corpus builder reads
and the same object a checkpoint records. Nothing here decides how a window is
built; it decides only how to get from a microphone to the tensor that
contract describes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.capture.health import CaptureHealth, DuplexValidator
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    PreprocessingDescriptor,
)
from wristsonar.runtime.frames import CaptureFrame
from wristsonar.signal.chirp import analytic_reference
from wristsonar.signal.echo_profile import differential_profiles, profile_from_frame
from wristsonar.types import EchoProfile

__all__ = ["CaptureProcessingError", "EchoWindowAssembler", "ProcessedFrame"]


class CaptureProcessingError(ValueError):
    """A live frame cannot belong to the WatchHand model's signal contract."""


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """One original and differential profile ending at the same timestamp.

    Ending at, which is the whole causal contract in three words. The
    differential is the difference between this frame and the one before it,
    so it is complete at this frame's timestamp and never later. The equality
    is checked rather than assumed because the failure it guards against, a
    differential that reaches one frame into the future, is invisible in the
    output and inflates offline accuracy by exactly the amount that cannot be
    reproduced online.
    """

    original: EchoProfile
    differential: EchoProfile

    def __post_init__(self) -> None:
        if not self.differential.differential:
            raise CaptureProcessingError("the second profile must be differential")
        if self.original.timestamp_s != self.differential.timestamp_s:
            raise CaptureProcessingError("original and differential timestamps differ")


class EchoWindowAssembler:
    """Process raw aligned PCM and emit causal 2 by crop by time windows."""

    def __init__(
        self,
        *,
        descriptor: PreprocessingDescriptor = WATCHHAND_PREPROCESSING,
    ) -> None:
        self._descriptor = descriptor
        self._reference = analytic_reference(descriptor.chirp)
        self._validator = DuplexValidator()
        self._previous: EchoProfile | None = None
        self._frames: deque[ProcessedFrame] = deque(maxlen=descriptor.window_frames)
        self._last_timestamp: float | None = None

    @property
    def descriptor(self) -> PreprocessingDescriptor:
        """The contract this assembler builds to, for a loader to check."""
        return self._descriptor

    @property
    def health(self) -> CaptureHealth:
        """Observed duplex health, including a discontinuity that reset state."""
        return self._validator.report()

    def push(
        self,
        pcm: NDArray[np.signedinteger] | NDArray[np.floating],
        *,
        timestamp_s: float,
        sample_rate: int,
        continuous: bool,
    ) -> CaptureFrame | None:
        """Consume one aligned chirp frame, yielding only full causal windows.

        A discontinuity clears differencing and temporal history. Otherwise the
        first post-restart frame would fabricate a large motion event.
        """
        chirp = self._descriptor.chirp
        if not np.isfinite(timestamp_s):
            raise CaptureProcessingError("timestamp must be finite")
        values = np.asarray(pcm)
        if values.ndim != 1:
            raise CaptureProcessingError("a microphone frame must be one dimensional")
        if values.size != chirp.n_samples:
            raise CaptureProcessingError(
                f"expected {chirp.n_samples} samples, got {values.size}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise CaptureProcessingError("PCM frame must be numeric")
        frame = np.asarray(values, dtype=np.float64)
        if not np.isfinite(frame).all():
            raise CaptureProcessingError("PCM frame contains non-finite values")
        if np.issubdtype(values.dtype, np.integer):
            frame /= float(np.iinfo(values.dtype).max + 1)

        monotone = self._last_timestamp is None or timestamp_s > self._last_timestamp
        connected = continuous and monotone
        self._validator.observe(
            sample_rate=sample_rate,
            frame_samples=int(values.size),
            continuous=connected,
        )
        self._last_timestamp = timestamp_s
        if not connected:
            self._previous = None
            self._frames.clear()
            # A restart begins a new capture segment. Preserve neither the
            # previous profile nor the validator's failed segment: otherwise
            # every later valid callback would remain rejected forever.
            self._validator = DuplexValidator()
            return None
        health = self._validator.report()
        if not health.accepted:
            raise CaptureProcessingError(health.reason)

        original = profile_from_frame(
            frame,
            self._reference,
            sample_rate,
            timestamp_s,
            n_bins=chirp.n_samples,
        )
        if self._previous is None:
            self._previous = original
            return None
        # The immediate difference, attributed to the later frame of its pair,
        # which is `DifferentialPhase.LATER` and is the only convention the
        # descriptor permits. There is no lag to apply here: the descriptor's
        # differential lag says where that same quantity sits inside the array
        # WatchHand shipped, and this path has no such array.
        differential = differential_profiles((self._previous, original))[0]
        self._previous = original
        self._frames.append(ProcessedFrame(original, differential))
        if len(self._frames) < self._descriptor.window_frames:
            return None
        crop = self._descriptor.crop_rows
        original_stack = np.stack(
            [crop(item.original.samples) for item in self._frames]
        )
        differential_stack = np.stack(
            [crop(item.differential.samples) for item in self._frames]
        )
        stacked = np.stack((original_stack.T, differential_stack.T)).astype(np.float32)
        return CaptureFrame(
            samples=self._descriptor.normalise_window(stacked),
            timestamp_s=timestamp_s,
            sample_rate=sample_rate,
        )
