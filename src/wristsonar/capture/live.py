"""The host path from a watch's raw PCM packet to a model-ready echo window."""

from __future__ import annotations

import numpy as np

from wristsonar.capture.processor import EchoWindowAssembler
from wristsonar.capture.synchronizer import PcmSynchronizer
from wristsonar.capture.wire import RawPcmWireFrame
from wristsonar.data.watchhand import WATCHHAND_CHIRP
from wristsonar.runtime.frames import CaptureFrame
from wristsonar.signal.chirp import windowed_chirp

__all__ = ["LiveCaptureError", "LiveCaptureProcessor"]


class LiveCaptureError(ValueError):
    """The watch stream and the public model's signal contract disagree."""


class LiveCaptureProcessor:
    """Process timestamped raw Watch PCM with no device-side DSP assumption."""

    def __init__(self) -> None:
        self._sample_rate: int | None = None
        self._sync = PcmSynchronizer(
            windowed_chirp(WATCHHAND_CHIRP),
            sample_rate=WATCHHAND_CHIRP.sample_rate,
        )
        self._windows = EchoWindowAssembler()

    def push(self, packet: RawPcmWireFrame) -> tuple[CaptureFrame, ...]:
        """Return zero or more causal model windows from one raw callback."""
        expected = WATCHHAND_CHIRP.sample_rate
        if packet.sample_rate != expected:
            raise LiveCaptureError(
                f"watch sent {packet.sample_rate} Hz; model requires {expected} Hz"
            )
        if self._sample_rate is None:
            self._sample_rate = packet.sample_rate
        elif packet.sample_rate != self._sample_rate:
            self._sync.reset()
            raise LiveCaptureError("watch changed sample rate within one session")

        result: list[CaptureFrame] = []
        for frame in self._sync.push(
            np.asarray(packet.samples), timestamp_s=packet.timestamp_s
        ):
            window = self._windows.push(
                frame.samples,
                timestamp_s=frame.timestamp_s,
                sample_rate=packet.sample_rate,
                continuous=frame.continuous,
            )
            if window is not None:
                result.append(window)
        return tuple(result)
