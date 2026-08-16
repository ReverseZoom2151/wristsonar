"""The host path from a watch's raw PCM packet to a model-ready echo window."""

from __future__ import annotations

import numpy as np

from wristsonar.capture.health import CaptureHealth
from wristsonar.capture.processor import EchoWindowAssembler
from wristsonar.capture.synchronizer import PcmSynchronizer, SyncStatus
from wristsonar.capture.wire import RawPcmWireFrame
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    PreprocessingDescriptor,
)
from wristsonar.runtime.frames import CaptureFrame
from wristsonar.signal.chirp import windowed_chirp

__all__ = ["LiveCaptureError", "LiveCaptureProcessor"]


class LiveCaptureError(ValueError):
    """The watch stream and the public model's signal contract disagree."""


class LiveCaptureProcessor:
    """Process timestamped raw Watch PCM with no device-side DSP assumption.

    The preprocessing contract is a parameter rather than a constant because
    the checkpoint decides it.  Hard-coding the defaults here while validating
    against a bundle's values means a checkpoint trained with anything else
    loads, connects, streams, and only fails once the first window is complete,
    which is after the user has finished setting up hardware.
    """

    def __init__(
        self,
        *,
        descriptor: PreprocessingDescriptor = WATCHHAND_PREPROCESSING,
        lock_timeout_s: float = 3.0,
        rate_tolerance: float = 0.02,
    ) -> None:
        if lock_timeout_s <= 0.0:
            raise ValueError("lock_timeout_s must be positive")
        if not 0.0 < rate_tolerance < 1.0:
            raise ValueError("rate_tolerance must lie in (0, 1)")
        self._descriptor = descriptor
        self._sample_rate: int | None = None
        self._lock_timeout_s = lock_timeout_s
        self._rate_tolerance = rate_tolerance
        self._sync = PcmSynchronizer(
            windowed_chirp(descriptor.chirp),
            sample_rate=descriptor.chirp.sample_rate,
        )
        self._windows = EchoWindowAssembler(descriptor=descriptor)

    @property
    def descriptor(self) -> PreprocessingDescriptor:
        """The preprocessing contract this host path is building to."""
        return self._descriptor

    @property
    def sync_status(self) -> SyncStatus:
        """What the chirp alignment is actually doing, ready to print."""
        return self._sync.status

    @property
    def health(self) -> CaptureHealth:
        """Observed duplex health for the frames that reached the assembler."""
        return self._windows.health

    def push(self, packet: RawPcmWireFrame) -> tuple[CaptureFrame, ...]:
        """Return zero or more causal model windows from one raw callback."""
        expected = self._descriptor.chirp.sample_rate
        if packet.sample_rate != expected:
            # The declared rate is a contract statement, not a measurement: it
            # is whatever the sender configured.  The measurement is below.
            raise LiveCaptureError(
                f"watch declares a {packet.sample_rate} Hz sample grid; the "
                f"model requires {expected} Hz"
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
        self._assert_stream_is_usable()
        return tuple(result)

    def _assert_stream_is_usable(self) -> None:
        """Fail loudly rather than stream silence.

        A capture path that emits nothing is the failure mode this whole module
        exists to make visible, so it is raised rather than logged: the operator
        is standing at the watch, and the only useful moment to tell them is
        now.
        """
        status = self._sync.status
        observed = status.observed_sample_rate
        if observed is not None:
            nominal = float(self._descriptor.chirp.sample_rate)
            if abs(observed - nominal) > self._rate_tolerance * nominal:
                raise LiveCaptureError(
                    f"watch declares {int(nominal)} Hz but delivered "
                    f"{observed:.0f} samples per second of monotonic time; the "
                    "capture path is resampling or losing buffers"
                )
        if not status.locked and status.unlocked_s > self._lock_timeout_s:
            raise LiveCaptureError(f"chirp synchronisation failed: {status.reason}")
