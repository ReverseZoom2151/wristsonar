"""Synchronise arbitrary microphone chunks to the known periodic FMCW chirp.

``AudioRecord`` callbacks are buffer boundaries chosen by Android, not chirp
boundaries chosen by Wristsonar.  Feeding them straight to a matched filter
would turn callback scheduling jitter into an apparent hand motion.  This
module establishes one alignment from the direct speaker-to-microphone path,
then emits only whole chirp frames on the transmitted cadence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.signal
from numpy.typing import NDArray

__all__ = [
    "AlignedPcmFrame",
    "PcmSynchronizer",
    "SynchronizationError",
]


class SynchronizationError(ValueError):
    """A PCM stream cannot be truthfully aligned to the configured chirp."""


@dataclass(frozen=True, slots=True)
class AlignedPcmFrame:
    """One microphone frame whose sample zero aligns to a transmitted chirp."""

    samples: NDArray[np.float32]
    timestamp_s: float
    continuous: bool


class PcmSynchronizer:
    """Find and preserve a periodic chirp alignment across PCM callbacks.

    The direct acoustic path is deliberately used only for timing.  It is not
    removed here, because the model must see exactly the signal product used
    during training.  A transport gap drops all partial state and forces a new
    lock rather than stitching unrelated chunks into a plausible false frame.
    """

    def __init__(
        self,
        reference: NDArray[np.floating],
        *,
        sample_rate: int,
        min_sync_score: float = 0.25,
        timing_tolerance_samples: float = 1.5,
    ) -> None:
        values = np.asarray(reference, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("reference must be a one-dimensional chirp of length >= 2")
        if not np.isfinite(values).all() or float(np.linalg.norm(values)) == 0.0:
            raise ValueError("reference must be finite and contain energy")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 < min_sync_score <= 1.0:
            raise ValueError("min_sync_score must lie in (0, 1]")
        if timing_tolerance_samples < 0.0:
            raise ValueError("timing_tolerance_samples cannot be negative")
        self._reference = values
        self._reference_norm = float(np.linalg.norm(values))
        self._sample_rate = sample_rate
        self._score = min_sync_score
        self._tolerance_s = timing_tolerance_samples / sample_rate
        self._samples: deque[float] = deque()
        self._start_s: float | None = None
        self._next_s: float | None = None
        self._locked = False
        self._fresh_segment = True

    @property
    def frame_samples(self) -> int:
        """The PCM samples in each emitted frame."""
        return int(self._reference.size)

    @property
    def locked(self) -> bool:
        """Whether the next whole frame is on an established chirp cadence."""
        return self._locked

    def reset(self) -> None:
        """Discard timing state after a route change, underrun, or sample-rate shift."""
        self._samples.clear()
        self._start_s = None
        self._next_s = None
        self._locked = False
        self._fresh_segment = True

    def push(
        self,
        pcm: NDArray[np.signedinteger] | NDArray[np.floating],
        *,
        timestamp_s: float,
    ) -> tuple[AlignedPcmFrame, ...]:
        """Append one timestamped callback and return any complete chirp frames.

        ``timestamp_s`` denotes the first sample in ``pcm``.  Inputs must be
        contiguous to within the stated tolerance.  A gap is not an exception:
        it starts a new segment and emits no frame until the chirp is located
        again, because that is the only response that cannot fabricate timing.
        """
        values = np.asarray(pcm)
        if values.ndim != 1 or values.size == 0:
            raise SynchronizationError("PCM callback must be a non-empty vector")
        if (
            not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all()
        ):
            raise SynchronizationError(
                "PCM callback must contain finite numeric samples"
            )
        if not np.isfinite(timestamp_s):
            raise SynchronizationError("timestamp must be finite")

        expected = self._expected_append_s()
        if expected is not None and abs(timestamp_s - expected) > self._tolerance_s:
            self.reset()
        if self._start_s is None:
            self._start_s = timestamp_s
        self._samples.extend(np.asarray(values, dtype=np.float64).tolist())
        self._next_s = timestamp_s + values.size / self._sample_rate
        return self._drain()

    def _expected_append_s(self) -> float | None:
        return self._next_s

    def _drain(self) -> tuple[AlignedPcmFrame, ...]:
        if not self._locked:
            self._acquire_lock()
        if not self._locked or self._start_s is None:
            return ()

        out: list[AlignedPcmFrame] = []
        n = self.frame_samples
        while len(self._samples) >= n:
            frame = np.fromiter(self._take(n), dtype=np.float64, count=n)
            out.append(
                AlignedPcmFrame(
                    samples=np.asarray(frame, dtype=np.float32),
                    timestamp_s=self._start_s,
                    continuous=not self._fresh_segment,
                )
            )
            self._start_s += n / self._sample_rate
            self._fresh_segment = False
        return tuple(out)

    def _acquire_lock(self) -> None:
        n = self.frame_samples
        if len(self._samples) < n or self._start_s is None:
            return
        values = np.fromiter(self._samples, dtype=np.float64, count=len(self._samples))
        # valid mode gives a score for every possible chirp start without FFT
        # wraparound.  Normalising each candidate by its own energy prevents a
        # loud Android callback from masquerading as a better alignment.
        correlation = scipy.signal.correlate(values, self._reference, mode="valid")
        energy = np.sqrt(
            scipy.signal.convolve(
                values * values,
                np.ones(n, dtype=np.float64),
                mode="valid",
            )
        )
        scores = np.abs(correlation) / np.maximum(energy * self._reference_norm, 1e-12)
        best = int(np.argmax(scores))
        if float(scores[best]) < self._score:
            # Keep enough tail to match a chirp split across the next callback.
            keep = min(len(self._samples), n - 1)
            self._discard(len(self._samples) - keep)
            return
        self._discard(best)
        self._locked = True

    def _discard(self, count: int) -> None:
        if count <= 0:
            return
        if self._start_s is None:
            raise RuntimeError("cannot discard samples without a stream timestamp")
        for _ in range(count):
            self._samples.popleft()
        self._start_s += count / self._sample_rate

    def _take(self, count: int) -> list[float]:
        return [self._samples.popleft() for _ in range(count)]
