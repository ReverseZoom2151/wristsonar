"""Record what a capture path actually delivered, frame by frame."""

from __future__ import annotations

from dataclasses import dataclass

from wristsonar.preprocess import WATCHHAND_CHIRP

__all__ = ["CaptureHealth", "DuplexValidator"]


@dataclass(frozen=True, slots=True)
class CaptureHealth:
    """What the capture path has actually delivered, rather than requested."""

    sample_rate: int
    frame_samples: int
    frames: int
    discontinuities: int
    accepted: bool
    reason: str


class DuplexValidator:
    """Bookkeeping for the parts of a stream this layer can honestly see.

    This class used to claim it detected Android silently resampling a 48 kHz
    request.  It cannot, and neither can anything else that only reads the
    declared rate: the watch sends a number, the host compares it to a
    constant, and a resampled stream carries exactly the same number.  Even on
    the device, ``AudioRecord.getSampleRate()`` returns the rate that was
    requested.

    Detecting resampling needs a clock, so it is done where a clock exists:
    ``PcmSynchronizer.observed_sample_rate`` divides the samples that actually
    arrived by elapsed monotonic time, and the Wear sender differences
    ``AudioRecord.getTimestamp()`` frame positions against boot time.  What is
    left for this class is the part it can check truthfully, and it checks only
    that: the declared rate never changes mid-session, the frame size matches
    the chirp, and the segment is continuous.
    """

    def __init__(self) -> None:
        self._rate: int | None = None
        self._last_samples: int | None = None
        self._frames = 0
        self._discontinuities = 0

    def observe(
        self, *, sample_rate: int, frame_samples: int, continuous: bool
    ) -> None:
        if sample_rate <= 0 or frame_samples <= 0:
            raise ValueError("sample rate and frame samples must be positive")
        if self._rate is None:
            self._rate = sample_rate
        elif sample_rate != self._rate:
            self._discontinuities += 1
        if not continuous:
            self._discontinuities += 1
        self._frames += 1
        self._last_samples = frame_samples

    def report(self) -> CaptureHealth:
        if self._rate is None or self._last_samples is None:
            return CaptureHealth(0, 0, 0, 0, False, "no frames observed")
        expected_rate = WATCHHAND_CHIRP.sample_rate
        expected_samples = WATCHHAND_CHIRP.n_samples
        samples = self._last_samples
        accepted = False
        if self._rate != expected_rate:
            reason = f"declared {self._rate} Hz; WatchHand requires {expected_rate} Hz"
        elif samples != expected_samples:
            reason = f"observed {samples} samples; chirp requires {expected_samples}"
        elif self._discontinuities:
            reason = f"{self._discontinuities} discontinuities observed"
        else:
            accepted = True
            reason = (
                f"{expected_rate} Hz, {expected_samples}-sample duplex stream "
                "is continuous"
            )
        return CaptureHealth(
            self._rate,
            samples,
            self._frames,
            self._discontinuities,
            accepted,
            reason,
        )
