"""Reject OS audio paths that cannot produce the WatchHand signal contract."""

from __future__ import annotations

from dataclasses import dataclass

from wristsonar.data.watchhand import WATCHHAND_CHIRP

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
    """Validate a stream before its echoes become model input.

    Android may accept a 48 kHz request then silently resample it.  A signal
    model trained on a 600-sample, 48 kHz chirp cannot compensate for that by
    itself: every range bin changes meaning.  This class checks observed frame
    boundaries, not configuration intent.
    """

    def __init__(self) -> None:
        self._rate: int | None = None
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
        if self._rate is None:
            return CaptureHealth(0, 0, 0, 0, False, "no frames observed")
        expected_rate = WATCHHAND_CHIRP.sample_rate
        expected_samples = WATCHHAND_CHIRP.n_samples
        samples = self._last_samples
        if self._rate != expected_rate:
            reason = f"observed {self._rate} Hz; WatchHand requires {expected_rate} Hz"
        elif samples != expected_samples:
            reason = f"observed {samples} samples; chirp requires {expected_samples}"
        elif self._discontinuities:
            reason = f"{self._discontinuities} discontinuities observed"
        else:
            reason = "48 kHz, 600-sample duplex stream is continuous"
        return CaptureHealth(
            self._rate,
            samples,
            self._frames,
            self._discontinuities,
            reason.startswith("48 kHz"),
            reason,
        )
