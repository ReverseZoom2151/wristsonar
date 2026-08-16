"""Synchronise arbitrary microphone chunks to the known periodic FMCW chirp.

``AudioRecord`` callbacks are buffer boundaries chosen by Android, not chirp
boundaries chosen by Wristsonar.  Feeding them straight to a matched filter
would turn callback scheduling jitter into an apparent hand motion.  This
module establishes an alignment from the direct speaker-to-microphone path,
then emits only whole chirp frames on the transmitted cadence.

Why sample counting, and not timestamps, carries the alignment
==============================================================

The watch stamps a callback on the capture thread, after ``AudioRecord.read``
has returned.  The clock it reads is accurate; the instant at which it manages
to read it is not, because the thread has to be scheduled first.  That is
milliseconds of jitter.  One range bin here is one sample, 20.8 microseconds,
so a single millisecond of stamp jitter is 48 bins of apparent range.

That leaves no usable timestamp tolerance.  A tolerance tight enough to
protect a sample-accurate alignment (the 1.5 samples, 31 microseconds, this
module used to demand) rejects every real callback, drops all state, and
emits nothing while still calling itself locked.  A tolerance loose enough to
accept real callbacks is by construction far too loose to detect a slip.  So
timestamps are demoted: they are no longer the alignment authority.

The sample stream is the authority instead.  Within one recording session
``AudioRecord.read`` returns consecutive samples with no holes; a hole appears
only when the ring buffer overflows, and that is a tens-of-milliseconds event,
not a sub-millisecond one.  Counting samples forward from one correlation lock
is therefore exact, and it is what this module does.

Open-loop counting alone would be a different lie, because it cannot see clock
drift, a dropped buffer, a shifted transmit cadence after an ``AudioTrack``
underrun, or a lock that was false from the start.  So the correlation is
re-scored on a fixed frame cadence.  A small residual slip is corrected in
place; a large one, or a collapsed score, ends the segment honestly.

Timestamps keep two jobs, both coarse enough to survive milliseconds of
jitter: spotting a gross transport or ring-buffer gap, and estimating the
effective sample rate over a multi-second baseline, where the jitter is
divided by seconds instead of by microseconds.

Finally, a stream that cannot lock says so.  ``locked`` is never True while
nothing is being emitted, and ``status`` carries a printable reason that the
caller is expected to surface rather than silently wait on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.signal
from numpy.typing import NDArray

__all__ = [
    "AlignedPcmFrame",
    "PcmSynchronizer",
    "SyncStatus",
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


@dataclass(frozen=True, slots=True)
class SyncStatus:
    """A printable account of why frames are, or are not, coming out.

    Every field is measured rather than assumed, so a caller can distinguish
    "no chirp in this audio at all" from "locked but the room is quiet".
    """

    locked: bool
    score: float
    frames: int
    relocks: int
    gaps: int
    unlocked_s: float
    observed_sample_rate: float | None
    reason: str


class PcmSynchronizer:
    """Acquire, count, and periodically re-verify a chirp alignment.

    The direct acoustic path is deliberately used only for timing.  It is not
    removed here, because the model must see exactly the signal product used
    during training.  A gross timestamp gap drops all partial state and forces
    a new lock rather than stitching unrelated chunks into a plausible false
    frame.

    ``min_sync_score`` is a normalised correlation, so it is comparable across
    devices and volumes.  The default was 0.25, which band-limited noise
    reaches often enough to lock on nothing; the direct speaker-to-microphone
    path is by far the loudest thing in this band on a watch body and scores
    close to 1.  The threshold is not the real defence against a false lock
    though, because a threshold is only consulted once.  ``relock_frames`` is:
    a lock that was wrong, or that has since become wrong, is discovered on
    the next re-score rather than believed for the rest of the session.
    """

    def __init__(
        self,
        reference: NDArray[np.floating],
        *,
        sample_rate: int,
        min_sync_score: float = 0.4,
        gap_tolerance_s: float = 0.030,
        relock_frames: int = 80,
        slip_tolerance_samples: int = 2,
        rate_baseline_s: float = 2.0,
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
        if gap_tolerance_s <= 0.0:
            raise ValueError("gap_tolerance_s must be positive")
        if relock_frames < 1:
            raise ValueError("relock_frames must be positive")
        if not 0 <= slip_tolerance_samples < values.size // 2:
            raise ValueError("slip_tolerance_samples must lie in [0, frame / 2)")
        if rate_baseline_s <= 0.0:
            raise ValueError("rate_baseline_s must be positive")
        self._reference = values
        self._reference_norm = float(np.linalg.norm(values))
        self._sample_rate = sample_rate
        self._score = min_sync_score
        self._gap_tolerance_s = gap_tolerance_s
        self._relock_frames = relock_frames
        self._slip_tolerance = slip_tolerance_samples
        self._rate_baseline_s = rate_baseline_s

        self._pending: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._segment_start_s: float | None = None
        self._next_s: float | None = None
        self._locked = False
        self._fresh_segment = True
        self._last_score = 0.0
        self._frames = 0
        self._relocks = 0
        self._gaps = 0
        self._since_relock = 0
        self._unlocked_samples = 0
        self._rate_anchor_s: float | None = None
        self._rate_samples = 0
        self._observed_rate: float | None = None

    @property
    def frame_samples(self) -> int:
        """The PCM samples in each emitted frame."""
        return int(self._reference.size)

    @property
    def locked(self) -> bool:
        """Whether the next whole frame is on a verified chirp cadence.

        False whenever nothing is being emitted, which is the whole point: the
        previous implementation could report True forever while returning no
        frames at all.
        """
        return self._locked

    @property
    def unlocked_s(self) -> float:
        """Seconds of audio consumed since the last successful lock."""
        return self._unlocked_samples / self._sample_rate

    @property
    def observed_sample_rate(self) -> float | None:
        """Samples delivered per second of wall clock, or None while measuring.

        This is the one thing the host can honestly say about resampling.  The
        rate on the wire is whatever the watch declares, so comparing it to
        48000 compares a constant to a constant.  Dividing the samples that
        actually arrived by the elapsed monotonic time does not, provided the
        baseline is long enough for millisecond stamp jitter to stop mattering.
        """
        return self._observed_rate

    @property
    def status(self) -> SyncStatus:
        """Everything measured about this stream, ready to print."""
        return SyncStatus(
            locked=self._locked,
            score=self._last_score,
            frames=self._frames,
            relocks=self._relocks,
            gaps=self._gaps,
            unlocked_s=self.unlocked_s,
            observed_sample_rate=self._observed_rate,
            reason=self._reason(),
        )

    def reset(self) -> None:
        """Discard timing state after a route change, underrun, or gap.

        Counters survive, because they describe the session rather than the
        segment, and a caller needs them precisely when segments keep dying.
        """
        self._pending = np.zeros(0, dtype=np.float64)
        self._segment_start_s = None
        self._next_s = None
        self._locked = False
        self._fresh_segment = True
        self._since_relock = 0
        self._rate_anchor_s = None
        self._rate_samples = 0

    def push(
        self,
        pcm: NDArray[np.signedinteger] | NDArray[np.floating],
        *,
        timestamp_s: float,
    ) -> tuple[AlignedPcmFrame, ...]:
        """Append one timestamped callback and return any complete chirp frames.

        ``timestamp_s`` is the capture time of the *first* sample in ``pcm``,
        which is what ``wristsonar.capture.wire`` documents and what the Wear
        sender computes.  It is used only to detect a gross gap and to estimate
        the effective sample rate; the alignment itself is carried by counting
        samples.  A gap is not an exception: it starts a new segment and emits
        no frame until the chirp is located again, because that is the only
        response that cannot fabricate timing.

        Integer PCM is scaled to [-1, 1) here, so that a caller feeding int16
        and a caller feeding float get frames on the same scale.
        """
        values = np.asarray(pcm)
        if values.ndim != 1 or values.size == 0:
            raise SynchronizationError("PCM callback must be a non-empty vector")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise SynchronizationError(
                "PCM callback must contain finite numeric samples"
            )
        if not np.isfinite(timestamp_s):
            raise SynchronizationError("timestamp must be finite")

        samples = _to_unit_scale(values)
        if (
            self._next_s is not None
            and abs(timestamp_s - self._next_s) > self._gap_tolerance_s
        ):
            self._gaps += 1
            self.reset()
        if self._segment_start_s is None:
            self._segment_start_s = timestamp_s
        self._observe_rate(timestamp_s, int(samples.size))
        self._pending = np.concatenate((self._pending, samples))
        self._next_s = timestamp_s + samples.size / self._sample_rate

        frames = self._drain()
        if self._locked:
            self._unlocked_samples = 0
        else:
            self._unlocked_samples += int(samples.size)
        return frames

    def _observe_rate(self, timestamp_s: float, size: int) -> None:
        """Accumulate a long-baseline samples-per-second estimate.

        Deliberately anchored at the first callback of the current segment and
        never re-anchored while the segment lives: the longer the baseline, the
        smaller the share of it that stamp jitter can be.
        """
        anchor = self._rate_anchor_s
        if anchor is None:
            self._rate_anchor_s = timestamp_s
            self._rate_samples = size
            return
        span_s = timestamp_s - anchor
        if span_s >= self._rate_baseline_s and self._rate_samples > 0:
            self._observed_rate = self._rate_samples / span_s
        self._rate_samples += size

    def _drain(self) -> tuple[AlignedPcmFrame, ...]:
        out: list[AlignedPcmFrame] = []
        n = self.frame_samples
        while True:
            if not self._locked and not self._acquire_lock():
                break
            if self._since_relock >= self._relock_frames:
                if self._pending.size < 2 * n:
                    break
                self._verify_lock()
                if not self._locked:
                    continue
            start = self._segment_start_s
            if self._pending.size < n or start is None:
                break
            frame = np.asarray(self._pending[:n], dtype=np.float32)
            self._pending = self._pending[n:]
            out.append(
                AlignedPcmFrame(
                    samples=frame,
                    timestamp_s=start,
                    continuous=not self._fresh_segment,
                )
            )
            self._segment_start_s = start + n / self._sample_rate
            self._fresh_segment = False
            self._frames += 1
            self._since_relock += 1
        return tuple(out)

    def _acquire_lock(self) -> bool:
        """Find the chirp anywhere in the pending buffer, or trim and wait."""
        n = self.frame_samples
        if self._pending.size < n or self._segment_start_s is None:
            return False
        offset, score = self._best_offset(self._pending)
        self._last_score = score
        if score < self._score:
            # Keep enough tail to match a chirp split across the next callback.
            keep = min(int(self._pending.size), n - 1)
            self._discard(int(self._pending.size) - keep)
            return False
        self._discard(offset)
        self._locked = True
        self._fresh_segment = True
        self._since_relock = 0
        return True

    def _verify_lock(self) -> None:
        """Re-score the correlation against the cadence we have been counting.

        Without this the lock is asserted once and never questioned again, so
        drift, a dropped buffer, a transmit cadence shifted by an output
        underrun, and a lock that was false from the first frame are all
        indistinguishable from a healthy stream.  Candidates are restricted to
        one frame period because that is the whole ambiguity of a periodic
        transmission.
        """
        n = self.frame_samples
        offset, score = self._best_offset(self._pending[: 2 * n], limit=n)
        self._last_score = score
        self._since_relock = 0
        if score < self._score:
            self._locked = False
            self._relocks += 1
            return
        slip = offset if offset <= n // 2 else offset - n
        if abs(slip) <= self._slip_tolerance:
            # A sample or two is correlation noise on a static direct path, not
            # motion of the frame grid.  Correcting it every second would break
            # continuity far more often than it would improve alignment.
            return
        self._relocks += 1
        self._discard(offset)
        self._fresh_segment = True

    def _best_offset(
        self, values: NDArray[np.float64], *, limit: int | None = None
    ) -> tuple[int, float]:
        """Best chirp start in ``values`` and its normalised correlation score.

        ``valid`` mode gives a score for every possible chirp start without FFT
        wraparound.  Normalising each candidate by its own energy prevents a
        loud Android callback from masquerading as a better alignment.
        """
        n = self.frame_samples
        correlation = scipy.signal.correlate(values, self._reference, mode="valid")
        energy = np.sqrt(
            scipy.signal.convolve(
                values * values,
                np.ones(n, dtype=np.float64),
                mode="valid",
            )
        )
        scores = np.abs(correlation) / np.maximum(energy * self._reference_norm, 1e-12)
        if limit is not None:
            scores = scores[:limit]
        best = int(np.argmax(scores))
        return best, float(scores[best])

    def _discard(self, count: int) -> None:
        if count <= 0:
            return
        if self._segment_start_s is None:
            raise RuntimeError("cannot discard samples without a stream timestamp")
        self._pending = self._pending[count:]
        self._segment_start_s += count / self._sample_rate

    def _reason(self) -> str:
        if self._locked:
            return (
                f"locked at correlation {self._last_score:.2f}; {self._frames} "
                f"frames, {self._relocks} re-locks, {self._gaps} gaps"
            )
        if self._frames == 0:
            return (
                f"no chirp in {self.unlocked_s:.2f} s of audio: best correlation "
                f"{self._last_score:.2f} is below the required {self._score:.2f}. "
                "Check that the watch is transmitting and that nothing in the "
                "capture path is filtering 18 to 21 kHz"
            )
        return (
            f"chirp lock lost after {self._frames} frames and {self._relocks} "
            f"re-locks: best correlation {self._last_score:.2f} is below the "
            f"required {self._score:.2f}"
        )


def _to_unit_scale(values: NDArray[Any]) -> NDArray[np.float64]:
    scaled = np.asarray(values, dtype=np.float64)
    if np.issubdtype(values.dtype, np.integer):
        return np.asarray(scaled / float(np.iinfo(values.dtype).max + 1))
    return scaled
