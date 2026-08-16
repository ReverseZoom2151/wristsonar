"""The Android callback boundary must never become a fake acoustic feature."""

from __future__ import annotations

import numpy as np
import pytest

from wristsonar.capture.synchronizer import (
    AlignedPcmFrame,
    PcmSynchronizer,
    SynchronizationError,
)
from wristsonar.signal.chirp import windowed_chirp
from wristsonar.types import ChirpConfig


def _stream(reference: np.ndarray, *, lead: int, frames: int) -> np.ndarray:
    return np.concatenate((np.zeros(lead), np.tile(reference, frames), np.zeros(11)))


def _feed(
    sync: PcmSynchronizer,
    source: np.ndarray,
    *,
    sample_rate: int,
    chunk: int,
    jitter_s: float = 0.0,
    seed: int = 7,
) -> list[AlignedPcmFrame]:
    """Push ``source`` in fixed chunks with optionally jittered timestamps."""
    rng = np.random.default_rng(seed)
    out: list[AlignedPcmFrame] = []
    for cursor in range(0, source.size - chunk + 1, chunk):
        offset = float(rng.uniform(-jitter_s, jitter_s)) if jitter_s else 0.0
        out.extend(
            sync.push(
                source[cursor : cursor + chunk],
                timestamp_s=cursor / sample_rate + offset,
            )
        )
    return out


def test_locks_an_arbitrary_callback_boundary_to_the_transmit_cadence() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    source = _stream(reference, lead=137, frames=4)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)

    output: list[AlignedPcmFrame] = []
    cursor = 0
    for size in (103, 701, 211, 985, 1000):
        if cursor >= source.size:
            break
        chunk = source[cursor : cursor + size]
        output.extend(sync.push(chunk, timestamp_s=cursor / config.sample_rate))
        cursor += chunk.size

    assert sync.locked
    assert len(output) == 4
    for frame in output:
        np.testing.assert_allclose(frame.samples, reference, atol=1e-6)
    assert output[0].timestamp_s == pytest.approx(137 / config.sample_rate)
    assert not output[0].continuous
    assert all(frame.continuous for frame in output[1:])


def test_millisecond_timestamp_jitter_does_not_stop_the_stream() -> None:
    """The regression this module was rewritten for.

    The watch stamps a callback after ``AudioRecord.read`` returns, so the
    stamp carries milliseconds of scheduling jitter.  Against the old 1.5
    sample (31 microsecond) tolerance every callback looked like a transport
    gap, every push reset the state, and the synchronizer reported ``locked``
    while emitting an unbroken run of nothing.
    """
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    source = np.tile(reference, 300)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)

    frames = _feed(
        sync, source, sample_rate=config.sample_rate, chunk=1_000, jitter_s=1e-3
    )

    assert sync.locked
    assert len(frames) >= 290
    assert sum(1 for frame in frames if not frame.continuous) == 1
    assert sync.status.gaps == 0
    assert sync.observed_sample_rate == pytest.approx(config.sample_rate, rel=0.01)


def test_a_stream_without_a_chirp_never_claims_to_be_locked() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    noise = np.random.default_rng(3).normal(size=48_000).astype(np.float32)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)

    frames = _feed(sync, noise, sample_rate=config.sample_rate, chunk=1_000)

    assert frames == []
    assert not sync.locked
    status = sync.status
    assert status.frames == 0
    assert status.unlocked_s == pytest.approx(1.0, rel=0.05)
    assert "no chirp" in status.reason


def test_a_silent_sample_drop_is_caught_by_the_periodic_re_lock() -> None:
    """Sample counting alone cannot see a hole the timestamps do not report.

    A dropped ring buffer of a few samples leaves the timestamps contiguous to
    well within any usable tolerance, so only re-scoring the correlation can
    find it.  Without the re-lock the frame grid stays wrong for the rest of
    the session and every range bin means something else.
    """
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    source = np.concatenate(
        (np.tile(reference, 100), np.tile(reference, 100)[173:])
    )
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)

    frames = _feed(sync, source, sample_rate=config.sample_rate, chunk=600)

    assert sync.locked
    assert sync.status.relocks == 1
    assert sum(1 for frame in frames if not frame.continuous) == 2
    # After the correction the emitted frames are chirps again, not a blend of
    # the tail of one chirp and the head of the next.
    np.testing.assert_allclose(frames[-1].samples, reference, atol=1e-5)


def test_a_steady_stream_does_not_manufacture_discontinuities() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    source = np.tile(reference, 400)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate, relock_frames=10)

    frames = _feed(
        sync, source, sample_rate=config.sample_rate, chunk=600, jitter_s=2e-3
    )

    assert sync.status.relocks == 0
    assert sum(1 for frame in frames if not frame.continuous) == 1


def test_integer_and_float_callbacks_produce_the_same_frames() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config)
    integers = np.rint(reference * 20_000).astype("<i2")
    floats = integers.astype(np.float64) / 32_768.0

    from_int = PcmSynchronizer(reference, sample_rate=config.sample_rate).push(
        np.tile(integers, 2), timestamp_s=0.0
    )
    from_float = PcmSynchronizer(reference, sample_rate=config.sample_rate).push(
        np.tile(floats, 2), timestamp_s=0.0
    )

    assert len(from_int) == len(from_float) == 2
    np.testing.assert_allclose(from_int[0].samples, from_float[0].samples, atol=1e-7)
    assert float(np.max(np.abs(from_int[0].samples))) < 1.0


def test_gap_forces_a_new_lock_and_marks_the_new_segment_discontinuous() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)
    first = np.tile(reference, 2)
    before_gap = sync.push(first, timestamp_s=0.0)
    assert len(before_gap) == 2

    # A one-second gap is far beyond any tolerance stamp jitter could justify.
    after_gap = sync.push(np.tile(reference, 2), timestamp_s=1.0)
    assert len(after_gap) == 2
    assert not after_gap[0].continuous
    assert after_gap[1].continuous
    assert sync.status.gaps == 1


def test_rejects_bad_callbacks_without_poisoning_later_input() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)
    with pytest.raises(SynchronizationError, match="non-empty"):
        sync.push(np.array([], dtype=np.float32), timestamp_s=0.0)
    frames = sync.push(reference, timestamp_s=0.0)
    assert len(frames) == 1
