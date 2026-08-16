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


def test_gap_forces_a_new_lock_and_marks_the_new_segment_discontinuous() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)
    first = np.tile(reference, 2)
    before_gap = sync.push(first, timestamp_s=0.0)
    assert len(before_gap) == 2

    # A one-second gap is far beyond the 1.5 sample timing tolerance.
    after_gap = sync.push(np.tile(reference, 2), timestamp_s=1.0)
    assert len(after_gap) == 2
    assert not after_gap[0].continuous
    assert after_gap[1].continuous


def test_rejects_bad_callbacks_without_poisoning_later_input() -> None:
    config = ChirpConfig()
    reference = windowed_chirp(config).astype(np.float32)
    sync = PcmSynchronizer(reference, sample_rate=config.sample_rate)
    with pytest.raises(SynchronizationError, match="non-empty"):
        sync.push(np.array([], dtype=np.float32), timestamp_s=0.0)
    frames = sync.push(reference, timestamp_s=0.0)
    assert len(frames) == 1
