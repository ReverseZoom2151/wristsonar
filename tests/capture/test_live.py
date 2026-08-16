"""The live host path composes transport, synchronization, and DSP safely."""

from __future__ import annotations

import numpy as np
import pytest

from wristsonar.capture.live import LiveCaptureError, LiveCaptureProcessor
from wristsonar.capture.wire import RawPcmWireFrame
from wristsonar.preprocess import WATCHHAND_PREPROCESSING
from wristsonar.runtime.frames import CaptureFrame
from wristsonar.signal.chirp import windowed_chirp
from wristsonar.types import ChirpConfig


def _packet(
    samples: np.ndarray, timestamp_ns: int, sample_rate: int = 48_000
) -> RawPcmWireFrame:
    return RawPcmWireFrame(np.asarray(samples, dtype="<i2"), timestamp_ns, sample_rate)


def _chirp16() -> np.ndarray:
    return np.rint(windowed_chirp(ChirpConfig()) * 6_000).astype("<i2")


def test_live_processor_reaches_a_causal_window_from_split_watch_callbacks() -> None:
    chirp = _chirp16()
    # The first frame establishes timing, the differential lag consumes a
    # couple more, and then the temporal model window fills.
    stream = np.tile(chirp, 120)
    processor = LiveCaptureProcessor()
    windows: list[CaptureFrame] = []
    cursor = 0
    for size in (211, 389, 777, 223, 1_000) * 100:
        if cursor >= stream.size:
            break
        chunk = stream[cursor : cursor + size]
        windows.extend(processor.push(_packet(chunk, cursor * 1_000_000_000 // 48_000)))
        cursor += chunk.size
    assert windows
    assert windows[-1].samples.shape == (
        2,
        WATCHHAND_PREPROCESSING.crop_bins,
        WATCHHAND_PREPROCESSING.window_frames,
    )


def test_live_processor_survives_realistic_watch_timestamp_jitter() -> None:
    """Milliseconds of stamp jitter used to produce exactly zero windows.

    The watch reads ``SystemClock.elapsedRealtimeNanos`` after
    ``AudioRecord.read`` returns, on a thread that has to be scheduled first.
    A pipeline that only works with a perfect clock does not work.
    """
    chirp = _chirp16()
    stream = np.tile(chirp, 300)
    rng = np.random.default_rng(11)
    processor = LiveCaptureProcessor()
    windows: list[CaptureFrame] = []
    for cursor in range(0, stream.size - 1_000 + 1, 1_000):
        jitter_ns = int(rng.uniform(-1e6, 1e6))
        stamp = 5_000_000 + cursor * 1_000_000_000 // 48_000 + jitter_ns
        windows.extend(processor.push(_packet(stream[cursor : cursor + 1_000], stamp)))

    assert windows
    status = processor.sync_status
    assert status.locked
    assert status.gaps == 0
    assert processor.health.accepted


def test_live_processor_gives_up_loudly_when_no_chirp_is_present() -> None:
    noise = np.rint(
        np.random.default_rng(5).normal(scale=2_000.0, size=60_000)
    ).astype("<i2")
    processor = LiveCaptureProcessor(lock_timeout_s=0.5)
    with pytest.raises(LiveCaptureError, match="synchronisation failed"):
        for cursor in range(0, noise.size, 1_000):
            stamp = cursor * 1_000_000_000 // 48_000
            processor.push(_packet(noise[cursor : cursor + 1_000], stamp))
    assert not processor.sync_status.locked


def test_live_processor_detects_resampling_the_declared_rate_cannot_show() -> None:
    """A resampled path still declares 48000, so only a clock can catch it.

    The packets below claim 48 kHz and are 48 kHz audio, but they arrive at a
    rate of 44100 samples per second of monotonic time.  Comparing the declared
    rate to a constant sees nothing; dividing samples by elapsed time does.
    """
    stream = np.tile(_chirp16(), 200)
    processor = LiveCaptureProcessor()
    with pytest.raises(LiveCaptureError, match="resampling or losing buffers"):
        for cursor in range(0, stream.size - 1_000 + 1, 1_000):
            stamp = cursor * 1_000_000_000 // 44_100
            processor.push(_packet(stream[cursor : cursor + 1_000], stamp))


def test_live_processor_refuses_the_wrong_declared_sample_rate() -> None:
    processor = LiveCaptureProcessor()
    with pytest.raises(LiveCaptureError, match="requires"):
        processor.push(_packet(np.ones(4, dtype="<i2"), 0, 44_100))
