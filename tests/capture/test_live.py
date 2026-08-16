"""The live host path composes transport, synchronization, and DSP safely."""

from __future__ import annotations

import numpy as np
import pytest

from wristsonar.capture.live import LiveCaptureError, LiveCaptureProcessor
from wristsonar.capture.wire import RawPcmWireFrame
from wristsonar.runtime.frames import CaptureFrame
from wristsonar.signal.chirp import windowed_chirp
from wristsonar.types import ChirpConfig


def _packet(
    samples: np.ndarray, timestamp_ns: int, sample_rate: int = 48_000
) -> RawPcmWireFrame:
    return RawPcmWireFrame(np.asarray(samples, dtype="<i2"), timestamp_ns, sample_rate)


def test_live_processor_reaches_a_causal_window_from_split_watch_callbacks() -> None:
    config = ChirpConfig()
    chirp = np.rint(windowed_chirp(config) * 6_000).astype("<i2")
    # The first frame establishes timing, the next establishes a differential
    # reference, and then 96 pairs fill the temporal model window.
    stream = np.tile(chirp, 100)
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
    assert windows[-1].samples.shape == (2, 60, 96)


def test_live_processor_refuses_the_wrong_observed_sample_rate() -> None:
    processor = LiveCaptureProcessor()
    with pytest.raises(LiveCaptureError, match="requires"):
        processor.push(_packet(np.ones(4, dtype="<i2"), 0, 44_100))
