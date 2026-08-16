from __future__ import annotations

import numpy as np
import pytest

from wristsonar.capture.processor import CaptureProcessingError, EchoWindowAssembler
from wristsonar.preprocess import WATCHHAND_CHIRP, PreprocessingDescriptor
from wristsonar.signal.chirp import windowed_chirp


def _descriptor(*, crop_bins: int, window_frames: int) -> PreprocessingDescriptor:
    return PreprocessingDescriptor(
        chirp=WATCHHAND_CHIRP,
        crop_bins=crop_bins,
        window_frames=window_frames,
    )


def _chirp() -> np.ndarray:
    return windowed_chirp(WATCHHAND_CHIRP).astype(np.float32)


def test_processor_emits_only_after_a_complete_causal_window() -> None:
    descriptor = _descriptor(crop_bins=8, window_frames=3)
    processor = EchoWindowAssembler(descriptor=descriptor)
    outputs = [
        processor.push(
            _chirp(),
            timestamp_s=float(index) / 80.0,
            sample_rate=48_000,
            continuous=True,
        )
        for index in range(6)
    ]

    # One frame is spent before a difference exists, then the window fills.
    # The descriptor's differential lag does not appear here: it indexes the
    # array WatchHand shipped, and this path builds its own.
    warmup = descriptor.window_frames
    assert outputs[:warmup] == [None] * warmup
    first = outputs[warmup]
    assert first is not None
    assert first.samples.shape == descriptor.window_shape
    assert first.timestamp_s == warmup / 80.0


def test_processor_clears_history_after_discontinuity() -> None:
    descriptor = _descriptor(crop_bins=8, window_frames=2)
    processor = EchoWindowAssembler(descriptor=descriptor)
    processor.push(_chirp(), timestamp_s=0.0, sample_rate=48_000, continuous=True)
    processor.push(_chirp(), timestamp_s=0.0125, sample_rate=48_000, continuous=True)

    assert processor.push(
        _chirp(), timestamp_s=0.025, sample_rate=48_000, continuous=False
    ) is None
    assert processor.push(
        _chirp(), timestamp_s=0.0375, sample_rate=48_000, continuous=True
    ) is None
    assert processor.health.accepted


def test_processor_rejects_an_incompatible_audio_callback() -> None:
    processor = EchoWindowAssembler()
    with pytest.raises(CaptureProcessingError, match="expected 600"):
        processor.push(
            np.zeros(599, dtype=np.int16),
            timestamp_s=0,
            sample_rate=48_000,
            continuous=True,
        )
    with pytest.raises(CaptureProcessingError, match="44100 Hz"):
        processor.push(
            np.zeros(600, dtype=np.int16),
            timestamp_s=0,
            sample_rate=44_100,
            continuous=True,
        )


def test_processor_takes_its_whole_contract_from_the_descriptor() -> None:
    """No local defaults left: the crop, the window and the lag all arrive."""
    descriptor = _descriptor(crop_bins=5, window_frames=2).with_bin_zero_offset(3)
    processor = EchoWindowAssembler(descriptor=descriptor)

    assert processor.descriptor is descriptor
    outputs = [
        processor.push(
            _chirp(),
            timestamp_s=float(index) / 80.0,
            sample_rate=48_000,
            continuous=True,
        )
        for index in range(8)
    ]
    emitted = [item for item in outputs if item is not None]

    assert emitted
    assert all(item.samples.shape == (2, 5, 2) for item in emitted)
    # Peak normalisation is part of the contract, so a live window lands on
    # the scale the training corpus taught the model to expect.
    assert float(np.max(np.abs(emitted[0].samples))) == pytest.approx(1.0)
