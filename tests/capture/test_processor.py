from __future__ import annotations

import numpy as np
import pytest

from wristsonar.capture.processor import CaptureProcessingError, EchoWindowAssembler
from wristsonar.signal.chirp import windowed_chirp
from wristsonar.types import ChirpConfig


def _chirp() -> np.ndarray:
    return windowed_chirp(
        ChirpConfig(18_000, 21_000, 600 / 48_000, 48_000)
    ).astype(np.float32)


def test_processor_emits_only_after_a_complete_causal_window() -> None:
    processor = EchoWindowAssembler(crop_bins=8, window_frames=3)
    outputs = [
        processor.push(
            _chirp(),
            timestamp_s=float(index) / 80.0,
            sample_rate=48_000,
            continuous=True,
        )
        for index in range(4)
    ]

    assert outputs[:3] == [None, None, None]
    assert outputs[3] is not None
    assert outputs[3].samples.shape == (2, 8, 3)
    assert outputs[3].timestamp_s == 3 / 80.0


def test_processor_clears_history_after_discontinuity() -> None:
    processor = EchoWindowAssembler(crop_bins=8, window_frames=2)
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
