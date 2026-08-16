from __future__ import annotations

import io

import numpy as np
import pytest

from wristsonar.runtime.frames import CaptureFrame, WireFormatError
from wristsonar.runtime.inference import InferenceError, RealtimeInference
from wristsonar.runtime.sinks import JsonLinesSink


def _capture(*, shape: tuple[int, int, int] = (2, 60, 96)) -> CaptureFrame:
    return CaptureFrame(
        samples=np.ones(shape, dtype=np.float32),
        timestamp_s=17.25,
        sample_rate=48_000,
    )


def test_capture_frame_round_trips_its_versioned_wire_format() -> None:
    capture = _capture()

    restored = CaptureFrame.from_jsonable(capture.jsonable())

    assert restored.timestamp_s == capture.timestamp_s
    assert restored.sample_rate == 48_000
    assert np.array_equal(restored.samples, capture.samples)


def test_capture_parser_rejects_pose_output_and_unknown_versions() -> None:
    with pytest.raises(WireFormatError, match=r"wristsonar\.capture"):
        CaptureFrame.from_jsonable({"type": "wristsonar.pose"})
    with pytest.raises(WireFormatError, match="unsupported capture protocol"):
        CaptureFrame.from_jsonable(
            {
                "type": "wristsonar.capture",
                "version": 2,
                "timestamp_s": 0,
                "sample_rate": 48_000,
                "samples": np.zeros((2, 60, 96)).tolist(),
            }
        )


def test_inference_emits_wrist_relative_pose_with_source_timestamp() -> None:
    stream = io.StringIO()
    calls: list[tuple[int, ...]] = []

    def predict(samples: np.ndarray) -> np.ndarray:
        calls.append(samples.shape)
        pose = np.zeros((21, 3), dtype=np.float32)
        pose[:, 0] = np.arange(21, dtype=np.float32) / 100.0
        pose[0] = (4.0, 5.0, 6.0)
        return pose

    runner = RealtimeInference(
        predict=predict,
        sink=JsonLinesSink(stream),
        model_id="pose-cnn/test",
        calibration_minutes=2,
        expected_shape=(2, 60, 96),
        clock=lambda: 20.5,
    )

    result = runner.process(_capture())

    assert calls == [(2, 60, 96)]
    assert result.timestamp_s == 20.5
    assert result.source_timestamp_s == 17.25
    assert np.array_equal(result.joints[0], np.zeros(3, dtype=np.float32))
    assert '"type":"wristsonar.pose"' in stream.getvalue()


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (_capture(shape=(2, 60, 95)), "capture shape"),
        (
            CaptureFrame(
                samples=np.zeros((2, 60, 96), dtype=np.float32),
                timestamp_s=0,
                sample_rate=44_100,
            ),
            "expects 48000 Hz",
        ),
    ],
)
def test_inference_rejects_capture_not_matching_training_contract(
    capture: CaptureFrame, message: str
) -> None:
    runner = RealtimeInference(
        predict=lambda samples: np.zeros((21, 3), dtype=np.float32),
        sink=JsonLinesSink(io.StringIO()),
        model_id="test",
        calibration_minutes=0,
        expected_shape=(2, 60, 96),
    )

    with pytest.raises(InferenceError, match=message):
        runner.process(capture)


def test_inference_rejects_a_model_that_returns_wrong_pose_shape() -> None:
    runner = RealtimeInference(
        predict=lambda samples: np.zeros((20, 3), dtype=np.float32),
        sink=JsonLinesSink(io.StringIO()),
        model_id="test",
        calibration_minutes=0,
        expected_shape=(2, 60, 96),
    )

    with pytest.raises(InferenceError, match="model returned"):
        runner.process(_capture())
