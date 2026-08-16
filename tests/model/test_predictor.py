from __future__ import annotations

import numpy as np
import pytest

from wristsonar.model.normalization import FeatureNormalizer, PoseNormalizer
from wristsonar.model.predictor import NormalizedPosePredictor, PredictorError


def _predictor() -> NormalizedPosePredictor:
    features = FeatureNormalizer.fit(np.ones((2, 2, 3, 4), dtype=np.float32))
    poses = np.zeros((2, 21, 3), dtype=np.float32)
    poses[:, :, 0] = np.arange(21, dtype=np.float32) / 100.0
    pose_normalizer = PoseNormalizer.fit(poses)

    def backend(batch: np.ndarray) -> np.ndarray:
        assert batch.shape == (1, 2, 3, 4)
        return np.zeros((1, 21, 3), dtype=np.float32)

    return NormalizedPosePredictor(backend, features=features, poses=pose_normalizer)


def test_predictor_converts_normalized_backend_output_to_physical_pose() -> None:
    output = _predictor()(np.ones((2, 3, 4), dtype=np.float32))

    assert output.shape == (21, 3)
    assert np.array_equal(output[0], np.zeros(3, dtype=np.float32))
    assert output[1, 0] > 0.0


def test_predictor_rejects_wrong_input_or_backend_shape() -> None:
    predictor = _predictor()
    with pytest.raises(PredictorError, match="acoustic window"):
        predictor(np.zeros((3, 3, 4), dtype=np.float32))

    broken = NormalizedPosePredictor(
        lambda batch: np.zeros((21, 3), dtype=np.float32),
        features=FeatureNormalizer.fit(np.ones((1, 2, 3, 4), dtype=np.float32)),
        poses=PoseNormalizer.fit(np.zeros((1, 21, 3), dtype=np.float32)),
    )
    with pytest.raises(PredictorError, match="must return"):
        broken(np.ones((2, 3, 4), dtype=np.float32))
