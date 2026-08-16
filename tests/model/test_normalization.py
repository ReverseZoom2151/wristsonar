from __future__ import annotations

import numpy as np
import pytest

from wristsonar.model.normalization import (
    FeatureNormalizer,
    NormalizationError,
    PoseNormalizer,
)


def test_feature_normalizer_fits_only_the_given_training_windows() -> None:
    train = np.zeros((2, 2, 3, 4), dtype=np.float32)
    train[:, 0] = 2.0
    train[1, 1] = 4.0
    held_out = np.full((1, 2, 3, 4), 1000.0, dtype=np.float32)

    normalizer = FeatureNormalizer.fit(train)

    assert normalizer.mean.reshape(2).tolist() == [2.0, 2.0]
    assert normalizer.transform(held_out).min() > 400.0
    restored = FeatureNormalizer.from_jsonable(normalizer.to_jsonable())
    assert np.array_equal(restored.mean, normalizer.mean)
    assert np.array_equal(restored.scale, normalizer.scale)


def test_feature_normalizer_rejects_wrong_channel_count() -> None:
    with pytest.raises(NormalizationError, match="nonempty"):
        FeatureNormalizer.fit(np.zeros((2, 1, 3, 4), dtype=np.float32))


def test_pose_normalizer_round_trips_and_removes_global_wrist_translation() -> None:
    pose = np.zeros((2, 21, 3), dtype=np.float32)
    pose[0, :, 0] = np.arange(21, dtype=np.float32) / 100.0
    pose[1] = pose[0] + np.asarray((3.0, -2.0, 7.0), dtype=np.float32)

    normalizer = PoseNormalizer.fit(pose)
    restored = normalizer.inverse(normalizer.transform(pose))

    assert np.allclose(restored[0], restored[1])
    assert np.array_equal(restored[:, 0], np.zeros((2, 3), dtype=np.float32))
    serialized = PoseNormalizer.from_jsonable(normalizer.to_jsonable())
    assert np.array_equal(serialized.mean, normalizer.mean)
    assert np.array_equal(serialized.scale, normalizer.scale)


def test_pose_normalizer_rejects_an_invalid_landmark_shape() -> None:
    with pytest.raises(NormalizationError, match="poses must"):
        PoseNormalizer.fit(np.zeros((1, 20, 3), dtype=np.float32))
