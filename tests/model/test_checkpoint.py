from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wristsonar.model import (
    CheckpointBundle,
    CheckpointMetadata,
    FeatureNormalizer,
    PoseNormalizer,
)
from wristsonar.protocol import GroundTruth, Split


def test_checkpoint_sidecar_round_trips_its_evaluation_context(tmp_path: Path) -> None:
    metadata = CheckpointMetadata(
        model="pose-cnn/1",
        dataset="watchhand",
        dataset_version="watchhand@ecommons-2026-07-07",
        split=Split.CROSS_USER,
        ground_truth=GroundTruth.VIDEO_FITTED,
        calibration_minutes=0,
        crop_bins=60,
        window_frames=96,
        sha256_data_manifest="a" * 64,
    )
    path = tmp_path / "weights.json"
    metadata.write(path)

    restored = CheckpointMetadata.read(path)
    assert restored == metadata
    assert restored.protocol(subjects=40).split is Split.CROSS_USER


def test_checkpoint_bundle_binds_weights_to_normalization_and_protocol(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "pose-cnn.pt"
    weights.write_bytes(b"weights-not-a-placeholder")
    metadata = CheckpointMetadata(
        model="pose-cnn/1",
        dataset="watchhand",
        dataset_version="watchhand@ecommons-2026-07-07",
        split=Split.CROSS_USER,
        ground_truth=GroundTruth.VIDEO_FITTED,
        calibration_minutes=0,
        crop_bins=60,
        window_frames=96,
        sha256_data_manifest="b" * 64,
    )
    feature = FeatureNormalizer.fit(np.ones((2, 2, 60, 96), dtype=np.float32))
    pose = PoseNormalizer.fit(np.zeros((2, 21, 3), dtype=np.float32))

    bundle = CheckpointBundle.for_weights(
        weights,
        metadata=metadata,
        feature_normalizer=feature,
        pose_normalizer=pose,
    )
    path = tmp_path / "pose-cnn.bundle.json"
    bundle.write(path)
    restored = CheckpointBundle.read(path)

    restored.verify_weights(weights)
    assert restored.metadata == metadata
    assert np.array_equal(restored.feature_normalizer.mean, feature.mean)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        restored.verify_weights(weights)
