from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wristsonar.model.checkpoint import CheckpointBundle, CheckpointMetadata
from wristsonar.model.load import (
    CheckpointLoadError,
    load_torch_checkpoint,
    pose_cnn_width,
)
from wristsonar.model.normalization import FeatureNormalizer, PoseNormalizer
from wristsonar.model.torch_model import ModelUnavailableError
from wristsonar.protocol import GroundTruth, Split


def test_model_identifier_requires_a_version_and_width() -> None:
    assert pose_cnn_width("pose-cnn/1,width=48") == 48
    with pytest.raises(CheckpointLoadError, match="explicit width"):
        pose_cnn_width("pose-cnn/1")
    with pytest.raises(CheckpointLoadError, match="positive"):
        pose_cnn_width("pose-cnn/1,width=0")


def test_loader_verifies_bundle_before_reporting_missing_optional_torch(
    tmp_path: Path,
    without_torch: None,
) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"not-a-torch-file-but-hashed")
    metadata = CheckpointMetadata(
        model="pose-cnn/1,width=32",
        dataset="watchhand",
        dataset_version="v1",
        split=Split.CROSS_USER,
        ground_truth=GroundTruth.VIDEO_FITTED,
        calibration_minutes=0,
        crop_bins=60,
        window_frames=96,
        sha256_data_manifest="a" * 64,
    )
    bundle = CheckpointBundle.for_weights(
        weights,
        metadata=metadata,
        feature_normalizer=FeatureNormalizer.fit(
            np.ones((1, 2, 60, 96), dtype=np.float32)
        ),
        pose_normalizer=PoseNormalizer.fit(np.zeros((1, 21, 3), dtype=np.float32)),
    )
    bundle_path = tmp_path / "model.bundle.json"
    bundle.write(bundle_path)

    with pytest.raises(ModelUnavailableError, match=r"\[train\]"):
        load_torch_checkpoint(weights, bundle_path)

    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_torch_checkpoint(weights, bundle_path)
