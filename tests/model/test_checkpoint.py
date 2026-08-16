from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wristsonar.model import (
    CheckpointBundle,
    CheckpointMetadata,
    FeatureNormalizer,
    PoseNormalizer,
)
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    Normalisation,
    PreprocessingDescriptor,
    PreprocessingMismatchError,
)
from wristsonar.protocol import GroundTruth, Split


def _metadata(
    *,
    preprocessing: PreprocessingDescriptor = WATCHHAND_PREPROCESSING,
    digest: str = "a" * 64,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        model="pose-cnn/1",
        dataset="watchhand",
        dataset_version="watchhand@ecommons-2026-07-07",
        split=Split.CROSS_USER,
        ground_truth=GroundTruth.VIDEO_FITTED,
        calibration_minutes=0,
        sha256_data_manifest=digest,
        preprocessing=preprocessing,
    )


def test_checkpoint_sidecar_round_trips_its_evaluation_context(tmp_path: Path) -> None:
    metadata = _metadata()
    path = tmp_path / "weights.json"
    metadata.write(path)

    restored = CheckpointMetadata.read(path)
    assert restored == metadata
    assert restored.protocol(subjects=40).split is Split.CROSS_USER
    # The crop and the window are no longer independent facts that could
    # disagree with the contract. They are read off it.
    assert restored.preprocessing == WATCHHAND_PREPROCESSING
    assert restored.crop_bins == WATCHHAND_PREPROCESSING.crop_bins
    assert restored.window_frames == WATCHHAND_PREPROCESSING.window_frames


def test_a_checkpoint_records_the_preprocessing_it_was_trained_under(
    tmp_path: Path,
) -> None:
    """Every field of the contract survives the round trip, not just the shapes."""
    trained_under = WATCHHAND_PREPROCESSING.with_bin_zero_offset(
        7
    ).with_differential_lag(1)
    path = tmp_path / "weights.json"
    _metadata(preprocessing=trained_under).write(path)

    restored = CheckpointMetadata.read(path)

    assert restored.preprocessing == trained_under
    assert restored.preprocessing.bin_zero_offset == 7
    assert restored.preprocessing.differential_lag == 1
    assert restored.preprocessing.window_normalisation is Normalisation.PEAK
    assert restored.preprocessing.chirp == WATCHHAND_PREPROCESSING.chirp


def test_a_checkpoint_without_a_preprocessing_record_is_refused(
    tmp_path: Path,
) -> None:
    """A version 1 sidecar cannot say how its windows were built, so it is not read.

    Defaulting the missing contract would be the same defect in a new place: a
    confident answer about a checkpoint nobody can reconstruct.
    """
    legacy = {
        "schema": "wristsonar.checkpoint/1",
        "model": "pose-cnn/1",
        "dataset": "watchhand",
        "dataset_version": "v1",
        "split": Split.CROSS_USER.value,
        "ground_truth": GroundTruth.VIDEO_FITTED.value,
        "calibration_minutes": 0.0,
        "crop_bins": 60,
        "window_frames": 96,
        "sha256_data_manifest": "a" * 64,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        CheckpointMetadata.read(path)


def test_a_non_causal_differential_convention_is_refused_at_read(
    tmp_path: Path,
) -> None:
    """A bundle claiming the earlier-frame convention describes a leaky window."""
    blob = _metadata().to_jsonable()
    preprocessing = json.loads(json.dumps(blob["preprocessing"]))
    preprocessing["differential_phase"] = "earlier"
    blob["preprocessing"] = preprocessing
    path = tmp_path / "leaky.json"
    path.write_text(json.dumps(blob), encoding="utf-8")

    with pytest.raises(PreprocessingMismatchError, match="causal"):
        CheckpointMetadata.read(path)


def test_checkpoint_bundle_binds_weights_to_normalization_and_protocol(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "pose-cnn.pt"
    weights.write_bytes(b"weights-not-a-placeholder")
    metadata = _metadata(digest="b" * 64)
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
    assert restored.preprocessing == WATCHHAND_PREPROCESSING
    assert np.array_equal(restored.feature_normalizer.mean, feature.mean)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        restored.verify_weights(weights)
