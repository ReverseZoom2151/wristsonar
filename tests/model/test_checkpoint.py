from __future__ import annotations

from pathlib import Path

from wristsonar.model import CheckpointMetadata
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
