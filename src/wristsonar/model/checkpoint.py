"""Checkpoint provenance. Weights are not useful without their measurement context."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from wristsonar.protocol import GroundTruth, Protocol, Split

__all__ = ["CHECKPOINT_SCHEMA", "CheckpointMetadata"]

CHECKPOINT_SCHEMA = "wristsonar.checkpoint/1"


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Everything required to decide whether a weight file can be evaluated."""

    model: str
    dataset: str
    dataset_version: str
    split: Split
    ground_truth: GroundTruth
    calibration_minutes: float
    crop_bins: int
    window_frames: int
    sha256_data_manifest: str
    schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if (
            not self.model
            or not self.dataset
            or not self.dataset_version
            or not self.sha256_data_manifest
        ):
            raise ValueError("model, dataset version and manifest digest are required")
        if self.calibration_minutes < 0 or self.crop_bins < 1 or self.window_frames < 1:
            raise ValueError("checkpoint dimensions and calibration must be positive")

    def protocol(self, *, subjects: int, held_out_poses: bool = False) -> Protocol:
        return Protocol(
            split=self.split,
            dataset=self.dataset,
            ground_truth=self.ground_truth,
            dataset_version=self.dataset_version,
            subjects=subjects,
            calibration_minutes=self.calibration_minutes,
            held_out_poses=held_out_poses,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), default=lambda x: x.value, indent=2) + "\n"
        )

    @classmethod
    def read(cls, path: Path) -> CheckpointMetadata:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema {raw.get('schema')!r}")
        return cls(
            model=str(raw["model"]),
            dataset=str(raw["dataset"]),
            dataset_version=str(raw["dataset_version"]),
            split=Split(raw["split"]),
            ground_truth=GroundTruth(raw["ground_truth"]),
            calibration_minutes=float(raw["calibration_minutes"]),
            crop_bins=int(raw["crop_bins"]),
            window_frames=int(raw["window_frames"]),
            sha256_data_manifest=str(raw["sha256_data_manifest"]),
        )
