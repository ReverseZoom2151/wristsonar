"""Checkpoint provenance. Weights are not useful without their measurement context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wristsonar.model.normalization import FeatureNormalizer, PoseNormalizer
from wristsonar.protocol import GroundTruth, Protocol, Split

__all__ = [
    "CHECKPOINT_BUNDLE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CheckpointBundle",
    "CheckpointMetadata",
]

CHECKPOINT_SCHEMA = "wristsonar.checkpoint/1"
CHECKPOINT_BUNDLE_SCHEMA = "wristsonar.checkpoint-bundle/1"


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
            json.dumps(self.to_jsonable(), indent=2) + "\n"
        )

    def to_jsonable(self) -> dict[str, object]:
        """JSON-safe form used by a legacy sidecar and a full bundle alike."""
        value: dict[str, object] = asdict(self)
        value["split"] = self.split.value
        value["ground_truth"] = self.ground_truth.value
        return value

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

    @classmethod
    def from_jsonable(cls, raw: dict[str, Any]) -> CheckpointMetadata:
        """Parse metadata embedded in a bundle or stored as a legacy sidecar."""
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


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    """The complete non-weight state required to use a learned checkpoint.

    Torch, ONNX and TFLite use different binary weight formats.  This sidecar
    stays format-neutral and binds each one to the immutable facts that do not
    belong inside an execution engine: the evaluation protocol, the exact
    dataset manifest and both train-fitted normalization transforms.
    """

    metadata: CheckpointMetadata
    feature_normalizer: FeatureNormalizer
    pose_normalizer: PoseNormalizer
    sha256_weights: str
    schema: str = CHECKPOINT_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CHECKPOINT_BUNDLE_SCHEMA:
            raise ValueError(f"unsupported checkpoint bundle schema {self.schema!r}")
        if len(self.sha256_weights) != 64:
            raise ValueError("sha256_weights must be a 64 character hex digest")

    @staticmethod
    def digest(path: Path) -> str:
        """SHA-256 of weights, streamed so multi-gigabyte artifacts remain safe."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def for_weights(
        cls,
        weights: Path,
        *,
        metadata: CheckpointMetadata,
        feature_normalizer: FeatureNormalizer,
        pose_normalizer: PoseNormalizer,
    ) -> CheckpointBundle:
        if not weights.is_file():
            raise FileNotFoundError(f"checkpoint weights do not exist: {weights}")
        return cls(
            metadata=metadata,
            feature_normalizer=feature_normalizer,
            pose_normalizer=pose_normalizer,
            sha256_weights=cls.digest(weights),
        )

    def verify_weights(self, weights: Path) -> None:
        actual = self.digest(weights)
        if actual != self.sha256_weights:
            raise ValueError(
                "checkpoint weight digest mismatch: expected "
                f"{self.sha256_weights}, got {actual}"
            )

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "metadata": self.metadata.to_jsonable(),
            "feature_normalizer": self.feature_normalizer.to_jsonable(),
            "pose_normalizer": self.pose_normalizer.to_jsonable(),
            "sha256_weights": self.sha256_weights,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_jsonable(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def read(cls, path: Path) -> CheckpointBundle:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != CHECKPOINT_BUNDLE_SCHEMA:
            raise ValueError(
                f"unsupported checkpoint bundle schema {raw.get('schema')!r}"
            )
        try:
            metadata = CheckpointMetadata.from_jsonable(raw["metadata"])
            features = FeatureNormalizer.from_jsonable(raw["feature_normalizer"])
            poses = PoseNormalizer.from_jsonable(raw["pose_normalizer"])
            return cls(
                metadata=metadata,
                feature_normalizer=features,
                pose_normalizer=poses,
                sha256_weights=str(raw["sha256_weights"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid checkpoint bundle: {error}") from error
