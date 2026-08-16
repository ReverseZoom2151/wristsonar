"""Checkpoint provenance. Weights are not useful without their measurement context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wristsonar.model.normalization import FeatureNormalizer, PoseNormalizer
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    PreprocessingDescriptor,
)
from wristsonar.protocol import GroundTruth, Protocol, Split

__all__ = [
    "CHECKPOINT_BUNDLE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CheckpointBundle",
    "CheckpointMetadata",
]

CHECKPOINT_SCHEMA = "wristsonar.checkpoint/2"
"""Version 2 adds the preprocessing descriptor.

Bumped rather than made optional. A version 1 sidecar records the crop and the
window and nothing else about how its windows were built, so there is no
answer to the only question that matters at load time, which is whether the
pipeline about to feed these weights builds the same tensor the training
corpus did. Refusing to read it is the honest response; the alternative is to
default the missing fields and be confidently wrong about a checkpoint nobody
can reconstruct.
"""

CHECKPOINT_BUNDLE_SCHEMA = "wristsonar.checkpoint-bundle/2"


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Everything required to decide whether a weight file can be evaluated."""

    model: str
    dataset: str
    dataset_version: str
    split: Split
    ground_truth: GroundTruth
    calibration_minutes: float
    sha256_data_manifest: str
    preprocessing: PreprocessingDescriptor = WATCHHAND_PREPROCESSING
    """How the training windows this model saw were built.

    Weights are a function of their preprocessing, and until this field
    existed the function's other argument was written down nowhere. Two
    reviewers found the same class of defect three times because of it. Stored
    here so that loading a checkpoint can compare it against the pipeline
    about to run, and refuse rather than produce well formed nonsense.
    """

    schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if (
            not self.model
            or not self.dataset
            or not self.dataset_version
            or not self.sha256_data_manifest
        ):
            raise ValueError("model, dataset version and manifest digest are required")
        if self.calibration_minutes < 0:
            raise ValueError("calibration minutes cannot be negative")

    @property
    def crop_bins(self) -> int:
        """Range bins per window, from the preprocessing contract."""
        return self.preprocessing.crop_bins

    @property
    def window_frames(self) -> int:
        """Frames per window, from the preprocessing contract."""
        return self.preprocessing.window_frames

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
        """JSON-safe form used by a standalone sidecar and a full bundle alike."""
        return {
            "schema": self.schema,
            "model": self.model,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "split": self.split.value,
            "ground_truth": self.ground_truth.value,
            "calibration_minutes": self.calibration_minutes,
            "sha256_data_manifest": self.sha256_data_manifest,
            "preprocessing": self.preprocessing.to_jsonable(),
        }

    @classmethod
    def read(cls, path: Path) -> CheckpointMetadata:
        return cls.from_jsonable(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_jsonable(cls, raw: dict[str, Any]) -> CheckpointMetadata:
        """Parse metadata embedded in a bundle or stored as a sidecar."""
        if raw.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"unsupported checkpoint schema {raw.get('schema')!r}; this "
                f"release reads {CHECKPOINT_SCHEMA}, which records the "
                "preprocessing its weights were trained under"
            )
        return cls(
            model=str(raw["model"]),
            dataset=str(raw["dataset"]),
            dataset_version=str(raw["dataset_version"]),
            split=Split(raw["split"]),
            ground_truth=GroundTruth(raw["ground_truth"]),
            calibration_minutes=float(raw["calibration_minutes"]),
            sha256_data_manifest=str(raw["sha256_data_manifest"]),
            preprocessing=PreprocessingDescriptor.from_jsonable(raw["preprocessing"]),
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

    @property
    def preprocessing(self) -> PreprocessingDescriptor:
        """The window contract these weights were trained under."""
        return self.metadata.preprocessing

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
