"""Train-only normalization statistics for acoustic windows and hand pose.

The values here are model parameters, not preprocessing convenience.  Fitting
them over training and test data leaks test-session acoustics into a result;
omitting them from a released checkpoint makes a live prediction nonportable.
This module therefore has no ``fit_transform(all_data)`` shortcut and makes
the fitted state explicitly serializable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wristsonar.types import N_JOINTS

__all__ = [
    "FEATURE_NORMALIZATION_SCHEMA",
    "POSE_NORMALIZATION_SCHEMA",
    "FeatureNormalizer",
    "NormalizationError",
    "PoseNormalizer",
]

FEATURE_NORMALIZATION_SCHEMA = "wristsonar.feature-normalization/1"
POSE_NORMALIZATION_SCHEMA = "wristsonar.pose-normalization/1"


class NormalizationError(ValueError):
    """Raised for incompatible, unfitted, or numerically unsafe statistics."""


def _finite(name: str, values: NDArray[np.floating]) -> None:
    if not np.isfinite(values).all():
        raise NormalizationError(f"{name} contains non-finite values")


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    """Per-channel standardization for ``(n, 2, bins, frames)`` windows.

    Per-channel rather than per-bin normalization preserves the relative echo
    structure that represents range.  A per-window normalization would erase
    real amplitude differences along with arbitrary microphone gain, so the
    fitted population statistics are intentionally saved and reused.
    """

    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.mean.shape != (1, 2, 1, 1) or self.scale.shape != (1, 2, 1, 1):
            raise NormalizationError(
                "feature statistics must have shape (1, 2, 1, 1), got "
                f"{self.mean.shape} and {self.scale.shape}"
            )
        _finite("feature mean", self.mean)
        _finite("feature scale", self.scale)
        if self.epsilon <= 0 or np.any(self.scale < self.epsilon):
            raise NormalizationError("feature scale must be at least epsilon")

    @classmethod
    def fit(
        cls, features: NDArray[np.floating], *, epsilon: float = 1e-6
    ) -> FeatureNormalizer:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] != 2:
            raise NormalizationError(
                "features must be nonempty (n, 2, bins, frames), got "
                f"{values.shape}"
            )
        _finite("training features", values)
        mean = values.mean(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
        scale = values.std(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
        return cls(
            mean=np.asarray(mean, dtype=np.float32),
            scale=np.asarray(np.maximum(scale, epsilon), dtype=np.float32),
            epsilon=epsilon,
        )

    def transform(self, features: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 4 or values.shape[1] != 2:
            raise NormalizationError(
                f"features must be (n, 2, bins, frames), got {values.shape}"
            )
        _finite("features", values)
        return np.ascontiguousarray((values - self.mean) / self.scale, dtype=np.float32)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": FEATURE_NORMALIZATION_SCHEMA,
            "epsilon": self.epsilon,
            "mean": self.mean.reshape(2).tolist(),
            "scale": self.scale.reshape(2).tolist(),
        }

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> FeatureNormalizer:
        if value.get("schema") != FEATURE_NORMALIZATION_SCHEMA:
            raise NormalizationError(
                f"unsupported feature normalization {value.get('schema')!r}"
            )
        try:
            return cls(
                mean=np.asarray(value["mean"], dtype=np.float32).reshape(1, 2, 1, 1),
                scale=np.asarray(value["scale"], dtype=np.float32).reshape(1, 2, 1, 1),
                epsilon=float(value["epsilon"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NormalizationError(
                f"invalid feature normalization: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class PoseNormalizer:
    """Train-only standardization for wrist-relative ``(n, 21, 3)`` poses."""

    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        expected = (1, N_JOINTS, 3)
        if self.mean.shape != expected or self.scale.shape != expected:
            raise NormalizationError(
                f"pose statistics must have shape {expected}, got "
                f"{self.mean.shape} and {self.scale.shape}"
            )
        _finite("pose mean", self.mean)
        _finite("pose scale", self.scale)
        if self.epsilon <= 0 or np.any(self.scale < self.epsilon):
            raise NormalizationError("pose scale must be at least epsilon")

    @staticmethod
    def wrist_relative(poses: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(poses, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (N_JOINTS, 3):
            raise NormalizationError(
                f"poses must be (n, {N_JOINTS}, 3), got {values.shape}"
            )
        if values.shape[0] < 1:
            raise NormalizationError("poses must contain at least one frame")
        _finite("poses", values)
        return np.ascontiguousarray(values - values[:, 0:1], dtype=np.float32)

    @classmethod
    def fit(
        cls, poses: NDArray[np.floating], *, epsilon: float = 1e-6
    ) -> PoseNormalizer:
        values = cls.wrist_relative(poses)
        mean = values.mean(axis=0, keepdims=True, dtype=np.float64)
        scale = values.std(axis=0, keepdims=True, dtype=np.float64)
        return cls(
            mean=np.asarray(mean, dtype=np.float32),
            scale=np.asarray(np.maximum(scale, epsilon), dtype=np.float32),
            epsilon=epsilon,
        )

    def transform(self, poses: NDArray[np.floating]) -> NDArray[np.float32]:
        values = self.wrist_relative(poses)
        return np.ascontiguousarray((values - self.mean) / self.scale, dtype=np.float32)

    def inverse(self, normalized: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(normalized, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (N_JOINTS, 3):
            raise NormalizationError(
                f"normalized poses must be (n, {N_JOINTS}, 3), got {values.shape}"
            )
        _finite("normalized poses", values)
        result = (values * self.scale) + self.mean
        # The output contract is wrist-relative even if a network's root output
        # carries harmless numerical residue.
        result = result - result[:, 0:1]
        return np.ascontiguousarray(result, dtype=np.float32)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": POSE_NORMALIZATION_SCHEMA,
            "epsilon": self.epsilon,
            "mean": self.mean.reshape(N_JOINTS, 3).tolist(),
            "scale": self.scale.reshape(N_JOINTS, 3).tolist(),
        }

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> PoseNormalizer:
        if value.get("schema") != POSE_NORMALIZATION_SCHEMA:
            raise NormalizationError(
                f"unsupported pose normalization {value.get('schema')!r}"
            )
        try:
            return cls(
                mean=np.asarray(value["mean"], dtype=np.float32).reshape(
                    1, N_JOINTS, 3
                ),
                scale=np.asarray(value["scale"], dtype=np.float32).reshape(
                    1, N_JOINTS, 3
                ),
                epsilon=float(value["epsilon"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NormalizationError(f"invalid pose normalization: {error}") from error
