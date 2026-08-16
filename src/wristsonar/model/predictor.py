"""Adapt a normalized learned model to the physical realtime pose contract."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from wristsonar.model.normalization import FeatureNormalizer, PoseNormalizer
from wristsonar.types import N_JOINTS

__all__ = ["NormalizedPosePredictor", "PredictorError"]

NormalizedBackend = Callable[[NDArray[np.float32]], NDArray[np.floating]]


class PredictorError(ValueError):
    """A learned backend violated the published tensor contract."""


class NormalizedPosePredictor:
    """Apply fitted transforms around a batch-oriented learned backend.

    The wrapped backend receives one batch of standardized feature windows and
    returns one batch of standardized 21 by 3 poses.  This narrow interface is
    compatible with Torch, ONNX Runtime, TFLite adapters, and test doubles,
    while preserving a single physical contract for the realtime layer.
    """

    def __init__(
        self,
        backend: NormalizedBackend,
        *,
        features: FeatureNormalizer,
        poses: PoseNormalizer,
    ) -> None:
        self._backend = backend
        self._features = features
        self._poses = poses

    def __call__(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        if window.ndim != 3 or window.shape[0] != 2:
            raise PredictorError(
                "one acoustic window must be (2, bins, frames), got "
                f"{window.shape}"
            )
        normalized = self._features.transform(window[None, ...])
        raw = np.asarray(self._backend(normalized), dtype=np.float32)
        if raw.shape != (1, N_JOINTS, 3):
            raise PredictorError(
                "normalized backend must return (1, 21, 3), got "
                f"{raw.shape}"
            )
        if not np.isfinite(raw).all():
            raise PredictorError("normalized backend returned non-finite values")
        return self._poses.inverse(raw)[0]
