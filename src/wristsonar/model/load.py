"""Load a provenance-bound Torch checkpoint as a realtime pose predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wristsonar.model.checkpoint import CheckpointBundle
from wristsonar.model.predictor import NormalizedPosePredictor
from wristsonar.model.torch_model import ModelUnavailableError, load_pose_cnn

__all__ = [
    "CheckpointLoadError",
    "LoadedPoseCheckpoint",
    "load_torch_checkpoint",
    "pose_cnn_width",
]

_MODEL_PREFIX = "pose-cnn/1,width="


class CheckpointLoadError(ValueError):
    """A bundle does not name an architecture this release can execute."""


def pose_cnn_width(model_id: str) -> int:
    """Parse the architecture identifier emitted by the training exporter."""
    if not model_id.startswith(_MODEL_PREFIX):
        raise CheckpointLoadError(
            "this loader supports only pose-cnn/1 checkpoints with an explicit "
            f"width, got {model_id!r}"
        )
    try:
        width = int(model_id.removeprefix(_MODEL_PREFIX))
    except ValueError as error:
        raise CheckpointLoadError(f"invalid pose-cnn model id {model_id!r}") from error
    if width < 1:
        raise CheckpointLoadError(f"pose-cnn width must be positive, got {width}")
    return width


@dataclass(frozen=True, slots=True)
class LoadedPoseCheckpoint:
    """A checked artifact and a callable that accepts one physical echo window."""

    bundle: CheckpointBundle
    predictor: NormalizedPosePredictor

    @property
    def model_id(self) -> str:
        return self.bundle.metadata.model


def load_torch_checkpoint(weights: Path, bundle_path: Path) -> LoadedPoseCheckpoint:
    """Verify, load and adapt a Torch checkpoint for `RealtimeInference`.

    Verification precedes the optional Torch import. A user missing Torch sees
    a dependency message only after the project has established that the two
    paths really name each other, rather than diagnosing an environment while
    silently accepting the wrong model file.
    """
    bundle = CheckpointBundle.read(bundle_path)
    bundle.verify_weights(weights)
    width = pose_cnn_width(bundle.metadata.model)
    try:
        model = load_pose_cnn(weights, width=width)
        import torch
    except ModelUnavailableError:
        raise

    def backend(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            output = model(torch.from_numpy(batch))
        return np.asarray(output.detach().cpu().numpy(), dtype=np.float32)

    return LoadedPoseCheckpoint(
        bundle=bundle,
        predictor=NormalizedPosePredictor(
            backend,
            features=bundle.feature_normalizer,
            poses=bundle.pose_normalizer,
        ),
    )
