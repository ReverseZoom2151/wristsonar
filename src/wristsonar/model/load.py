"""Load a provenance-bound Torch checkpoint as a realtime pose predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wristsonar.model.checkpoint import CheckpointBundle
from wristsonar.model.predictor import NormalizedPosePredictor
from wristsonar.model.torch_model import ModelUnavailableError, load_pose_cnn
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    PreprocessingDescriptor,
)

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

    @property
    def preprocessing(self) -> PreprocessingDescriptor:
        """The contract the caller's capture path was checked against."""
        return self.bundle.preprocessing


def load_torch_checkpoint(
    weights: Path,
    bundle_path: Path,
    *,
    preprocessing: PreprocessingDescriptor = WATCHHAND_PREPROCESSING,
) -> LoadedPoseCheckpoint:
    """Verify, load and adapt a Torch checkpoint for `RealtimeInference`.

    `preprocessing` is the contract of the pipeline that is about to feed
    these weights, and loading refuses when it differs from the one the
    checkpoint was trained under. The default is the project's own contract,
    which is what `LiveCaptureProcessor` assembles windows to, so the common
    path is checked rather than exempt. A caller running a non-default capture
    configuration has to say so here, and finds out at load time instead of
    inside a pose stream that looks plausible.

    Verification precedes the optional Torch import. A user missing Torch sees
    a dependency message only after the project has established that the two
    paths really name each other, rather than diagnosing an environment while
    silently accepting the wrong model file.
    """
    bundle = CheckpointBundle.read(bundle_path)
    preprocessing.require_match(
        bundle.preprocessing,
        ours="the capture pipeline",
        theirs=f"checkpoint {bundle_path.name}",
    )
    bundle.verify_weights(weights)
    width = pose_cnn_width(bundle.metadata.model)
    # load_pose_cnn imports Torch on its first line and converts a missing
    # install into ModelUnavailableError, so once it returns the import below
    # cannot fail. This used to sit in a try whose handler re-raised exactly
    # what it caught, which protected nothing while implying the bare import
    # was the line at risk.
    model = load_pose_cnn(weights, width=width)
    import torch

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
