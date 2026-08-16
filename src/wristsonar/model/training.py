"""Optional Torch training and export for the causal acoustic pose baseline.

This module deliberately imports Torch only inside execution paths.  The data
contract, signal tests and evaluation harness must stay usable on a machine
that has not installed a multi-gigabyte ML framework.  When Torch is present,
the implementation keeps every state that affects a result explicit: seed,
training-only normalizers, validation set, optimizer configuration, weights and
the checkpoint bundle that binds them together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wristsonar.model.checkpoint import CheckpointBundle, CheckpointMetadata
from wristsonar.model.normalization import FeatureNormalizer, PoseNormalizer
from wristsonar.model.predictor import NormalizedPosePredictor
from wristsonar.model.torch_model import ModelUnavailableError, create_pose_cnn
from wristsonar.types import N_JOINTS

__all__ = [
    "TrainedPoseModel",
    "TrainingConfig",
    "TrainingError",
    "TrainingHistory",
    "export_torch_checkpoint",
    "train_pose_cnn",
]


class TrainingError(ValueError):
    """Inputs cannot define an honest supervised training run."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """All optimization choices that change a released baseline."""

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 32
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.width < 1:
            raise TrainingError("epochs, batch_size and width must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise TrainingError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if not self.device:
            raise TrainingError("device must not be empty")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """MSE in normalized pose space, one entry per completed epoch."""

    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.train_loss or len(self.train_loss) != len(self.validation_loss):
            raise TrainingError(
                "history must contain one train and validation loss per epoch"
            )
        if not np.isfinite(self.train_loss).all() or not np.isfinite(
            self.validation_loss
        ).all():
            raise TrainingError("training history contains non-finite loss")


@dataclass(slots=True)
class TrainedPoseModel:
    """A trained backend plus the transforms required to use it physically."""

    model: Any
    features: FeatureNormalizer
    poses: PoseNormalizer
    history: TrainingHistory
    width: int

    def predictor(self) -> NormalizedPosePredictor:
        """Build a host-safe, physical-pose predictor from this trained model."""
        try:
            import torch
        except ImportError as error:
            raise ModelUnavailableError(
                "install training support with: pip install -e '.[train]'"
            ) from error

        self.model.eval()

        def backend(batch: NDArray[np.float32]) -> NDArray[np.float32]:
            with torch.inference_mode():
                value = self.model(torch.from_numpy(batch))
            return np.asarray(value.detach().cpu().numpy(), dtype=np.float32)

        return NormalizedPosePredictor(
            backend, features=self.features, poses=self.poses
        )


def _arrays(
    features: NDArray[np.floating], targets: NDArray[np.floating]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 4 or x.shape[0] < 1 or x.shape[1] != 2:
        raise TrainingError(
            f"features must be nonempty (n, 2, bins, frames), got {x.shape}"
        )
    if y.shape != (x.shape[0], N_JOINTS, 3):
        raise TrainingError(
            f"targets must be ({x.shape[0]}, {N_JOINTS}, 3), got {y.shape}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise TrainingError("features and targets must be finite")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def train_pose_cnn(
    train_features: NDArray[np.floating],
    train_targets: NDArray[np.floating],
    validation_features: NDArray[np.floating],
    validation_targets: NDArray[np.floating],
    *,
    config: TrainingConfig | None = None,
) -> TrainedPoseModel:
    """Fit the public causal CNN baseline without leaking validation state.

    Normalizers are fit exclusively on ``train_*``.  Validation drives only a
    reported loss curve; it does not trigger early stopping or architecture
    selection, because silently tuning until a held-out number improves is a
    second training loop under another name.
    """
    active_config = TrainingConfig() if config is None else config
    train_x, train_y = _arrays(train_features, train_targets)
    valid_x, valid_y = _arrays(validation_features, validation_targets)
    if train_x.shape[1:] != valid_x.shape[1:]:
        raise TrainingError(
            "train and validation features disagree on input shape: "
            f"{train_x.shape[1:]} against {valid_x.shape[1:]}"
        )

    try:
        import torch
        from torch.nn import functional as functional
    except ImportError as error:
        raise ModelUnavailableError(
            "install training support with: pip install -e '.[train]'"
        ) from error

    feature_normalizer = FeatureNormalizer.fit(train_x)
    pose_normalizer = PoseNormalizer.fit(train_y)
    x = feature_normalizer.transform(train_x)
    y = pose_normalizer.transform(train_y)
    vx = feature_normalizer.transform(valid_x)
    vy = pose_normalizer.transform(valid_y)

    torch.manual_seed(active_config.seed)
    model = create_pose_cnn(
        channels=2, width=active_config.width, seed=active_config.seed
    )
    device = torch.device(active_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    generator = torch.Generator(device="cpu").manual_seed(active_config.seed)

    for _ in range(active_config.epochs):
        model.train()
        order = torch.randperm(x.shape[0], generator=generator).numpy()
        total = 0.0
        seen = 0
        for start in range(0, x.shape[0], active_config.batch_size):
            rows = order[start : start + active_config.batch_size]
            xb = torch.from_numpy(x[rows]).to(device)
            yb = torch.from_numpy(y[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            count = int(rows.shape[0])
            total += float(loss.detach().cpu()) * count
            seen += count
        train_losses.append(total / seen)

        model.eval()
        with torch.inference_mode():
            validation = functional.mse_loss(
                model(torch.from_numpy(vx).to(device)), torch.from_numpy(vy).to(device)
            )
        validation_losses.append(float(validation.detach().cpu()))

    model.cpu()
    return TrainedPoseModel(
        model=model,
        features=feature_normalizer,
        poses=pose_normalizer,
        history=TrainingHistory(tuple(train_losses), tuple(validation_losses)),
        width=active_config.width,
    )


def export_torch_checkpoint(
    trained: TrainedPoseModel,
    weights: Path,
    bundle: Path,
    *,
    metadata: CheckpointMetadata,
) -> CheckpointBundle:
    """Write CPU-portable Torch weights and their mandatory provenance bundle."""
    expected_model = f"pose-cnn/1,width={trained.width}"
    if metadata.model != expected_model:
        raise TrainingError(
            f"metadata model must be {expected_model!r}, got {metadata.model!r}"
        )
    if metadata.crop_bins < 1 or metadata.window_frames < 1:
        raise TrainingError("metadata crop and window dimensions must be positive")
    try:
        import torch
    except ImportError as error:
        raise ModelUnavailableError(
            "install training support with: pip install -e '.[train]'"
        ) from error

    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained.model.state_dict(), weights)
    artifact = CheckpointBundle.for_weights(
        weights,
        metadata=metadata,
        feature_normalizer=trained.features,
        pose_normalizer=trained.poses,
    )
    artifact.write(bundle)
    return artifact
